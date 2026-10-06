"""Metadata-only resource inventory and explicit project memory lifecycle.

The catalog is flat. GUI owners supply viewer handles through narrow callbacks;
this service never constructs Qt widgets or loads a cube merely to list it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

from .data_workspace import temporary_storage_usage
from .mapped_archive import array_storage_allocations
from .performance import load_resource_limits
from .project_archive import ArchiveMember
from .raw_dgs_cache import clear_reduced_event_cache

CACHE_EXCLUSIONS_KEY = "resource_cache_exclusions"


@dataclass(frozen=True)
class ResourceRow:
    key: str
    name: str
    kind: str
    state: str
    ram_bytes: int = 0
    reclaimable_bytes: int = 0
    project_disk_bytes: int = 0
    temporary_disk_bytes: int = 0
    users: tuple[str, ...] = ()
    parents: tuple[str, ...] = ()
    children: tuple[str, ...] = ()
    can_unload: bool = False
    can_load: bool = False
    can_delete: bool = False
    reason: str = ""


@dataclass(frozen=True)
class ResourceSnapshot:
    rows: tuple[ResourceRow, ...]
    used_bytes: int
    reserved_bytes: int
    limit_bytes: int
    process_bytes: int
    temporary_directory: str
    temporary_disk_bytes: int
    cpu_limit: int
    ram_limit_mb: int


@dataclass
class _Binding:
    row: ResourceRow
    payload: Any = None
    load: Any = None
    unload: Any = None
    delete: Any = None
    member: str | None = None


@dataclass(frozen=True)
class ResourceLoadPlan:
    """Validated metadata-only load actions prepared before a worker starts."""
    project_identity: int
    entries: tuple[_Binding, ...]


def _node_identity(key):
    while isinstance(key, tuple) and key:
        key = key[0]
    return key


class _StorageInventory:
    """Walk each live payload once and index its shared allocations by owner."""

    def __init__(self):
        self.payloads = {}
        self.sizes = {}
        self.resources = {}
        self.viewers = {}
        self.resource_owners = {}
        self.viewer_owners = {}

    def allocations(self, payload):
        identity = id(payload)
        if identity not in self.payloads:
            # Retain payloads only until this scan finishes so ids cannot be reused.
            values = array_storage_allocations(payload)
            self.payloads[identity] = (payload, values)
            for token, size in values.items():
                self.sizes.setdefault(token, size)
        return self.payloads[identity][1]

    def size(self, payload):
        return sum(self.allocations(payload).values())

    def add(self, key, payload, *, viewer=False):
        values = self.allocations(payload)
        targets, owners = ((self.viewers, self.viewer_owners) if viewer else
                           (self.resources, self.resource_owners))
        targets[key] = values
        for token in values:
            owners.setdefault(token, set()).add(key)

    def related(self, key, *, viewers=False):
        owners = self.viewer_owners if viewers else self.resource_owners
        return set().union(*(owners.get(token, ()) for token in self.resources[key]))

    def reclaimable(self, key, removed_viewers):
        # Closing a sharing viewer can also release its private transformed arrays.
        tokens = set(self.resources[key])
        for index in removed_viewers:
            tokens.update(self.viewers[index])
        return sum(self.sizes[token] for token in tokens
                   if self.resource_owners.get(token, set()) <= {key}
                   and self.viewer_owners.get(token, set()) <= removed_viewers)


def excluded_project_cache(project, member):
    """Whether an explicit cache deletion should survive the next project save."""
    return str(member) in project.settings.get(CACHE_EXCLUSIONS_KEY, ())


def project_binning_member(project, kind, group, target, binning_id):
    """Resolve the persisted identity of a configured binning without loading it."""
    from .project_archive import binning_artifact_member
    if kind == "dataset":
        cache_id = f"dataset-{target.id}-{binning_id}"
    else:
        node = getattr(target, "node", None)
        owner = node.id if node is not None else f"root-{project.data_groups.index(group)}"
        cache_id = f"composite-{owner}-{binning_id}"
    return binning_artifact_member(cache_id)


class ProjectResources:
    """Inspect, load, unload, or forget recomputable project resources.

    ``viewer_payloads`` returns ``(viewer label, numerical payload)`` pairs;
    ``release_viewers`` closes/detaches viewers sharing selected payloads.
    ``busy`` prevents resource mutation while a project operation is active.
    These callbacks make the same lifecycle available to scripts without Qt.
    """

    def __init__(self, project, *, viewer_payloads=lambda: (), release_viewers=lambda values: None,
                 busy=lambda: False, changed=lambda: None, recipes=lambda: ()):
        self.project = project
        self.viewer_payloads = viewer_payloads
        self.release_viewers = release_viewers
        self.busy = busy
        self.changed = changed
        self.recipes = recipes
        self._bindings: dict[str, _Binding] = {}
        self._disk_index = None
        self._disk_members = {}
        self._temporary_sizes = {}

    def _temporary_size(self, path, identity):
        key = (str(path), identity)
        if key not in self._temporary_sizes:
            try:
                self._temporary_sizes[key] = Path(path).stat().st_size
            except OSError:
                return 0
        return self._temporary_sizes[key]

    def _members(self):
        path = getattr(self.project, "_project_path", None)
        try:
            stat = Path(path).stat()
            identity = (str(path), stat.st_ino, stat.st_size, stat.st_mtime_ns)
        except (OSError, TypeError):
            return {}
        if identity != self._disk_index:
            try:
                with ZipFile(path) as archive:
                    self._disk_members = {info.filename: info.compress_size for info in archive.infolist()}
                self._disk_index = identity
            except (OSError, BadZipFile):
                return {}
        return self._disk_members

    def snapshot(self):
        """Return current rows and deduplicated bytes without decoding archives."""
        from . import project_composites as comp
        from . import project_data
        from .project_caches import MODEL_OVERLAY_CACHE
        from .resource_budget import snapshot_memory

        storage = _StorageInventory()
        _bytes = storage.size
        bindings = {}
        members = self._members()
        recipes = tuple(self.recipes())
        caches = (("binning", project_data._VIEWER_VIEW_CACHE),
                  ("composite", comp._COMPOSITE_DATA_CACHE))
        records = {id(cache): cache.resource_records() for _name, cache in caches}
        labels = {member: label for _cache, _key, label, member, _loader in recipes}
        recipe_loaders = {(id(cache), key): loader for cache, key, _label, _member, loader in recipes}
        datasets = {}
        nodes = {}
        for group in self.project.data_groups:
            for node in (group, *group.iter_subgroups()):
                nodes[id(node)] = node
            for dataset in group.iter_datasets():
                datasets[dataset.id] = dataset
                source = dataset.metadata.get("source_file")
                analysis = dataset.metadata.get("analysis_output") or dataset.metadata.get("derived_from_analysis")
                kind = "Analysis output" if analysis else "Source data"
                artifact_member = dataset.metadata.get("project_artifact_path") or dataset.metadata.get("analysis_artifact_path")
                if artifact_member:
                    backing_available = artifact_member in members
                else:
                    backing_available = bool(source and Path(source).expanduser().is_file())
                reloadable = bool(backing_available and dataset.data_matches_source)
                from .project_imports import _source_availability
                can_load = backing_available and (bool(artifact_member)
                    or _source_availability(dataset, operation="load")[1])
                reason = ""
                if dataset.data is not None and not reloadable:
                    reason = ("The source backing is unavailable; retain this loaded copy until it is saved or exported."
                              if dataset.data_matches_source else "Unsaved source data: save or export before unloading.")
                elif not backing_available and (source or artifact_member):
                    reason = "The source backing is unavailable. Restore its file or project artifact before loading."
                key = f"source:{dataset.id}"
                bindings[key] = _Binding(ResourceRow(
                    key, dataset.name, kind, "In RAM" if dataset.data is not None else "Not loaded",
                    ram_bytes=_bytes(dataset.data), parents=(group.name,),
                    project_disk_bytes=members.get(artifact_member, 0),
                    can_unload=dataset.data is not None and reloadable,
                    can_load=dataset.data is None and can_load,
                    reason=reason,
                ), dataset.data, load=lambda d=dataset: self._load_source(d), unload=dataset.unload_data)
                cache = getattr(dataset, "_raw_dgs_reduction_cache", None)
                if cache is not None:
                    content = cache.content
                    member = content.member if isinstance(content, ArchiveMember) else None
                    disk = members.get(member, 0) if member else 0
                    temporary = 0
                    if isinstance(content, Path):
                        temporary = getattr(cache, "disk_bytes", 0) or self._temporary_size(content, id(cache))
                    key = f"reduction:{dataset.id}"
                    bindings[key] = _Binding(ResourceRow(
                        key, dataset.name, "Reduced events", "Saved in project" if member else "Temporary cache",
                        project_disk_bytes=disk, temporary_disk_bytes=temporary, parents=(group.name,),
                        can_delete=True, reason="Reduced events are streamed in bounded chunks; they are not loaded as a whole.",
                    ), delete=lambda d=dataset: clear_reduced_event_cache(d), member=member)

        for cache_name, cache in caches:
            for record in records[id(cache)]:
                owner = None
                if cache_name == "binning" and isinstance(record.key, str):
                    owner = datasets.get(record.key.split(":", 1)[0])
                elif cache_name == "composite":
                    owner = nodes.get(_node_identity(record.key))
                if owner is None:
                    continue
                key = f"{cache_name}:{record.key!r}"
                member = record.project_member
                loaded = record.data is not None
                disk = members.get(member, 0) if member else 0
                temporary = 0
                if record.disk_path:
                    temporary = self._temporary_size(record.disk_path, record.signature)
                kind = "Prepared view" if "viewer-" in str(record.key) else "Histogram"
                recipe_loader = recipe_loaders.get((id(cache), record.key))
                state = "In RAM" if loaded else ("Saved in project" if member else "Temporary cache")
                if not loaded and not record.available:
                    state = "Needs recompute" if recipe_loader else "Unavailable"
                label = record.label
                if label.startswith("Cached histogram "):
                    label = labels.get(member, owner.name)
                    metadata = getattr(record.data, "metadata", {})
                    binning_name = metadata.get("binning_name")
                    if binning_name and binning_name not in label:
                        label = f"{label} · {binning_name}"
                    if isinstance(record.key, tuple) and len(record.key) > 1:
                        stage = str(record.key[1])
                        if stage in ("unsubtracted", "background", "viewer-finish", "viewer-source", "viewer-prepared"):
                            stage = {"viewer-finish": "spectral view", "viewer-source": "prepared source",
                                     "viewer-prepared": "prepared view"}.get(stage, stage)
                            label = f"{label} ({stage})"
                    elif kind == "Prepared view":
                        label = f"{label} (prepared view)"
                bindings[key] = _Binding(ResourceRow(
                    key, label, kind, state,
                    ram_bytes=_bytes(record.data) + record.compressed_bytes,
                    project_disk_bytes=disk, temporary_disk_bytes=temporary,
                    parents=(owner.name,), can_unload=loaded or bool(record.compressed_bytes),
                    can_load=not loaded and bool(recipe_loader or
                        record.available and (member or temporary or record.compressed_bytes)),
                    can_delete=bool(member or temporary),
                    reason="" if record.available else "The saved cache is unavailable; recompute from its recipe.",
                ), record.data, load=lambda c=cache, k=record.key, replay=recipe_loader: self._load_cached(c, k, replay),
                   unload=lambda c=cache, k=record.key: c.unload(k),
                   delete=lambda c=cache, k=record.key: c.delete_backing(k), member=member)

        represented_members = {binding.member for binding in bindings.values() if binding.member}
        available_keys = {identity: {record.key for record in cache_records if record.available}
                          for identity, cache_records in records.items()}
        for recipe in recipes:
            cache, cache_key, label, member, loader = recipe
            if member in represented_members:
                continue
            # A runtime cube is already represented by its cache row.
            if cache_key in available_keys.get(id(cache), ()):
                continue
            key = f"recipe:{member}"
            bindings[key] = _Binding(ResourceRow(key, label, "Histogram", "Needs recompute",
                can_load=True, reason="Load computes this binning from its retained recipe and sources."),
                load=loader, member=member)
            represented_members.add(member)

        for dataset in datasets.values():
            prepared = project_data._PREPARED_POINT_LIST_CACHE.get(dataset.id)
            if prepared is not None:
                key = f"points:{dataset.id}"
                bindings[key] = _Binding(ResourceRow(key, dataset.name, "Prepared points", "In RAM",
                    ram_bytes=_bytes(prepared), parents=(dataset.name,), can_unload=True), prepared,
                    unload=lambda d=dataset: project_data._PREPARED_POINT_LIST_CACHE.pop(d.id, None))
        for group in self.project.data_groups:
            overlay = MODEL_OVERLAY_CACHE.get(id(group))
            if overlay is not None:
                key = f"overlay:{id(group)}"
                bindings[key] = _Binding(ResourceRow(key, group.name, "Model overlays", "In RAM",
                    ram_bytes=_bytes(overlay), parents=(group.name,), can_unload=True), overlay,
                    unload=lambda g=group: MODEL_OVERLAY_CACHE.pop(id(g), None))
            for name, model in group.models.items():
                payload = getattr(model, "_nfit_electronic_model_cache", None)
                if payload is not None:
                    key = f"model:{id(model)}"
                    bindings[key] = _Binding(ResourceRow(key, name, "Evaluated model", "In RAM",
                        ram_bytes=_bytes(payload), parents=(group.name,), can_unload=True), payload,
                        unload=lambda m=model: m.__dict__.pop("_nfit_electronic_model_cache", None))
            if _bytes(group.fits):
                key = f"fits:{id(group)}"
                bindings[key] = _Binding(ResourceRow(key, group.name, "Fit results", "In RAM",
                    ram_bytes=_bytes(group.fits), parents=(group.name,),
                    reason="Fit results are project content, not disposable caches."), group.fits)
            for analysis in group.analyses:
                if _bytes(analysis):
                    key = f"analysis:{analysis.id}"
                    bindings[key] = _Binding(ResourceRow(key, analysis.name, "Analysis results", "In RAM",
                        ram_bytes=_bytes(analysis), parents=(group.name,),
                        reason="Analysis records are project content; output datasets have separate rows."), analysis)

        from . import electronic_structure
        for name, store, lock in (
            ("Fourier coefficients", electronic_structure._FOURIER_COEFFICIENT_CACHE, electronic_structure._FOURIER_CACHE_LOCK),
            ("Hamiltonian components", electronic_structure._HAMILTONIAN_COMPONENT_CACHE, electronic_structure._HAMILTONIAN_COMPONENT_CACHE_LOCK),
        ):
            with lock:
                payload = tuple(store.values())
            if _bytes(payload):
                key = f"shared-model:{name}"
                def unload(s=store, guard=lock):
                    with guard:
                        s.clear()
                bindings[key] = _Binding(ResourceRow(key, name, "Shared model cache", "In RAM",
                    ram_bytes=_bytes(payload), can_unload=True,
                    reason="Shared evaluated intermediates may be recomputed by models in any open project."),
                    payload, unload=unload)

        viewers = tuple(self.viewer_payloads())
        for key, binding in bindings.items():
            storage.add(key, binding.payload)
        for index, (label, payload) in enumerate(viewers):
            storage.add(index, payload, viewer=True)
            if any(storage.resource_owners.get(token) for token in storage.viewers[index]):
                continue
            key = f"viewer:{index}:{id(payload)}"
            bindings[key] = _Binding(ResourceRow(key, label, "Viewer data", "In RAM",
                ram_bytes=_bytes(payload), users=(label,), can_unload=True,
                reason="Removing this data closes its viewer; saved plot settings are retained."), payload,
                unload=lambda: None)
            storage.add(key, payload)
        compressed = sum(max(0, b.row.ram_bytes - _bytes(b.payload)) for b in bindings.values()
                         if b.row.kind in ("Histogram", "Prepared view"))
        used_arrays = sum(storage.sizes.values())
        used = used_arrays + compressed
        rows = []
        order = {key: index for index, key in enumerate(bindings)}
        busy = self.busy()
        for binding in bindings.values():
            row = binding.row
            sharing_viewers = storage.related(row.key, viewers=True)
            users = tuple(viewers[index][0] for index in sorted(sharing_viewers)) if binding.payload is not None else row.users
            freed = 0
            if row.can_unload:
                # Viewers sharing this row will be closed before its owner releases it.
                freed = storage.reclaimable(row.key, sharing_viewers)
                if row.kind in ("Histogram", "Prepared view"):
                    freed += max(0, row.ram_bytes - _bytes(binding.payload))
            related = storage.related(row.key) - {row.key}
            children = tuple(bindings[key].row.name for key in sorted(related, key=order.__getitem__))
            rows.append(replace(row, reclaimable_bytes=freed, users=users, children=children,
                can_unload=row.can_unload and not busy, can_load=row.can_load and not busy,
                can_delete=row.can_delete and not busy,
                reason="An operation is using project resources." if busy else row.reason))
        # Store callbacks and identities, never numerical payloads between scans.
        self._bindings = {key: replace(binding, payload=None) for key, binding in bindings.items()}
        memory = snapshot_memory()
        limits = load_resource_limits()
        temporary = sum(size for _, size in temporary_storage_usage())
        directory = self.project.settings.get("temporary_storage_directory") or ""
        return ResourceSnapshot(tuple(rows), used, memory.reserved_bytes, memory.limit_bytes,
            memory.process_bytes, directory, temporary, limits["cpu_limit"], limits["ram_limit_mb"])

    @staticmethod
    def _load_source(dataset):
        from .project_dataset_io import _load_nfit_dataset_file
        from .project_imports import _ensure_dataset_data_loaded
        return _ensure_dataset_data_loaded(dataset, dataset_file_loader=_load_nfit_dataset_file)

    @staticmethod
    def _load_cached(cache, key, replay=None):
        result = cache.get(key)
        if result is not None:
            return result
        if replay is not None:
            return replay()
        raise OSError("The saved cache is unavailable. Recompute its binning from the retained sources.")

    def _selected(self, keys, capability):
        if self.busy():
            raise RuntimeError("Wait for the active project operation before managing resources.")
        self.snapshot()
        selected = [self._bindings[key] for key in keys]
        if any(not getattr(b.row, capability) for b in selected):
            raise ValueError("The selected resources do not all support this action.")
        return selected

    def unload(self, keys):
        """Release selected memory, closing users first; retain all disk backings."""
        selected = self._selected(keys, "can_unload")
        # Reconstruct the selected payload references only for this action.
        values = self._selected_payloads(keys)
        self.release_viewers(values)
        for binding in selected:
            binding.unload()
        # The inventory itself must not keep the released payloads alive.
        self._bindings.clear()

    def load(self, keys):
        """Decode selected resources through the same guarded public loaders."""
        self.execute_load(self.prepare_load(keys))

    def prepare_load(self, keys):
        """Validate a selection before locking the GUI for its background load."""
        return ResourceLoadPlan(id(self.project), tuple(self._selected(keys, "can_load")))

    def execute_load(self, plan, *, notify=True):
        """Execute a validated load plan and report whether saved settings changed.

        GUI workers set ``notify=False`` and notify their GUI owner after joining.
        Scripting callers retain the ordinary synchronous change callback.
        """
        if plan.project_identity != id(self.project):
            raise ValueError("The load plan belongs to a different project.")
        changed = False
        for binding in plan.entries:
            binding.load()
            if binding.member:
                exclusions = set(self.project.settings.get(CACHE_EXCLUSIONS_KEY, ()))
                if binding.member in exclusions:
                    exclusions.remove(binding.member)
                    self.project.settings[CACHE_EXCLUSIONS_KEY] = sorted(exclusions)
                    changed = True
        self._bindings.clear()
        if changed and notify:
            self.changed()
        return changed

    def delete(self, keys):
        """Forget selected cache backings; the next save removes project members.

        Loaded cubes remain usable. Source data, models, fit results, and analysis
        artifacts are never deleted by this cache action.
        """
        exclusions = set(self.project.settings.get(CACHE_EXCLUSIONS_KEY, ()))
        for binding in self._selected(keys, "can_delete"):
            binding.delete()
            if binding.member:
                exclusions.add(binding.member)
        self.project.settings[CACHE_EXCLUSIONS_KEY] = sorted(exclusions)
        self._bindings.clear()
        self.changed()

    def _selected_payloads(self, keys):
        from . import project_composites, project_data
        selected = set(keys)
        values = []
        for name, cache in (("binning", project_data._VIEWER_VIEW_CACHE),
                            ("composite", project_composites._COMPOSITE_DATA_CACHE)):
            for record in cache.resource_records():
                if f"{name}:{record.key!r}" in selected and record.data is not None:
                    values.append(record.data)
        for group in self.project.data_groups:
            for dataset in group.iter_datasets():
                if f"source:{dataset.id}" in selected and dataset.data is not None:
                    values.append(dataset.data)
                if f"points:{dataset.id}" in selected:
                    values.append(project_data._PREPARED_POINT_LIST_CACHE.get(dataset.id))
            from .project_caches import MODEL_OVERLAY_CACHE
            if f"overlay:{id(group)}" in selected:
                values.append(MODEL_OVERLAY_CACHE.get(id(group)))
            for model in group.models.values():
                if f"model:{id(model)}" in selected:
                    values.append(getattr(model, "_nfit_electronic_model_cache", None))
        from . import electronic_structure
        for name, store in (("Fourier coefficients", electronic_structure._FOURIER_COEFFICIENT_CACHE),
                            ("Hamiltonian components", electronic_structure._HAMILTONIAN_COMPONENT_CACHE)):
            if f"shared-model:{name}" in selected:
                values.append(tuple(store.values()))
        for index, (_label, payload) in enumerate(self.viewer_payloads()):
            if f"viewer:{index}:{id(payload)}" in selected:
                values.append(payload)
        return tuple(values)
