"""Lossless, lazy project selections shared between independent processes.

The clipboard contains configuration or an archive reference, never Python
pickles. The destination adopts an independent file before installing a
selection, so closing its source cannot remove pasted numerical data.
"""

from __future__ import annotations

import copy
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from .analysis.artifacts import write_dataset_artifact
from .cache_utils import dataset_content_signature
from .data_workspace import record_temporary_file, temporary_data_directory
from .operation_control import report_operation
from .pipeline import BackgroundSpec, DataGroup, DatasetEntry, DatasetGroup, MaskSpec
from .project_archive import (
    ArchiveMember,
    dataset_artifact_member,
    read_project_manifest,
    write_project_manifest,
)
from .project_clipboard import (
    ProjectClipboardPayload,
    _clone_background,
    _snapshot_dataset_group,
    _snapshot_item,
    paste_payload,
)
from .project_io import NfitProject, _project_from_manifest, _project_manifest_for_save
from .raw_dgs_cache import bind_project_reduced_event_caches, project_reduced_event_artifacts

TRANSFER_SCHEMA = "nfit.project-selection"
TRANSFER_VERSION = 1
TRANSFER_MIME = "application/x-nfit-project-selection+json"
_ORIGIN = "project_transfer_source_ids"


@dataclass
class ProjectSelection:
    project: NfitProject
    payload: ProjectClipboardPayload
    description: dict[str, Any]
    path: Path | None = None
    owner: Any = None
    source_project: NfitProject | None = None


def _snapshot_workspace(source):
    result = copy.copy(source)
    result.datasets = [_snapshot_item(item) for item in source.datasets]
    result.subgroups = [_snapshot_dataset_group(item) for item in source.subgroups]
    for name in ("metadata", "masks", "lattice_parameters", "models", "fits", "analyses", "plots", "resolution"):
        if hasattr(source, name):
            setattr(result, name, copy.deepcopy(getattr(source, name)))
    result.backgrounds = [_clone_background(item) for item in source.backgrounds]
    for original, snapshot in zip(source.iter_datasets(), result.iter_datasets(), strict=True):
        snapshot._data_revision = original.data_revision
    return result


def _outer_masks(root, selected, inherited=()):
    if root is selected:
        return list(inherited)
    if any(item is selected for item in root.datasets):
        return [*inherited, *root.masks]
    for child in root.subgroups:
        found = _outer_masks(child, selected, (*inherited, *root.masks))
        if found is not None:
            return found
    return None


def prepare_project_selection(project, role, items, *, source_group=None):
    """Snapshot selected configuration and its background dependencies.

    Numerical arrays remain immutable and shared until an archive is written.
    Complete workspaces retain analyses, fits, plots and their dependencies.
    """
    items = tuple(items)
    if not items:
        raise ValueError("No project objects selected")
    kind = {"datasets": "dataset_tree", "dataset_page": "dataset", "masks": "mask",
            "group_mask": "mask", "group_masks": "mask", "backgrounds": "background",
            "group_background": "background", "group_backgrounds": "background",
            "models": "model", "fits": "fit", "fit_timeline": "fit",
            "analyses": "analysis", "plots": "plot"}.get(role, role)
    if kind == "group":
        roots = [_snapshot_workspace(item) for item in items]
        selected = roots
        source_roots = list(items)
    else:
        if source_group is None:
            source_group = next((root for root in project.data_groups
                                 if any(item is candidate for item in items
                                        for candidate in [*root.iter_datasets(), *root.iter_subgroups(),
                                                          *root.models.values(), *root.fits, *root.analyses, *root.plots])), None)
        root = DataGroup("Copied selection")
        if source_group is not None:
            root.name = source_group.name
            root.lattice_parameters = copy.deepcopy(source_group.lattice_parameters)
            root.spacegroup = source_group.spacegroup
            root.masks = copy.deepcopy(source_group.masks)
            root.metadata = copy.deepcopy(source_group.metadata)
        if kind == "dataset_tree":
            root.masks = []
        roots = [root]
        source_roots = [source_group or root]
        selected = [_snapshot_item(item) for item in items]
        for original, item in zip(items, selected, strict=True):
            if isinstance(item, DatasetEntry):
                if source_group is not None and kind != "dataset_tree":
                    item.masks = copy.deepcopy(_outer_masks(source_group, original) or []) + item.masks
                root.datasets.append(item)
            elif isinstance(item, DatasetGroup):
                if source_group is not None and kind != "dataset_tree":
                    outer = _outer_masks(source_group, original) or []
                    item.masks = copy.deepcopy(outer) + item.masks
                root.subgroups.append(item)
            elif kind == "dataset_tree" and isinstance(item, MaskSpec):
                root.masks.append(item)
            elif kind == "dataset_tree" and isinstance(item, BackgroundSpec):
                root.backgrounds.append(item)
            elif kind == "mask":
                # Context masks do not become additional selected objects.
                root.masks = selected
            elif kind == "background":
                root.backgrounds = selected
            elif kind == "model":
                root.models[item.name] = item
            elif kind == "fit":
                root.fits = selected
            elif kind == "analysis":
                root.analyses = selected
            elif kind == "plot":
                root.plots = selected
            else:
                raise ValueError(f"Unsupported project selection: {kind}")
        if kind in {"dataset", "dataset_tree", "dataset_group"} and source_group is not None:
            # Include measured background sources, not the unrelated runs of a
            # workspace. Iterating to a fixed point also handles nested links.
            originals = {item.id: item for item in source_group.iter_datasets()}
            groups = {item.id: item for item in source_group.iter_subgroups()}
            while True:
                owned = [root, *root.iter_subgroups(), *root.iter_datasets()]
                known_data = {item.id for item in root.iter_datasets()}
                known_groups = {item.id for item in root.iter_subgroups()}
                changed = False
                for owner in owned:
                    for background in owner.backgrounds:
                        if background.source_group_id and background.source_group_id not in known_groups:
                            source = groups.get(background.source_group_id)
                            if source is None:
                                raise ValueError(f"Missing background group: {background.name}")
                            cloned = _snapshot_dataset_group(source)
                            cloned.masks = copy.deepcopy(_outer_masks(source_group, source) or []) + cloned.masks
                            root.subgroups.append(cloned)
                            selected.append(cloned)
                            known_groups.update([cloned.id, *(child.id for child in cloned.iter_subgroups())])
                            known_data.update(child.id for child in cloned.iter_datasets())
                            changed = True
                        elif background.source_dataset_id and background.source_dataset_id not in known_data:
                            source = originals.get(background.source_dataset_id)
                            if source is None:
                                raise ValueError(f"Missing background dataset: {background.name}")
                            cloned = _snapshot_item(source)
                            cloned.masks = copy.deepcopy(_outer_masks(source_group, source) or []) + cloned.masks
                            root.datasets.append(cloned)
                            selected.append(cloned)
                            known_data.add(cloned.id)
                            changed = True
                if not changed:
                    break
            if kind == "dataset_group" and any(not isinstance(item, DatasetGroup) for item in selected):
                kind = "dataset_tree"
    # Resolve background pointers without consulting or loading source data.
    from .project_clipboard import _link_backgrounds
    for root in roots:
        _link_backgrounds([root, *root.iter_subgroups(), *root.iter_datasets()], root)
    packet = NfitProject(roots, settings={"cache_binnings": True})
    if hasattr(project, "_path_resolution"):
        packet._path_resolution = project._path_resolution
    if hasattr(project, "_project_path"):
        packet._project_path = project._project_path
    packet._project_transfer_assets = dict(getattr(project, "_project_transfer_assets", {}))
    identities = {item.id: list(dataset_content_signature(item)) for root in roots for item in root.iter_datasets()}
    description = {"schema": TRANSFER_SCHEMA, "version": TRANSFER_VERSION, "kind": kind,
                   "selected_ids": [getattr(item, "id", None) for item in selected],
                   "identities": identities}
    if kind == "dataset_tree":
        description["tree_items"] = [
            {"type": "mask", "index": roots[0].masks.index(item)} if isinstance(item, MaskSpec) else
            {"type": "background", "index": roots[0].backgrounds.index(item)} if isinstance(item, BackgroundSpec) else
            {"type": "object", "id": item.id} for item in selected]
    return ProjectSelection(packet, ProjectClipboardPayload(kind, tuple(selected)), description,
                            source_project=NfitProject(source_roots))


def selection_needs_archive(selection):
    """Configuration-only selections can use the ordinary system clipboard."""
    from .project_clipboard import _iter_fits

    return any(item.data is not None or item._raw_dgs_reduction_cache is not None
               or item.metadata.get("project_artifact_path") or item.metadata.get("derived_from_analysis")
               for root in selection.project.data_groups for item in root.iter_datasets()) or any(
                   fit.channels for root in selection.project.data_groups for entry in root.fits for fit in _iter_fits(entry))


def selection_manifest(selection, path):
    payload = _project_manifest_for_save(selection.project, path)
    payload["project_selection"] = selection.description
    return payload


def write_project_selection(selection, path, *, collect_binnings=None):
    """Write a self-contained selection without reducing or rebinning sources."""
    target = Path(path).expanduser().absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    with temporary_data_directory(target, prefix="nfit-transfer-write-") as directory:
        dataset_ids = {item.id for root in selection.project.data_groups for item in root.iter_datasets()}
        scopes = {(index, node.id) for index, root in enumerate(selection.project.data_groups)
                  for node in root.iter_subgroups()}
        if selection.payload.kind == "group":
            scopes.update((index, None) for index in range(len(selection.project.data_groups)))
        artifacts, entries = ({}, []) if collect_binnings is None else collect_binnings(
            selection.source_project or selection.project, Path(directory), dataset_ids=dataset_ids, scope_ids=scopes)
        selection.project.settings["binning_cache_entries"] = entries
        reduced = project_reduced_event_artifacts(selection.project)
        for root in selection.project.data_groups:
            for dataset in root.iter_datasets():
                reference = getattr(dataset, "_project_artifact_source", None)
                member = dataset.metadata.get("project_artifact_path") or dataset.metadata.get("analysis_artifact_path")
                if reference is None and member and dataset.metadata.get("_project_path"):
                    reference = ArchiveMember(Path(dataset.metadata["_project_path"]), member)
                if dataset.data is not None and not dataset.data_matches_source:
                    member = dataset_artifact_member(dataset.id)
                    output = Path(directory) / f"{dataset.id}.npz"
                    write_dataset_artifact(dataset.data, output)
                    artifacts[member] = output
                    dataset.metadata.update(source_file=member, project_artifact_path=member)
                    dataset.replace_data(None, source_backed=True)
                elif reference is not None:
                    artifacts[member] = reference
        # Analysis tables/figures are already immutable archive assets. Include
        # them for whole-workspace copies without loading their numerical data.
        source_paths = {Path(item.metadata["_project_path"])
                        for root in selection.project.data_groups for item in root.iter_datasets()
                        if item.metadata.get("_project_path")}
        if hasattr(selection.project, "_project_path"):
            source_paths.add(Path(selection.project._project_path))
        analysis_ids = {item.id for root in selection.project.data_groups for item in root.analyses}
        requested_assets = {output.artifact_path for root in selection.project.data_groups for item in root.analyses
                            if item.result is not None for output in item.result.outputs if output.artifact_path}
        for member in requested_assets:
            pending = getattr(selection.project, "_project_transfer_assets", {}).get(member)
            if pending is not None:
                artifacts.setdefault(member, pending)
        for source in source_paths:
            if source.is_file():
                with zipfile.ZipFile(source) as archive:
                    for name in archive.namelist():
                        if name in requested_assets or any(name.startswith(f"assets/analyses/{identity}/") for identity in analysis_ids):
                            artifacts.setdefault(name, ArchiveMember(source, name))
        write_project_manifest(target, selection_manifest(selection, target), preserve_existing=False,
                               binning_artifacts=artifacts, reduced_event_artifacts=reduced)
    selection.path = target
    record_temporary_file(target)
    return target


def read_project_selection(path=None, *, manifest=None):
    """Decode selected configuration; scientific arrays and reduced events stay lazy."""
    path = Path(path).absolute() if path is not None else None
    payload = read_project_manifest(path) if manifest is None else manifest
    description = payload.get("project_selection", {})
    if description.get("schema") != TRANSFER_SCHEMA or description.get("version") != TRANSFER_VERSION:
        raise ValueError("Unsupported nfit project clipboard format")
    project = _project_from_manifest(payload, path or Path.cwd() / ".nfit-clipboard")
    if path is not None:
        project._project_path = path
        bind_project_reduced_event_caches(project, path)
        for root in project.data_groups:
            for dataset in root.iter_datasets():
                member = dataset.metadata.get("project_artifact_path") or dataset.metadata.get("analysis_artifact_path")
                if member:
                    dataset._project_artifact_source = ArchiveMember(path, member)
    kind = description["kind"]
    if kind == "group":
        items = project.data_groups
    else:
        root = project.data_groups[0]
        if kind in {"dataset", "dataset_tree", "dataset_group"}:
            by_id = {item.id: item for item in [*root.iter_datasets(), *root.iter_subgroups()]}
            if "tree_items" in description:
                items = [root.masks[ref["index"]] if ref["type"] == "mask" else
                         root.backgrounds[ref["index"]] if ref["type"] == "background" else
                         by_id[ref["id"]] for ref in description["tree_items"]]
            else:
                items = [by_id[identity] for identity in description["selected_ids"]]
        else:
            items = {"mask": root.masks, "background": root.backgrounds, "model": list(root.models.values()),
                     "fit": root.fits, "analysis": root.analyses, "plot": root.plots}[kind]
            selected = set(description["selected_ids"])
            if kind == "fit":
                items = [item for item in items if item.id in selected]
    return ProjectSelection(project, ProjectClipboardPayload(kind, tuple(items)), description, path)


def save_project_selection(selection, path):
    """Persist an existing clipboard packet without decoding scientific arrays."""
    target = Path(path).expanduser().absolute()
    artifacts = {}
    if selection.path is not None:
        with zipfile.ZipFile(selection.path) as archive:
            artifacts = {name: ArchiveMember(selection.path, name) for name in archive.namelist()
                         if name.startswith("assets/") and not name.endswith("/")}
    write_project_manifest(target, selection_manifest(selection, target), preserve_existing=False,
                           binning_artifacts=artifacts)
    return target


def project_selection_script(selection_path, *, kind="group"):
    """Render editable public-API replay for a persisted project selection."""
    role = {"group": "project", "dataset": "datasets", "dataset_tree": "datasets", "dataset_group": "datasets",
            "mask": "group_masks", "background": "group_backgrounds", "model": "models", "fit": "fits",
            "analysis": "analyses", "plot": "plots"}[kind]
    lines = ["from pathlib import Path", "from nfit import DataGroup, NfitProject, import_project_items, load_project, save_project",
             "", f"selection = Path({str(Path(selection_path).absolute())!r})", "destination = Path('destination.nfit')",
             "project = load_project(destination) if destination.exists() else NfitProject([DataGroup('Workspace')])",
             "# Choose the destination workspace/subfolder below. Referenced sources must exist there."]
    if kind == "group":
        lines.append("import_project_items(project, selection)")
    else:
        lines.extend(["workspace = project.data_groups[0]",
                      f"import_project_items(project, selection, target_role={role!r}, data_group=workspace)"])
    lines.extend(["save_project(project, destination)", ""])
    return "\n".join(lines)


def adopt_project_selection(path, owner_path):
    """Adopt a private immutable archive, using a hard link where possible."""
    source = Path(path).absolute()
    owner_path = Path(owner_path).absolute()
    # An unsaved destination may choose the clipboard packet as its storage
    # hint. Never nest its lease inside another operation's disposable folder.
    for parent in reversed(owner_path.parents):
        if parent.name == ".nfit-work":
            owner_path = parent.parent / f".nfit-unsaved-project-{uuid4().hex}"
            break
    owner = temporary_data_directory(owner_path, prefix="nfit-pasted-selection-")
    destination = Path(owner.name) / "selection.nfit"
    try:
        before = source.stat()
        try:
            os.link(source, destination)
        except OSError:
            from .storage_budget import reserve_disk_space
            with reserve_disk_space(destination.parent, before.st_size, operation="Copying project selection"):
                with source.open("rb") as reader, destination.open("xb") as writer:
                    copied = 0
                    while block := reader.read(8 * 1024**2):
                        writer.write(block)
                        copied += len(block)
                        report_operation("Copying project selection", completed=copied, total=before.st_size)
        after = source.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise OSError("The copied project selection changed during paste; copy it again")
        selection = read_project_selection(destination)
        selection.owner = owner
        record_temporary_file(destination, before.st_size)
        return selection
    except BaseException:
        owner.cleanup()
        raise


def selection_paste_capability(selection, target_role, *, data_group=None):
    from .project_clipboard import paste_capability
    return paste_capability(_remap_external_references(selection.payload, data_group), target_role,
                            data_group=data_group)


def _remap_external_references(payload, destination):
    if destination is None:
        return payload
    dataset_ids, group_ids, fit_ids = {}, {}, {}
    for item in destination.iter_datasets():
        for identity in item.metadata.get(_ORIGIN, []):
            dataset_ids[identity] = item.id
    for item in destination.iter_subgroups():
        for identity in item.metadata.get(_ORIGIN, []):
            group_ids[identity] = item.id
    from .project_clipboard import (
        _iter_fits,
        _remap_analysis_source_id,
        _remap_backgrounds,
        _remap_fit_snapshot_backgrounds,
    )
    for entry in destination.fits:
        for item in _iter_fits(entry):
            for identity in item.metadata.get(_ORIGIN, []):
                fit_ids[identity] = item.id
    for item in payload.items:
        if isinstance(item, BackgroundSpec):
            _remap_backgrounds([type("Owner", (), {"backgrounds": [item]})()], dataset_ids, group_ids)
        elif payload.kind == "analysis":
            item.input_dataset_ids = [_remap_analysis_source_id(identity, dataset_ids, group_ids) for identity in item.input_dataset_ids]
        elif payload.kind == "fit":
            for fit in _iter_fits(item):
                _remap_fit_snapshot_backgrounds(fit.snapshot, dataset_ids, group_ids)
        elif payload.kind == "plot":
            for reference in item.sources:
                reference.dataset_id = dataset_ids.get(reference.dataset_id, reference.dataset_id)
                reference.fit_id = fit_ids.get(reference.fit_id, reference.fit_id)
            configs = item.settings.get("source_rebin_configs")
            if isinstance(configs, dict):
                item.settings["source_rebin_configs"] = {dataset_ids.get(key, key): value for key, value in configs.items()}
            composite = item.settings.get("source_composite", {})
            if composite.get("dataset_group_id"):
                composite["dataset_group_id"] = group_ids.get(composite["dataset_group_id"], composite["dataset_group_id"])
    return payload


def _adopt_analysis_assets(selection, destination, result):
    if selection.path is None:
        return
    prefix = f"assets/analyses/{uuid4().hex}-"

    def renamed(value):
        return prefix + value.removeprefix("assets/analyses/") if isinstance(value, str) and value.startswith("assets/analyses/") else value

    with zipfile.ZipFile(selection.path) as archive:
        assets = destination.__dict__.setdefault("_project_transfer_assets", {})
        for member in archive.namelist():
            if member.startswith("assets/analyses/"):
                assets[renamed(member)] = ArchiveMember(selection.path, member)
    for root in (item for item in result.items if isinstance(item, DataGroup)):
        for item in root.iter_datasets():
            for key in ("source_file", "analysis_artifact_path", "project_artifact_path"):
                if key in item.metadata:
                    item.metadata[key] = renamed(item.metadata[key])
        for analysis in root.analyses:
            if analysis.result is not None:
                for output in analysis.result.outputs:
                    output.artifact_path = renamed(output.artifact_path)


def paste_project_selection(selection, destination, *, target_role, data_group=None,
                            dataset_node=None, dataset=None, install_binnings=None):
    """Install an adopted selection with independent IDs and lazy cache ownership."""
    for root in selection.project.data_groups:
        from .project_clipboard import _iter_fits
        fits = [item for entry in root.fits for item in _iter_fits(entry)]
        for item in [*root.iter_subgroups(), *root.iter_datasets(), *fits]:
            aliases = list(item.metadata.get(_ORIGIN, []))
            item.metadata[_ORIGIN] = list(dict.fromkeys([*aliases, item.id]))[-32:]
    payload = _remap_external_references(selection.payload, data_group)
    result = paste_payload(payload, target_role=target_role, data_group=data_group, dataset_node=dataset_node,
                           dataset=dataset, project_groups=destination.data_groups)
    # The public clone remaps stable IDs. Give every newly pasted data asset a
    # destination-specific member so repeated copies of edited data cannot
    # overwrite the backing of an earlier copy.
    copied = []
    for item in result.items:
        if isinstance(item, (DataGroup, DatasetGroup)):
            copied.extend(item.iter_datasets())
        elif isinstance(item, DatasetEntry):
            copied.append(item)
    for item in copied:
        if hasattr(item, "_project_artifact_source") and not item.metadata.get("derived_from_analysis"):
            member = dataset_artifact_member(item.id)
            item.metadata.update(project_artifact_path=member, source_file=member)
    if selection.owner is not None:
        # Keep the destination's private file alive until it closes, even if
        # this clipboard entry is replaced or its originating GUI quits.
        destination.__dict__.setdefault("_project_transfer_owners", []).append(selection.owner)
    _adopt_analysis_assets(selection, destination, result)
    if install_binnings is not None:
        install_binnings(selection, destination, result, data_group)
    return result


def project_transfer_artifacts(project):
    """Collect pasted data assets for the normal atomic single-file save."""
    artifacts = dict(getattr(project, "_project_transfer_assets", {}))
    for root in project.data_groups:
        for dataset in root.iter_datasets():
            source = getattr(dataset, "_project_artifact_source", None)
            if source is not None:
                member = dataset.metadata.get("project_artifact_path") or dataset.metadata.get("analysis_artifact_path")
                if member:
                    artifacts[member] = source
    return artifacts


def adopt_saved_transfer_artifacts(project, path):
    """Publish new backing only after a complete, successful project save."""
    for root in project.data_groups:
        for dataset in root.iter_datasets():
            member = dataset.metadata.get("project_artifact_path") or dataset.metadata.get("analysis_artifact_path")
            if member and hasattr(dataset, "_project_artifact_source"):
                dataset._project_artifact_source = ArchiveMember(Path(path), member)
    project.__dict__.pop("_project_transfer_assets", None)
