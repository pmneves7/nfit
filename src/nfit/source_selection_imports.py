"""Import selected measurements and optional expression groups without Qt.

Expressions preserve source appearances and acquisition grouping. Repeated files are
the same measurement, rather than extra independent observations.
"""

from __future__ import annotations

import copy
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat

from .pipeline import DataGroup, DatasetGroup
from .source_lineage import (
    SOURCE_SELECTION_LINEAGE_KEY as SOURCE_SELECTION_LINEAGE_KEY,
)
from .source_lineage import (
    source_identity,
)
from .source_lineage import (
    validate_source_selection_combination as validate_source_selection_combination,
)
from .source_selection import SourceSelection, SourceSelectionPlan, resolve_source_selection


def _source_family(path):
    from .corelli import is_corelli_raw_nexus_file
    from .mdevent import is_mdevent_file
    from .raw_dgs import is_raw_dgs_nexus_file

    if is_corelli_raw_nexus_file(path):
        return "corelli"
    if is_raw_dgs_nexus_file(path):
        return "raw_dgs"
    if is_mdevent_file(path):
        return "mdevent"
    return "ordinary"


def _native_group(paths, family, *, name, normalization_path, mask_path, flux_path,
                 progress_callback, reduction_config=None):
    common = dict(normalization_path=normalization_path, mask_path=mask_path,
                  name=name, progress_callback=progress_callback)
    if family == "raw_dgs":
        from .raw_dgs import raw_dgs_dataset_group

        policies = {key: value for key, value in (reduction_config or {}).items()
                    if key in {"trajectory_energy_policy", "monitor_variance_policy",
                               "event_precision_policy", "symmetry_variance_policy"}}
        return raw_dgs_dataset_group(paths, **common, **policies)
    if family == "corelli":
        from .corelli import corelli_dataset_group

        common.pop("normalization_path")
        return corelli_dataset_group(paths, solid_angle_path=normalization_path,
                                     flux_path=flux_path, **common)
    from .mdevent import append_mdevent_file, mdevent_dataset_group

    policies = {key: value for key, value in (reduction_config or {}).items()
                if key in {"trajectory_energy_policy", "event_precision_policy", "symmetry_variance_policy"}}
    group = mdevent_dataset_group(paths[0], **common, **policies)
    for path in paths[1:]:
        append_mdevent_file(group, path)
    return group


def _import_source_selection(
    parent: DataGroup | DatasetGroup,
    selection: SourceSelection | SourceSelectionPlan,
    *,
    data_type: str = "single_crystal_inelastic",
    name: str | None = None,
    missing: str = "error",
    preserve_groups: bool = False,
    importer_name: str | None = None,
    importer_options: dict | None = None,
    normalization_path: str | Path | None = None,
    mask_path: str | Path | None = None,
    flux_path: str | Path | None = None,
    progress_callback=None,
    reuse_sources=None,
    templates=None,
    coadd=False,
) -> DatasetGroup:
    """Append a resolved acquisition collection atomically and return it.

    By default selected sources form one existing dataset group. With
    ``preserve_groups=True``, each expression group becomes an ordinary child
    dataset group. Native sources share their reduction configuration; ordinary
    files retain registered importer behavior, including multiple streams.
    ``missing='skip'`` preserves skipped appearances in provenance. Empty child
    groups are retained only when preserving expression grouping.
    A native group requires one reduction format. Raw DGS runs may come from
    different instruments: their importer checks each run's geometry. Native
    formats and ordinary files cannot share one flat import group; preserve
    expression groups to import different formats into separate subfolders.

    Flat import treats the expression as unique physical source membership,
    including repetitions written with ``+``. Appearance indices remain saved.
    When preserving groups, repeated raw DGS/MDEvent sources within a group use
    their multiplicity as a coefficient: numerator/exposure scale by ``m`` and
    variance by ``m²``.
    This cannot create independent counting precision. Repeated CORELLI or
    ordinary sources in one slot are rejected until their adapters can retain
    this covariance. Across slots, aliases retain one physical source identity.
    """
    from .project_dataset_io import _load_nfit_dataset_file
    from .project_imports import (
        GROUP_COMPOSITE_KEY,
        _single_normalization_companion,
        import_dataset_paths,
    )

    if not isinstance(parent, (DataGroup, DatasetGroup)):
        raise TypeError("parent must be a data group or dataset group")
    if not isinstance(preserve_groups, bool):
        raise TypeError("preserve_groups must be a boolean")
    if missing not in {"error", "skip"}:
        raise ValueError("missing must be 'error' or 'skip'")
    if isinstance(selection, SourceSelection):
        plan = resolve_source_selection(selection)
    elif isinstance(selection, SourceSelectionPlan):
        plan = selection
    else:
        raise TypeError("selection must be SourceSelection or SourceSelectionPlan")
    if not plan.slots:
        raise ValueError("the selection contains no acquisition slots")
    absent = {str(item.path) for item in plan.appearances
              if item.path is not None and not Path(item.path).is_file()}
    if absent and missing == "error":
        raise FileNotFoundError("missing selected sources: " + ", ".join(sorted(absent)))

    # No parent mutation occurs until every group and importer has succeeded.
    names = {group.name for group in parent.subgroups}
    base = name or "Source selection"
    unique = base
    index = 2
    while unique in names:
        unique = f"{base} ({index})"
        index += 1
    collection = DatasetGroup(unique)
    scratch = DataGroup(unique, subgroups=[collection])
    if isinstance(parent, DataGroup):
        scratch.lattice_parameters = copy.deepcopy(parent.lattice_parameters)
        scratch.spacegroup = parent.spacegroup
    family_cache = {path: records[0][0] for path, records in (reuse_sources or {}).items()}
    used_ids = set()
    blocks = ([(slot, slot.appearance_indices) for slot in plan.slots] if preserve_groups
              else [(None, tuple(range(len(plan.appearances))))])
    for slot, appearance_indices in blocks:
        appearances = [plan.appearances[index] for index in appearance_indices]
        indices_by_file = {}
        for index, item in zip(appearance_indices, appearances, strict=True):
            if item.path is None or str(item.path) in absent:
                continue
            canonical = str(Path(item.path).expanduser().resolve())
            indices_by_file.setdefault(canonical, []).append(index)
        paths = list(indices_by_file)
        counts = Counter({path: len(indices) for path, indices in indices_by_file.items()})
        title = f"Group {slot.depth}:{slot.stack}" if slot is not None else unique
        for path in paths:
            if path not in family_cache:
                family_cache[path] = _source_family(path)
        families = {family_cache[path] for path in paths}
        if len(families) > 1 and families != {"ordinary"}:
            raise ValueError("one acquisition group cannot mix native reduction formats and ordinary files")
        family = next(iter(families), "ordinary")
        if (preserve_groups or coadd) and any(value > 1 for value in counts.values()) and family not in {"raw_dgs", "mdevent"}:
            raise ValueError(f"repeated {family} sources in a summed group require a covariance-aware adapter")
        template = (templates or {}).get(None if slot is None else (slot.depth, slot.stack))
        template_family = next((key for key in ("raw_dgs", "corelli", "mdevent")
                                if template is not None and isinstance(template.metadata.get(key), dict)), "ordinary")
        if paths and template is not None and template_family != family:
            raise ValueError("a source edit cannot change an existing group's reduction format; import a separate group")
        if paths and family != "ordinary":
            if flux_path is not None and family != "corelli":
                raise ValueError("an incident flux calibration applies only to CORELLI groups")
            resolved_normalization, resolved_mask = normalization_path, mask_path
            companion = None
            if normalization_path is None and mask_path is None and template is None and not reuse_sources:
                companion = _single_normalization_companion(paths[0])
                if companion is not None:
                    resolved_normalization = resolved_mask = companion
            reusable = {path: (reuse_sources or {}).get(path, ()) for path in paths}
            new_paths = [path for path in paths if not reusable[path]]
            owner = template or next((records[0][1] for records in reusable.values() if records), None)
            if owner is None:
                child = _native_group(paths, family, name=title, normalization_path=resolved_normalization,
                                     mask_path=resolved_mask, flux_path=flux_path,
                                     progress_callback=progress_callback)
            else:
                child = copy.copy(owner)
                child.metadata = copy.deepcopy(owner.metadata)
                child.subgroups = []
                child.datasets = []
                config = child.metadata[family]
                added = None
                if new_paths:
                    added = _native_group(new_paths, family, name=title,
                                          normalization_path=config.get("normalization_file"),
                                          mask_path=config.get("mask_file"), flux_path=config.get("flux_file"),
                                          progress_callback=progress_callback, reduction_config=config)
                new_by_file = {} if added is None else {
                    path: [entry for entry in added.datasets if str(Path(entry.metadata["source_file"]).resolve()) == path]
                    for path in new_paths}
                for path in paths:
                    records = reusable[path]
                    if records:
                        preferred = [record for record in records if record[1] is owner]
                        records = preferred or records
                        # One file can represent several MDE experiments or ordinary streams.
                        identities = {}
                        for record in records:
                            identities.setdefault(_entry_identity(record[2]), record)
                        for _, old_owner, original in identities.values():
                            entry = copy.copy(original) if original.id not in used_ids else original.copy()
                            used_ids.add(entry.id)
                            entry.metadata = copy.deepcopy(original.metadata)
                            entry.metadata["_selection_previous_coefficient"] = original.metadata.get(SOURCE_SELECTION_LINEAGE_KEY, {}).get("coaddition_coefficient", 1)
                            child.datasets.append(entry)
                            _preserve_run_settings(old_owner, child, entry, original)
                    else:
                        child.datasets.extend(new_by_file[path])
                config["source_files"] = paths
                config["event_count"] = sum(entry.metadata.get("event_count", 0) for entry in child.datasets)
                if added is not None:
                    for old_axis, new_axis in zip(config.get("dimensions", ()), added.metadata[family].get("dimensions", ()), strict=False):
                        if "lower" in old_axis and "lower" in new_axis:
                            old_axis["lower"] = min(old_axis["lower"], new_axis["lower"])
                            old_axis["upper"] = max(old_axis["upper"], new_axis["upper"])
                if template is None:
                    child.id = DatasetGroup(title).id
                    child.name = title
            if companion is not None:
                child.metadata["source_selection_calibration"] = {
                    "discovery": "single_van_companion", "source_file": paths[0],
                    "normalization_file": str(companion), "mask_file": str(companion),
                }
            if family == "mdevent" and scratch.lattice_parameters is None:
                scratch.lattice_parameters = copy.deepcopy(child.metadata["mdevent"]["lattice_parameters"])
            if owner is None:
                child.metadata[GROUP_COMPOSITE_KEY] = {
                    "enabled": True, "auto_rebin": False, "stale": True,
                    "fractional": False, "mean_weighting": "uniform",
                    "coordinate_mode": "powder" if data_type == "powder_inelastic" else "hkle",
                }
            for dataset in child.datasets:
                dataset.data_type = data_type
            if preserve_groups:
                collection.subgroups.append(child)
            else:
                collection = child
                scratch.subgroups = [collection]
        else:
            if paths and any(value is not None for value in (normalization_path, mask_path, flux_path)):
                raise ValueError("native calibration paths cannot be applied to ordinary source importers")
            child = copy.copy(template) if template is not None else (DatasetGroup(title, enabled=bool(paths)) if preserve_groups else collection)
            child.metadata = copy.deepcopy(child.metadata)
            child.datasets = []
            child.subgroups = []
            if template is not None:
                for old_stream in template.subgroups:
                    if old_stream.metadata.get("source_stream"):
                        stream_group = copy.copy(old_stream)
                        stream_group.metadata = copy.deepcopy(old_stream.metadata)
                        stream_group.datasets, stream_group.subgroups = [], []
                        child.subgroups.append(stream_group)
            if not preserve_groups:
                collection = child
                scratch.subgroups = [child]
            if preserve_groups:
                collection.subgroups.append(child)
            if paths:
                for path in paths:
                    records = (reuse_sources or {}).get(path, ())
                    distinct = {}
                    for record in records:
                        distinct.setdefault(_entry_identity(record[2]), record)
                    for _, old_owner, original in distinct.values():
                        entry = copy.copy(original) if original.id not in used_ids else original.copy()
                        used_ids.add(entry.id)
                        entry.metadata = copy.deepcopy(original.metadata)
                        entry.metadata["_selection_previous_coefficient"] = original.metadata.get(SOURCE_SELECTION_LINEAGE_KEY, {}).get("coaddition_coefficient", 1)
                        target = child
                        if old_owner.metadata.get("source_stream"):
                            target = next((stream for stream in child.subgroups
                                           if stream.metadata.get("source_stream") == old_owner.metadata["source_stream"]
                                           and stream.metadata.get("importer") == old_owner.metadata.get("importer")), None)
                            if target is None:
                                target = copy.copy(old_owner)
                                target.id = DatasetGroup(old_owner.name).id
                                target.metadata = copy.deepcopy(old_owner.metadata)
                                target.datasets, target.subgroups = [], []
                                child.subgroups.append(target)
                        target.datasets.append(entry)
                new_paths = [path for path in paths if path not in (reuse_sources or {})]
                import_dataset_paths(scratch, new_paths, into=child, data_type=data_type,
                                     importer_name=importer_name, importer_options=importer_options,
                                     stream_group_mode="reuse", progress_callback=progress_callback,
                                     dataset_file_loader=_load_nfit_dataset_file)
        if slot is not None:
            child.metadata["source_selection_group"] = slot.to_dict()
            child.metadata["source_selection_group"]["skipped_appearance_indices"] = [
                index for index, item in zip(appearance_indices, appearances, strict=True)
                if item.path is not None and str(item.path) in absent
            ]
        for dataset in child.iter_datasets():
            source = str(Path(dataset.metadata["source_file"]).expanduser().resolve())
            multiplicity = counts[source]
            experiment = dataset.metadata.get("mdevent_experiment_index")
            importer = stream = None
            if family == "ordinary":
                options = dataset.metadata.get("import_options", dataset.metadata.get("importer_options", {}))
                stream = options.get("stream") if isinstance(options, dict) else None
                stream = stream or dataset.metadata.get("source_stream")
                if stream:
                    importer = str(dataset.metadata.get("importer", importer_name or ""))
            identity = source_identity(source, experiment_index=experiment, importer=importer, stream=stream)
            dataset.metadata[SOURCE_SELECTION_LINEAGE_KEY] = {
                "version": 1, "source_identity": identity, "multiplicity": multiplicity,
                "appearance_indices": list(indices_by_file[source]),
                "coaddition_coefficient": multiplicity if preserve_groups or coadd else 1,
            }
            if (preserve_groups or coadd) and multiplicity > 1:
                prior = dataset.metadata.pop("_selection_previous_coefficient", 1)
                dataset.fit_weight = dataset.fit_weight / prior * multiplicity
            elif "_selection_previous_coefficient" in dataset.metadata:
                dataset.fit_weight /= dataset.metadata.pop("_selection_previous_coefficient")
        if paths and family != "ordinary":
            from .reduction_recipes import ensure_reduction_recipe

            selected = plan.to_dict()
            selected["preserve_groups"] = preserve_groups
            if companion is not None:
                selected["calibration_discovery"] = copy.deepcopy(child.metadata["source_selection_calibration"])
            if slot is not None:
                selected["selected_group"] = slot.to_dict()
            ensure_reduction_recipe(child, source_selection=selected)
    collection.metadata["source_selection"] = plan.to_dict()
    collection.metadata["source_selection"].update(
        import_missing_policy=missing, preserve_groups=preserve_groups,
        import_skipped_files=sorted(absent),
        import_settings={"data_type": data_type, "importer_name": importer_name,
                         "importer_options": copy.deepcopy(importer_options)},
    )
    collection.enabled = bool(collection.datasets) or any(child.enabled for child in collection.subgroups)
    from .data_workspace import inherit_data_workspace

    inherit_data_workspace(collection, parent)
    parent.subgroups.append(collection)
    if isinstance(parent, DataGroup):
        if parent.lattice_parameters is None:
            parent.lattice_parameters = scratch.lattice_parameters
        if parent.spacegroup is None:
            parent.spacegroup = scratch.spacegroup
    return collection


def source_selection_script(selection, **import_settings) -> str:
    """Export an editable standalone selector and public import operation."""
    if isinstance(selection, SourceSelectionPlan):
        selection = selection.selection
    if not isinstance(selection, SourceSelection):
        raise TypeError("selection must be SourceSelection or SourceSelectionPlan")
    settings = copy.deepcopy(import_settings)
    if "progress_callback" in settings:
        raise ValueError("progress callbacks are runtime objects, not saved import settings")
    for key in ("normalization_path", "mask_path", "flux_path"):
        if settings.get(key) is not None:
            settings[key] = str(settings[key])
    try:
        settings = json.loads(json.dumps(settings, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("saved import settings must be finite JSON values") from error
    return f'''"""Import source membership and optional groups from an editable run expression."""
from nfit import DataGroup
from nfit.source_selection import SourceSelection
from nfit.source_selection_imports import import_source_selection

SOURCE_SELECTION = {pformat(selection.to_dict(), sort_dicts=False)}
IMPORT_SETTINGS = {pformat(settings, sort_dicts=False)}

def run():
    parent = DataGroup("Imported sources")
    selection = SourceSelection.from_dict(SOURCE_SELECTION)
    group = import_source_selection(parent, selection, **IMPORT_SETTINGS)
    return parent, group

if __name__ == "__main__":
    parent, group = run()
    print(group.name)
'''


@dataclass(frozen=True)
class SourceSelectionEdit:
    """Result of an atomic membership edit, with unchanged reduction caches retained."""

    added_dataset_ids: tuple[str, ...]
    removed_dataset_ids: tuple[str, ...]
    retained_dataset_ids: tuple[str, ...]
    membership_changed: bool
    settings_changed: bool


def _entry_identity(entry):
    lineage = entry.metadata.get(SOURCE_SELECTION_LINEAGE_KEY, {})
    return lineage.get("source_identity") or source_identity(
        entry.metadata["source_file"], experiment_index=entry.metadata.get("mdevent_experiment_index"),
        importer=entry.metadata.get("importer"),
        stream=entry.metadata.get("import_options", {}).get("stream"),
    )


def _preserve_run_settings(old_owner, new_owner, entry, original):
    """Moving a run preserves its effective acquisition settings as overrides."""
    from .reduction_recipes import (
        REDUCTION_OVERRIDES_KEY,
        effective_reduction_config,
        reduction_settings_schema,
    )

    before = effective_reduction_config(old_owner, original)
    after = effective_reduction_config(new_owner, entry)
    overrides = copy.deepcopy(entry.metadata.get(REDUCTION_OVERRIDES_KEY, {}))
    for setting in reduction_settings_schema(new_owner):
        if setting.scope != "coordinates" and before.get(setting.key) != after.get(setting.key):
            overrides[setting.key] = copy.deepcopy(before.get(setting.key))
    if overrides:
        entry.metadata[REDUCTION_OVERRIDES_KEY] = overrides


def import_source_selection(
    parent: DataGroup | DatasetGroup, selection: SourceSelection | SourceSelectionPlan,
    *, data_type: str = "single_crystal_inelastic", name: str | None = None,
    missing: str = "error", preserve_groups: bool = False,
    importer_name: str | None = None, importer_options: dict | None = None,
    normalization_path: str | Path | None = None, mask_path: str | Path | None = None,
    flux_path: str | Path | None = None, progress_callback=None,
) -> DatasetGroup:
    """Import a run expression into one ordinary dataset group atomically.

    See :func:`update_source_selection` to edit a saved collection while retaining
    existing entries and reduced-event caches. Settings include ``data_type``,
    ``name``, ``missing``, ``preserve_groups``, importer and calibration paths.
    """
    return _import_source_selection(
        parent, selection, data_type=data_type, name=name, missing=missing,
        preserve_groups=preserve_groups, importer_name=importer_name, importer_options=importer_options,
        normalization_path=normalization_path, mask_path=mask_path, flux_path=flux_path,
        progress_callback=progress_callback,
    )


def update_source_selection(
    group: DatasetGroup,
    selection: SourceSelection | SourceSelectionPlan,
    *, parent: DataGroup | None = None,
    missing: str | None = None,
    preserve_groups: bool | None = None,
    reference_roots=None,
    progress_callback=None,
) -> SourceSelectionEdit:
    """Replace selected membership atomically, preserving unchanged source entries.

    Only new files are inspected by their importer. Existing IDs, masks, per-run
    settings, loaded immutable data and lazy reduced caches remain attached to
    retained entries. Changed collections and their ancestors become stale;
    unrelated branches and retained per-run histograms remain reusable. If
    provided, ``parent`` must be the owning top-level data group. Pass all
    project data groups as ``reference_roots`` to check external background
    links; by default the owning parent (or detached group) is checked. Saved
    links, including disabled links, must be edited before their sources can
    be removed. Reference validation precedes every saved-object mutation.

    Expression subfolders are ordinary DatasetGroups. Their scientific settings
    are retained by expression-group index. Moving a run preserves its effective
    reduction settings as per-run overrides. Coordinate transforms belong to
    the destination group. Only source-managed subfolders can be replaced; an
    unrelated nested collection requires a separate source selection.
    """
    from .project_composites import GROUP_COMPOSITE_BINNINGS_KEY
    from .project_imports import GROUP_COMPOSITE_KEY
    if not isinstance(group, DatasetGroup):
        raise TypeError("group must be an existing dataset group")
    previous = group.metadata.get("source_selection", {})
    if not isinstance(previous, dict):
        previous = {}
    if parent is not None and not isinstance(parent, DataGroup):
        raise TypeError("parent must be the owning top-level data group")
    if parent is not None and not any(node is group for node in parent.iter_subgroups()):
        raise ValueError("parent does not own the selected dataset group")
    for child in group.iter_subgroups():
        if not (isinstance(child.metadata.get("source_selection_group"), dict)
                or child.metadata.get("source_stream")):
            raise ValueError("edit sources in each nested collection rather than replacing unrelated subfolders")
    old_nodes = [group, *group.iter_subgroups()]
    old_members = {node.id: [(entry.id, entry.fit_weight) for entry in node.datasets] for node in old_nodes}
    reuse = {}
    before = list(group.iter_datasets())
    for owner in old_nodes:
        family = next((key for key in ("raw_dgs", "mdevent", "corelli")
                       if isinstance(owner.metadata.get(key), dict)), "ordinary")
        for entry in owner.datasets:
            path = entry.metadata.get("source_file")
            if not path:
                raise ValueError("source membership editing requires file-backed entries")
            canonical = str(Path(path).expanduser().resolve())
            reuse.setdefault(canonical, []).append((family, owner, entry))
    had_groups = previous.get("preserve_groups", any("source_selection_group" in child.metadata for child in group.subgroups))
    desired_groups = had_groups if preserve_groups is None else preserve_groups
    templates = {None: group} if not had_groups else {
        (child.metadata["source_selection_group"]["depth"],
         child.metadata["source_selection_group"]["stack"]): child for child in group.subgroups}
    if had_groups != desired_groups:
        templates = {}
    selection_plan = resolve_source_selection(selection) if isinstance(selection, SourceSelection) else selection
    if not isinstance(selection_plan, SourceSelectionPlan):
        raise TypeError("selection must be SourceSelection or SourceSelectionPlan")
    policy = previous.get("import_missing_policy", "error") if missing is None else missing
    if policy not in {"error", "skip"}:
        raise ValueError("missing must be 'error' or 'skip'")
    if not isinstance(desired_groups, bool):
        raise TypeError("preserve_groups must be a boolean")
    scratch = DataGroup("Source edit staging")
    if parent is not None:
        scratch.lattice_parameters = copy.deepcopy(parent.lattice_parameters)
        scratch.spacegroup = parent.spacegroup
    if selection_plan.clear:
        candidate = copy.copy(group)
        candidate.metadata = copy.deepcopy(group.metadata)
        candidate.datasets, candidate.subgroups = [], []
        for key in ("raw_dgs", "mdevent", "corelli"):
            if isinstance(candidate.metadata.get(key), dict):
                candidate.metadata[key]["source_files"] = []
                candidate.metadata[key]["event_count"] = 0
        if any(key in candidate.metadata for key in ("raw_dgs", "mdevent", "corelli")):
            from .reduction_recipes import ensure_reduction_recipe

            ensure_reduction_recipe(candidate, source_selection=selection_plan.to_dict())
        candidate.metadata["source_selection"] = selection_plan.to_dict()
        candidate.metadata["source_selection"].update(
            preserve_groups=desired_groups, import_missing_policy=policy,
            import_settings=copy.deepcopy(previous.get("import_settings", {})),
        )
    else:
        settings = previous.get("import_settings", {})
        data_type = settings.get("data_type", before[0].data_type if before else "single_crystal_inelastic")
        candidate = _import_source_selection(
            scratch, selection_plan, name=group.name, data_type=data_type,
            missing=policy, preserve_groups=desired_groups,
            importer_name=settings.get("importer_name"), importer_options=settings.get("importer_options"),
            reuse_sources=reuse, templates=templates, progress_callback=progress_callback,
            coadd="source_selection_group" in group.metadata and not desired_groups,
        )
    if "source_selection_group" in group.metadata and not desired_groups:
        record = copy.deepcopy(group.metadata["source_selection_group"])
        record.update(run_numbers=[item.run_number for item in selection_plan.appearances],
                      appearance_indices=list(range(len(selection_plan.appearances))),
                      empty=not any(item.path for item in selection_plan.appearances))
        candidate.metadata["source_selection_group"] = record
        candidate.metadata["source_selection"]["coaddition_mode"] = "correlated_sum"
        if isinstance(candidate.metadata.get("reduction_recipe"), dict):
            candidate.metadata["reduction_recipe"]["source_selection"]["selected_group"] = copy.deepcopy(record)
    for owner in (candidate, *candidate.iter_subgroups()):
        for key in ("raw_dgs", "mdevent", "corelli"):
            if key in owner.metadata and not owner.datasets:
                owner.metadata[key]["source_files"] = []
                owner.metadata[key]["event_count"] = 0
                from .reduction_recipes import ensure_reduction_recipe

                ensure_reduction_recipe(owner, source_selection=selection_plan.to_dict())
    # Detached candidates have validated every new source and recipe. No saved
    # object is changed before this point.
    old_by_id = {entry.id: entry for entry in before}
    after = list(candidate.iter_datasets())
    before_order = [(entry.id, entry.fit_weight) for entry in before]
    after_order = [(entry.id, entry.fit_weight) for entry in after]
    before_structure = [(node.id, old_members[node.id]) for node in old_nodes[1:]]
    after_structure = [(node.id, [(entry.id, entry.fit_weight) for entry in node.datasets])
                       for node in candidate.iter_subgroups()]
    membership_changed = (before_order != after_order or had_groups != desired_groups
                          or before_structure != after_structure)
    settings_changed = membership_changed or previous != candidate.metadata["source_selection"]
    roots = (parent or group,) if reference_roots is None else tuple(reference_roots)
    _validate_removed_source_references(group, candidate, (*roots, group))
    for owner in (candidate, *candidate.iter_subgroups()):
        final = []
        for staged in owner.datasets:
            original = old_by_id.get(staged.id)
            if original is not None:
                original.metadata = staged.metadata
                original.fit_weight = staged.fit_weight
                final.append(original)
            else:
                final.append(staged)
        owner.datasets = final
    old_children = {child.id: child for child in old_nodes[1:]}

    def retain_folders(owner):
        for index, child in enumerate(owner.subgroups):
            retain_folders(child)
            old = old_children.get(child.id)
            if old is not None:
                old.datasets, old.subgroups = child.datasets, child.subgroups
                old.metadata = child.metadata
                owner.subgroups[index] = old

    retain_folders(candidate)
    metadata = copy.deepcopy(group.metadata)
    for key in ("raw_dgs", "mdevent", "corelli", "reduction_recipe", "source_selection", "source_selection_calibration", "source_selection_group"):
        metadata.pop(key, None)
        if key in candidate.metadata:
            metadata[key] = candidate.metadata[key]
    group.datasets, group.subgroups, group.metadata = candidate.datasets, candidate.subgroups, metadata
    if not after:
        group.enabled = False
    if membership_changed:
        def mark(node):
            config = node.metadata.get(GROUP_COMPOSITE_KEY)
            if isinstance(config, dict):
                config["stale"] = True
            registry = node.metadata.get(GROUP_COMPOSITE_BINNINGS_KEY, {})
            for item in registry.get("items", ()):
                if isinstance(item.get("config"), dict):
                    item["config"]["stale"] = True

        mark(group)
        # Mark changed native subfolders; retaining a folder with identical
        # membership keeps its complete numerical cache signature reusable.
        for child in group.iter_subgroups():
            if old_members.get(child.id) != [(entry.id, entry.fit_weight) for entry in child.datasets]:
                mark(child)
        if parent is not None:
            def ancestors(node):
                found = node is group
                for child in node.subgroups:
                    found = ancestors(child) or found
                if found:
                    mark(node)
                return found
            ancestors(parent)
    from .data_workspace import inherit_data_workspace

    inherit_data_workspace(group, parent if parent is not None else group)
    return SourceSelectionEdit(
        tuple(entry.id for entry in after if entry.id not in old_by_id),
        tuple(entry.id for entry in before if entry.id not in {item.id for item in after}),
        tuple(entry.id for entry in after if entry.id in old_by_id),
        membership_changed, settings_changed,
    )


def _validate_removed_source_references(group, candidate, roots):
    """Reject dangling saved background references before an atomic edit commits."""
    retained_ids = {entry.id for entry in candidate.iter_datasets()}
    removed_ids = {entry.id for entry in group.iter_datasets()} - retained_ids
    retained_groups = {node.id for node in candidate.iter_subgroups()}
    removed_groups = {node.id for node in group.iter_subgroups()} - retained_groups
    seen = set()
    for root in roots:
        if not isinstance(root, (DataGroup, DatasetGroup)):
            raise TypeError("reference_roots must contain data groups or dataset groups")
        owners = (root, *root.iter_subgroups(), *root.iter_datasets())
        for owner in owners:
            if getattr(owner, "id", None) in removed_ids | removed_groups:
                # Links owned by removed members disappear with those members;
                # only surviving scientific configuration can become dangling.
                continue
            if id(owner) in seen:
                continue
            seen.add(id(owner))
            for background in owner.backgrounds:
                if (background.source_dataset_id in removed_ids
                        or background.source_group_id in removed_groups):
                    raise ValueError(
                        f"source removal would leave background {background.name!r} on "
                        f"{owner.name!r} without its source; edit or remove that background link first"
                    )
