"""Import selected measurements and optional expression groups without Qt.

Expressions preserve source appearances and acquisition grouping. Repeated files are
the same measurement, rather than extra independent observations.
"""

from __future__ import annotations

import copy
import json
from collections import Counter
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
                 progress_callback):
    common = dict(normalization_path=normalization_path, mask_path=mask_path,
                  name=name, progress_callback=progress_callback)
    if family == "raw_dgs":
        from .raw_dgs import raw_dgs_dataset_group

        return raw_dgs_dataset_group(paths, **common)
    if family == "corelli":
        from .corelli import corelli_dataset_group

        common.pop("normalization_path")
        return corelli_dataset_group(paths, solid_angle_path=normalization_path,
                                     flux_path=flux_path, **common)
    from .mdevent import append_mdevent_file, mdevent_dataset_group

    group = mdevent_dataset_group(paths[0], **common)
    for path in paths[1:]:
        append_mdevent_file(group, path)
    return group


def import_source_selection(
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
    family_cache = {}
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
        if preserve_groups and any(value > 1 for value in counts.values()) and family not in {"raw_dgs", "mdevent"}:
            raise ValueError(f"repeated {family} sources in a summed group require a covariance-aware adapter")
        if paths and family != "ordinary":
            if flux_path is not None and family != "corelli":
                raise ValueError("an incident flux calibration applies only to CORELLI groups")
            resolved_normalization, resolved_mask = normalization_path, mask_path
            companion = None
            if normalization_path is None and mask_path is None:
                companion = _single_normalization_companion(paths[0])
                if companion is not None:
                    resolved_normalization = resolved_mask = companion
            child = _native_group(paths, family, name=title, normalization_path=resolved_normalization,
                                 mask_path=resolved_mask, flux_path=flux_path,
                                 progress_callback=progress_callback)
            if companion is not None:
                child.metadata["source_selection_calibration"] = {
                    "discovery": "single_van_companion", "source_file": paths[0],
                    "normalization_file": str(companion), "mask_file": str(companion),
                }
            if family == "mdevent" and scratch.lattice_parameters is None:
                scratch.lattice_parameters = copy.deepcopy(child.metadata["mdevent"]["lattice_parameters"])
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
            child = DatasetGroup(title, enabled=bool(paths)) if preserve_groups else collection
            if preserve_groups:
                collection.subgroups.append(child)
            if paths:
                import_dataset_paths(scratch, paths, into=child, data_type=data_type,
                                     importer_name=importer_name, importer_options=importer_options,
                                     stream_group_mode="new", progress_callback=progress_callback,
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
                "coaddition_coefficient": multiplicity if preserve_groups else 1,
            }
            if preserve_groups and multiplicity > 1:
                dataset.fit_weight *= multiplicity
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
    )
    collection.enabled = bool(collection.datasets) or any(child.enabled for child in collection.subgroups)
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
