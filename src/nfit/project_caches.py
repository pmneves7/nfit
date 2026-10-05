"""Explicit cache lifecycle for project data and evaluated model intermediates."""

from collections import OrderedDict

from . import electronic_structure, project_data
from .analysis import fingerprint
from .raw_dgs_cache import clear_reduced_event_cache

# Compatibility aliases in project_gui refer to these same stores.
MODEL_OVERLAY_CACHE: OrderedDict[int, dict] = OrderedDict()
MODEL_OVERLAY_ERRORS: OrderedDict[int, dict] = OrderedDict()
PROJECT_CACHE_BINNINGS_KEY = "cache_binnings"
PROJECT_BINNING_CACHE_ENTRIES_KEY = "binning_cache_entries"


def release_project_memory(project):
    """Retire runtime caches when a project closes, without editing its recipes."""
    nodes = [node for group in project.data_groups for node in (group, *group.iter_subgroups())]
    datasets = [dataset for group in project.data_groups for dataset in group.iter_datasets()]
    ids = {dataset.id for dataset in datasets}
    node_ids = {id(node) for node in nodes}
    def owner(key):
        while isinstance(key, tuple) and key:
            key = key[0]
        return key
    project_data._VIEWER_VIEW_CACHE.discard_matching(
        lambda key: isinstance(key, str) and key.split(":", 1)[0] in ids)
    project_data._COMPOSITE_DATA_CACHE.discard_matching(lambda key: owner(key) in node_ids)
    for dataset in datasets:
        project_data._PREPARED_POINT_LIST_CACHE.pop(dataset.id, None)
    for node in nodes:
        MODEL_OVERLAY_CACHE.pop(id(node), None)
        MODEL_OVERLAY_ERRORS.pop(id(node), None)
        for component in node.models.values() if hasattr(node, "models") else ():
            component.__dict__.pop("_nfit_electronic_model_cache", None)


def _mark_binnings_stale(container, fit_key, registry_key) -> None:
    """Mark existing recipes only, without normalization or data preparation."""
    config = container.get(fit_key)
    if isinstance(config, dict):
        config["stale"] = True
    registry = container.get(registry_key)
    if isinstance(registry, dict):
        for item in registry.get("items", ()):
            if isinstance(item, dict) and isinstance(item.get("config"), dict):
                item["config"]["stale"] = True


def clear_project_caches(project) -> None:
    """Discard recomputable project caches without changing canonical data.

    Disable embedded binning caches so the next save removes cached artifacts
    rather than rebuilding them. Shared electronic geometry and fingerprint
    memoization is also flushed; other projects' numerical results stay intact.
    """
    groups = list(project.data_groups)
    nodes = list(groups)
    for group in groups:
        nodes.extend(group.iter_subgroups())
    datasets = [dataset for group in groups for dataset in group.iter_datasets()]
    dataset_ids = {dataset.id for dataset in datasets}
    node_ids = {id(node) for node in nodes}

    project_data._VIEWER_VIEW_CACHE.discard_matching(
        lambda key: isinstance(key, str) and key.split(":", 1)[0] in dataset_ids
    )
    project_data._COMPOSITE_DATA_CACHE.discard_matching(
        lambda key: (key[0] if isinstance(key, tuple) else key) in node_ids
    )
    for dataset in datasets:
        project_data._PREPARED_POINT_LIST_CACHE.pop(dataset.id, None)
        clear_reduced_event_cache(dataset)
        _mark_binnings_stale(
            dataset.parameters, project_data.DATASET_REBIN_KEY,
            project_data.DATASET_REBIN_BINNINGS_KEY,
        )
    for node in nodes:
        _mark_binnings_stale(
            node.metadata, project_data.GROUP_COMPOSITE_KEY,
            project_data.GROUP_COMPOSITE_BINNINGS_KEY,
        )
        for component in node.models.values() if hasattr(node, "models") else ():
            component.__dict__.pop("_nfit_electronic_model_cache", None)
        MODEL_OVERLAY_CACHE.pop(id(node), None)
        MODEL_OVERLAY_ERRORS.pop(id(node), None)

    with electronic_structure._FOURIER_CACHE_LOCK:
        electronic_structure._FOURIER_COEFFICIENT_CACHE.clear()
    with electronic_structure._HAMILTONIAN_COMPONENT_CACHE_LOCK:
        electronic_structure._HAMILTONIAN_COMPONENT_CACHE.clear()
    electronic_structure._mesh_orbit_representatives.cache_clear()
    fingerprint._DATASET_FINGERPRINT_CACHE.clear()
    fingerprint._DATA_ARRAY_HASH_CACHE.clear()
    project.settings.pop(PROJECT_BINNING_CACHE_ENTRIES_KEY, None)
    project.settings[PROJECT_CACHE_BINNINGS_KEY] = False
