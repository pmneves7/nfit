"""Public, metadata-only crystal orientation ownership and application.

Native reduction collections own the UB of their source runs. Ordinary data
already expressed in HKL retain their numerical coordinates when their crystal
metadata changes; a metadata edit does not reproject a baked histogram.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import numpy as np

from .pipeline import DataGroup, DatasetEntry, DatasetGroup
from .ub_orientation import (
    ipns_ub_from_mantid,
    mantid_ub_from_ipns,
    ub_setup_metadata,
)
from .ub_orientation import (
    ub_matrix_from_metadata as _matrix_from_metadata,
)
from .ub_orientation import (
    ub_setup_from_matrix as _setup,
)
from .ub_orientation import (
    with_ub_setup_metadata as with_ub_setup_metadata,
)


def _owner_path(root, target):
    if target is root:
        return [root]

    def find(node, path):
        for dataset in node.datasets:
            if dataset is target:
                return [*path, dataset]
        for child in node.subgroups:
            if child is target:
                return [*path, child]
            found = find(child, [*path, child])
            if found:
                return found
        return None

    path = find(root, [root])
    if path is None:
        raise ValueError("UB target must belong to the supplied workspace")
    return path


def _native_owner(root, target):
    from .reduction_recipes import reduction_family

    path = _owner_path(root, target)
    if isinstance(target, DatasetEntry) and len(path) > 1:
        parent = path[-2]
        if isinstance(parent, DatasetGroup) and reduction_family(parent):
            return parent, path[:-1]
    return target, path


def get_ub_setup(root: DataGroup, target: DataGroup | DatasetGroup | DatasetEntry) -> dict[str, Any] | None:
    """Read the effective UB and derive its canonical lattice and u/v values.

    This never loads source files. A native run reads its collection's shared
    transform. Explicit ordinary dataset metadata precedes imported payload
    metadata. A workspace matrix describes that workspace's composite only;
    it never replaces a child collection or dataset's transform.
    """
    from .reduction_recipes import effective_reduction_config, reduction_family

    owner, _ = _native_owner(root, target)
    native = isinstance(owner, DatasetGroup) and reduction_family(owner)
    if native:
        matrix = effective_reduction_config(owner).get("ub_matrix")
        if matrix is not None:
            if native == "macs-step-nexus":
                matrix = mantid_ub_from_ipns(matrix)
            return _setup(matrix)
    # Matrix-only scripts can restore automatic MACS orientation without
    # touching a previous dialog snapshot. That snapshot is not source UB.
    matrix = (
        None if native and owner.metadata.get("ub_setup_override")
        else _matrix_from_metadata(owner.metadata)
    )
    if matrix is not None and not owner.metadata.get("ub_setup_override") and (
        native == "macs-step-nexus" or owner.metadata.get("importer") == "macs_nexus"
    ):
        matrix = mantid_ub_from_ipns(matrix)
    if matrix is None and isinstance(owner, DatasetEntry):
        source_metadata = getattr(owner.data, "metadata", {}) or {}
        matrix = _matrix_from_metadata(source_metadata)
        if matrix is not None and source_metadata.get("importer") == "macs_nexus" and not source_metadata.get("ub_setup_override"):
            matrix = mantid_ub_from_ipns(matrix)
    if matrix is None and isinstance(owner, DatasetGroup):
        for dataset in owner.datasets:
            matrix = _matrix_from_metadata(dataset.metadata)
            source_metadata = dataset.metadata
            if matrix is None:
                source_metadata = getattr(dataset.data, "metadata", {}) or {}
                matrix = _matrix_from_metadata(source_metadata)
            if matrix is not None:
                if (native == "macs-step-nexus" or source_metadata.get("importer") == "macs_nexus") and not source_metadata.get("ub_setup_override"):
                    matrix = mantid_ub_from_ipns(matrix)
                break
    return None if matrix is None else _setup(matrix)


def _mark_stale(owner):
    from .project_composites import GROUP_COMPOSITE_BINNINGS_KEY
    from .project_rebinning import DATASET_REBIN_BINNINGS_KEY

    metadata = owner.parameters if isinstance(owner, DatasetEntry) else owner.metadata
    key = "rebin" if isinstance(owner, DatasetEntry) else "composite"
    registry = DATASET_REBIN_BINNINGS_KEY if isinstance(owner, DatasetEntry) else GROUP_COMPOSITE_BINNINGS_KEY
    config = metadata.get(key)
    if isinstance(config, dict):
        config["stale"] = True
    for item in metadata.get(registry, {}).get("items", ()):
        if isinstance(item.get("config"), dict):
            item["config"]["stale"] = True


def set_ub_setup(root: DataGroup, target: DataGroup | DatasetGroup | DatasetEntry,
                 setup: Mapping[str, Any] | None = None, *, ub: Any = None) -> dict[str, Any] | None:
    """Apply an authoritative matrix through the public reduction ownership API.

    Pass ``ub=matrix`` or a setup mapping containing ``ub_matrix``. Passing
    ``ub=None`` restores automatic source orientation for a MACS collection.
    Lattice and u/v are derived from that matrix, so scripts need only retain
    the matrix.
    Native run targets update their collection, invalidating coordinate-dependent
    histograms while preserving reusable raw laboratory events. MACS runs are
    reimported with the changed transform. Ordinary datasets and scope composites
    receive physical metadata for masks and viewers without changing stored HKL
    coordinates, intensities, or the source payload. Scope edits do not propagate
    into independent child reduction collections.
    """
    from .reduction_recipes import reduction_family, set_reduction_settings

    if ub is not None and setup is not None:
        raise ValueError("Supply either ub or setup, not both")
    owner, path = _native_owner(root, target)
    native = isinstance(owner, DatasetGroup) and reduction_family(owner)
    matrix = ub if setup is None else setup["ub_matrix"]
    if matrix is None:
        if native != "macs-step-nexus":
            raise ValueError("Only MACS collections support automatic source UB; an explicit matrix is required")
        set_reduction_settings(owner, {"ub_matrix": None})
        owner.metadata["macs"].pop("lattice_parameters", None)
        for key in ub_setup_metadata(_setup(np.eye(3))):
            owner.metadata.pop(key, None)
        for item in path:
            _mark_stale(item)
        return get_ub_setup(root, owner)
    canonical = _setup(matrix)
    if native:
        reduction_matrix = (
            ipns_ub_from_mantid(canonical["ub_matrix"]).tolist()
            if native == "macs-step-nexus" else canonical["ub_matrix"]
        )
        set_reduction_settings(owner, {"ub_matrix": reduction_matrix})
        for key in ("raw_dgs", "mdevent", "macs"):
            if isinstance(owner.metadata.get(key), dict):
                owner.metadata[key]["lattice_parameters"] = copy.deepcopy(canonical["lattice_parameters"])
    owner.metadata.update(ub_setup_metadata(canonical))
    if owner is root:
        root.lattice_parameters = copy.deepcopy(canonical["lattice_parameters"])
    for item in path:
        _mark_stale(item)
    return copy.deepcopy(canonical)
