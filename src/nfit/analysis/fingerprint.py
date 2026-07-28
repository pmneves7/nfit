from __future__ import annotations

import hashlib
import json
import weakref
from collections import OrderedDict
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..pipeline import DataGroup, DatasetEntry

_DATASET_FINGERPRINT_CACHE: OrderedDict[str, tuple[Any, str]] = OrderedDict()
_DATASET_FINGERPRINT_CACHE_LIMIT = 128
_DATA_ARRAY_HASH_CACHE: OrderedDict[
    int,
    tuple[weakref.ReferenceType[Any], Any, dict[str, Any]],
] = OrderedDict()
_DATA_ARRAY_HASH_CACHE_LIMIT = 64


def _identity_signature(value: Any) -> Any:
    """Represent configuration values cheaply while treating arrays as immutable."""

    if is_dataclass(value):
        return tuple(
            (item.name, _identity_signature(getattr(value, item.name))) for item in fields(value)
        )
    if isinstance(value, np.ndarray):
        return ("array", id(value), value.shape, str(value.dtype))
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return tuple(
            (str(key), _identity_signature(item))
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_identity_signature(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return (type(value).__name__, id(value))


def _fingerprint_cache_signature(dataset: DatasetEntry, group: DataGroup) -> Any:
    source = dataset.metadata.get("source_file")
    source_state = None
    if (
        source
        and Path(source).exists()
        and (dataset.data is None or dataset.data_matches_source)
    ):
        stat = Path(source).stat()
        source_state = (str(Path(source).resolve()), stat.st_size, stat.st_mtime_ns)
    data = dataset.data
    arrays = []
    if data is not None:
        for name in (
            "signal",
            "errors",
            "mask",
            "num_events",
            "H",
            "K",
            "L",
            "E",
            "intensity",
            "sigma",
        ):
            value = getattr(data, name, None)
            if isinstance(value, np.ndarray):
                arrays.append((name, id(value), value.shape, str(value.dtype)))
        columns = getattr(data, "columns", None)
        if isinstance(columns, dict):
            arrays.extend(
                (f"column:{name}", id(value), np.asarray(value).shape, str(np.asarray(value).dtype))
                for name, value in columns.items()
            )
    metadata = {
        key: value
        for key, value in dataset.metadata.items()
        if key not in {"source_file", "import_status", "export_file"}
    }
    return (
        source_state,
        dataset.data_cache_token,
        dataset.data_matches_source,
        tuple(arrays),
        dataset.id,
        dataset.kind,
        dataset.data_type,
        _identity_signature(dataset.parameters),
        _identity_signature(metadata),
        dataset.scale_factor,
        dataset.enabled,
        _identity_signature(dataset.masks),
        _identity_signature(group.masks),
        _identity_signature(group.lattice_parameters),
        group.spacegroup,
        _identity_signature(group.metadata.get("crystal")),
        _identity_signature(dataset.transforms),
    )


def _dataset_array_hashes(data: Any) -> dict[str, Any]:
    """Hash immutable dataset arrays once, independently of mutable settings."""

    key = id(data)
    signature = _identity_signature(data)
    cached = _DATA_ARRAY_HASH_CACHE.get(key)
    if cached is not None and cached[0]() is data and cached[1] == signature:
        _DATA_ARRAY_HASH_CACHE.move_to_end(key)
        return cached[2]
    arrays: dict[str, Any] = {}
    for name in (
        "signal",
        "errors",
        "mask",
        "num_events",
        "H",
        "K",
        "L",
        "E",
        "intensity",
        "sigma",
    ):
        value = getattr(data, name, None)
        if isinstance(value, np.ndarray):
            arrays[name] = array_hash(value)
    columns = getattr(data, "columns", None)
    if isinstance(columns, dict):
        arrays["columns"] = {name: array_hash(value) for name, value in columns.items()}
    _DATA_ARRAY_HASH_CACHE[key] = (weakref.ref(data), signature, arrays)
    _DATA_ARRAY_HASH_CACHE.move_to_end(key)
    while len(_DATA_ARRAY_HASH_CACHE) > _DATA_ARRAY_HASH_CACHE_LIMIT:
        _DATA_ARRAY_HASH_CACHE.popitem(last=False)
    return arrays


def canonical_json(value: Any) -> str:
    return json.dumps(_normalize(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def recipe_hash(
    operation_key: str, operation_version: int, parameters: dict[str, Any], input_ids: list[str]
) -> str:
    return sha256_json(
        {
            "operation": operation_key,
            "version": operation_version,
            "parameters": parameters,
            "inputs": input_ids,
        }
    )


def array_hash(array: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(repr(value.shape).encode())
    digest.update(value.view(np.uint8))
    return digest.hexdigest()


def file_fingerprint(path: str | Path, *, content_hash: bool = False) -> dict[str, Any]:
    source = Path(path)
    stat = source.stat()
    result: dict[str, Any] = {
        "path": str(source.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if content_hash:
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        result["sha256"] = digest.hexdigest()
    return result


def dataset_entry_fingerprint(dataset: DatasetEntry, group: DataGroup) -> str:
    cache_key = dataset.id
    cache_signature = _fingerprint_cache_signature(dataset, group)
    cached = _DATASET_FINGERPRINT_CACHE.get(cache_key)
    if cached is not None and cached[0] == cache_signature:
        _DATASET_FINGERPRINT_CACHE.move_to_end(cache_key)
        return cached[1]
    source = dataset.metadata.get("source_file")
    source_state = None
    if (
        source
        and Path(source).exists()
        and (dataset.data is None or dataset.data_matches_source)
    ):
        source_state = file_fingerprint(
            source, content_hash=bool(dataset.metadata.get("derived_from_analysis"))
        )
    arrays = {}
    if source_state is None and dataset.data is not None:
        arrays = _dataset_array_hashes(dataset.data)
    if dataset.transforms:
        raise ValueError("analysis fingerprints require materialized dataset transforms")
    result = sha256_json(
        {
            "id": dataset.id,
            "kind": dataset.kind,
            "data_type": dataset.data_type,
            "source": source_state,
            "arrays": arrays,
            "parameters": dataset.parameters,
            "metadata": {
                key: value
                for key, value in dataset.metadata.items()
                if key not in {"source_file", "import_status", "export_file"}
            },
            "scale_factor": dataset.scale_factor,
            "enabled": dataset.enabled,
            "masks": [asdict(mask) for mask in dataset.masks],
            "group_masks": [asdict(mask) for mask in group.masks],
            "lattice": group.lattice_parameters,
            "spacegroup": group.spacegroup,
            "crystal": group.metadata.get("crystal"),
        }
    )
    _DATASET_FINGERPRINT_CACHE[cache_key] = (cache_signature, result)
    _DATASET_FINGERPRINT_CACHE.move_to_end(cache_key)
    while len(_DATASET_FINGERPRINT_CACHE) > _DATASET_FINGERPRINT_CACHE_LIMIT:
        _DATASET_FINGERPRINT_CACHE.popitem(last=False)
    return result


def _normalize(value: Any) -> Any:
    if is_dataclass(value):
        return _normalize(asdict(value))
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "shape": list(value.shape), "sha256": array_hash(value)}
    if isinstance(value, np.generic):
        return _normalize(value.item())
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("fingerprints require finite numeric values")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"cannot fingerprint {type(value).__name__}")
