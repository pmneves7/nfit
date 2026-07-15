from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..pipeline import DataGroup, DatasetEntry


def canonical_json(value: Any) -> str:
    return json.dumps(_normalize(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def recipe_hash(operation_key: str, operation_version: int, parameters: dict[str, Any], input_ids: list[str]) -> str:
    return sha256_json({"operation": operation_key, "version": operation_version, "parameters": parameters, "inputs": input_ids})


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
    result: dict[str, Any] = {"path": str(source.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if content_hash:
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        result["sha256"] = digest.hexdigest()
    return result


def dataset_entry_fingerprint(dataset: DatasetEntry, group: DataGroup) -> str:
    source = dataset.metadata.get("source_file")
    source_state = None
    if source and Path(source).exists():
        source_state = file_fingerprint(
            source, content_hash=bool(dataset.metadata.get("derived_from_analysis"))
        )
    arrays = {}
    if source_state is None and dataset.data is not None:
        for name in (
            "signal", "errors", "mask", "num_events", "H", "K", "L", "E",
            "intensity", "sigma",
        ):
            value = getattr(dataset.data, name, None)
            if isinstance(value, np.ndarray):
                arrays[name] = array_hash(value)
        columns = getattr(dataset.data, "columns", None)
        if isinstance(columns, dict):
            arrays["columns"] = {name: array_hash(value) for name, value in columns.items()}
    if dataset.transforms:
        raise ValueError("analysis fingerprints require materialized dataset transforms")
    return sha256_json(
        {
            "id": dataset.id, "kind": dataset.kind, "data_type": dataset.data_type,
            "source": source_state, "arrays": arrays, "parameters": dataset.parameters,
            "metadata": {key: value for key, value in dataset.metadata.items() if key not in {"source_file", "import_status", "export_file"}}, "scale_factor": dataset.scale_factor,
            "enabled": dataset.enabled, "masks": [asdict(mask) for mask in dataset.masks],
            "group_masks": [asdict(mask) for mask in group.masks],
            "lattice": group.lattice_parameters, "spacegroup": group.spacegroup,
            "crystal": group.metadata.get("crystal"),
        }
    )


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
