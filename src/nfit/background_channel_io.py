"""Binary persistence for sparse background visualization recovery data."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from .background_channels import BACKGROUND_EXCEPTIONS_KEY

_FIELDS = "background_original_fields_json"
_PREFIX = "background_original_"


def background_metadata_payload(metadata: dict) -> tuple[dict, dict[str, np.ndarray]]:
    """Separate numerical recovery arrays from ordinary JSON metadata."""
    exceptions = metadata.get(BACKGROUND_EXCEPTIONS_KEY)
    if not isinstance(exceptions, dict) or not exceptions:
        return metadata, {}
    remaining = dict(metadata)
    remaining.pop(BACKGROUND_EXCEPTIONS_KEY)
    arrays = {
        f"{_PREFIX}{name}": np.asarray(value)
        for name, value in exceptions.items()
    }
    arrays[_FIELDS] = np.asarray(json.dumps(list(exceptions)))
    return remaining, arrays


def restore_background_metadata(metadata: dict, archive: Any) -> dict:
    """Restore sparse arrays without making them writable or expanding them."""
    if _FIELDS not in archive:
        return metadata
    fields = json.loads(str(np.asarray(archive[_FIELDS]).item()))
    exceptions = {}
    for name in fields:
        array = np.asarray(archive[f"{_PREFIX}{name}"])
        if array.dtype.hasobject:
            raise ValueError("background recovery arrays must be numerical")
        array.setflags(write=False)
        exceptions[name] = array
    return {**metadata, BACKGROUND_EXCEPTIONS_KEY: exceptions}
