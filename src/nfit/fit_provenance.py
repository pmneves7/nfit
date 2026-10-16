"""Content fingerprints for continuing a recorded scientific fit target.

The digest is deliberately conservative: changes to prepared observations,
their statistical payloads, or model configuration require a fresh chain.
Arrays are hashed in bounded chunks rather than serialized into Python lists.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import fields, is_dataclass
from pathlib import PurePath
from typing import Any

import numpy as np


def model_configuration_for_target(configuration: Mapping[str, Any]) -> dict[str, Any]:
    """Copy scientific settings, omitting certificates produced by evaluation.

    Meshes, interpolation tolerances and backend choices still determine the
    numerical target. Their derived certification records do not: evaluating an
    unchanged model must not invalidate its own posterior or search identity.
    """
    derived = {"response_sampling_certificate", "response_q_interpolation_certificates"}
    return {key: deepcopy(value) for key, value in configuration.items() if key not in derived}


def _hash_value(digest: Any, value: Any) -> None:
    """Encode typed values unambiguously without copying a whole array."""
    if value is None:
        digest.update(b"N")
    elif isinstance(value, (bool, np.bool_)):
        digest.update(b"T" if value else b"F")
    elif isinstance(value, (int, np.integer)):
        _hash_value(digest, str(int(value)))
        digest.update(b"I")
    elif isinstance(value, (float, np.floating)):
        digest.update(b"D" + struct.pack("!d", float(value)))
    elif isinstance(value, (str, PurePath)):
        encoded = str(value).encode("utf-8")
        digest.update(b"S" + struct.pack("!Q", len(encoded)) + encoded)
    elif isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ValueError("Fit provenance cannot contain object-valued arrays")
        digest.update(b"A")
        _hash_value(digest, value.dtype.str)
        _hash_value(digest, value.shape)
        # nditer handles strided and mapped arrays with a bounded copy buffer.
        with np.nditer(
            value,
            flags=["external_loop", "buffered", "zerosize_ok"],
            op_flags=["readonly"],
            order="C",
            buffersize=131072,
        ) as chunks:
            for chunk in chunks:
                digest.update(chunk.tobytes(order="C"))
    elif isinstance(value, Mapping):
        digest.update(b"M" + struct.pack("!Q", len(value)))
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Fit provenance mappings require string keys")
        for key in sorted(value):
            _hash_value(digest, key)
            _hash_value(digest, value[key])
    elif isinstance(value, (tuple, list)):
        digest.update(b"L" + struct.pack("!Q", len(value)))
        for item in value:
            _hash_value(digest, item)
    elif is_dataclass(value) and not isinstance(value, type):
        _hash_value(
            digest,
            {field.name: getattr(value, field.name) for field in fields(value) if field.init},
        )
    else:
        raise ValueError(
            f"Fit provenance requires serializable scientific configuration; "
            f"found {type(value).__name__}"
        )


def fit_problem_fingerprint(
    problem: Any,
    *,
    model_configuration: Any = None,
    require_positive_sigma: bool = True,
) -> str:
    """Fingerprint the prepared observations and declared fitting target.

    Declaratively compiled problems carry their model configuration. For custom
    callable models, callers must supply a serializable ``model_configuration``
    describing the complete model and resolution behavior; opaque Python
    callables cannot provide persistent scientific identity automatically.
    Varied starting values are excluded because they do not change the target.
    Fixed values, bounds, constraints, bindings, and all point metadata are
    included. No original dataset payload is retained by this function.
    """
    if model_configuration is None:
        model_configuration = problem.metadata.get("fit_model_configuration")
    if model_configuration is None:
        raise ValueError("A fit target fingerprint requires declared model configuration")
    digest = hashlib.sha256()
    _hash_value(digest, "nfit-fit-target-v1")
    _hash_value(digest, model_configuration)
    _hash_value(digest, bool(require_positive_sigma))
    _hash_value(
        digest,
        [
            {
                "name": spec.name,
                "value": None if spec.vary else spec.value,
                "min": spec.min,
                "max": spec.max,
                "vary": spec.vary,
                "unit": spec.unit,
            }
            for spec in problem.parameter_specs
        ],
    )
    _hash_value(digest, problem.derived_parameters)
    for dataset in problem.datasets:
        _hash_value(
            digest,
            {
                "name": dataset.name,
                "weight": dataset.weight,
                "likelihood": dataset.likelihood,
                "parameter_bindings": dataset.parameter_bindings,
                "data_scale_parameter": dataset.data_scale_parameter,
            },
        )
        points = dataset.prepared_valid(require_positive_sigma=require_positive_sigma)
        _hash_value(
            digest,
            {
                name: getattr(points, name)
                for name in (
                    "H",
                    "K",
                    "L",
                    "E",
                    "intensity",
                    "sigma",
                    "mask",
                    "temperature",
                    "magnetic_field",
                    "metadata",
                    "normalization_denominator",
                    "measurement_payload",
                    "source_dependencies",
                )
            },
        )
    return digest.hexdigest()
