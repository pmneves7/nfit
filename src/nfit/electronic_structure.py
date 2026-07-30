"""Material-independent tight-binding models and electronic-structure results.

The canonical Fourier convention is the Wannier gauge

``H(k) = sum_R H(R) exp(2 pi i k.R)``.

Orbital centres are metadata and do not enter this phase. Energies are stored
in meV, direct-space vectors in Angstrom, and reduced wavevectors are
dimensionless.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, fields, replace
from functools import cached_property
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from pprint import pformat
from threading import RLock
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .cache_utils import lru_store
from .electronic_backends import ElectronicBackend, evaluate_eigensystem

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
ELECTRONIC_ENERGY_UNITS = ("eV", "meV")

_FOURIER_CACHE_MAX_BYTES = 256 * 1024**2
_FOURIER_CACHE_MAX_ENTRIES = 64
_FOURIER_COEFFICIENT_CACHE: OrderedDict[tuple[str, str], ComplexArray] = (
    OrderedDict()
)
_FOURIER_CACHE_LOCK = RLock()
_HAMILTONIAN_COMPONENT_CACHE_MAX_BYTES = 512 * 1024**2
_HAMILTONIAN_COMPONENT_CACHE_MAX_ENTRIES = 32
_HAMILTONIAN_COMPONENT_CACHE: OrderedDict[
    tuple[str, str], tuple[tuple[str, ...], ComplexArray]
] = OrderedDict()
_HAMILTONIAN_COMPONENT_CACHE_LOCK = RLock()


def _numeric_digest(value: ArrayLike, dtype: Any) -> str:
    array = np.ascontiguousarray(value, dtype=dtype)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _fourier_coefficients(
    translations: NDArray[np.int64],
    weights: FloatArray,
    wavevectors: FloatArray,
) -> ComplexArray:
    """Return cached, parameter-independent Fourier coefficients."""

    structure_digest = _numeric_digest(translations, np.int64) + _numeric_digest(
        weights, np.float64
    )
    key = (structure_digest, _numeric_digest(wavevectors, np.float64))
    with _FOURIER_CACHE_LOCK:
        cached = _FOURIER_COEFFICIENT_CACHE.get(key)
        if cached is not None:
            _FOURIER_COEFFICIENT_CACHE.move_to_end(key)
            return cached
    coefficients = np.exp(
        2j * np.pi * wavevectors @ np.asarray(translations, dtype=float).T
    )
    coefficients *= np.asarray(weights, dtype=float)[None, :]
    coefficients.setflags(write=False)
    with _FOURIER_CACHE_LOCK:
        lru_store(
            _FOURIER_COEFFICIENT_CACHE,
            key,
            coefficients,
            _FOURIER_CACHE_MAX_ENTRIES,
            max_array_bytes=_FOURIER_CACHE_MAX_BYTES,
        )
    return coefficients


def _momentum_hamiltonian_components(
    model: ElectronicModel,
    wavevectors: FloatArray,
) -> tuple[tuple[str, ...], ComplexArray] | None:
    """Return cached ``H_0(k)`` and named ``dH(k)/dp`` matrices.

    The cache key excludes the current parameter values, so every trial point
    in a fit can reuse the same Fourier-transformed Hamiltonian basis.
    """

    component_bytes = (
        16
        * (len(model.parameter_blocks) + 1)
        * int(wavevectors.shape[0])
        * model.n_basis
        * model.n_basis
    )
    if component_bytes > _HAMILTONIAN_COMPONENT_CACHE_MAX_BYTES:
        return None
    key = (model.structure_digest, _numeric_digest(wavevectors, np.float64))
    with _HAMILTONIAN_COMPONENT_CACHE_LOCK:
        cached = _HAMILTONIAN_COMPONENT_CACHE.get(key)
        if cached is not None:
            _HAMILTONIAN_COMPONENT_CACHE.move_to_end(key)
            return cached
    names = tuple(model.parameter_blocks)
    real_space = np.stack(
        (
            model.hamiltonian_blocks,
            *(model.parameter_blocks[name] for name in names),
        ),
        axis=0,
    )
    coefficients = _fourier_coefficients(
        model.translations,
        model.interpolation_weights,
        wavevectors,
    )
    components = np.einsum(
        "kr,crij->ckij",
        coefficients,
        real_space,
        optimize=True,
    )
    components.setflags(write=False)
    result = (names, components)
    with _HAMILTONIAN_COMPONENT_CACHE_LOCK:
        lru_store(
            _HAMILTONIAN_COMPONENT_CACHE,
            key,
            result,
            _HAMILTONIAN_COMPONENT_CACHE_MAX_ENTRIES,
            max_array_bytes=_HAMILTONIAN_COMPONENT_CACHE_MAX_BYTES,
        )
    return result


def normalize_electronic_energy_unit(unit: str) -> Literal["eV", "meV"]:
    """Return the canonical spelling of a supported electronic energy unit."""

    normalized = str(unit).strip().casefold()
    if normalized == "ev":
        return "eV"
    if normalized == "mev":
        return "meV"
    raise ValueError("electronic energy unit must be 'eV' or 'meV'")


def electronic_energy_to_meV(values: ArrayLike, unit: str) -> float | FloatArray:
    """Convert explicitly declared electronic energies to canonical meV."""

    canonical = normalize_electronic_energy_unit(unit)
    result = np.asarray(values, dtype=float) * (1000.0 if canonical == "eV" else 1.0)
    if result.ndim == 0:
        return float(result)
    return result


def electronic_energy_from_meV(values: ArrayLike, unit: str) -> float | FloatArray:
    """Convert canonical meV energies to an electronic input or display unit."""

    canonical = normalize_electronic_energy_unit(unit)
    result = np.asarray(values, dtype=float) / (1000.0 if canonical == "eV" else 1.0)
    if result.ndim == 0:
        return float(result)
    return result


def set_electronic_energy_unit(component: Any, unit: str) -> str:
    """Set a component's electronic input/display unit without rescaling data."""

    canonical = normalize_electronic_energy_unit(unit)
    config = getattr(component, "config", None)
    if not isinstance(config, dict):
        raise TypeError("component must provide a mutable config dictionary")
    config["electronic_energy_unit"] = canonical
    return canonical


def _readonly(array: ArrayLike, dtype: Any) -> np.ndarray:
    result = np.array(array, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, np.ndarray):
        result = np.array(value, copy=True)
        result.setflags(write=False)
        return result
    if isinstance(value, np.generic):
        return value.item()
    return deepcopy(value)


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return deepcopy(value)


def _complex_payload(values: ComplexArray) -> dict[str, Any]:
    real = np.where(values.real == 0.0, 0.0, values.real)
    imaginary = np.where(values.imag == 0.0, 0.0, values.imag)
    return {"real": real.tolist(), "imag": imaginary.tolist()}


def _complex_array(payload: Mapping[str, Any]) -> ComplexArray:
    return np.asarray(payload["real"], dtype=float) + 1j * np.asarray(
        payload["imag"], dtype=float
    )


@dataclass(frozen=True)
class BasisState:
    """Metadata for one arbitrary orthonormal electronic basis state."""

    label: str
    site: str = ""
    species: str = ""
    orbital: str = ""
    correlated_shell: str = ""
    spin: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "site": self.site,
            "species": self.species,
            "orbital": self.orbital,
            "correlated_shell": self.correlated_shell,
            "spin": self.spin,
            "metadata": _thaw(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BasisState:
        return cls(
            label=str(payload["label"]),
            site=str(payload.get("site", "")),
            species=str(payload.get("species", "")),
            orbital=str(payload.get("orbital", "")),
            correlated_shell=str(payload.get("correlated_shell", "")),
            spin=str(payload.get("spin", "")),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class ElectronicModel:
    """Immutable orthonormal tight-binding Hamiltonian, canonically in meV."""

    direct_lattice: FloatArray
    basis: tuple[BasisState, ...]
    translations: NDArray[np.int64]
    hamiltonian_blocks: ComplexArray
    interpolation_weights: FloatArray
    orbital_centers: FloatArray
    periodic_axes: tuple[int, ...] = (0, 1, 2)
    parameter_values: Mapping[str, float] = field(default_factory=dict)
    parameter_blocks: Mapping[str, ComplexArray] = field(default_factory=dict)
    spin_operators: ComplexArray | None = None
    energy_zero_meV: float = 0.0
    provenance: Mapping[str, Any] = field(default_factory=dict)
    fourier_gauge: str = "wannier"

    def __post_init__(self) -> None:
        lattice = _readonly(self.direct_lattice, float)
        translations = _readonly(self.translations, np.int64)
        blocks = _readonly(self.hamiltonian_blocks, np.complex128)
        weights = _readonly(self.interpolation_weights, float)
        centers = _readonly(self.orbital_centers, float)
        spin = (
            None
            if self.spin_operators is None
            else _readonly(self.spin_operators, np.complex128)
        )
        object.__setattr__(self, "direct_lattice", lattice)
        object.__setattr__(self, "translations", translations)
        object.__setattr__(self, "hamiltonian_blocks", blocks)
        object.__setattr__(self, "interpolation_weights", weights)
        object.__setattr__(self, "orbital_centers", centers)
        object.__setattr__(self, "spin_operators", spin)
        object.__setattr__(
            self,
            "parameter_values",
            MappingProxyType(
                {str(name): float(value) for name, value in self.parameter_values.items()}
            ),
        )
        object.__setattr__(
            self,
            "parameter_blocks",
            MappingProxyType(
                {
                    str(name): _readonly(values, np.complex128)
                    for name, values in self.parameter_blocks.items()
                }
            ),
        )
        object.__setattr__(self, "provenance", _freeze(self.provenance))
        self._validate()

    @property
    def n_basis(self) -> int:
        return len(self.basis)

    @property
    def dimension(self) -> int:
        return len(self.periodic_axes)

    @cached_property
    def reciprocal_lattice(self) -> FloatArray:
        """Return reciprocal vectors as columns in Angstrom^-1."""

        result = 2.0 * np.pi * np.linalg.inv(self.direct_lattice).T
        result.setflags(write=False)
        return result

    def _validate(self) -> None:
        if self.direct_lattice.shape != (3, 3):
            raise ValueError("direct_lattice must have shape (3, 3)")
        if not np.all(np.isfinite(self.direct_lattice)):
            raise ValueError("direct_lattice must be finite")
        if abs(float(np.linalg.det(self.direct_lattice))) < 1.0e-12:
            raise ValueError("direct_lattice must be invertible")
        if not self.basis or len({item.label for item in self.basis}) != len(self.basis):
            raise ValueError("basis-state labels must be nonempty and unique")
        if self.translations.ndim != 2 or self.translations.shape[1] != 3:
            raise ValueError("translations must have shape (n_R, 3)")
        expected = (self.translations.shape[0], self.n_basis, self.n_basis)
        if self.hamiltonian_blocks.shape != expected:
            raise ValueError(f"hamiltonian_blocks must have shape {expected}")
        if self.interpolation_weights.shape != (self.translations.shape[0],):
            raise ValueError("interpolation_weights must contain one value per block")
        if np.any(~np.isfinite(self.interpolation_weights)) or np.any(
            self.interpolation_weights <= 0.0
        ):
            raise ValueError("interpolation weights must be finite and positive")
        if self.orbital_centers.shape != (self.n_basis, 3):
            raise ValueError("orbital_centers must have shape (n_basis, 3)")
        if not self.periodic_axes or any(axis not in (0, 1, 2) for axis in self.periodic_axes):
            raise ValueError("periodic_axes must contain one to three Cartesian indices")
        if len(set(self.periodic_axes)) != len(self.periodic_axes):
            raise ValueError("periodic_axes must be unique")
        nonperiodic = set((0, 1, 2)) - set(self.periodic_axes)
        if nonperiodic and np.any(self.translations[:, sorted(nonperiodic)] != 0):
            raise ValueError("translations must vanish along nonperiodic axes")
        if self.spin_operators is not None:
            if self.spin_operators.shape != (3, self.n_basis, self.n_basis):
                raise ValueError("spin_operators must have shape (3, n_basis, n_basis)")
            for operator in self.spin_operators:
                if not np.allclose(operator, operator.conj().T, rtol=1e-10, atol=1e-10):
                    raise ValueError("spin operators must be Hermitian")
        if set(self.parameter_values) != set(self.parameter_blocks):
            raise ValueError("parameter values and parameter Hamiltonian terms must match")
        if any(not np.isfinite(value) for value in self.parameter_values.values()):
            raise ValueError("electronic-model parameters must be finite")
        for name, blocks in self.parameter_blocks.items():
            if blocks.shape != expected:
                raise ValueError(
                    f"parameter term {name!r} must have shape {expected}"
                )
        if self.fourier_gauge != "wannier":
            raise ValueError("the canonical electronic model uses the Wannier gauge")

        scale = max(1.0, float(np.max(np.abs(self.hamiltonian_blocks))))
        self._validate_hermitian_blocks(self.hamiltonian_blocks, scale, "H")
        for name, blocks in self.parameter_blocks.items():
            self._validate_hermitian_blocks(blocks, scale, f"dH/d{name}")

    def _validate_hermitian_blocks(
        self, blocks: ComplexArray, scale: float, label: str
    ) -> None:
        weighted = {
            tuple(vector): weight * block
            for vector, weight, block in zip(
                self.translations,
                self.interpolation_weights,
                blocks,
                strict=True,
            )
        }
        if len(weighted) != len(self.translations):
            raise ValueError("translations must be unique")
        for vector, block in weighted.items():
            partner = weighted.get(tuple(-np.asarray(vector, dtype=int)))
            if partner is None or not np.allclose(
                partner, block.conj().T, rtol=1e-9, atol=1e-9 * scale
            ):
                raise ValueError(
                    f"{label}(-R) = {label}(R)^dagger is violated for R={vector}"
                )

    def with_parameters(
        self, *, energy_unit: str, **values: float
    ) -> ElectronicModel:
        """Return a model with named hopping parameters converted from ``energy_unit``."""

        unknown = set(values) - set(self.parameter_values)
        if unknown:
            raise KeyError(f"unknown electronic-model parameter {sorted(unknown)[0]!r}")
        updated = dict(self.parameter_values)
        updated.update(
            {
                name: float(electronic_energy_to_meV(value, energy_unit))
                for name, value in values.items()
            }
        )
        return self._with_parameter_values_meV(updated)

    def _with_parameter_values_meV(
        self,
        values: Mapping[str, float],
    ) -> ElectronicModel:
        """Reuse validated immutable structure with new canonical parameter values."""

        if set(values) != set(self.parameter_values):
            raise ValueError("parameter values must preserve the model parameter names")
        normalized = {str(name): float(value) for name, value in values.items()}
        if any(not np.isfinite(value) for value in normalized.values()):
            raise ValueError("electronic-model parameters must be finite")
        if normalized == dict(self.parameter_values):
            return self

        updated = object.__new__(type(self))
        for model_field in fields(self):
            value = (
                MappingProxyType(normalized)
                if model_field.name == "parameter_values"
                else getattr(self, model_field.name)
            )
            object.__setattr__(updated, model_field.name, value)
        if "structure_digest" in self.__dict__:
            object.__setattr__(
                updated,
                "structure_digest",
                self.__dict__["structure_digest"],
            )
        return updated

    @cached_property
    def resolved_hamiltonian_blocks(self) -> ComplexArray:
        """Return immutable real-space blocks with named coefficients applied."""

        resolved = np.array(self.hamiltonian_blocks, copy=True)
        for name, value in self.parameter_values.items():
            resolved += value * self.parameter_blocks[name]
        resolved.setflags(write=False)
        return resolved

    def hamiltonian(self, reduced_k: ArrayLike) -> ComplexArray:
        """Evaluate ``H(k)`` at one or more reduced wavevectors."""

        wavevectors = np.asarray(reduced_k, dtype=float)
        single = wavevectors.ndim == 1
        wavevectors = np.atleast_2d(wavevectors)
        if wavevectors.shape[1] != 3 or not np.all(np.isfinite(wavevectors)):
            raise ValueError("reduced_k must have shape (n_k, 3) and be finite")
        component_result = _momentum_hamiltonian_components(
            self,
            wavevectors,
        )
        if component_result is None:
            coefficients = _fourier_coefficients(
                self.translations,
                self.interpolation_weights,
                wavevectors,
            )
            result = np.einsum(
                "kr,rij->kij",
                coefficients,
                self.resolved_hamiltonian_blocks,
                optimize=True,
            )
        else:
            names, components = component_result
            coefficients = np.asarray(
                [1.0, *(self.parameter_values[name] for name in names)],
                dtype=float,
            )
            result = np.tensordot(coefficients, components, axes=(0, 0))
        if not np.allclose(result, result.swapaxes(1, 2).conj(), rtol=1e-9, atol=1e-8):
            raise ValueError("interpolated Hamiltonian is not Hermitian")
        return result[0] if single else result

    def _scientific_structure_payload(self) -> dict[str, Any]:
        """Return the parameter-value-independent scientific definition."""

        return {
            "schema_version": 1,
            "direct_lattice": self.direct_lattice.tolist(),
            "basis": [item.to_dict() for item in self.basis],
            "translations": self.translations.tolist(),
            "hamiltonian_blocks": _complex_payload(self.hamiltonian_blocks),
            "interpolation_weights": self.interpolation_weights.tolist(),
            "orbital_centers": self.orbital_centers.tolist(),
            "periodic_axes": list(self.periodic_axes),
            "parameter_blocks": {
                name: _complex_payload(values)
                for name, values in self.parameter_blocks.items()
            },
            "spin_operators": (
                None
                if self.spin_operators is None
                else _complex_payload(self.spin_operators)
            ),
            "energy_zero_meV": float(self.energy_zero_meV),
            "fourier_gauge": self.fourier_gauge,
        }

    @cached_property
    def structure_digest(self) -> str:
        """Hash the immutable Hamiltonian structure, excluding trial values."""

        encoded = json.dumps(
            self._scientific_structure_payload(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _scientific_payload(self) -> dict[str, Any]:
        return {
            **self._scientific_structure_payload(),
            "parameter_values": dict(self.parameter_values),
        }

    @cached_property
    def content_digest(self) -> str:
        """Hash one parameter point without reserializing fixed model arrays."""

        encoded = json.dumps(
            {
                "structure_digest": self.structure_digest,
                "parameter_values": dict(self.parameter_values),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _legacy_content_digest(self) -> str:
        """Return the pre-0.49 digest accepted in existing project files."""

        encoded = json.dumps(
            self._scientific_payload(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        payload = self._scientific_payload()
        payload["canonical_energy_unit"] = "meV"
        payload["provenance"] = _thaw(self.provenance)
        payload["content_digest"] = self.content_digest
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ElectronicModel:
        unit = str(payload.get("canonical_energy_unit", "meV"))
        if normalize_electronic_energy_unit(unit) != "meV":
            raise ValueError("serialized electronic models must use canonical meV")
        model = cls(
            direct_lattice=np.asarray(payload["direct_lattice"], dtype=float),
            basis=tuple(BasisState.from_dict(item) for item in payload["basis"]),
            translations=np.asarray(payload["translations"], dtype=np.int64),
            hamiltonian_blocks=_complex_array(payload["hamiltonian_blocks"]),
            interpolation_weights=np.asarray(
                payload["interpolation_weights"], dtype=float
            ),
            orbital_centers=np.asarray(payload["orbital_centers"], dtype=float),
            periodic_axes=tuple(int(value) for value in payload["periodic_axes"]),
            parameter_values={
                str(name): float(value)
                for name, value in payload.get("parameter_values", {}).items()
            },
            parameter_blocks={
                str(name): _complex_array(values)
                for name, values in payload.get("parameter_blocks", {}).items()
            },
            spin_operators=(
                None
                if payload.get("spin_operators") is None
                else _complex_array(payload["spin_operators"])
            ),
            energy_zero_meV=float(payload.get("energy_zero_meV", 0.0)),
            provenance=dict(payload.get("provenance", {})),
            fourier_gauge=str(payload.get("fourier_gauge", "wannier")),
        )
        expected = payload.get("content_digest")
        if expected is not None and str(expected) not in {
            model.content_digest,
            model._legacy_content_digest(),
        }:
            raise ValueError("electronic-model content digest does not match its payload")
        return model


def reduce_electronic_model_to_primitive(
    model: ElectronicModel,
    primitive_lattice: ArrayLike,
    *,
    tolerance: float = 1.0e-8,
) -> ElectronicModel:
    """Fold a three-dimensional supercell model onto a primitive lattice.

    Basis centers and real-space Hamiltonian blocks are transformed exactly.
    The operation is conservative: it returns ``model`` when the supplied
    lattice has the same volume and raises if the basis cannot be partitioned
    into complete translation-equivalent groups.
    """

    primitive = np.asarray(primitive_lattice, dtype=float)
    if primitive.shape != (3, 3) or not np.all(np.isfinite(primitive)):
        raise ValueError("primitive_lattice must be a finite 3x3 matrix")
    if tuple(model.periodic_axes) != (0, 1, 2):
        return model
    conventional_volume = abs(float(np.linalg.det(model.direct_lattice)))
    primitive_volume = abs(float(np.linalg.det(primitive)))
    if primitive_volume <= tolerance:
        raise ValueError("primitive_lattice must be invertible")
    multiplicity_float = conventional_volume / primitive_volume
    multiplicity = int(round(multiplicity_float))
    if not np.isclose(multiplicity_float, multiplicity, atol=tolerance):
        raise ValueError("primitive lattice must have an integer cell multiplicity")
    if multiplicity <= 1:
        return model

    transform = np.linalg.solve(model.direct_lattice, primitive)
    inverse_transform = np.linalg.inv(transform)
    primitive_centers = (
        inverse_transform @ np.asarray(model.orbital_centers, dtype=float).T
    ).T
    wrapped_centers = primitive_centers - np.floor(
        primitive_centers + tolerance
    )
    cell_shifts = np.rint(primitive_centers - wrapped_centers).astype(np.int64)

    def basis_key(index: int) -> tuple[Any, ...]:
        state = model.basis[index]
        return (
            str(state.metadata.get("manifold", "")),
            state.orbital,
            state.species,
            state.correlated_shell,
            state.spin,
            tuple(float(value) for value in np.round(wrapped_centers[index], 9)),
        )

    key_to_index: dict[tuple[Any, ...], int] = {}
    old_to_new: list[int] = []
    representative_indices: list[int] = []
    group_sizes: dict[int, int] = {}
    for old_index in range(model.n_basis):
        key = basis_key(old_index)
        new_index = key_to_index.get(key)
        if new_index is None:
            new_index = len(representative_indices)
            key_to_index[key] = new_index
            representative_indices.append(old_index)
        old_to_new.append(new_index)
        group_sizes[new_index] = group_sizes.get(new_index, 0) + 1
    if any(size != multiplicity for size in group_sizes.values()):
        raise ValueError(
            "electronic basis is not complete under primitive translations"
        )

    reduced_size = len(representative_indices)
    translation_transform = inverse_transform
    block_maps: dict[str, dict[tuple[int, int, int], ComplexArray]] = {
        "hamiltonian": {}
    }
    block_maps.update({name: {} for name in model.parameter_blocks})

    def fold_blocks(
        source: ComplexArray,
        destination: dict[tuple[int, int, int], ComplexArray],
    ) -> None:
        for translation, weight, block in zip(
            model.translations,
            model.interpolation_weights,
            source,
            strict=True,
        ):
            primitive_translation = translation_transform @ translation
            rounded_translation = np.rint(primitive_translation).astype(np.int64)
            if not np.allclose(
                primitive_translation,
                rounded_translation,
                atol=tolerance,
            ):
                raise ValueError(
                    "conventional translation is not integral in the "
                    "primitive lattice"
                )
            for old_row, old_column in np.argwhere(
                np.abs(block) > tolerance
            ):
                new_row = old_to_new[int(old_row)]
                new_column = old_to_new[int(old_column)]
                new_translation = (
                    rounded_translation
                    + cell_shifts[int(old_column)]
                    - cell_shifts[int(old_row)]
                )
                key = tuple(int(value) for value in new_translation)
                target = destination.setdefault(
                    key,
                    np.zeros(
                        (reduced_size, reduced_size),
                        dtype=np.complex128,
                    ),
                )
                target[new_row, new_column] += (
                    float(weight) * block[old_row, old_column] / multiplicity
                )

    fold_blocks(model.hamiltonian_blocks, block_maps["hamiltonian"])
    for name, source in model.parameter_blocks.items():
        fold_blocks(source, block_maps[name])
    all_translations = sorted(
        {
            translation
            for blocks in block_maps.values()
            for translation in blocks
        }
        or {(0, 0, 0)}
    )
    zero = np.zeros((reduced_size, reduced_size), dtype=np.complex128)

    spin_operators = None
    if model.spin_operators is not None:
        reduced_spin = np.zeros(
            (3, reduced_size, reduced_size),
            dtype=np.complex128,
        )
        for axis, operator in enumerate(model.spin_operators):
            for old_row, old_column in np.argwhere(
                np.abs(operator) > tolerance
            ):
                shift = (
                    cell_shifts[int(old_column)]
                    - cell_shifts[int(old_row)]
                )
                if np.any(shift):
                    raise ValueError(
                        "spin operator couples different primitive cells"
                    )
                reduced_spin[
                    axis,
                    old_to_new[int(old_row)],
                    old_to_new[int(old_column)],
                ] += operator[old_row, old_column] / multiplicity
        spin_operators = reduced_spin

    provenance = {
        **dict(model.provenance),
        "primitive_reduction": {
            "enabled": True,
            "cell_multiplicity": multiplicity,
            "input_basis_size": model.n_basis,
            "output_basis_size": reduced_size,
        },
    }
    return ElectronicModel(
        direct_lattice=primitive,
        basis=tuple(model.basis[index] for index in representative_indices),
        translations=np.asarray(all_translations, dtype=np.int64),
        hamiltonian_blocks=np.asarray(
            [
                block_maps["hamiltonian"].get(translation, zero)
                for translation in all_translations
            ],
            dtype=np.complex128,
        ),
        interpolation_weights=np.ones(len(all_translations), dtype=float),
        orbital_centers=np.asarray(
            [wrapped_centers[index] for index in representative_indices],
            dtype=float,
        ),
        periodic_axes=model.periodic_axes,
        parameter_values=dict(model.parameter_values),
        parameter_blocks={
            name: np.asarray(
                [
                    blocks.get(translation, zero)
                    for translation in all_translations
                ],
                dtype=np.complex128,
            )
            for name, blocks in block_maps.items()
            if name != "hamiltonian"
        },
        spin_operators=spin_operators,
        energy_zero_meV=model.energy_zero_meV,
        provenance=provenance,
        fourier_gauge=model.fourier_gauge,
    )


def build_electronic_model(
    *,
    direct_lattice: ArrayLike,
    basis: Sequence[BasisState | Mapping[str, Any] | str],
    hoppings: Mapping[tuple[int, int, int], ArrayLike],
    orbital_centers: ArrayLike | None = None,
    periodic_axes: Sequence[int] = (0, 1, 2),
    interpolation_weights: Mapping[tuple[int, int, int], float] | None = None,
    parameter_values: Mapping[str, float] | None = None,
    parameter_hoppings: Mapping[
        str, Mapping[tuple[int, int, int], ArrayLike]
    ] | None = None,
    spin_operators: ArrayLike | None = None,
    energy_unit: str = "eV",
    energy_zero: float = 0.0,
    add_hermitian_conjugates: bool = True,
    provenance: Mapping[str, Any] | None = None,
) -> ElectronicModel:
    """Build a manual model, converting declared input energies to meV.

    Missing nonzero ``-R`` blocks are generated when
    ``add_hermitian_conjugates`` is true. Explicit inconsistent partners and a
    non-Hermitian onsite block always raise an error. ``hoppings``,
    ``parameter_values``, and ``energy_zero`` use ``energy_unit``; parameter
    hopping blocks are dimensionless selectors.
    """

    input_unit = normalize_electronic_energy_unit(energy_unit)
    scale_to_meV = 1000.0 if input_unit == "eV" else 1.0
    states = tuple(
        item
        if isinstance(item, BasisState)
        else BasisState(label=item)
        if isinstance(item, str)
        else BasisState.from_dict(item)
        for item in basis
    )
    n_basis = len(states)

    def _blocks(
        raw: Mapping[tuple[int, int, int], ArrayLike],
        label: str,
        *,
        energy_scale: float = 1.0,
    ) -> dict[tuple[int, int, int], ComplexArray]:
        result = {
            tuple(int(value) for value in vector): np.asarray(
                matrix, dtype=np.complex128
            ) * energy_scale
            for vector, matrix in raw.items()
        }
        for vector, matrix in list(result.items()):
            if len(vector) != 3 or matrix.shape != (n_basis, n_basis):
                raise ValueError(
                    f"every {label} block must use R=(i,j,k) and shape (n,n)"
                )
            partner = tuple(-np.asarray(vector, dtype=int))
            if partner == vector:
                if not np.allclose(matrix, matrix.conj().T, rtol=1e-10, atol=1e-10):
                    raise ValueError(f"the R=0 {label} block must be Hermitian")
            elif partner not in result and add_hermitian_conjugates:
                result[partner] = matrix.conj().T
        return result

    blocks = _blocks(hoppings, "hopping", energy_scale=scale_to_meV)
    parameter_maps = {
        str(name): _blocks(terms, f"parameter {name!r}")
        for name, terms in (parameter_hoppings or {}).items()
    }
    values = {
        str(name): float(electronic_energy_to_meV(value, input_unit))
        for name, value in (parameter_values or {}).items()
    }
    if set(values) != set(parameter_maps):
        raise ValueError("parameter_values and parameter_hoppings must have the same keys")
    ordered = sorted(
        set(blocks).union(*(set(terms) for terms in parameter_maps.values()))
    )
    zero = np.zeros((n_basis, n_basis), dtype=np.complex128)
    weight_map = interpolation_weights or {}
    resolved_weights = {}
    for key in ordered:
        partner = tuple(-np.asarray(key, dtype=int))
        resolved_weights[key] = float(
            weight_map.get(key, weight_map.get(partner, 1.0))
        )
    source_provenance = dict(provenance or {"source": "manual"})
    source_provenance.update(
        {
            "source_energy_unit": input_unit,
            "canonical_energy_unit": "meV",
            "energy_conversion": (
                "eV to meV (x1000)"
                if input_unit == "eV"
                else "meV to meV (x1)"
            ),
        }
    )
    return ElectronicModel(
        direct_lattice=np.asarray(direct_lattice, dtype=float),
        basis=states,
        translations=np.asarray(ordered, dtype=np.int64),
        hamiltonian_blocks=np.asarray([blocks.get(key, zero) for key in ordered]),
        interpolation_weights=np.asarray(
            [resolved_weights[key] for key in ordered], dtype=float
        ),
        orbital_centers=(
            np.zeros((n_basis, 3), dtype=float)
            if orbital_centers is None
            else np.asarray(orbital_centers, dtype=float)
        ),
        periodic_axes=tuple(int(axis) for axis in periodic_axes),
        parameter_values=values,
        parameter_blocks={
            name: np.asarray([terms.get(key, zero) for key in ordered])
            for name, terms in parameter_maps.items()
        },
        spin_operators=(
            None if spin_operators is None else np.asarray(spin_operators)
        ),
        energy_zero_meV=float(electronic_energy_to_meV(energy_zero, input_unit)),
        provenance=source_provenance,
    )


@dataclass(frozen=True)
class WavevectorSampling:
    """One serializable path or integration mesh in reduced coordinates."""

    kind: Literal["path", "mesh"]
    reduced_coordinates: FloatArray
    weights: FloatArray | None = None
    path_distance_inv_angstrom: FloatArray | None = None
    labels: tuple[tuple[int, str], ...] = ()
    mesh_shape: tuple[int, ...] = ()
    shift: tuple[float, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        coordinates = _readonly(self.reduced_coordinates, float)
        object.__setattr__(self, "reduced_coordinates", coordinates)
        object.__setattr__(self, "provenance", _freeze(self.provenance))
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError("wavevector coordinates must have shape (n, 3)")
        if self.kind not in {"path", "mesh"}:
            raise ValueError("wavevector sampling kind must be 'path' or 'mesh'")
        if self.kind == "path":
            distance = _readonly(self.path_distance_inv_angstrom, float)
            object.__setattr__(self, "path_distance_inv_angstrom", distance)
            if distance.shape != (coordinates.shape[0],):
                raise ValueError("a path requires one physical distance per wavevector")
        else:
            weights = _readonly(self.weights, float)
            object.__setattr__(self, "weights", weights)
            if weights.shape != (coordinates.shape[0],) or not np.isclose(
                weights.sum(), 1.0
            ):
                raise ValueError("a mesh requires normalized integration weights")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "reduced_coordinates": self.reduced_coordinates.tolist(),
            "weights": None if self.weights is None else self.weights.tolist(),
            "path_distance_inv_angstrom": (
                None
                if self.path_distance_inv_angstrom is None
                else self.path_distance_inv_angstrom.tolist()
            ),
            "labels": [[index, label] for index, label in self.labels],
            "mesh_shape": list(self.mesh_shape),
            "shift": list(self.shift),
            "provenance": _thaw(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> WavevectorSampling:
        return cls(
            kind=str(payload["kind"]),
            reduced_coordinates=np.asarray(
                payload["reduced_coordinates"], dtype=float
            ),
            weights=(
                None
                if payload.get("weights") is None
                else np.asarray(payload["weights"], dtype=float)
            ),
            path_distance_inv_angstrom=(
                None
                if payload.get("path_distance_inv_angstrom") is None
                else np.asarray(payload["path_distance_inv_angstrom"], dtype=float)
            ),
            labels=tuple(
                (int(index), str(label))
                for index, label in payload.get("labels", ())
            ),
            mesh_shape=tuple(int(value) for value in payload.get("mesh_shape", ())),
            shift=tuple(float(value) for value in payload.get("shift", ())),
            provenance=dict(payload.get("provenance", {})),
        )


def band_path(
    model: ElectronicModel,
    nodes: Sequence[ArrayLike],
    *,
    labels: Sequence[str] | None = None,
    break_before: Sequence[bool] | None = None,
    points_per_inv_angstrom: float = 80.0,
    coordinate_reciprocal_lattice: ArrayLike | None = None,
) -> WavevectorSampling:
    """Resolve a manual high-symmetry path at uniform physical point density.

    Nodes normally use the model's reduced reciprocal coordinates. Pass a
    ``3x3`` column-vector ``coordinate_reciprocal_lattice`` to describe nodes
    in another reciprocal basis, such as the primitive basis associated with
    a centered conventional-cell Hamiltonian. Returned reduced coordinates
    are always converted to the model basis used by
    :meth:`ElectronicModel.hamiltonian`.

    ``points_per_inv_angstrom`` sets the number of interpolation intervals per
    inverse Angstrom of physical reciprocal-space distance. Each connected
    segment receives at least one interval.
    """

    node_array = np.asarray(nodes, dtype=float)
    if node_array.ndim != 2 or node_array.shape[0] < 2:
        raise ValueError("a band path requires at least two nodes")
    if node_array.shape[1] == model.dimension:
        padded = np.zeros((node_array.shape[0], 3), dtype=float)
        padded[:, model.periodic_axes] = node_array
        node_array = padded
    if node_array.shape[1] != 3:
        raise ValueError("path nodes must use three reduced coordinates")
    point_density = float(points_per_inv_angstrom)
    if not np.isfinite(point_density) or point_density <= 0.0:
        raise ValueError("points_per_inv_angstrom must be finite and positive")
    names = (
        tuple(str(value) for value in labels)
        if labels is not None
        else tuple(f"K{index}" for index in range(node_array.shape[0]))
    )
    if len(names) != node_array.shape[0]:
        raise ValueError("path labels must match the number of nodes")
    breaks = (
        tuple(False for _ in range(node_array.shape[0]))
        if break_before is None
        else tuple(bool(value) for value in break_before)
    )
    if len(breaks) != node_array.shape[0] or breaks[0]:
        raise ValueError(
            "break_before must match the nodes and begin with False"
        )
    if coordinate_reciprocal_lattice is None:
        coordinate_reciprocal = model.reciprocal_lattice
    else:
        coordinate_reciprocal = np.asarray(
            coordinate_reciprocal_lattice,
            dtype=float,
        )
        if (
            coordinate_reciprocal.shape != (3, 3)
            or not np.all(np.isfinite(coordinate_reciprocal))
        ):
            raise ValueError(
                "coordinate_reciprocal_lattice must be a finite 3x3 "
                "column-vector matrix"
            )
    physical_nodes = node_array @ coordinate_reciprocal.T
    points = [node_array[0]]
    point_breaks = [False]
    label_points: list[tuple[int, str]] = [(0, names[0])]
    segment_intervals: list[int] = []
    for segment in range(node_array.shape[0] - 1):
        if breaks[segment + 1]:
            points.append(node_array[segment + 1])
            point_breaks.append(True)
            label_points.append((len(points) - 1, names[segment + 1]))
            segment_intervals.append(0)
            continue
        segment_length = float(
            np.linalg.norm(physical_nodes[segment + 1] - physical_nodes[segment])
        )
        intervals = max(1, int(np.ceil(point_density * segment_length)))
        segment_intervals.append(intervals)
        for step in range(1, intervals + 1):
            fraction = step / intervals
            points.append(
                (1.0 - fraction) * node_array[segment]
                + fraction * node_array[segment + 1]
            )
            point_breaks.append(False)
        label_points.append((len(points) - 1, names[segment + 1]))
    input_coordinates = np.asarray(points, dtype=float)
    physical = input_coordinates @ coordinate_reciprocal.T
    coordinates = physical @ np.linalg.inv(model.reciprocal_lattice).T
    increments = np.linalg.norm(np.diff(physical, axis=0), axis=1)
    increments[np.asarray(point_breaks[1:], dtype=bool)] = 0.0
    distance = np.concatenate([[0.0], np.cumsum(increments)])
    return WavevectorSampling(
        "path",
        coordinates,
        path_distance_inv_angstrom=distance,
        labels=tuple(label_points),
        provenance={
            "provider": "manual",
            "points_per_inv_angstrom": point_density,
            "segment_intervals": segment_intervals,
            "break_before": list(breaks),
            "coordinate_reciprocal_lattice_inv_angstrom": (
                coordinate_reciprocal.tolist()
            ),
        },
    )


def k_mesh(
    model: ElectronicModel,
    shape: Sequence[int],
    *,
    shift: Sequence[float] | None = None,
    symmetry: Literal["auto", "full", "reduced"] = "full",
) -> WavevectorSampling:
    """Build a uniform periodic integration mesh for the model dimension.

    ``"auto"`` reduces only when nfit can certify the model and mesh.
    ``"full"`` always retains the original mesh. ``"reduced"`` raises rather
    than silently falling back when reduction cannot be certified.
    """

    dimensions = tuple(int(value) for value in shape)
    if len(dimensions) == 3:
        dimensions = tuple(dimensions[axis] for axis in model.periodic_axes)
    if len(dimensions) != model.dimension or any(value < 1 for value in dimensions):
        raise ValueError("mesh shape must contain one positive size per periodic axis")
    offsets = (
        tuple(0.0 for _ in dimensions)
        if shift is None
        else tuple(float(value) for value in shift)
    )
    if len(offsets) == 3 and model.dimension != 3:
        offsets = tuple(offsets[axis] for axis in model.periodic_axes)
    if len(offsets) != model.dimension:
        raise ValueError("mesh shift must match the model dimension")
    axes = [
        (np.arange(size, dtype=float) + offset) / size
        for size, offset in zip(dimensions, offsets, strict=True)
    ]
    local = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(
        -1, model.dimension
    )
    coordinates = np.zeros((local.shape[0], 3), dtype=float)
    coordinates[:, model.periodic_axes] = local
    full = WavevectorSampling(
        "mesh",
        coordinates,
        weights=np.full(local.shape[0], 1.0 / local.shape[0]),
        mesh_shape=dimensions,
        shift=offsets,
        provenance={"provider": "uniform"},
    )
    policy = str(symmetry).strip().lower()
    if policy not in {"auto", "full", "reduced"}:
        raise ValueError("mesh symmetry policy must be auto, full, or reduced")
    if policy == "full":
        return full
    return _symmetry_reduced_k_mesh(
        model,
        full,
        required=policy == "reduced",
    )


def _symmetry_reduced_k_mesh(
    model: ElectronicModel,
    full: WavevectorSampling,
    *,
    required: bool,
) -> WavevectorSampling:
    metadata = model.provenance.get("reciprocal_symmetry", {})

    def unchanged(reason: str) -> WavevectorSampling:
        if required:
            raise ValueError(f"mesh symmetry reduction is not certified: {reason}")
        return replace(
            full,
            provenance={
                **dict(full.provenance),
                "symmetry_reduction": {
                    "policy": "auto",
                    "applied": False,
                    "reason": reason,
                },
            },
        )

    if model.dimension != 3:
        return unchanged("only three-dimensional meshes are currently reduced")
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("certified_by") != "nfit_orbital_builder"
    ):
        return unchanged("model has no nfit-certified reciprocal symmetry")
    raw_rotations = metadata.get("rotations", ())
    rotations = []
    for raw in raw_rotations:
        rotation = np.asarray(raw, dtype=np.int64)
        if rotation.shape != (3, 3):
            return unchanged("certified rotation metadata is invalid")
        rotations.append(rotation)
    if not rotations:
        return unchanged("certified rotation metadata is empty")

    shape = np.asarray(full.mesh_shape, dtype=np.int64)
    shift = np.asarray(full.shift, dtype=float)
    indices = np.indices(tuple(shape), dtype=np.int64).reshape(3, -1).T
    coordinates = (indices + shift[None, :]) / shape[None, :]
    representatives = np.arange(indices.shape[0], dtype=np.int64)
    transformed_count = 0
    signs = (1, -1) if bool(metadata.get("includes_time_reversal")) else (1,)
    for rotation in rotations:
        for sign in signs:
            transformed = sign * (coordinates @ rotation.T)
            transformed_indices = transformed * shape[None, :] - shift[None, :]
            rounded = np.rint(transformed_indices).astype(np.int64)
            if not np.allclose(transformed_indices, rounded, atol=1.0e-9):
                continue
            wrapped = np.mod(rounded, shape[None, :])
            flat = np.ravel_multi_index(wrapped.T, tuple(shape))
            representatives = np.minimum(representatives, flat)
            transformed_count += 1
    if transformed_count < 2:
        return unchanged("mesh shape or shift is incompatible with the symmetry")

    unique, counts = np.unique(
        representatives,
        return_counts=True,
    )
    # A complete operation list maps every member directly onto the minimum
    # orbit representative. This assertion prevents an incomplete metadata set
    # from silently producing incorrect multiplicities.
    if np.any(representatives[unique] != unique):
        return unchanged("certified operations do not close on this mesh")
    reduced_coordinates = full.reduced_coordinates[unique]
    weights = counts.astype(float) / float(indices.shape[0])
    return WavevectorSampling(
        "mesh",
        reduced_coordinates,
        weights=weights,
        mesh_shape=full.mesh_shape,
        shift=full.shift,
        provenance={
            **dict(full.provenance),
            "symmetry_reduction": {
                "policy": "reduced" if required else "auto",
                "applied": True,
                "full_size": int(indices.shape[0]),
                "irreducible_size": int(unique.size),
                "operation_count": int(transformed_count),
            },
        },
    )


@dataclass(frozen=True)
class BandResult:
    sampling: WavevectorSampling
    energies_meV: FloatArray
    chemical_potential_meV: float
    eigenvectors: ComplexArray | None
    projected_weights: Mapping[str, FloatArray]
    model_digest: str
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "energies_meV", _readonly(self.energies_meV, float))
        if self.eigenvectors is not None:
            object.__setattr__(
                self, "eigenvectors", _readonly(self.eigenvectors, np.complex128)
            )
        object.__setattr__(
            self,
            "projected_weights",
            MappingProxyType(
                {
                    str(label): _readonly(values, float)
                    for label, values in self.projected_weights.items()
                }
            ),
        )
        object.__setattr__(self, "provenance", _freeze(self.provenance))


def _projection_indices(
    model: ElectronicModel,
    projections: Mapping[str, Sequence[int]] | None,
) -> dict[str, NDArray[np.int64]]:
    if projections is None:
        return {}
    result: dict[str, NDArray[np.int64]] = {}
    for label, raw_indices in projections.items():
        indices = np.asarray(raw_indices, dtype=np.int64)
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= model.n_basis):
            raise ValueError(f"projection {label!r} contains an invalid basis index")
        result[str(label)] = indices
    return result


def calculate_bands(
    model: ElectronicModel,
    sampling: WavevectorSampling,
    *,
    chemical_potential_meV: float = 0.0,
    projections: Mapping[str, Sequence[int]] | None = None,
    include_eigenvectors: bool = True,
    backend: ElectronicBackend | str | None = None,
    workers: int | None = None,
    max_batch_bytes: int = 256 * 1024**2,
) -> BandResult:
    """Diagonalize an arbitrary model on a path or mesh."""

    projection_indices = _projection_indices(model, projections)
    need_eigenvectors = bool(include_eigenvectors or projection_indices)
    eigensystem = evaluate_eigensystem(
        model,
        sampling.reduced_coordinates,
        eigenvectors=need_eigenvectors,
        backend=backend,
        workers=workers,
        max_batch_bytes=max_batch_bytes,
    )
    energies = eigensystem.eigenvalues
    eigenvectors = eigensystem.eigenvectors
    projected: dict[str, FloatArray] = {}
    if projection_indices and eigenvectors is None:  # pragma: no cover - defensive
        raise RuntimeError("orbital projections require eigenvectors")
    for label, indices in projection_indices.items():
        projected[label] = np.sum(
            np.abs(eigenvectors[:, indices, :]) ** 2, axis=1
        )
    return BandResult(
        sampling=sampling,
        energies_meV=_readonly(energies, float),
        chemical_potential_meV=float(chemical_potential_meV),
        eigenvectors=(
            _readonly(eigenvectors, np.complex128)
            if include_eigenvectors
            else None
        ),
        projected_weights=projected,
        model_digest=model.content_digest,
        provenance={
            "backend": eigensystem.provenance["resolved_backend"],
            "precision": "float64/complex128",
            "fourier_gauge": model.fourier_gauge,
            "execution": dict(eigensystem.provenance),
            "projection_groups": {
                label: indices.tolist()
                for label, indices in projection_indices.items()
            },
            "sampling": sampling.to_dict(),
        },
    )


@dataclass(frozen=True)
class DensityOfStatesResult:
    energy_meV: FloatArray
    total_per_meV_cell: FloatArray
    projected_per_meV_cell: Mapping[str, FloatArray]
    chemical_potential_meV: float
    broadening_meV: float
    mesh: WavevectorSampling
    model_digest: str
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "energy_meV", _readonly(self.energy_meV, float))
        object.__setattr__(
            self, "total_per_meV_cell", _readonly(self.total_per_meV_cell, float)
        )
        object.__setattr__(
            self,
            "projected_per_meV_cell",
            MappingProxyType(
                {
                    str(label): _readonly(values, float)
                    for label, values in self.projected_per_meV_cell.items()
                }
            ),
        )
        object.__setattr__(self, "provenance", _freeze(self.provenance))


def density_of_states(
    model: ElectronicModel,
    mesh: WavevectorSampling,
    energy_meV: ArrayLike,
    *,
    broadening_meV: float,
    method: Literal["gaussian", "tetrahedron"] = "gaussian",
    chemical_potential_meV: float = 0.0,
    projections: Mapping[str, Sequence[int]] | None = None,
    max_chunk_bytes: int = 64 * 1024**2,
    backend: ElectronicBackend | str | None = None,
    workers: int | None = None,
    max_batch_bytes: int = 256 * 1024**2,
) -> DensityOfStatesResult:
    """Calculate total and projected density of states.

    ``method="gaussian"`` replaces each sampled state with a normalized
    Gaussian. ``method="tetrahedron"`` uses ASE's linear tetrahedron
    integration on a complete uniform three-dimensional mesh.
    """

    if mesh.kind != "mesh" or mesh.weights is None:
        raise ValueError("density of states requires an integration mesh")
    energy = np.asarray(energy_meV, dtype=float)
    sigma = float(broadening_meV)
    selected_method = str(method).strip().lower()
    if selected_method not in {"gaussian", "tetrahedron"}:
        raise ValueError("density-of-states method must be gaussian or tetrahedron")
    if energy.ndim != 1 or energy.size < 2 or not np.all(np.isfinite(energy)):
        raise ValueError("energy must be a finite 1D grid with at least two points")
    if np.any(np.diff(energy) <= 0.0):
        raise ValueError("density-of-states energies must be strictly increasing")
    if selected_method == "gaussian" and (
        not np.isfinite(sigma) or sigma <= 0.0
    ):
        raise ValueError("Gaussian DOS broadening must be positive")
    bands = calculate_bands(
        model,
        mesh,
        chemical_potential_meV=chemical_potential_meV,
        projections=projections,
        include_eigenvectors=False,
        backend=backend,
        workers=workers,
        max_batch_bytes=max_batch_bytes,
    )
    if selected_method == "tetrahedron":
        if (
            model.dimension != 3
            or len(mesh.mesh_shape) != 3
            or len(mesh.shift) != 3
        ):
            raise ValueError(
                "tetrahedron DOS requires a three-dimensional electronic model"
            )
        if int(np.prod(mesh.mesh_shape)) != mesh.reduced_coordinates.shape[0]:
            raise ValueError(
                "tetrahedron DOS requires a complete uniform integration mesh"
            )
        expected_axes = [
            (np.arange(size, dtype=float) + offset) / size
            for size, offset in zip(
                mesh.mesh_shape,
                mesh.shift,
                strict=True,
            )
        ]
        expected_local = np.stack(
            np.meshgrid(*expected_axes, indexing="ij"),
            axis=-1,
        ).reshape(-1, 3)
        expected_coordinates = np.zeros_like(mesh.reduced_coordinates)
        expected_coordinates[:, model.periodic_axes] = expected_local
        if not np.allclose(
            mesh.reduced_coordinates,
            expected_coordinates,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "tetrahedron DOS requires the ordered uniform mesh from k_mesh"
            )
        expected_weight = 1.0 / mesh.reduced_coordinates.shape[0]
        if not np.allclose(
            mesh.weights,
            expected_weight,
            rtol=1.0e-12,
            atol=1.0e-15,
        ):
            raise ValueError(
                "tetrahedron DOS does not support symmetry-reduced mesh weights"
            )
        spacing = np.diff(energy)
        if not np.allclose(
            spacing,
            spacing[0],
            rtol=1.0e-10,
            atol=max(abs(float(spacing[0])) * 1.0e-12, 1.0e-14),
        ):
            raise ValueError("tetrahedron DOS requires a uniform energy grid")
        try:
            import ase
            from ase.dft.dos import linear_tetrahedron_integration
        except ImportError as exc:  # pragma: no cover - declared dependency
            raise ImportError("tetrahedron DOS requires ASE") from exc

        grid_shape = (*mesh.mesh_shape, model.n_basis)
        eigenvalues = bands.energies_meV.reshape(grid_shape)
        labels = tuple(bands.projected_weights)
        weight_channels = np.stack(
            [
                np.ones(grid_shape, dtype=float),
                *(
                    bands.projected_weights[label].reshape(grid_shape)
                    for label in labels
                ),
            ],
            axis=-1,
        )
        integrated = np.asarray(
            linear_tetrahedron_integration(
                model.direct_lattice.T,
                eigenvalues,
                energy,
                weights=weight_channels,
            ),
            dtype=float,
        )
        return DensityOfStatesResult(
            energy_meV=_readonly(energy, float),
            total_per_meV_cell=_readonly(integrated[0], float),
            projected_per_meV_cell={
                label: _readonly(integrated[index + 1], float)
                for index, label in enumerate(labels)
            },
            chemical_potential_meV=float(chemical_potential_meV),
            broadening_meV=0.0,
            mesh=mesh,
            model_digest=model.content_digest,
            provenance={
                "backend": bands.provenance["backend"],
                "precision": "float64/complex128",
                "method": "tetrahedron",
                "provider": "ASE",
                "provider_version": str(
                    getattr(ase, "__version__", "unknown")
                ),
                "execution": dict(bands.provenance["execution"]),
                "projection_groups": {
                    label: list(indices)
                    for label, indices in bands.provenance[
                        "projection_groups"
                    ].items()
                },
            },
        )

    flat_energy = bands.energies_meV.reshape(-1)
    state_weight = np.repeat(np.asarray(mesh.weights), model.n_basis)
    projected_flat = {
        label: values.reshape(-1) for label, values in bands.projected_weights.items()
    }
    total = np.zeros(energy.shape, dtype=float)
    projected = {label: np.zeros(energy.shape, dtype=float) for label in projected_flat}
    bytes_per_state = max(energy.size * 8, 8)
    chunk = max(1, int(max_chunk_bytes) // bytes_per_state)
    normalization = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    for start in range(0, flat_energy.size, chunk):
        stop = min(start + chunk, flat_energy.size)
        kernel = normalization * np.exp(
            -0.5
            * ((energy[None, :] - flat_energy[start:stop, None]) / sigma) ** 2
        )
        weights = state_weight[start:stop, None]
        total += np.sum(weights * kernel, axis=0)
        for label, values in projected_flat.items():
            projected[label] += np.sum(
                weights * values[start:stop, None] * kernel, axis=0
            )
    return DensityOfStatesResult(
        energy_meV=_readonly(energy, float),
        total_per_meV_cell=_readonly(total, float),
        projected_per_meV_cell={
            label: _readonly(values, float) for label, values in projected.items()
        },
        chemical_potential_meV=float(chemical_potential_meV),
        broadening_meV=sigma,
        mesh=mesh,
        model_digest=model.content_digest,
        provenance={
            "backend": bands.provenance["backend"],
            "precision": "float64/complex128",
            "method": "gaussian",
            "execution": dict(bands.provenance["execution"]),
            "max_chunk_bytes": int(max_chunk_bytes),
            "projection_groups": {
                label: list(indices)
                for label, indices in bands.provenance[
                    "projection_groups"
                ].items()
            },
        },
    )


@dataclass(frozen=True)
class FermiSurfaceSheet:
    band_index: int
    vertices_reduced: FloatArray
    vertices_inv_angstrom: FloatArray
    connectivity: NDArray[np.int64]
    projected_weights: Mapping[str, FloatArray]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "vertices_reduced", _readonly(self.vertices_reduced, float)
        )
        object.__setattr__(
            self,
            "vertices_inv_angstrom",
            _readonly(self.vertices_inv_angstrom, float),
        )
        object.__setattr__(
            self, "connectivity", _readonly(self.connectivity, np.int64)
        )
        object.__setattr__(
            self,
            "projected_weights",
            MappingProxyType(
                {
                    str(label): _readonly(values, float)
                    for label, values in self.projected_weights.items()
                }
            ),
        )


@dataclass(frozen=True)
class FermiSurfaceResult:
    target_energy_meV: float
    dimension: int
    periodic_axes: tuple[int, ...]
    sheets: tuple[FermiSurfaceSheet, ...]
    mesh_shape: tuple[int, ...]
    model_digest: str
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance", _freeze(self.provenance))


def _crossing_fraction(value_a: float, value_b: float, target: float) -> float:
    delta = value_b - value_a
    return 0.5 if abs(delta) < 1.0e-15 else float((target - value_a) / delta)


def _fermi_vertices_1d(
    coordinates: FloatArray, values: FloatArray, target: float
) -> tuple[FloatArray, NDArray[np.int64]]:
    vertices = []
    for index in range(values.size - 1):
        a, b = values[index], values[index + 1]
        if np.isclose(a, target, rtol=0.0, atol=1.0e-12):
            vertices.append(coordinates[index])
        elif (a - target) * (b - target) < 0.0:
            fraction = _crossing_fraction(a, b, target)
            vertices.append(
                coordinates[index]
                + fraction * (coordinates[index + 1] - coordinates[index])
            )
    if np.isclose(values[-1], target, rtol=0.0, atol=1.0e-12):
        vertices.append(coordinates[-1])
    result = np.asarray(vertices, dtype=float).reshape(-1, 3)
    return result, np.arange(result.shape[0], dtype=np.int64)[:, None]


_MARCHING_SQUARES = {
    1: ((3, 0),),
    2: ((0, 1),),
    3: ((3, 1),),
    4: ((1, 2),),
    5: ((3, 2), (0, 1)),
    6: ((0, 2),),
    7: ((3, 2),),
    8: ((2, 3),),
    9: ((0, 2),),
    10: ((0, 3), (1, 2)),
    11: ((1, 2),),
    12: ((1, 3),),
    13: ((0, 1),),
    14: ((3, 0),),
}


def _fermi_vertices_2d(
    coordinates: FloatArray,
    values: FloatArray,
    target: float,
) -> tuple[FloatArray, NDArray[np.int64]]:
    nx, ny = values.shape
    vertices: list[np.ndarray] = []
    segments: list[tuple[int, int]] = []
    edge_vertices = ((0, 1), (1, 2), (2, 3), (3, 0))
    for i in range(nx - 1):
        for j in range(ny - 1):
            points = (
                coordinates[i, j],
                coordinates[i + 1, j],
                coordinates[i + 1, j + 1],
                coordinates[i, j + 1],
            )
            cell_values = (
                values[i, j],
                values[i + 1, j],
                values[i + 1, j + 1],
                values[i, j + 1],
            )
            case = sum(
                (1 << corner) for corner, value in enumerate(cell_values) if value >= target
            )
            for edge_a, edge_b in _MARCHING_SQUARES.get(case, ()):
                segment_indices = []
                for edge in (edge_a, edge_b):
                    corner_a, corner_b = edge_vertices[edge]
                    fraction = _crossing_fraction(
                        cell_values[corner_a], cell_values[corner_b], target
                    )
                    point = points[corner_a] + fraction * (
                        points[corner_b] - points[corner_a]
                    )
                    segment_indices.append(len(vertices))
                    vertices.append(point)
                segments.append((segment_indices[0], segment_indices[1]))
    return (
        np.asarray(vertices, dtype=float).reshape(-1, 3),
        np.asarray(segments, dtype=np.int64).reshape(-1, 2),
    )


def _fermi_vertices_3d(
    coordinates: FloatArray,
    values: FloatArray,
    target: float,
) -> tuple[FloatArray, NDArray[np.int64]]:
    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover - declared core dependency
        raise ImportError("3D Fermi surfaces require pyvista") from exc
    grid = pv.StructuredGrid(
        coordinates[..., 0], coordinates[..., 1], coordinates[..., 2]
    )
    grid.point_data["energy"] = np.asarray(values, dtype=float).ravel(order="F")
    surface = grid.contour([float(target)], scalars="energy").triangulate()
    faces = np.asarray(surface.faces, dtype=np.int64).reshape(-1, 4)[:, 1:]
    return np.asarray(surface.points, dtype=float), faces


def fermi_surface(
    model: ElectronicModel,
    mesh_shape: Sequence[int],
    *,
    target_energy_meV: float = 0.0,
    projections: Mapping[str, Sequence[int]] | None = None,
    backend: ElectronicBackend | str | None = None,
    workers: int | None = None,
    max_batch_bytes: int = 256 * 1024**2,
) -> FermiSurfaceResult:
    """Extract 1D Fermi points, 2D contours, or 3D triangulated surfaces."""

    shape = tuple(int(value) for value in mesh_shape)
    if len(shape) == 3:
        shape = tuple(shape[axis] for axis in model.periodic_axes)
    if len(shape) != model.dimension or any(value < 2 for value in shape):
        raise ValueError("Fermi mesh needs at least two points per periodic axis")
    axes = [np.linspace(0.0, 1.0, value + 1) for value in shape]
    local_grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    coordinates = np.zeros((*local_grid.shape[:-1], 3), dtype=float)
    coordinates[..., model.periodic_axes] = local_grid
    flat = coordinates.reshape(-1, 3)
    mesh_eigensystem = evaluate_eigensystem(
        model,
        flat,
        eigenvectors=False,
        backend=backend,
        workers=workers,
        max_batch_bytes=max_batch_bytes,
    )
    energies = mesh_eigensystem.eigenvalues.reshape(
        *coordinates.shape[:-1], model.n_basis
    )
    projection_indices = _projection_indices(model, projections)
    sheets = []
    for band_index in range(model.n_basis):
        values = energies[..., band_index]
        if float(np.nanmin(values)) > target_energy_meV or float(
            np.nanmax(values)
        ) < target_energy_meV:
            continue
        if model.dimension == 1:
            vertices, connectivity = _fermi_vertices_1d(
                flat, values.reshape(-1), target_energy_meV
            )
        elif model.dimension == 2:
            vertices, connectivity = _fermi_vertices_2d(
                coordinates, values, target_energy_meV
            )
        else:
            vertices, connectivity = _fermi_vertices_3d(
                coordinates, values, target_energy_meV
            )
        if vertices.size == 0:
            continue
        projected: dict[str, FloatArray] = {}
        if projection_indices:
            vertex_eigensystem = evaluate_eigensystem(
                model,
                vertices,
                eigenvectors=True,
                backend=backend,
                workers=workers,
                max_batch_bytes=max_batch_bytes,
            )
            vectors = vertex_eigensystem.eigenvectors
            if vectors is None:  # pragma: no cover - defensive
                raise RuntimeError("Fermi-surface projections require eigenvectors")
            for label, indices in projection_indices.items():
                projected[label] = np.sum(
                    np.abs(vectors[:, indices, band_index]) ** 2, axis=1
                )
        sheets.append(
            FermiSurfaceSheet(
                band_index,
                _readonly(vertices, float),
                _readonly(vertices @ model.reciprocal_lattice.T, float),
                _readonly(connectivity, np.int64),
                projected,
            )
        )
    return FermiSurfaceResult(
        target_energy_meV=float(target_energy_meV),
        dimension=model.dimension,
        periodic_axes=model.periodic_axes,
        sheets=tuple(sheets),
        mesh_shape=shape,
        model_digest=model.content_digest,
        provenance={
            "backend": mesh_eigensystem.provenance["resolved_backend"],
            "precision": "float64/complex128",
            "execution": dict(mesh_eigensystem.provenance),
            "method": {
                1: "linear",
                2: "marching_squares",
                3: "marching_cubes",
            }[model.dimension],
            "projection_groups": {
                label: indices.tolist()
                for label, indices in projection_indices.items()
            },
        },
    )


def save_electronic_model(model: ElectronicModel, path: str | Path) -> None:
    """Save a portable, digest-protected canonical electronic model as JSON."""

    Path(path).write_text(json.dumps(model.to_dict(), indent=2, sort_keys=True))


def load_electronic_model(path: str | Path) -> ElectronicModel:
    """Load and validate a portable canonical electronic-model JSON file."""

    return ElectronicModel.from_dict(json.loads(Path(path).read_text()))


def tight_binding_structure_script(
    crystal: Mapping[str, Any],
    periodic_axes: Sequence[int],
    *,
    group_name: str = "Electronic",
    model_name: str = "tight_binding",
    electronic_energy_unit: str = "eV",
    use_primitive_cell: bool = True,
    hopping_parameterization: str = "slater_koster",
    band_path_nodes: Sequence[Mapping[str, Any]] = (),
    band_path_convention: str = "manual",
    band_path_metadata: Mapping[str, Any] | None = None,
    orbital_manifolds: Sequence[Mapping[str, Any]] = (),
    onsite_terms: Sequence[Mapping[str, Any]] = (),
    hopping_cutoff_angstrom: float | None = None,
    hopping_terms: Sequence[Mapping[str, Any]] = (),
    spin_treatment: str = "auto",
    soc_terms: Sequence[Mapping[str, Any]] = (),
    parameter_values_meV: Mapping[str, float] | None = None,
    fit_parameters: Mapping[str, bool] | None = None,
    parameter_limits_meV: Mapping[str, Sequence[float | None]] | None = None,
    parameter_sharing: Mapping[str, Mapping[str, Any]] | None = None,
    expected_model_digest: str = "",
) -> str:
    """Return an editable script that rebuilds tight-binding builder state."""

    from .crystal import validate_crystal

    payload = deepcopy(dict(crystal))
    validate_crystal(payload)
    axes = tuple(int(axis) for axis in periodic_axes)
    display_unit = normalize_electronic_energy_unit(electronic_energy_unit)
    if not axes or len(set(axes)) != len(axes) or any(
        axis not in (0, 1, 2) for axis in axes
    ):
        raise ValueError(
            "periodic_axes must contain one to three unique indices from 0, 1, 2"
        )
    lines = [
        '"""Rebuild tight-binding crystal geometry without GUI state."""',
        "",
        (
            "from nfit import (DataGroup, create_model_component, "
            "set_electronic_energy_unit, set_model_crystal)"
        ),
    ]
    provenance = payload.get("provenance", {})
    if (
        isinstance(provenance, Mapping)
        and provenance.get("source") == "cif"
        and provenance.get("path")
        and provenance.get("sha256")
    ):
        lines.extend(
            [
                "from nfit import crystal_from_cif",
                "",
                f"crystal = crystal_from_cif({str(provenance['path'])!r})",
                "assert crystal['provenance']['sha256'] == "
                f"{str(provenance['sha256'])!r}",
            ]
        )
    else:
        lines.extend(["", f"crystal = {pformat(payload, sort_dicts=True)}"])
    lines.extend(
        [
            "",
            f"group = DataGroup({str(group_name)!r})",
            f"model = create_model_component(group, {str(model_name)!r}, type='tight_binding')",
            f"set_model_crystal(model, crystal, group=group, periodic_axes={axes!r})",
            f"set_electronic_energy_unit(model, {display_unit!r})",
            f"model.config['use_primitive_cell'] = {bool(use_primitive_cell)!r}",
            (
                "model.config['hopping_parameterization'] = "
                f"{str(hopping_parameterization)!r}"
            ),
            f"model.config['band_path'] = {pformat(list(band_path_nodes), sort_dicts=True)}",
            f"model.config['band_path_convention'] = {str(band_path_convention)!r}",
            (
                "model.config['band_path_metadata'] = "
                f"{pformat(dict(band_path_metadata or {}), sort_dicts=True)}"
            ),
        ]
    )
    if orbital_manifolds:
        lines.extend(
            [
                "from nfit import configure_tight_binding_builder",
                "",
                f"orbital_manifolds = {pformat(list(orbital_manifolds), sort_dicts=True)}",
                f"onsite_terms = {pformat(list(onsite_terms), sort_dicts=True)}",
                f"hopping_terms = {pformat(list(hopping_terms), sort_dicts=True)}",
                f"soc_terms = {pformat(list(soc_terms), sort_dicts=True)}",
                "electronic_model = configure_tight_binding_builder(",
                "    model,",
                "    manifolds=orbital_manifolds,",
                "    onsite_terms=onsite_terms,",
                (
                    "    hopping_cutoff_angstrom="
                    f"{None if hopping_cutoff_angstrom is None else float(hopping_cutoff_angstrom)!r},"
                ),
                "    hopping_terms=hopping_terms,",
                f"    spin_treatment={str(spin_treatment)!r},",
                "    soc_terms=soc_terms,",
                ")",
            ]
        )
        if expected_model_digest:
            lines.append(
                f"assert electronic_model.content_digest == {str(expected_model_digest)!r}"
            )
        if any(
            value
            for value in (
                parameter_values_meV,
                fit_parameters,
                parameter_limits_meV,
                parameter_sharing,
            )
        ):
            lines.extend(
                [
                    "from nfit import set_tight_binding_parameter_state",
                    "",
                    "electronic_model = set_tight_binding_parameter_state(",
                    "    model,",
                    f"    values_meV={pformat(dict(parameter_values_meV or {}), sort_dicts=True)},",
                    f"    fit_parameters={pformat(dict(fit_parameters or {}), sort_dicts=True)},",
                    f"    limits_meV={pformat(dict(parameter_limits_meV or {}), sort_dicts=True)},",
                    f"    sharing={pformat(dict(parameter_sharing or {}), sort_dicts=True)},",
                    ")",
                ]
            )
            if expected_model_digest:
                lines.append(
                    f"assert electronic_model.content_digest == {str(expected_model_digest)!r}"
                )
    lines.append("")
    return "\n".join(lines)


def _source_record(path: Path, role: str) -> dict[str, Any]:
    content = path.read_bytes()
    return {
        "role": role,
        "path": str(path.resolve()),
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _nfit_version() -> str:
    try:
        return version("nfit")
    except PackageNotFoundError:  # pragma: no cover - source tree without metadata
        return "unknown"


def _parse_win_lattice(path: Path) -> FloatArray:
    lines = path.read_text().splitlines()
    start = next(
        (
            index
            for index, line in enumerate(lines)
            if line.strip().lower() == "begin unit_cell_cart"
        ),
        None,
    )
    if start is None:
        raise ValueError(f"{path.name} does not contain unit_cell_cart")
    index = start + 1
    unit = "ang"
    if lines[index].strip().lower() in {"ang", "bohr"}:
        unit = lines[index].strip().lower()
        index += 1
    vectors = np.asarray(
        [[float(value) for value in lines[index + row].split()] for row in range(3)]
    )
    if unit == "bohr":
        vectors *= 0.529177210903
    return vectors.T


def _parse_centres_xyz(
    path: Path, n_basis: int, direct_lattice: FloatArray
) -> FloatArray:
    lines = path.read_text().splitlines()[2:]
    centers = []
    for line in lines:
        fields = line.split()
        if len(fields) >= 4 and fields[0].casefold() in {"x", "w", "wf"}:
            centers.append([float(value) for value in fields[1:4]])
    if len(centers) < n_basis:
        raise ValueError(
            f"{path.name} contains {len(centers)} Wannier centres, expected {n_basis}"
        )
    cartesian = np.asarray(centers[:n_basis], dtype=float)
    return cartesian @ np.linalg.inv(direct_lattice).T


def _parse_hr_section(
    lines: list[str], start: int
) -> tuple[int, list[tuple[int, int, int]], list[int], list[ComplexArray], int]:
    n_basis = int(lines[start].split()[0])
    n_vectors = int(lines[start + 1].split()[0])
    index = start + 2
    degeneracies: list[int] = []
    while len(degeneracies) < n_vectors:
        degeneracies.extend(int(value) for value in lines[index].split())
        index += 1
    degeneracies = degeneracies[:n_vectors]
    vectors: list[tuple[int, int, int]] = []
    blocks: list[ComplexArray] = []
    for _ in range(n_vectors):
        while index < len(lines) and not lines[index].strip():
            index += 1
        fields = lines[index].split()
        block = np.zeros((n_basis, n_basis), dtype=np.complex128)
        if len(fields) >= 7:
            vector = tuple(int(value) for value in fields[:3])
            for element in range(n_basis * n_basis):
                row = fields if element == 0 else lines[index].split()
                current = tuple(int(value) for value in row[:3])
                if current != vector:
                    raise ValueError("Wannier90 Hamiltonian block changed R unexpectedly")
                m, n = int(row[3]) - 1, int(row[4]) - 1
                block[m, n] = float(row[5]) + 1j * float(row[6])
                index += 1
        elif len(fields) == 3:
            vector = tuple(int(value) for value in fields)
            index += 1
            for _element in range(n_basis * n_basis):
                row = lines[index].split()
                m, n = int(row[0]) - 1, int(row[1]) - 1
                block[m, n] = float(row[2]) + 1j * float(row[3])
                index += 1
        else:
            raise ValueError("unrecognized Wannier90 Hamiltonian record")
        vectors.append(vector)
        blocks.append(block)
    return n_basis, vectors, degeneracies, blocks, index


def _parse_tb_centers(
    lines: list[str], start: int, n_basis: int, vectors: Sequence[tuple[int, int, int]]
) -> FloatArray:
    centers = np.zeros((n_basis, 3), dtype=float)
    found = np.zeros(n_basis, dtype=bool)
    index = start
    for _vector_index in range(len(vectors)):
        while index < len(lines) and not lines[index].strip():
            index += 1
        vector = tuple(int(value) for value in lines[index].split()[:3])
        index += 1
        for _ in range(n_basis * n_basis):
            fields = lines[index].split()
            m, n = int(fields[0]) - 1, int(fields[1]) - 1
            if vector == (0, 0, 0) and m == n and len(fields) >= 8:
                centers[m] = [float(fields[2]), float(fields[4]), float(fields[6])]
                found[m] = True
            index += 1
    if not np.all(found):
        raise ValueError("Wannier90 tb.dat does not contain every diagonal R=0 centre")
    return centers


def _parse_wsvec(path: Path) -> dict[tuple[int, int, int, int, int], list[np.ndarray]]:
    lines = path.read_text().splitlines()[1:]
    index = 0
    result: dict[tuple[int, int, int, int, int], list[np.ndarray]] = {}
    while index < len(lines):
        if not lines[index].strip():
            index += 1
            continue
        fields = lines[index].split()
        if len(fields) < 5:
            raise ValueError("invalid Wannier90 wsvec key")
        key = (
            int(fields[0]),
            int(fields[1]),
            int(fields[2]),
            int(fields[3]) - 1,
            int(fields[4]) - 1,
        )
        count = int(lines[index + 1].split()[0])
        translations = [
            np.asarray(
                [int(value) for value in lines[index + 2 + offset].split()[:3]],
                dtype=int,
            )
            for offset in range(count)
        ]
        result[key] = translations
        index += 2 + count
    return result


def import_wannier90(
    path: str | Path,
    *,
    direct_lattice: ArrayLike | None = None,
    orbital_centers: ArrayLike | None = None,
    basis: Sequence[BasisState | Mapping[str, Any] | str] | None = None,
    periodic_axes: Sequence[int] | None = None,
    wsvec_path: str | Path | None = None,
) -> ElectronicModel:
    """Import ``seedname_hr.dat`` or ``seedname_tb.dat`` without Wannier90.

    The importer converts eV to meV, applies Wigner--Seitz degeneracies and
    optional pair-dependent ``wsvec`` replicas, records source digests, and
    validates the resulting Hermitian interpolation.
    """

    source = Path(path)
    name = source.name
    is_tb = name.endswith("_tb.dat")
    is_hr = name.endswith("_hr.dat")
    if not (is_tb or is_hr):
        raise ValueError("Wannier90 source must end in _hr.dat or _tb.dat")
    lines = source.read_text().splitlines()
    sources = [_source_record(source, "wannier90_tb" if is_tb else "wannier90_hr")]
    if is_tb:
        lattice = np.asarray(
            [[float(value) for value in lines[row].split()[:3]] for row in range(1, 4)]
        ).T
        n_basis, vectors, degeneracies, blocks, end = _parse_hr_section(lines, 4)
        cartesian_centers = _parse_tb_centers(lines, end, n_basis, vectors)
        centers = cartesian_centers @ np.linalg.inv(lattice).T
    else:
        n_basis, vectors, degeneracies, blocks, _end = _parse_hr_section(lines, 1)
        seed = name[: -len("_hr.dat")]
        win = source.with_name(f"{seed}.win")
        centres_file = source.with_name(f"{seed}_centres.xyz")
        if direct_lattice is None:
            if not win.is_file():
                raise ValueError("hr.dat import requires direct_lattice or seedname.win")
            lattice = _parse_win_lattice(win)
            sources.append(_source_record(win, "wannier90_input"))
        else:
            lattice = np.asarray(direct_lattice, dtype=float)
        if orbital_centers is None:
            if not centres_file.is_file():
                raise ValueError(
                    "hr.dat import requires orbital_centers or seedname_centres.xyz"
                )
            centers = _parse_centres_xyz(centres_file, n_basis, lattice)
            sources.append(_source_record(centres_file, "wannier_centres"))
        else:
            centers = np.asarray(orbital_centers, dtype=float)
    if direct_lattice is not None and is_tb:
        supplied = np.asarray(direct_lattice, dtype=float)
        if not np.allclose(supplied, lattice, rtol=1e-9, atol=1e-9):
            raise ValueError("supplied lattice disagrees with the tb.dat lattice")
    if orbital_centers is not None and is_tb:
        centers = np.asarray(orbital_centers, dtype=float)

    if wsvec_path is None:
        seed = name[: -len("_tb.dat")] if is_tb else name[: -len("_hr.dat")]
        candidate = source.with_name(f"{seed}_wsvec.dat")
        wsvec = candidate if candidate.is_file() else None
    else:
        wsvec = Path(wsvec_path)
    corrections = {} if wsvec is None else _parse_wsvec(wsvec)
    if wsvec is not None:
        sources.append(_source_record(wsvec, "wannier90_wsvec"))

    effective: dict[tuple[int, int, int], ComplexArray] = {}
    for vector, degeneracy, block in zip(vectors, degeneracies, blocks, strict=True):
        for m in range(n_basis):
            for n in range(n_basis):
                value = block[m, n] * (1000.0 / degeneracy)
                if value == 0.0:
                    continue
                replicas = corrections.get((*vector, m, n), [np.zeros(3, dtype=int)])
                for replica in replicas:
                    resolved = tuple(np.asarray(vector, dtype=int) + replica)
                    effective.setdefault(
                        resolved, np.zeros((n_basis, n_basis), dtype=np.complex128)
                    )[m, n] += value / len(replicas)
    states = (
        tuple(BasisState(f"w{index + 1}") for index in range(n_basis))
        if basis is None
        else tuple(
            item
            if isinstance(item, BasisState)
            else BasisState(item)
            if isinstance(item, str)
            else BasisState.from_dict(item)
            for item in basis
        )
    )
    if len(states) != n_basis:
        raise ValueError("basis metadata must match the Wannier function count")
    if periodic_axes is None:
        active = tuple(
            axis
            for axis in range(3)
            if any(vector[axis] != 0 for vector in effective)
        )
        periodic = active or (0, 1, 2)
    else:
        periodic = tuple(int(axis) for axis in periodic_axes)
    ordered = sorted(effective)
    model = ElectronicModel(
        direct_lattice=lattice,
        basis=states,
        translations=np.asarray(ordered, dtype=np.int64),
        hamiltonian_blocks=np.asarray([effective[key] for key in ordered]),
        interpolation_weights=np.ones(len(ordered), dtype=float),
        orbital_centers=centers,
        periodic_axes=periodic,
        energy_zero_meV=0.0,
        provenance={
            "source": "wannier90",
            "importer": "nfit.electronic_structure.import_wannier90",
            "importer_schema_version": 1,
            "nfit_version": _nfit_version(),
            "format": "tb.dat" if is_tb else "hr.dat",
            "source_energy_unit": "eV",
            "canonical_energy_unit": "meV",
            "energy_conversion": "eV to meV (x1000)",
            "source_fourier_gauge": "wannier",
            "canonical_fourier_gauge": "wannier",
            "wigner_seitz_degeneracies": degeneracies,
            "wsvec_applied": wsvec is not None,
            "files": sources,
        },
    )
    return model


def electronic_model_from_component(
    component: Any,
    *,
    parameter_values_meV: Mapping[str, float] | None = None,
) -> ElectronicModel:
    """Resolve a tight-binding component and optional trial parameter values.

    The stored source digest is checked before overrides are applied. Builder
    parameters are then installed by replacing only the immutable model's
    named coefficient mapping, so its symmetry-generated Hamiltonian blocks
    remain reusable by electronic-response calculations.
    """

    if getattr(component, "type", None) != "tight_binding":
        raise TypeError("electronic model resolution requires a tight_binding component")
    config = component.config if isinstance(component.config, dict) else {}
    source_path = str(config.get("source_path", "")).strip()
    if source_path:
        raw_axes = config.get("periodic_axes", [])
        model = import_wannier90(
            source_path,
            periodic_axes=(
                None
                if not raw_axes
                else tuple(int(value) for value in raw_axes)
            ),
        )
    else:
        payload = config.get("model_data")
        if bool(config.get("model_stale", False)) or not isinstance(
            payload, dict
        ) or not payload:
            if config.get("orbital_manifolds"):
                from .electronic_builder import resolve_tight_binding_builder

                model = resolve_tight_binding_builder(component)
                config = component.config
            else:
                raise ValueError(
                    "tight-binding model has no source; import Wannier90 data "
                    "or add orbital manifolds to the structure-first builder"
                )
        else:
            expected_digest = str(config.get("model_digest", "")).strip()
            cached = getattr(
                component,
                "_nfit_electronic_model_cache",
                None,
            )
            if (
                isinstance(cached, tuple)
                and len(cached) == 2
                and cached[0] == expected_digest
                and isinstance(cached[1], ElectronicModel)
            ):
                model = cached[1]
            else:
                model = ElectronicModel.from_dict(payload)
                component._nfit_electronic_model_cache = (
                    model.content_digest,
                    model,
                )
    expected = str(config.get("model_digest", "")).strip()
    if expected and expected != model.content_digest:
        raise ValueError("tight-binding source no longer matches its stored digest")

    values = {
        name: float(component.parameters.get(name, value))
        for name, value in model.parameter_values.items()
    }
    for name, value in (parameter_values_meV or {}).items():
        if name not in model.parameter_values:
            raise KeyError(f"unknown electronic-model parameter {name!r}")
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError("electronic-model parameter values must be finite")
        values[name] = numeric
    if values == dict(model.parameter_values):
        return model
    return model._with_parameter_values_meV(values)
