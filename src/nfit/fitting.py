from __future__ import annotations

import ast
import math
import os
import re
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from multiprocessing.pool import ThreadPool
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .dataset import PointData4D
from .rebin import rebin_nd, rebin_nd_symmetry

try:  # pragma: no cover - exercised only when SciPy is importable.
    from scipy.optimize import least_squares as _scipy_least_squares
except Exception:  # SciPy may be absent or have a broken compiled dependency.
    _scipy_least_squares = None

try:  # pragma: no cover - exercised only when SciPy is importable.
    from scipy.optimize import differential_evolution as _scipy_differential_evolution
except Exception:  # SciPy may be absent or have a broken compiled dependency.
    _scipy_differential_evolution = None


FloatArray = NDArray[np.float64]
ModelFunction = Callable[[PointData4D, dict[str, float]], FloatArray]
DataTransform = Callable[[PointData4D], PointData4D]
ResolutionFunction = Callable[[PointData4D, FloatArray, dict[str, float]], FloatArray]
ProgressCallback = Callable[[dict[str, Any]], None]
Range = tuple[float | None, float | None]
ProjectionSpec = str | Sequence[float]
ParameterBinding = str | float | int


def _resolve_parallel_workers(value: Any) -> int:
    """Resolve user-facing worker counts.

    ``-1`` means "auto": use a conservative number based on available CPUs.
    Thread pools are useful for expensive NumPy/SciPy-heavy evaluations, but
    oversubscribing can easily make small fits slower.
    """

    requested = int(value or 1)
    if requested == -1:
        available = os.cpu_count() or 1
        return max(1, min(available - 1, 8))
    return max(1, requested)


@dataclass(frozen=True)
class ParameterSpec:
    """Named scalar fitting parameter.

    Bounds use the same units as the parameter value. Fixed parameters are
    included in the parameter dictionary passed to model functions but excluded
    from the optimizer vector.
    """

    name: str
    value: float
    min: float | None = None
    max: float | None = None
    vary: bool = True
    unit: str = ""
    description: str = ""


@dataclass(frozen=True)
class ModelSpec:
    """Callable physics model with descriptive metadata.

    The callable predicts measured intensity at the coordinates in
    :class:`PointData4D`. It may wrap phenomenological formulae, ab initio
    calculations, Monte Carlo simulations, or any other model that can be
    evaluated for a trial parameter dictionary.
    """

    name: str
    predict: ModelFunction
    kind: str = "phenomenological"
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __call__(self, data: PointData4D, params: dict[str, float]) -> FloatArray:
        """Evaluate the model at the supplied data coordinates."""

        return np.asarray(self.predict(data, params), dtype=float)


@dataclass(frozen=True)
class ResolutionSpec:
    """Optional instrument-resolution model.

    The callable receives the data coordinates, unconvolved model values, and
    trial parameters. It returns values in the same measured-intensity units.
    Use :func:`identity_resolution` when resolution effects are intentionally
    disabled or deferred.
    """

    name: str
    apply: ResolutionFunction | None = None
    approximation: str = "none"
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.apply is None:
            object.__setattr__(self, "apply", identity_resolution)

    def __call__(
        self,
        data: PointData4D,
        model_values: FloatArray,
        params: dict[str, float],
    ) -> FloatArray:
        """Apply the resolution function to model values."""

        return np.asarray(self.apply(data, model_values, params), dtype=float)


@dataclass(frozen=True)
class DerivedParameter:
    """A parameter computed from other parameters instead of the optimizer.

    The legacy linear form is ``base + sign * offset``. An exact relationship
    may instead supply ``expression`` and ``dependencies``; the target is then
    removed from the optimizer and evaluated from the remaining parameters.
    Backtick-delimited names allow readable qualified references such as
    ``10 / `Model 1.scale``` when component names contain spaces.
    """

    name: str
    base: str | float
    offset: str | None = None
    sign: float = 1.0
    expression: str | None = None
    dependencies: tuple[str, ...] = ()

    def evaluate(self, params: dict[str, float]) -> float:
        if self.expression is not None:
            return evaluate_parameter_expression(self.expression, params)
        base = float(params[self.base]) if isinstance(self.base, str) else float(self.base)
        if self.offset is None:
            return base
        return base + float(self.sign) * float(params[self.offset])


_EXPRESSION_FUNCTIONS: dict[str, Callable[..., float]] = {
    "abs": abs,
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
}
_BACKTICK_PARAMETER = re.compile(r"`([^`]+)`")


def parameter_expression_names(expression: str) -> tuple[str, ...]:
    """Return parameter references used by a safe arithmetic expression."""

    tree, aliases = _parameter_expression_tree(expression)
    names: list[str] = []

    def visit(node: ast.AST) -> None:
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _EXPRESSION_FUNCTIONS:
                raise ValueError("constraint expressions only allow common scalar functions")
            for argument in node.args:
                visit(argument)
            if node.keywords:
                raise ValueError("constraint expression functions do not accept keywords")
            return
        reference = _expression_reference(node, aliases)
        if reference is not None:
            if reference not in names:
                names.append(reference)
            return
        if isinstance(node, ast.Expression):
            visit(node.body)
        elif isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)
        ):
            visit(node.left)
            visit(node.right)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            visit(node.operand)
        elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return
        else:
            raise ValueError("constraint expressions only allow arithmetic and scalar functions")

    visit(tree)
    return tuple(names)


def evaluate_parameter_expression(expression: str, params: Mapping[str, float]) -> float:
    """Safely evaluate a constraint expression against named parameters."""

    tree, aliases = _parameter_expression_tree(expression)

    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        reference = _expression_reference(node, aliases)
        if reference is not None:
            if reference not in params:
                raise ValueError(f"constraint expression references unknown parameter {reference!r}")
            return float(params[reference])
        if isinstance(node, ast.UnaryOp):
            value = evaluate(node.operand)
            if isinstance(node.op, ast.UAdd):
                return value
            if isinstance(node.op, ast.USub):
                return -value
        if isinstance(node, ast.BinOp):
            left, right = evaluate(node.left), evaluate(node.right)
            operations = {
                ast.Add: lambda: left + right,
                ast.Sub: lambda: left - right,
                ast.Mult: lambda: left * right,
                ast.Div: lambda: left / right,
                ast.Pow: lambda: left**right,
                ast.Mod: lambda: left % right,
            }
            operation = operations.get(type(node.op))
            if operation is not None:
                return float(operation())
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            function = _EXPRESSION_FUNCTIONS.get(node.func.id)
            if function is not None and not node.keywords:
                return float(function(*(evaluate(argument) for argument in node.args)))
        raise ValueError("unsupported syntax in constraint expression")

    value = float(evaluate(tree))
    if not np.isfinite(value):
        raise ValueError("constraint expression produced a non-finite value")
    return value


def _parameter_expression_tree(expression: str) -> tuple[ast.Expression, dict[str, str]]:
    aliases: dict[str, str] = {}

    def replace(match: re.Match[str]) -> str:
        alias = f"__parameter_{len(aliases)}"
        aliases[alias] = match.group(1).strip()
        return alias

    source = _BACKTICK_PARAMETER.sub(replace, str(expression).strip())
    if not source:
        raise ValueError("constraint expression cannot be empty")
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid constraint expression: {exc.msg}") from exc
    return tree, aliases


def _expression_reference(node: ast.AST, aliases: Mapping[str, str]) -> str | None:
    if isinstance(node, ast.Name):
        if node.id in _EXPRESSION_FUNCTIONS:
            return None
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        parts: list[str] = []
        current: ast.AST = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(aliases.get(current.id, current.id))
            return ".".join(reversed(parts))
    return None


@dataclass(frozen=True)
class FitDataset:
    """One dataset participating in a simultaneous fit.

    Parameters
    ----------
    name:
        Stable label used in diagnostics and per-dataset chi-squared values.
    data:
        Flattened measured points.
    weight:
        Multiplicative contribution to the global chi-squared objective. A
        weight of 1.0 preserves the usual statistical weighting by ``sigma``.
    transforms:
        Ordered preprocessing callables. Use these for masking, rebinning,
        normalization corrections, or other dataset-local preparation.
    resolution:
        Optional instrument resolution model for this dataset.
    parameter_bindings:
        Mapping from local model/resolution parameter names to global parameter
        names or fixed scalar values. This lets arbitrary groups of datasets
        share one fitted value while other groups use independently fitted
        values.
    model:
        Optional dataset-specific model. When set, it overrides the problem's
        shared model for this dataset, which lets different datasets fit
        different model compositions in one simultaneous problem.
    data_scale_parameter:
        Optional optimizer parameter name for a data-side scale factor. When
        set, residuals compare ``scale * intensity`` to the model and use
        ``abs(scale) * sigma`` for uncertainty, matching the fixed dataset
        scale convention.
    metadata:
        Free-form instrument, scan, normalization, and provenance notes.
    """

    name: str
    data: PointData4D
    weight: float = 1.0
    transforms: Sequence[DataTransform] = field(default_factory=tuple)
    resolution: Any = None
    parameter_bindings: dict[str, ParameterBinding] = field(default_factory=dict)
    model: Any = None
    data_scale_parameter: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    model_jacobian: Any = None
    """Optional ``(data, params) -> {qualified_name: d(model)/d(param)}`` giving
    exact model gradients for analytic least-squares. When every dataset in a
    problem supplies one, the optimizer uses them instead of finite
    differences."""
    _prepared_valid_cache: dict[bool, PointData4D] = field(
        default_factory=dict, init=False, compare=False, repr=False
    )

    def prepared(self) -> PointData4D:
        """Return data after applying all dataset-local transforms."""

        prepared = self.data
        for transform in self.transforms:
            prepared = transform(prepared)
            if not isinstance(prepared, PointData4D):
                raise TypeError("dataset transforms must return PointData4D")
        return prepared

    def prepared_valid(self, *, require_positive_sigma: bool) -> PointData4D:
        """Return the transformed, validity-filtered points, memoized per fit.

        Transforms and validity filtering depend only on ``self.data``, not on
        trial parameters, so the result is stable for the lifetime of the
        dataset. Optimizers re-evaluate the model thousands of times; caching
        here avoids re-running the transform chain each iteration and, crucially,
        returns the *same* ``PointData4D`` object every time so per-dataset model
        precomputations keyed on data identity (e.g. the RPA phase geometry)
        actually hit their cache instead of rebuilding.
        """

        cached = self._prepared_valid_cache.get(require_positive_sigma)
        if cached is None:
            cached = self.prepared().valid(require_positive_sigma=require_positive_sigma)
            self._prepared_valid_cache[require_positive_sigma] = cached
        return cached

    def apply_resolution(
        self,
        data: PointData4D,
        model_values: FloatArray,
        params: dict[str, float],
    ) -> FloatArray:
        """Apply this dataset's resolution model, if one is configured."""

        if self.resolution is None:
            return identity_resolution(data, model_values, params)
        if isinstance(self.resolution, ResolutionSpec):
            return self.resolution(data, model_values, params)
        return np.asarray(self.resolution(data, model_values, params), dtype=float)


@dataclass(frozen=True)
class OptimizationConfig:
    """Settings for deterministic parameter optimization."""

    method: str = "least_squares"
    require_positive_sigma: bool = True
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SamplerConfig:
    """Settings for posterior sampling with the optional emcee backend."""

    method: str = "emcee"
    n_walkers: int | None = None
    n_steps: int | None = None
    burn_in: int = 0
    thin: int = 1
    random_seed: int | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class SamplingResult:
    """Container for posterior/uncertainty samples."""

    samples: FloatArray
    variable_names: list[str]
    log_probability: FloatArray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    chain: FloatArray | None = None
    log_probability_chain: FloatArray | None = None


class SamplingCancelled(RuntimeError):
    """Raised when posterior sampling is cancelled after producing samples."""

    def __init__(self, message: str, result: SamplingResult) -> None:
        super().__init__(message)
        self.result = result


@dataclass(frozen=True)
class FitProblem:
    """Complete simultaneous-fitting problem definition.

    A problem combines one physics model, one shared parameter set, and any
    number of weighted datasets. Dataset-specific behavior belongs in
    ``FitDataset.transforms`` and ``FitDataset.resolution``; shared physics
    belongs in ``model``. ``model`` may be ``None`` when every dataset carries
    its own ``FitDataset.model``.

    ``derived_parameters`` are evaluated from the optimizer/fixed parameters on
    every residual evaluation and merged into the parameter dictionary passed
    to models. Their names must not collide with ``parameter_specs`` names.
    """

    datasets: Sequence[FitDataset]
    model: ModelSpec | ModelFunction | None
    parameter_specs: Sequence[ParameterSpec]
    derived_parameters: Sequence[DerivedParameter] = field(default_factory=tuple)
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.datasets) == 0:
            raise ValueError("FitProblem requires at least one dataset")
        names = [dataset.name for dataset in self.datasets]
        if len(names) != len(set(names)):
            raise ValueError("FitProblem dataset names must be unique")
        for dataset in self.datasets:
            if dataset.weight < 0.0:
                raise ValueError("dataset weights must be nonnegative")
            if self.model is None and dataset.model is None:
                raise ValueError(
                    f"dataset {dataset.name!r} has no model and the problem has no shared model"
                )
        spec_names = {spec.name for spec in self.parameter_specs}
        derived_names = {derived.name for derived in self.derived_parameters}
        available_names = spec_names | derived_names
        for derived in self.derived_parameters:
            if derived.name in spec_names:
                raise ValueError(
                    f"derived parameter {derived.name!r} collides with a parameter spec"
                )
            references = (
                derived.dependencies
                if derived.expression is not None
                else (derived.base, derived.offset)
            )
            for reference in references:
                if isinstance(reference, str) and reference not in spec_names:
                    if reference not in available_names:
                        raise ValueError(
                            f"derived parameter {derived.name!r} references unknown parameter "
                            f"{reference!r}"
                        )
        self.resolve_parameters({spec.name: float(spec.value) for spec in self.parameter_specs})

    def predict(self, data: PointData4D, params: dict[str, float]) -> FloatArray:
        """Evaluate the shared physics model."""

        if self.model is None:
            raise ValueError("FitProblem has no shared model; use the dataset models")
        if isinstance(self.model, ModelSpec):
            return self.model(data, params)
        return np.asarray(self.model(data, params), dtype=float)

    def resolve_parameters(self, params: dict[str, float]) -> dict[str, float]:
        """Return ``params`` with derived parameters evaluated and merged."""

        if not self.derived_parameters:
            return params
        resolved = dict(params)
        pending = list(self.derived_parameters)
        while pending:
            progressed = False
            for derived in list(pending):
                references = (
                    derived.dependencies
                    if derived.expression is not None
                    else tuple(
                        reference
                        for reference in (derived.base, derived.offset)
                        if isinstance(reference, str)
                    )
                )
                if any(reference not in resolved for reference in references):
                    continue
                resolved[derived.name] = derived.evaluate(resolved)
                pending.remove(derived)
                progressed = True
            if not progressed:
                names = ", ".join(derived.name for derived in pending)
                raise ValueError(f"cyclic derived parameter constraints: {names}")
        return resolved


@dataclass
class FitResult:
    """Result returned by deterministic optimization."""

    params: dict[str, float]
    success: bool
    message: str
    cost: float
    chi2: float
    reduced_chi2: float
    residuals: FloatArray
    model_values: FloatArray
    covariance: FloatArray | None
    stderr: dict[str, float] | None
    variable_names: list[str]
    fixed_params: dict[str, float]
    dataset_chi2: dict[str, float] = field(default_factory=dict)
    dataset_reduced_chi2: dict[str, float] = field(default_factory=dict)
    dataset_sizes: dict[str, int] = field(default_factory=dict)
    dataset_weights: dict[str, float] = field(default_factory=dict)
    dataset_residuals: dict[str, FloatArray] = field(default_factory=dict)
    dataset_model_values: dict[str, FloatArray] = field(default_factory=dict)


def identity_resolution(
    data: PointData4D,
    model_values: ArrayLike,
    params: dict[str, float],
) -> FloatArray:
    """Resolution placeholder that returns model values unchanged.

    The ``data`` and ``params`` arguments are accepted so this function has the
    same signature as instrument-specific resolution functions.
    """

    del data, params
    return np.asarray(model_values, dtype=float)


def reciprocal_basis_from_lattice_parameters(
    a: float,
    b: float,
    c: float,
    alpha: float,
    beta: float,
    gamma: float,
    *,
    include_2pi: bool = True,
) -> FloatArray:
    """Return a matrix that converts HKL in RLU to Cartesian momentum.

    The returned ``3x3`` matrix has reciprocal-lattice basis vectors as columns.
    Multiplying it by ``[H, K, L]`` gives ``Q`` in inverse angstroms when
    ``include_2pi=True``, which is the convention normally used for neutron
    scattering momentum transfer.
    """

    lengths = np.asarray([a, b, c], dtype=float)
    if np.any(lengths <= 0.0):
        raise ValueError("lattice lengths must be positive")
    angles = np.deg2rad(np.asarray([alpha, beta, gamma], dtype=float))
    if np.any((angles <= 0.0) | (angles >= np.pi)):
        raise ValueError("lattice angles must be between 0 and 180 degrees")

    alpha_r, beta_r, gamma_r = angles
    avec = np.array([a, 0.0, 0.0], dtype=float)
    bvec = np.array([b * np.cos(gamma_r), b * np.sin(gamma_r), 0.0], dtype=float)
    cx = c * np.cos(beta_r)
    cy = c * (np.cos(alpha_r) - np.cos(beta_r) * np.cos(gamma_r)) / np.sin(gamma_r)
    cz2 = c * c - cx * cx - cy * cy
    if cz2 <= 0.0:
        raise ValueError("lattice parameters produce a non-positive unit-cell volume")
    cvec = np.array([cx, cy, np.sqrt(cz2)], dtype=float)

    volume = float(np.dot(avec, np.cross(bvec, cvec)))
    factor = 2.0 * np.pi if include_2pi else 1.0
    astar = factor * np.cross(bvec, cvec) / volume
    bstar = factor * np.cross(cvec, avec) / volume
    cstar = factor * np.cross(avec, bvec) / volume
    return np.column_stack([astar, bstar, cstar])


def direct_basis_from_lattice_parameters(
    a: float,
    b: float,
    c: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> FloatArray:
    """Return the 3x3 matrix whose columns are the direct lattice vectors (Å).

    Uses the same Cartesian frame as
    :func:`reciprocal_basis_from_lattice_parameters` (``a`` along ``x``), so
    direct-lattice directions ``[u v w]`` and reciprocal directions ``(H K L)``
    convert into one consistent Cartesian frame.
    """

    lengths = np.asarray([a, b, c], dtype=float)
    if np.any(lengths <= 0.0):
        raise ValueError("lattice lengths must be positive")
    angles = np.deg2rad(np.asarray([alpha, beta, gamma], dtype=float))
    if np.any((angles <= 0.0) | (angles >= np.pi)):
        raise ValueError("lattice angles must be between 0 and 180 degrees")
    alpha_r, beta_r, gamma_r = angles
    avec = np.array([a, 0.0, 0.0], dtype=float)
    bvec = np.array([b * np.cos(gamma_r), b * np.sin(gamma_r), 0.0], dtype=float)
    cx = c * np.cos(beta_r)
    cy = c * (np.cos(alpha_r) - np.cos(beta_r) * np.cos(gamma_r)) / np.sin(gamma_r)
    cz2 = c * c - cx * cx - cy * cy
    if cz2 <= 0.0:
        raise ValueError("lattice parameters produce a non-positive unit-cell volume")
    cvec = np.array([cx, cy, np.sqrt(cz2)], dtype=float)
    return np.column_stack([avec, bvec, cvec])


def magnetic_field_vector(
    magnitude_tesla: float,
    direction: ArrayLike,
    frame: str,
    lattice: Mapping[str, Any],
) -> FloatArray:
    """Return the applied field as a Cartesian 3-vector in Tesla.

    ``direction`` is a crystallographic direction in the given ``frame``:
    ``"uvw"`` interprets it as a direct-lattice direction ``[u v w]`` (the
    usual experimental statement, e.g. field along ``[1 1 1]``), ``"hkl"`` as
    a reciprocal-lattice direction ``(H K L)``. The direction is normalized in
    Cartesian coordinates and scaled by ``magnitude_tesla``; only the
    direction's orientation matters, not its length. For cubic lattices the
    two frames give the same directions; for lower symmetry they differ.
    """

    magnitude = float(magnitude_tesla)
    vector = np.asarray(direction, dtype=float).reshape(3)
    if not np.all(np.isfinite(vector)) or float(np.linalg.norm(vector)) == 0.0:
        raise ValueError("magnetic field direction must be a finite nonzero 3-vector")
    a = float(lattice["a"])
    b = float(lattice["b"])
    c = float(lattice["c"])
    alpha = float(lattice.get("alpha", 90.0))
    beta = float(lattice.get("beta", 90.0))
    gamma = float(lattice.get("gamma", 90.0))
    frame_name = str(frame).lower()
    if frame_name == "uvw":
        basis = direct_basis_from_lattice_parameters(a, b, c, alpha, beta, gamma)
    elif frame_name == "hkl":
        basis = reciprocal_basis_from_lattice_parameters(a, b, c, alpha, beta, gamma)
    else:
        raise ValueError(f"unknown magnetic field frame {frame!r}; use 'uvw' or 'hkl'")
    cartesian = basis @ vector
    return magnitude * cartesian / float(np.linalg.norm(cartesian))


def attach_lattice_parameters(
    data: PointData4D,
    *,
    a: float,
    b: float,
    c: float,
    alpha: float = 90.0,
    beta: float = 90.0,
    gamma: float = 90.0,
    include_2pi: bool = True,
) -> PointData4D:
    """Return ``data`` with lattice metadata for RLU-to-``|Q|`` conversion."""

    matrix = reciprocal_basis_from_lattice_parameters(
        a,
        b,
        c,
        alpha,
        beta,
        gamma,
        include_2pi=include_2pi,
    )
    out = _copy_point_data(data, mask=data.mask)
    out.metadata["lattice_parameters"] = {
        "a": float(a),
        "b": float(b),
        "c": float(c),
        "alpha": float(alpha),
        "beta": float(beta),
        "gamma": float(gamma),
        "angle_units": "degree",
        "include_2pi": include_2pi,
    }
    out.metadata["rlu_to_inv_angstrom_matrix"] = matrix.tolist()
    return out


def attach_ub_matrix(
    data: PointData4D,
    ub_matrix: ArrayLike,
    *,
    key: str = "ub_matrix",
) -> PointData4D:
    """Return ``data`` with a sample UB or equivalent HKL-to-Q matrix attached.

    The matrix must be ``3x3`` and is assumed to convert ``[H,K,L]`` to
    Cartesian momentum transfer in inverse angstroms. Mantid NeXus files often
    store this as an ``orientation_matrix`` in the sample oriented-lattice tree.
    """

    matrix = _as_3x3_matrix(ub_matrix, name=key)
    out = _copy_point_data(data, mask=data.mask)
    out.metadata[key] = matrix.tolist()
    out.metadata["rlu_to_inv_angstrom_matrix"] = matrix.tolist()
    return out


def q_vectors_inv_angstrom(
    data: PointData4D,
    matrix: ArrayLike | None = None,
) -> FloatArray:
    """Return Cartesian ``Q`` vectors in inverse angstroms.

    If ``matrix`` is omitted, the helper first treats data tagged with inverse
    angstrom coordinate units as already Cartesian. Otherwise it looks for
    ``rlu_to_inv_angstrom_matrix``, ``ub_matrix``, or ``orientation_matrix`` in
    ``data.metadata``.
    """

    if matrix is None and _metadata_coordinate_units_are_inv_angstrom(data.metadata):
        return np.column_stack([data.H, data.K, data.L]).astype(float, copy=False)

    transform = _resolve_q_transform(data, matrix)
    hkl = np.column_stack([data.H, data.K, data.L])
    return hkl @ transform.T


def q_modulus_inv_angstrom(
    data: PointData4D,
    matrix: ArrayLike | None = None,
) -> FloatArray:
    """Return ``|Q|`` in inverse angstroms."""

    return np.linalg.norm(q_vectors_inv_angstrom(data, matrix=matrix), axis=1)


def apply_mask(data: PointData4D, mask: ArrayLike) -> PointData4D:
    """Return a copy of ``data`` with an additional boolean analysis mask.

    ``True`` values remain eligible for analysis. The new mask is combined with
    the existing mask using logical AND, so earlier masking decisions are
    preserved.
    """

    mask_arr = np.asarray(mask, dtype=bool)
    if mask_arr.shape != data.H.shape:
        raise ValueError(f"mask shape {mask_arr.shape} does not match data shape {data.H.shape}")
    combined = np.asarray(data.mask, dtype=bool) & mask_arr
    return _copy_point_data(data, mask=combined)


def mask_by_coordinate_range(
    data: PointData4D,
    *,
    H: Range | None = None,
    K: Range | None = None,
    L: Range | None = None,
    E: Range | None = None,
) -> PointData4D:
    """Mask points outside inclusive coordinate ranges.

    Bounds may be ``None`` on either side. For example ``E=(2.0, None)`` keeps
    points with energy transfer greater than or equal to 2 meV.
    """

    keep = np.ones(data.size, dtype=bool)
    for name, bounds in {"H": H, "K": K, "L": L, "E": E}.items():
        if bounds is None:
            continue
        lower, upper = bounds
        values = getattr(data, name)
        if lower is not None:
            keep &= values >= float(lower)
        if upper is not None:
            keep &= values <= float(upper)
    return apply_mask(data, keep)


def mask_out_energy_q_range(
    data: PointData4D,
    *,
    energy: Range | None = None,
    q_modulus: Range | None = None,
    q_matrix: ArrayLike | None = None,
) -> PointData4D:
    """Mask out points satisfying an energy range, ``|Q|`` range, or both.

    When both ranges are supplied, a point is masked only if it lies in both
    ranges. ``|Q|`` is evaluated in inverse angstroms and therefore requires
    direct inverse-angstrom coordinates, attached lattice metadata, or an
    explicit ``q_matrix``.
    """

    reject = np.ones(data.size, dtype=bool)
    if energy is None and q_modulus is None:
        return _copy_point_data(data, mask=data.mask)
    if energy is not None:
        reject &= _in_range(data.E, energy)
    if q_modulus is not None:
        reject &= _in_range(q_modulus_inv_angstrom(data, matrix=q_matrix), q_modulus)
    return apply_mask(data, ~reject)


def mask_out_box(
    data: PointData4D,
    *,
    dimensions: Sequence[ProjectionSpec],
    center: Sequence[float],
    half_widths: Sequence[float],
) -> PointData4D:
    """Mask out an N-dimensional box in projected coordinate space.

    ``dimensions`` may contain ``"H"``, ``"K"``, ``"L"``, ``"E"``, or numeric
    projection vectors. Three-component vectors project only reciprocal-space
    coordinates; four-component vectors project ``(H,K,L,E)``. A 2D box masks
    all points whose selected two projected coordinates fall inside the box,
    regardless of the other dimensions.
    """

    values = _project_dimensions(data, dimensions)
    center_arr = _as_length("center", center, len(dimensions))
    widths = _as_length("half_widths", half_widths, len(dimensions))
    if np.any(widths <= 0.0):
        raise ValueError("box half_widths must be positive")
    reject = np.all(np.abs(values - center_arr) <= widths, axis=1)
    return apply_mask(data, ~reject)


def mask_out_ellipsoid(
    data: PointData4D,
    *,
    dimensions: Sequence[ProjectionSpec],
    center: Sequence[float],
    radii: Sequence[float],
) -> PointData4D:
    """Mask out an N-dimensional ellipsoid in projected coordinate space."""

    values = _project_dimensions(data, dimensions)
    center_arr = _as_length("center", center, len(dimensions))
    radii_arr = _as_length("radii", radii, len(dimensions))
    if np.any(radii_arr <= 0.0):
        raise ValueError("ellipsoid radii must be positive")
    scaled = (values - center_arr) / radii_arr
    reject = np.sum(scaled * scaled, axis=1) <= 1.0
    return apply_mask(data, ~reject)


def mask_out_phonon_cone(
    data: PointData4D,
    *,
    center: Sequence[float] | Sequence[Sequence[float]],
    slope: float,
    energy_range: Range | None = None,
    q_matrix: ArrayLike | None = None,
    center_units: Literal["rlu", "inv_angstrom"] = "rlu",
    energy_offset: float = 0.0,
    radius_offset: float = 0.0,
    use_absolute_energy: bool = True,
    energy_range_uses_absolute: bool = True,
) -> PointData4D:
    """Mask phonon-like cones centered on one or more reciprocal-space points.

    ``slope`` is ``dE/d|Q|`` in ``meV / inverse angstrom``. Points are rejected
    when their distance from any ``center`` in reciprocal space is less than
    ``abs(E - energy_offset) / slope + radius_offset``. ``energy_range`` limits
    the energies where the cone is applied, which is useful for stopping the
    acoustic-phonon mask once optical branches or unrelated features dominate.
    """

    if slope <= 0.0:
        raise ValueError("phonon slope must be positive")
    if radius_offset < 0.0:
        raise ValueError("radius_offset must be nonnegative")

    center_arr = np.asarray(center, dtype=float)
    if center_arr.shape == (3,):
        center_arr = center_arr.reshape(1, 3)
    if center_arr.ndim != 2 or center_arr.shape[0] == 0 or center_arr.shape[1] != 3:
        raise ValueError("phonon center must be one 3-vector or a nonempty list of 3-vectors")
    if not np.all(np.isfinite(center_arr)):
        raise ValueError("phonon centers must contain only finite coordinates")
    q_vectors = q_vectors_inv_angstrom(data, matrix=q_matrix)
    if center_units == "rlu":
        transform = _resolve_q_transform(data, q_matrix)
        centers_q = center_arr @ transform.T
    elif center_units == "inv_angstrom":
        centers_q = center_arr
    else:
        raise ValueError("center_units must be 'rlu' or 'inv_angstrom'")

    energy_delta = data.E - float(energy_offset)
    radius_energy = np.abs(energy_delta) if use_absolute_energy else energy_delta
    cone_radius = radius_energy / float(slope) + float(radius_offset)
    reject = np.zeros(data.size, dtype=bool)
    for center_q in centers_q:
        reject |= np.linalg.norm(q_vectors - center_q, axis=1) <= cone_radius

    if energy_range is not None:
        energy_for_range = np.abs(data.E) if energy_range_uses_absolute else data.E
        reject &= _in_range(energy_for_range, energy_range)
    return apply_mask(data, ~reject)


def make_mask_transform(
    *,
    mask: ArrayLike | None = None,
    H: Range | None = None,
    K: Range | None = None,
    L: Range | None = None,
    E: Range | None = None,
) -> DataTransform:
    """Create a reusable masking transform for :class:`FitDataset`."""

    def transform(data: PointData4D) -> PointData4D:
        out = data if mask is None else apply_mask(data, mask)
        return mask_by_coordinate_range(out, H=H, K=K, L=L, E=E)

    return transform


def make_energy_q_mask_transform(**kwargs: Any) -> DataTransform:
    """Create a transform from :func:`mask_out_energy_q_range`."""

    def transform(data: PointData4D) -> PointData4D:
        return mask_out_energy_q_range(data, **kwargs)

    return transform


def make_box_mask_transform(**kwargs: Any) -> DataTransform:
    """Create a transform from :func:`mask_out_box`."""

    def transform(data: PointData4D) -> PointData4D:
        return mask_out_box(data, **kwargs)

    return transform


def make_ellipsoid_mask_transform(**kwargs: Any) -> DataTransform:
    """Create a transform from :func:`mask_out_ellipsoid`."""

    def transform(data: PointData4D) -> PointData4D:
        return mask_out_ellipsoid(data, **kwargs)

    return transform


def make_phonon_mask_transform(**kwargs: Any) -> DataTransform:
    """Create a transform from :func:`mask_out_phonon_cone`."""

    def transform(data: PointData4D) -> PointData4D:
        return mask_out_phonon_cone(data, **kwargs)

    return transform


def rebin_point_data(
    data: PointData4D,
    *,
    lower: ArrayLike | None = None,
    upper: ArrayLike | None = None,
    step_size: ArrayLike | None = None,
    num_bins: ArrayLike | None = None,
    fractional: bool = True,
    normalize: bool = True,
    mean_weighting: str = "inverse_variance",
    max_batch_bytes: int = 192 * 1024 * 1024,
    progress_callback: ProgressCallback | None = None,
    symmetry_operations: Sequence[ArrayLike] | None = None,
) -> PointData4D:
    """Rebin flattened ``(H,K,L,E)`` point data onto a regular 4D grid.

    Empty bins are retained with ``mask=False`` so later calls to
    :meth:`PointData4D.valid` drop them. This helper is intentionally simple;
    more specialized projection/integration workflows can be expressed as
    custom :class:`FitDataset` transforms.
    """

    source = data.valid(require_positive_sigma=False)
    if source.size == 0:
        raise ValueError("no valid data points remain before rebinning")

    coords = np.column_stack(source.coordinates())
    kwargs = dict(
        data_errs=source.sigma,
        lower=lower,
        upper=upper,
        step_size=step_size,
        num_bins=num_bins,
        fractional=fractional,
        normalize=normalize,
        mean_weighting=mean_weighting,
        max_batch_bytes=max_batch_bytes,
        progress_callback=progress_callback,
    )
    result = (
        rebin_nd_symmetry(source.intensity, coords, symmetry_operations, **kwargs)
        if symmetry_operations is not None
        else rebin_nd(source.intensity, coords, **kwargs)
    )
    if result.bin_centers_list is None:
        raise RuntimeError("rebinning did not produce bin centers")
    if result.binned_data is None or result.binned_data_errs is None or result.n_samples is None:
        raise RuntimeError("rebinning did not produce binned data")

    H_grid, K_grid, L_grid, E_grid = np.meshgrid(*result.bin_centers_list, indexing="ij")
    mask = np.isfinite(result.binned_data) & np.isfinite(result.binned_data_errs)
    mask &= result.n_samples > 0.0

    temperature: float | FloatArray | None
    if isinstance(source.temperature, np.ndarray):
        temperature = None
    else:
        temperature = source.temperature

    metadata = dict(data.metadata)
    metadata["rebin"] = {
        "lower": None if lower is None else np.asarray(lower, dtype=float).tolist(),
        "upper": None if upper is None else np.asarray(upper, dtype=float).tolist(),
        "step_size": np.asarray(result.step_size, dtype=float).tolist(),
        "num_bins": np.asarray(result.num_bins, dtype=int).tolist(),
        "fractional": fractional,
        "normalize": normalize,
        "mean_weighting": str(mean_weighting),
        "max_batch_bytes": int(max_batch_bytes),
    }
    if isinstance(source.temperature, np.ndarray):
        metadata["temperature_note"] = "pointwise temperature was dropped during rebinning"

    return PointData4D(
        H_grid.ravel(),
        K_grid.ravel(),
        L_grid.ravel(),
        E_grid.ravel(),
        result.binned_data.ravel(),
        result.binned_data_errs.ravel(),
        mask=mask.ravel(),
        temperature=temperature,
        magnetic_field=None if source.magnetic_field is None else np.array(source.magnetic_field),
        metadata=metadata,
    )


def make_rebin_transform(**kwargs: Any) -> DataTransform:
    """Create a reusable 4D rebinning transform for :class:`FitDataset`."""

    def transform(data: PointData4D) -> PointData4D:
        return rebin_point_data(data, **kwargs)

    return transform


def pack_parameters(
    specs: Sequence[ParameterSpec],
) -> tuple[FloatArray, tuple[FloatArray, FloatArray], list[str], dict[str, float]]:
    """Pack variable parameters for SciPy least squares."""

    x0: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    names: list[str] = []
    fixed: dict[str, float] = {}

    seen: set[str] = set()
    for spec in specs:
        if spec.name in seen:
            raise ValueError(f"duplicate parameter name {spec.name!r}")
        seen.add(spec.name)
        lo = -np.inf if spec.min is None else float(spec.min)
        hi = np.inf if spec.max is None else float(spec.max)
        if lo > hi:
            raise ValueError(f"lower bound exceeds upper bound for {spec.name!r}")
        if not (lo <= spec.value <= hi):
            raise ValueError(f"initial value for {spec.name!r} is outside bounds")
        if spec.vary:
            names.append(spec.name)
            x0.append(float(spec.value))
            lower.append(lo)
            upper.append(hi)
        else:
            fixed[spec.name] = float(spec.value)

    return (
        np.asarray(x0, dtype=float),
        (np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)),
        names,
        fixed,
    )


def unpack_parameters(x: Sequence[float], names: Sequence[str], fixed: dict[str, float]) -> dict[str, float]:
    """Combine optimizer vector values with fixed parameters."""

    params = dict(fixed)
    params.update({name: float(value) for name, value in zip(names, x)})
    return params


def fit_least_squares(
    data: PointData4D,
    model: ModelFunction,
    parameter_specs: Sequence[ParameterSpec],
    *,
    require_positive_sigma: bool = True,
    kwargs: dict | None = None,
) -> FitResult:
    """Fit one measured-intensity model to one :class:`PointData4D`.

    This is the backwards-compatible convenience wrapper around
    :func:`fit_problem_least_squares`. For simultaneous fitting, construct a
    :class:`FitProblem` directly.
    """

    model_kwargs = {} if kwargs is None else dict(kwargs)

    def wrapped_model(valid_data: PointData4D, params: dict[str, float]) -> FloatArray:
        return np.asarray(model(valid_data, params, **model_kwargs), dtype=float)

    problem = FitProblem(
        datasets=[FitDataset(name="data", data=data)],
        model=wrapped_model,
        parameter_specs=parameter_specs,
    )
    config = OptimizationConfig(require_positive_sigma=require_positive_sigma)
    return fit_problem_least_squares(problem, config=config)


def fit_problem_least_squares(
    problem: FitProblem,
    *,
    config: OptimizationConfig | None = None,
    progress_callback: ProgressCallback | None = None,
) -> FitResult:
    """Fit all datasets in a :class:`FitProblem` by weighted least squares.

    The global residual vector concatenates per-dataset residuals

    ``sqrt(dataset.weight) * (I_obs - I_model) / sigma``.

    The reported ``chi2`` is therefore the weighted objective contribution
    minimized by the optimizer. Per-dataset values in ``dataset_chi2`` use the
    same convention.
    """

    opt = OptimizationConfig() if config is None else config
    if opt.method != "least_squares":
        raise ValueError("only method='least_squares' is implemented")

    x0, bounds, names, fixed = pack_parameters(problem.parameter_specs)

    def residual_fn(x: FloatArray) -> FloatArray:
        params = unpack_parameters(x, names, fixed)
        evaluation = _evaluate_problem(
            problem,
            params,
            require_positive_sigma=opt.require_positive_sigma,
        )
        return evaluation.residuals

    jacobian_fn: Callable[[FloatArray], FloatArray] | None = None
    if problem_supports_analytic_jacobian(problem):
        def jacobian_fn(x: FloatArray) -> FloatArray:
            params = unpack_parameters(x, names, fixed)
            return _evaluate_problem_jacobian(
                problem,
                params,
                names,
                require_positive_sigma=opt.require_positive_sigma,
            )

    if len(names) == 0:
        residuals = residual_fn(np.asarray([], dtype=float))
        params = problem.resolve_parameters(dict(fixed))
        evaluation = _evaluate_problem(
            problem,
            params,
            require_positive_sigma=opt.require_positive_sigma,
        )
        chi2 = float(np.sum(residuals * residuals))
        dof = sum(evaluation.dataset_sizes.values())
        return FitResult(
            params=params,
            success=True,
            message="no variable parameters",
            cost=0.5 * chi2,
            chi2=chi2,
            reduced_chi2=float(chi2 / dof) if dof > 0 else np.nan,
            residuals=residuals,
            model_values=evaluation.model_values,
            covariance=None,
            stderr=None,
            variable_names=[],
            fixed_params=fixed,
            dataset_chi2=evaluation.dataset_chi2,
            dataset_reduced_chi2=evaluation.dataset_reduced_chi2,
            dataset_sizes=evaluation.dataset_sizes,
            dataset_weights=evaluation.dataset_weights,
            dataset_residuals=evaluation.dataset_residuals,
            dataset_model_values=evaluation.dataset_model_values,
        )

    x_start = x0
    init_config = opt.kwargs.get("initialization") if isinstance(opt.kwargs, dict) else None
    if isinstance(init_config, dict) and init_config.get("method") == "differential_evolution":
        x_start = initialize_problem_differential_evolution(
            problem,
            config=init_config,
            require_positive_sigma=opt.require_positive_sigma,
            progress_callback=progress_callback,
        )

    result = _run_least_squares(
        residual_fn,
        x0=x_start,
        bounds=bounds,
        kwargs=_least_squares_kwargs(opt.kwargs),
        progress_callback=progress_callback,
        names=names,
        fixed=fixed,
        jac=jacobian_fn,
    )
    params = problem.resolve_parameters(unpack_parameters(result.x, names, fixed))
    evaluation = _evaluate_problem(
        problem,
        params,
        require_positive_sigma=opt.require_positive_sigma,
    )
    residuals = evaluation.residuals
    chi2 = float(np.sum(residuals * residuals))
    dof = sum(evaluation.dataset_sizes.values()) - len(names)
    reduced_chi2 = chi2 / dof if dof > 0 else np.nan
    covariance = _covariance_from_jacobian(result.jac, chi2=chi2, dof=dof)
    stderr = None
    if covariance is not None:
        stderr = {
            name: float(np.sqrt(covariance[i, i]))
            for i, name in enumerate(names)
            if covariance[i, i] >= 0
        }

    return FitResult(
        params=params,
        success=bool(result.success),
        message=str(result.message),
        cost=float(result.cost),
        chi2=chi2,
        reduced_chi2=float(reduced_chi2),
        residuals=residuals,
        model_values=evaluation.model_values,
        covariance=covariance,
        stderr=stderr,
        variable_names=list(names),
        fixed_params=fixed,
        dataset_chi2=evaluation.dataset_chi2,
        dataset_reduced_chi2=evaluation.dataset_reduced_chi2,
        dataset_sizes=evaluation.dataset_sizes,
        dataset_weights=evaluation.dataset_weights,
        dataset_residuals=evaluation.dataset_residuals,
        dataset_model_values=evaluation.dataset_model_values,
    )


def evaluate_problem_model(
    problem: FitProblem,
    dataset_name: str,
    params: dict[str, float],
    data: PointData4D | None = None,
) -> FloatArray:
    """Evaluate one dataset's model at arbitrary coordinates.

    Derived parameters and the dataset's parameter bindings are resolved
    exactly as during optimization. ``data`` defaults to the dataset's own
    prepared points; pass a different :class:`PointData4D` to evaluate the
    fitted model over a full unmasked grid for visualization.
    """

    for dataset in problem.datasets:
        if dataset.name == dataset_name:
            break
    else:
        raise KeyError(f"unknown dataset {dataset_name!r}")
    target = dataset.prepared() if data is None else data
    resolved = problem.resolve_parameters(params)
    dataset_params = _apply_parameter_bindings(resolved, dataset.parameter_bindings)
    model = dataset.model if dataset.model is not None else problem.model
    if isinstance(model, ModelSpec):
        return model(target, dataset_params)
    if model is None:
        raise ValueError(f"dataset {dataset_name!r} has no model")
    return np.asarray(model(target, dataset_params), dtype=float)


def sample_problem_parameters(
    problem: FitProblem,
    config: SamplerConfig | None = None,
    *,
    initial_params: dict[str, float] | None = None,
    initial_walkers: FloatArray | None = None,
    require_positive_sigma: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> SamplingResult:
    """Sample fitted parameters with the configured posterior sampler.

    The default ``method="emcee"`` backend is optional. Install the package
    extra that provides ``emcee`` before using this function. The likelihood is
    the Gaussian chi-squared implied by the current weighted residual vector,
    with uniform priors from finite parameter bounds.
    """

    sampler = SamplerConfig() if config is None else config
    if sampler.method != "emcee":
        raise ValueError("only method='emcee' is implemented")
    x0, bounds, names, fixed = pack_parameters(problem.parameter_specs)
    if len(names) == 0:
        raise ValueError("emcee sampling requires at least one variable parameter")
    try:
        import emcee  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - depends on optional package.
        raise ImportError(
            "emcee posterior sampling requires the 'emcee' package. "
            "Install or update the nfit environment before sampling."
        ) from exc

    lower, upper = bounds
    if initial_params is not None:
        for index, name in enumerate(names):
            if name in initial_params:
                x0[index] = float(initial_params[name])
    x0 = _nudge_inside_bounds(np.clip(x0, lower, upper), bounds)

    n_dim = len(names)
    n_walkers = int(sampler.n_walkers or max(32, 2 * n_dim + 2))
    if n_walkers < 2 * n_dim:
        raise ValueError("emcee requires at least 2 * n_parameters walkers")
    n_steps = int(sampler.n_steps or 1000)
    if n_steps <= 0:
        raise ValueError("emcee n_steps must be positive")
    burn_in = int(max(0, sampler.burn_in))
    thin = int(max(1, sampler.thin))
    rng = np.random.default_rng(sampler.random_seed)
    if initial_walkers is None:
        p0 = _initial_walker_positions(x0, bounds, n_walkers, rng)
    else:
        p0 = np.asarray(initial_walkers, dtype=float)
        if p0.shape != (n_walkers, n_dim):
            raise ValueError(
                "initial_walkers must have shape "
                f"({n_walkers}, {n_dim}); got {tuple(p0.shape)}"
            )
        if np.any(p0 < lower) or np.any(p0 > upper):
            raise ValueError("initial_walkers must lie inside finite parameter bounds")

    def log_probability(x: FloatArray) -> float:
        if np.any(x < lower) or np.any(x > upper):
            return -np.inf
        params = unpack_parameters(x, names, fixed)
        try:
            evaluation = _evaluate_problem(
                problem,
                params,
                require_positive_sigma=require_positive_sigma,
            )
        except Exception:
            return -np.inf
        chi2 = float(np.dot(evaluation.residuals, evaluation.residuals))
        return -0.5 * chi2

    backend_kwargs = dict(sampler.kwargs)
    worker_request = backend_kwargs.pop("workers", backend_kwargs.pop("parallel_workers", 1))
    workers = _resolve_parallel_workers(worker_request)
    pool = ThreadPool(workers) if workers > 1 else None
    emcee_sampler: Any | None = None

    def result_from_sampler(*, completed: bool) -> SamplingResult:
        if emcee_sampler is None:
            raise RuntimeError("emcee sampler has not been initialized")
        chain = np.asarray(emcee_sampler.get_chain(), dtype=float)
        log_prob_chain = np.asarray(emcee_sampler.get_log_prob(), dtype=float)
        if chain.ndim != 3 or chain.shape[0] == 0:
            raise RuntimeError("emcee sampling was cancelled before any samples were recorded")
        actual_steps = int(chain.shape[0])
        samples = np.asarray(
            emcee_sampler.get_chain(discard=burn_in, thin=thin, flat=True), dtype=float
        )
        log_prob = np.asarray(
            emcee_sampler.get_log_prob(discard=burn_in, thin=thin, flat=True),
            dtype=float,
        )
        acceptance_fraction = np.asarray(emcee_sampler.acceptance_fraction, dtype=float)
        try:
            # ``quiet=False`` gives us AutocorrError.tau for chains that are
            # still too short, so users can estimate a useful target length.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                autocorrelation_time = [
                    float(value) for value in emcee_sampler.get_autocorr_time(quiet=False)
                ]
        except Exception as exc:
            tau = np.asarray(getattr(exc, "tau", []), dtype=float)
            autocorrelation_time = [float(value) for value in tau]
        finite_tau = [value for value in autocorrelation_time if np.isfinite(value) and value > 0.0]
        recommended_steps = (
            int(np.ceil(50.0 * max(finite_tau))) if finite_tau else None
        )
        recommended_additional_steps = (
            max(0, recommended_steps - actual_steps)
            if recommended_steps is not None
            else None
        )
        metadata: dict[str, Any] = {
            "method": "emcee",
            "n_walkers": n_walkers,
            "n_steps": actual_steps,
            "requested_n_steps": n_steps,
            "completed": bool(completed),
            "cancelled": not bool(completed),
            "burn_in": burn_in,
            "thin": thin,
            "random_seed": sampler.random_seed,
            "workers": workers,
            "worker_request": "auto" if int(worker_request or 1) == -1 else workers,
            "acceptance_fraction_mean": float(np.mean(acceptance_fraction)),
            "acceptance_fraction_min": float(np.min(acceptance_fraction)),
            "acceptance_fraction_max": float(np.max(acceptance_fraction)),
        }
        metadata["autocorrelation_time"] = autocorrelation_time
        metadata["autocorrelation_recommended_steps"] = recommended_steps
        metadata["autocorrelation_recommended_additional_steps"] = recommended_additional_steps
        if finite_tau:
            tau_text = ", ".join(f"{value:.3g}" for value in autocorrelation_time)
            autocorrelation_message = (
                f"emcee integrated autocorrelation time (steps): [{tau_text}]; "
                f"recommended total chain length: at least {recommended_steps:,} steps "
                f"(50 x max tau = {max(finite_tau):.3g}); "
                f"estimated additional steps: {recommended_additional_steps:,}."
            )
        else:
            autocorrelation_message = (
                "emcee integrated autocorrelation time is unavailable for this chain; "
                "run more steps before using it to estimate a chain length."
            )
        print(autocorrelation_message)
        return SamplingResult(
            samples=samples,
            variable_names=list(names),
            log_probability=log_prob,
            metadata=metadata,
            chain=chain,
            log_probability_chain=log_prob_chain,
        )

    try:
        if pool is not None:
            backend_kwargs["pool"] = pool
        emcee_sampler = emcee.EnsembleSampler(
            n_walkers, n_dim, log_probability, **backend_kwargs
        )
        sampler_start = time.perf_counter()
        for iteration, state in enumerate(emcee_sampler.sample(p0, iterations=n_steps), start=1):
            if progress_callback is not None:
                mean = np.mean(state.coords, axis=0)
                elapsed = time.perf_counter() - sampler_start
                try:
                    progress_callback(
                        {
                            "stage": "emcee",
                            "iteration": iteration,
                            "total": n_steps,
                            "elapsed_seconds": elapsed,
                            "seconds_per_step": elapsed / max(iteration, 1),
                            "parameters": {name: float(value) for name, value in zip(names, mean)},
                            "message": f"emcee step {iteration}/{n_steps}",
                        }
                    )
                except Exception as exc:
                    raise SamplingCancelled(
                        "emcee posterior sampling cancelled; partial samples were saved.",
                        result_from_sampler(completed=False),
                    ) from exc
        result = result_from_sampler(completed=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return result


def initialize_problem_differential_evolution(
    problem: FitProblem,
    *,
    config: dict[str, Any] | None = None,
    require_positive_sigma: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> FloatArray:
    """Return a least-squares start vector found by differential evolution."""

    if _scipy_differential_evolution is None:
        raise RuntimeError("differential evolution initialization requires SciPy")
    x0, bounds, names, fixed = pack_parameters(problem.parameter_specs)
    if len(names) == 0:
        return x0
    lower, upper = bounds
    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
        unbounded = [
            name
            for name, lo, hi in zip(names, lower, upper)
            if not np.isfinite(lo) or not np.isfinite(hi)
        ]
        raise ValueError(
            "differential evolution requires finite bounds for all variable parameters: "
            + ", ".join(unbounded)
        )

    options = {} if config is None else dict(config)
    options.pop("method", None)
    seed = options.pop("random_seed", options.pop("seed", None))
    workers = _resolve_parallel_workers(options.pop("workers", options.pop("parallel_workers", 1)))
    options.setdefault("polish", False)
    options.setdefault("updating", "immediate")
    pool = None
    if workers > 1:
        pool = ThreadPool(workers)
        options["workers"] = pool.map
        options["updating"] = "deferred"
    bounds_list = [(float(lo), float(hi)) for lo, hi in zip(lower, upper)]
    iteration = 0
    initialization_start = time.perf_counter()

    def objective(x: FloatArray) -> float:
        params = unpack_parameters(x, names, fixed)
        evaluation = _evaluate_problem(
            problem,
            params,
            require_positive_sigma=require_positive_sigma,
        )
        return float(np.dot(evaluation.residuals, evaluation.residuals))

    def callback(xk: FloatArray, convergence: float | None = None) -> bool:
        nonlocal iteration
        iteration += 1
        if progress_callback is not None:
            params = unpack_parameters(xk, names, fixed)
            elapsed = time.perf_counter() - initialization_start
            progress_callback(
                {
                    "stage": "initialization",
                    "iteration": iteration,
                    "elapsed_seconds": elapsed,
                    "seconds_per_step": elapsed / max(iteration, 1),
                    "parameters": {name: float(params[name]) for name in names},
                    "cost": objective(np.asarray(xk, dtype=float)),
                    "convergence": None if convergence is None else float(convergence),
                    "message": f"differential evolution generation {iteration}",
                }
            )
        return False

    try:
        objective(x0)
    except Exception as exc:
        preview = ", ".join(
            f"{name}={value:.6g}" for name, value in zip(names, x0[: len(names)])
        )
        raise RuntimeError(
            "differential evolution initialization could not evaluate the "
            "model at the starting parameters"
            + (f" ({preview})" if preview else "")
            + f": {exc}"
        ) from exc

    try:
        result = _scipy_differential_evolution(
            objective,
            bounds_list,
            seed=seed,
            callback=callback,
            **options,
        )
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    if progress_callback is not None:
        elapsed = time.perf_counter() - initialization_start
        progress_callback(
            {
                "stage": "initialization",
                "iteration": iteration,
                "elapsed_seconds": elapsed,
                "seconds_per_step": elapsed / iteration if iteration else None,
                "parameters": {name: float(value) for name, value in zip(names, result.x)},
                "cost": float(result.fun),
                "message": "differential evolution initialization finished",
            }
        )
    return np.asarray(result.x, dtype=float)


@dataclass
class _ProblemEvaluation:
    residuals: FloatArray
    model_values: FloatArray
    dataset_chi2: dict[str, float]
    dataset_reduced_chi2: dict[str, float]
    dataset_sizes: dict[str, int]
    dataset_weights: dict[str, float]
    dataset_residuals: dict[str, FloatArray]
    dataset_model_values: dict[str, FloatArray]


def _evaluate_problem(
    problem: FitProblem,
    params: dict[str, float],
    *,
    require_positive_sigma: bool,
) -> _ProblemEvaluation:
    residual_blocks: list[FloatArray] = []
    model_blocks: list[FloatArray] = []
    dataset_chi2: dict[str, float] = {}
    dataset_reduced_chi2: dict[str, float] = {}
    dataset_sizes: dict[str, int] = {}
    dataset_weights: dict[str, float] = {}
    dataset_residuals: dict[str, FloatArray] = {}
    dataset_model_values: dict[str, FloatArray] = {}

    params = problem.resolve_parameters(params)
    for dataset in problem.datasets:
        prepared = dataset.prepared_valid(require_positive_sigma=require_positive_sigma)
        if prepared.size == 0:
            raise ValueError(f"dataset {dataset.name!r} has no valid points after preprocessing")

        dataset_params = _apply_parameter_bindings(params, dataset.parameter_bindings)
        model_values = _evaluate_dataset_model(problem, dataset, prepared, dataset_params)
        if model_values.shape != prepared.intensity.shape:
            raise ValueError(
                f"model/resolution returned shape {model_values.shape} for dataset {dataset.name!r}; "
                f"expected {prepared.intensity.shape}"
            )

        intensity = np.asarray(prepared.intensity, dtype=float)
        sigma = np.asarray(prepared.sigma, dtype=float)
        if dataset.data_scale_parameter:
            scale = float(params[dataset.data_scale_parameter])
            intensity = intensity * scale
            sigma = sigma * max(abs(scale), np.finfo(float).tiny)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            residual = (intensity - model_values) / sigma
        weighted_residual = np.sqrt(dataset.weight) * residual
        residual_blocks.append(weighted_residual)
        model_blocks.append(model_values)

        with np.errstate(over="ignore", invalid="ignore"):
            chi2 = float(np.sum(weighted_residual * weighted_residual))
        dataset_chi2[dataset.name] = chi2
        dataset_reduced_chi2[dataset.name] = chi2 / prepared.size
        dataset_sizes[dataset.name] = prepared.size
        dataset_weights[dataset.name] = float(dataset.weight)
        dataset_residuals[dataset.name] = weighted_residual
        dataset_model_values[dataset.name] = model_values

    return _ProblemEvaluation(
        residuals=np.concatenate(residual_blocks),
        model_values=np.concatenate(model_blocks),
        dataset_chi2=dataset_chi2,
        dataset_reduced_chi2=dataset_reduced_chi2,
        dataset_sizes=dataset_sizes,
        dataset_weights=dataset_weights,
        dataset_residuals=dataset_residuals,
        dataset_model_values=dataset_model_values,
    )


def problem_supports_analytic_jacobian(problem: FitProblem) -> bool:
    """Return whether every dataset can supply an exact model Jacobian.

    Requires a ``model_jacobian`` on each dataset and no instrument resolution
    (a resolution operator would have to be applied to the gradient columns as
    well, which the bare model Jacobian does not do).
    """

    if any(derived.expression is not None for derived in problem.derived_parameters):
        return False
    return all(
        dataset.model_jacobian is not None and dataset.resolution is None
        for dataset in problem.datasets
    )


def _sensitivity_resolver(
    problem: FitProblem, variable_names: set[str]
) -> Callable[[str], dict[str, float]]:
    """Map any resolved-parameter name to its optimizer-variable sensitivities.

    Returns a memoized function ``name -> {variable_name: d(name)/d(variable)}``.
    Optimizer variables map to themselves; derived (constraint) parameters
    expand through their linear ``base + sign * offset`` definition; fixed and
    unknown names contribute nothing.
    """

    derived_by_name = {derived.name: derived for derived in problem.derived_parameters}
    cache: dict[str, dict[str, float]] = {}

    def resolve(name: str) -> dict[str, float]:
        cached = cache.get(name)
        if cached is not None:
            return cached
        result: dict[str, float] = {}
        if name in variable_names:
            result = {name: 1.0}
        elif name in derived_by_name:
            derived = derived_by_name[name]
            if isinstance(derived.base, str):
                for variable, coeff in resolve(derived.base).items():
                    result[variable] = result.get(variable, 0.0) + coeff
            if derived.offset is not None:
                for variable, coeff in resolve(derived.offset).items():
                    result[variable] = result.get(variable, 0.0) + derived.sign * coeff
        cache[name] = result
        return result

    return resolve


def _evaluate_problem_jacobian(
    problem: FitProblem,
    params: dict[str, float],
    names: Sequence[str],
    *,
    require_positive_sigma: bool,
) -> FloatArray:
    """Assemble the residual Jacobian ``d(residual)/d(variable)`` analytically.

    Rows follow the same dataset order and weighting as :func:`_evaluate_problem`
    (``residual = sqrt(weight) (I_obs - I_model) / sigma``); columns follow
    ``names``. Per-dataset/grouped sharing (parameter bindings) and inequality
    constraints (derived parameters) are resolved through the same chain rule
    the residual uses.
    """

    resolved = problem.resolve_parameters(params)
    resolve = _sensitivity_resolver(problem, set(names))
    name_index = {name: index for index, name in enumerate(names)}

    blocks: list[FloatArray] = []
    for dataset in problem.datasets:
        prepared = dataset.prepared_valid(require_positive_sigma=require_positive_sigma)
        block = np.zeros((prepared.size, len(names)), dtype=float)
        dataset_params = _apply_parameter_bindings(resolved, dataset.parameter_bindings)
        columns = dataset.model_jacobian(prepared, dataset_params)
        sigma = np.asarray(prepared.sigma, dtype=float)
        weight_factor = np.sqrt(dataset.weight)
        scale_magnitude = 1.0
        if dataset.data_scale_parameter:
            scale = float(resolved[dataset.data_scale_parameter])
            scale_magnitude = max(abs(scale), np.finfo(float).tiny)
        residual_factor = -weight_factor / (scale_magnitude * sigma)
        for qualified, d_model in columns.items():
            binding = dataset.parameter_bindings.get(qualified)
            if isinstance(binding, str):
                target = binding
            elif binding is not None:
                continue  # bound to a fixed scalar: contributes no column
            else:
                target = qualified
            sensitivities = resolve(target)
            if not sensitivities:
                continue
            residual_column = residual_factor * np.asarray(d_model, dtype=float)
            for variable, coeff in sensitivities.items():
                block[:, name_index[variable]] += coeff * residual_column

        if dataset.data_scale_parameter:
            scale_sensitivities = resolve(dataset.data_scale_parameter)
            if scale_sensitivities:
                model_values = _evaluate_dataset_model(
                    problem, dataset, prepared, dataset_params
                )
                scale_direction = np.copysign(1.0, scale)
                scale_column = (
                    weight_factor
                    * scale_direction
                    * model_values
                    / (scale_magnitude * scale_magnitude * sigma)
                )
                for variable, coeff in scale_sensitivities.items():
                    block[:, name_index[variable]] += coeff * scale_column
        blocks.append(block)

    if not blocks:
        return np.zeros((0, len(names)), dtype=float)
    return np.concatenate(blocks, axis=0)


def _evaluate_dataset_model(
    problem: FitProblem,
    dataset: FitDataset,
    prepared: PointData4D,
    params: dict[str, float],
) -> FloatArray:
    resolution = dataset.resolution
    model = dataset.model if dataset.model is not None else problem.model

    def predict(data: PointData4D, trial_params: dict[str, float]) -> FloatArray:
        if isinstance(model, ModelSpec):
            return model(data, trial_params)
        if model is not None and model is not problem.model:
            return np.asarray(model(data, trial_params), dtype=float)
        return problem.predict(data, trial_params)

    if hasattr(resolution, "evaluate_model"):
        values = resolution.evaluate_model(prepared, predict, params)  # type: ignore[union-attr]
        return np.asarray(values, dtype=float)

    raw_model = np.asarray(predict(prepared, params), dtype=float)
    if raw_model.shape != prepared.intensity.shape:
        raise ValueError(
            f"model returned shape {raw_model.shape} for dataset {dataset.name!r}; "
            f"expected {prepared.intensity.shape}"
        )
    return dataset.apply_resolution(prepared, raw_model, params)


def _apply_parameter_bindings(
    params: dict[str, float],
    bindings: dict[str, ParameterBinding],
) -> dict[str, float]:
    if not bindings:
        return params
    resolved = dict(params)
    for local_name, binding in bindings.items():
        if isinstance(binding, str):
            if binding not in params:
                raise KeyError(
                    f"parameter binding for {local_name!r} references unknown parameter {binding!r}"
                )
            resolved[local_name] = float(params[binding])
        else:
            resolved[local_name] = float(binding)
    return resolved


def _copy_point_data(data: PointData4D, *, mask: ArrayLike) -> PointData4D:
    if isinstance(data.temperature, np.ndarray):
        temperature: float | FloatArray | None = data.temperature.copy()
    else:
        temperature = data.temperature
    return PointData4D(
        data.H.copy(),
        data.K.copy(),
        data.L.copy(),
        data.E.copy(),
        data.intensity.copy(),
        data.sigma.copy(),
        mask=np.asarray(mask, dtype=bool).copy(),
        temperature=temperature,
        magnetic_field=None if data.magnetic_field is None else np.array(data.magnetic_field),
        metadata=dict(data.metadata),
    )


def _in_range(values: ArrayLike, bounds: Range) -> NDArray[np.bool_]:
    lower, upper = bounds
    arr = np.asarray(values, dtype=float)
    mask = np.ones(arr.shape, dtype=bool)
    if lower is not None:
        mask &= arr >= float(lower)
    if upper is not None:
        mask &= arr <= float(upper)
    return mask


def _project_dimensions(data: PointData4D, dimensions: Sequence[ProjectionSpec]) -> FloatArray:
    if len(dimensions) == 0:
        raise ValueError("at least one dimension must be supplied")
    columns = [_project_one_dimension(data, dimension) for dimension in dimensions]
    return np.column_stack(columns)


def _project_one_dimension(data: PointData4D, dimension: ProjectionSpec) -> FloatArray:
    if isinstance(dimension, str):
        key = dimension.upper()
        if key not in {"H", "K", "L", "E"}:
            raise ValueError("dimension strings must be one of 'H', 'K', 'L', or 'E'")
        return np.asarray(getattr(data, key), dtype=float)

    weights = np.asarray(dimension, dtype=float)
    if weights.shape == (3,):
        coords = np.column_stack([data.H, data.K, data.L])
    elif weights.shape == (4,):
        coords = np.column_stack([data.H, data.K, data.L, data.E])
    else:
        raise ValueError("projection dimensions must have length 3 or 4")
    return coords @ weights


def _as_length(name: str, values: Sequence[float], expected: int) -> FloatArray:
    arr = np.asarray(values, dtype=float)
    if arr.shape != (expected,):
        raise ValueError(f"{name} must contain {expected} values")
    return arr


def _metadata_coordinate_units_are_inv_angstrom(metadata: dict[str, Any]) -> bool:
    candidates = [
        metadata.get("coordinate_units"),
        metadata.get("momentum_units"),
        metadata.get("q_units"),
    ]
    normalized = set()
    for value in candidates:
        if value is None:
            continue
        text = str(value).strip().lower()
        text = text.replace("å", "angstrom")
        text = text.replace(" ", "").replace("_", "").replace("-", "")
        normalized.add(text)
    return bool(
        normalized
        & {
            "a^1",
            "1/a",
            "angstrom^1",
            "1/angstrom",
            "1/ang",
            "inversea",
            "inverseas",
            "inverseangstrom",
            "inverseangstroms",
        }
    )


def _resolve_q_transform(data: PointData4D, matrix: ArrayLike | None = None) -> FloatArray:
    if matrix is not None:
        return _as_3x3_matrix(matrix, name="q_matrix")
    for key in ("rlu_to_inv_angstrom_matrix", "ub_matrix", "orientation_matrix"):
        if key in data.metadata:
            return _as_3x3_matrix(data.metadata[key], name=key)
    oriented_lattice = data.metadata.get("oriented_lattice")
    if isinstance(oriented_lattice, dict):
        for key in ("rlu_to_inv_angstrom_matrix", "ub_matrix", "orientation_matrix"):
            if key in oriented_lattice:
                return _as_3x3_matrix(oriented_lattice[key], name=f"oriented_lattice.{key}")
    raise ValueError(
        "|Q| masks require inverse-angstrom coordinates or an attached lattice/UB matrix"
    )


def _as_3x3_matrix(value: ArrayLike, *, name: str) -> FloatArray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"{name} must be a 3x3 matrix")
    return matrix


def _covariance_from_jacobian(jacobian: FloatArray, *, chi2: float, dof: int) -> FloatArray | None:
    if dof <= 0 or jacobian.size == 0:
        return None
    try:
        _, singular_values, vt = np.linalg.svd(jacobian, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    eps = np.finfo(float).eps
    threshold = eps * max(jacobian.shape) * singular_values[0]
    keep = singular_values > threshold
    if not np.any(keep):
        return None
    vt = vt[keep]
    singular_values = singular_values[keep]
    cov = (vt.T / (singular_values * singular_values)) @ vt
    return cov * (chi2 / dof)


def _least_squares_kwargs(kwargs: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(kwargs, dict):
        return {}
    ignored = {"initialization", "sampler"}
    return {name: value for name, value in kwargs.items() if name not in ignored}


def _initial_walker_positions(
    center: FloatArray,
    bounds: tuple[FloatArray, FloatArray],
    n_walkers: int,
    rng: np.random.Generator,
) -> FloatArray:
    lower, upper = bounds
    walkers = np.empty((n_walkers, center.size), dtype=float)
    for index, value in enumerate(center):
        lo = lower[index]
        hi = upper[index]
        if np.isfinite(lo) and np.isfinite(hi):
            scale = max((hi - lo) * 1.0e-4, 1.0e-8)
        else:
            scale = max(abs(value) * 1.0e-4, 1.0e-8)
        walkers[:, index] = value + rng.normal(0.0, scale, size=n_walkers)
    return _nudge_inside_bounds(np.clip(walkers, lower, upper), bounds)


@dataclass
class _LeastSquaresResult:
    x: FloatArray
    jac: FloatArray
    success: bool
    message: str
    cost: float


def _run_least_squares(
    residual_fn: Callable[[FloatArray], FloatArray],
    *,
    x0: FloatArray,
    bounds: tuple[FloatArray, FloatArray],
    kwargs: dict[str, Any] | None = None,
    progress_callback: ProgressCallback | None = None,
    names: Sequence[str] = (),
    fixed: dict[str, float] | None = None,
    jac: Callable[[FloatArray], FloatArray] | None = None,
) -> _LeastSquaresResult:
    optimizer_kwargs = {} if kwargs is None else dict(kwargs)
    x0 = _nudge_inside_bounds(x0, bounds)
    wrapped_residual_fn = residual_fn
    if progress_callback is not None:
        evaluation = 0
        fit_start = time.perf_counter()

        def wrapped_residual_fn(x: FloatArray) -> FloatArray:
            nonlocal evaluation
            evaluation += 1
            residual = residual_fn(x)
            if evaluation == 1 or evaluation % 10 == 0:
                params = unpack_parameters(x, names, {} if fixed is None else fixed)
                elapsed = time.perf_counter() - fit_start
                progress_callback(
                    {
                        "stage": "least_squares",
                        "iteration": evaluation,
                        "elapsed_seconds": elapsed,
                        "seconds_per_step": elapsed / max(evaluation, 1),
                        "parameters": {name: float(params[name]) for name in names},
                        "cost": 0.5 * float(np.dot(residual, residual)),
                        "message": f"least-squares residual evaluation {evaluation}",
                    }
                )
            return residual

    if _scipy_least_squares is not None:
        if jac is not None:
            optimizer_kwargs.setdefault("jac", jac)
        scipy_result = _scipy_least_squares(
            wrapped_residual_fn,
            x0=x0,
            bounds=bounds,
            **optimizer_kwargs,
        )
        return _LeastSquaresResult(
            x=np.asarray(scipy_result.x, dtype=float),
            jac=np.asarray(scipy_result.jac, dtype=float),
            success=bool(scipy_result.success),
            message=str(scipy_result.message),
            cost=float(scipy_result.cost),
        )
    supported = {"max_nfev", "xtol", "ftol"}
    unsupported = set(optimizer_kwargs) - supported
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise ValueError(f"NumPy least-squares fallback does not support optimizer kwargs: {names}")
    return _numpy_least_squares(wrapped_residual_fn, x0=x0, bounds=bounds, **optimizer_kwargs)


def _nudge_inside_bounds(
    x0: FloatArray,
    bounds: tuple[FloatArray, FloatArray],
) -> FloatArray:
    """Move initial values strictly inside finite bounds.

    Recent SciPy releases can satisfy ``ftol`` immediately when a start value
    sits exactly on a bound, returning without optimizing. A nudge of the
    finite-difference step scale keeps the optimizer productive without
    meaningfully changing the start point.
    """

    lower, upper = bounds
    if x0.size == 0:
        return x0
    out = np.asarray(x0, dtype=float).copy()
    step = np.maximum(1.0e-8, 1.0e-8 * np.abs(out))
    span = upper - lower
    finite_span = np.isfinite(span)
    step = np.where(finite_span, np.minimum(step, 0.25 * span), step)
    at_lower = np.isfinite(lower) & (out <= lower + step)
    at_upper = np.isfinite(upper) & (out >= upper - step)
    out = np.where(at_lower, lower + step, out)
    out = np.where(at_upper & ~at_lower, upper - step, out)
    return out


def _numpy_least_squares(
    residual_fn: Callable[[FloatArray], FloatArray],
    *,
    x0: FloatArray,
    bounds: tuple[FloatArray, FloatArray],
    max_nfev: int = 400,
    xtol: float = 1e-8,
    ftol: float = 1e-8,
) -> _LeastSquaresResult:
    """Small Levenberg-Marquardt fallback for environments without SciPy.

    This is intentionally modest. It is good enough for the package's synthetic
    validation tests, while SciPy remains the preferred optimizer when present.
    """

    lower, upper = bounds
    x = np.clip(np.asarray(x0, dtype=float), lower, upper)
    residual = residual_fn(x)
    cost = 0.5 * float(np.dot(residual, residual))
    damping = 1e-3
    n_eval = 1

    success = False
    message = "maximum function evaluations reached in NumPy fallback"
    jac = _finite_difference_jacobian(residual_fn, x, residual, bounds)

    while n_eval < max_nfev:
        jac = _finite_difference_jacobian(residual_fn, x, residual, bounds)
        gradient = jac.T @ residual
        hessian = jac.T @ jac
        step = _solve_damped_step(hessian, gradient, damping)
        if np.linalg.norm(step) <= xtol * (xtol + np.linalg.norm(x)):
            success = True
            message = "xtol reached in NumPy fallback"
            break

        trial = np.clip(x - step, lower, upper)
        if np.allclose(trial, x, rtol=0.0, atol=xtol):
            success = True
            message = "bounded step became smaller than xtol in NumPy fallback"
            break

        trial_residual = residual_fn(trial)
        n_eval += 1
        trial_cost = 0.5 * float(np.dot(trial_residual, trial_residual))
        if trial_cost < cost:
            if abs(cost - trial_cost) <= ftol * max(1.0, cost):
                x = trial
                residual = trial_residual
                cost = trial_cost
                success = True
                message = "ftol reached in NumPy fallback"
                break
            x = trial
            residual = trial_residual
            cost = trial_cost
            damping = max(damping / 5.0, 1e-12)
        else:
            damping = min(damping * 10.0, 1e12)

    jac = _finite_difference_jacobian(residual_fn, x, residual, bounds)
    return _LeastSquaresResult(x=x, jac=jac, success=success, message=message, cost=cost)


def _finite_difference_jacobian(
    residual_fn: Callable[[FloatArray], FloatArray],
    x: FloatArray,
    residual: FloatArray,
    bounds: tuple[FloatArray, FloatArray],
) -> FloatArray:
    lower, upper = bounds
    jac = np.empty((residual.size, x.size), dtype=float)
    for i, value in enumerate(x):
        step = np.sqrt(np.finfo(float).eps) * max(1.0, abs(value))
        plus = x.copy()
        minus = x.copy()
        plus[i] = min(upper[i], value + step)
        minus[i] = max(lower[i], value - step)
        if plus[i] == minus[i]:
            jac[:, i] = 0.0
        elif plus[i] == value:
            jac[:, i] = (residual - residual_fn(minus)) / (value - minus[i])
        elif minus[i] == value:
            jac[:, i] = (residual_fn(plus) - residual) / (plus[i] - value)
        else:
            jac[:, i] = (residual_fn(plus) - residual_fn(minus)) / (plus[i] - minus[i])
    return jac


def _solve_damped_step(hessian: FloatArray, gradient: FloatArray, damping: float) -> FloatArray:
    matrix = hessian + damping * np.eye(hessian.shape[0])
    try:
        return np.linalg.solve(matrix, gradient)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(matrix) @ gradient
