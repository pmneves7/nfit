from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .dataset import PointData4D
from .rebin import rebin_nd

try:  # pragma: no cover - exercised only when SciPy is importable.
    from scipy.optimize import least_squares as _scipy_least_squares
except Exception:  # SciPy may be absent or have a broken compiled dependency.
    _scipy_least_squares = None


FloatArray = NDArray[np.float64]
ModelFunction = Callable[[PointData4D, dict[str, float]], FloatArray]
DataTransform = Callable[[PointData4D], PointData4D]
ResolutionFunction = Callable[[PointData4D, FloatArray, dict[str, float]], FloatArray]
Range = tuple[float | None, float | None]
ProjectionSpec = str | Sequence[float]
ParameterBinding = str | float | int


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
    metadata:
        Free-form instrument, scan, normalization, and provenance notes.
    """

    name: str
    data: PointData4D
    weight: float = 1.0
    transforms: Sequence[DataTransform] = field(default_factory=tuple)
    resolution: Any = None
    parameter_bindings: dict[str, ParameterBinding] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def prepared(self) -> PointData4D:
        """Return data after applying all dataset-local transforms."""

        prepared = self.data
        for transform in self.transforms:
            prepared = transform(prepared)
            if not isinstance(prepared, PointData4D):
                raise TypeError("dataset transforms must return PointData4D")
        return prepared

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
    """Settings placeholder for uncertainty sampling.

    The current package defines the configuration and result containers so
    scripts can be structured now. A concrete MCMC backend will be wired in
    once we choose an implementation such as emcee, PyMC, NumPyro, or a custom
    sampler.
    """

    method: str = "mcmc"
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


@dataclass(frozen=True)
class FitProblem:
    """Complete simultaneous-fitting problem definition.

    A problem combines one physics model, one shared parameter set, and any
    number of weighted datasets. Dataset-specific behavior belongs in
    ``FitDataset.transforms`` and ``FitDataset.resolution``; shared physics
    belongs in ``model``.
    """

    datasets: Sequence[FitDataset]
    model: ModelSpec | ModelFunction
    parameter_specs: Sequence[ParameterSpec]
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

    def predict(self, data: PointData4D, params: dict[str, float]) -> FloatArray:
        """Evaluate the shared physics model."""

        if isinstance(self.model, ModelSpec):
            return self.model(data, params)
        return np.asarray(self.model(data, params), dtype=float)


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
    """Return ``data`` with lattice metadata for RLU-to-|Q| conversion."""

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
    center: Sequence[float],
    slope: float,
    energy_range: Range | None = None,
    q_matrix: ArrayLike | None = None,
    center_units: Literal["rlu", "inv_angstrom"] = "rlu",
    energy_offset: float = 0.0,
    radius_offset: float = 0.0,
    use_absolute_energy: bool = True,
    energy_range_uses_absolute: bool = True,
) -> PointData4D:
    """Mask a phonon-like cone centered on a reciprocal-space point.

    ``slope`` is ``dE/d|Q|`` in ``meV / inverse angstrom``. Points are rejected
    when their distance from ``center`` in reciprocal space is less than
    ``abs(E - energy_offset) / slope + radius_offset``. ``energy_range`` limits
    the energies where the cone is applied, which is useful for stopping the
    acoustic-phonon mask once optical branches or unrelated features dominate.
    """

    if slope <= 0.0:
        raise ValueError("phonon slope must be positive")
    if radius_offset < 0.0:
        raise ValueError("radius_offset must be nonnegative")

    q_vectors = q_vectors_inv_angstrom(data, matrix=q_matrix)
    center_arr = np.asarray(center, dtype=float)
    if center_arr.shape != (3,):
        raise ValueError("phonon center must contain exactly three coordinates")
    if center_units == "rlu":
        transform = _resolve_q_transform(data, q_matrix)
        center_q = transform @ center_arr
    elif center_units == "inv_angstrom":
        center_q = center_arr
    else:
        raise ValueError("center_units must be 'rlu' or 'inv_angstrom'")

    energy_delta = data.E - float(energy_offset)
    radius_energy = np.abs(energy_delta) if use_absolute_energy else energy_delta
    cone_radius = radius_energy / float(slope) + float(radius_offset)
    reject = np.linalg.norm(q_vectors - center_q, axis=1) <= cone_radius

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
    fractional: bool = False,
    normalize: bool = True,
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
    result = rebin_nd(
        source.intensity,
        coords,
        data_errs=source.sigma,
        lower=lower,
        upper=upper,
        step_size=step_size,
        num_bins=num_bins,
        fractional=fractional,
        normalize=normalize,
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

    if len(names) == 0:
        residuals = residual_fn(np.asarray([], dtype=float))
        params = dict(fixed)
        evaluation = _evaluate_problem(
            problem,
            params,
            require_positive_sigma=opt.require_positive_sigma,
        )
        chi2 = float(np.sum(residuals * residuals))
        return FitResult(
            params=params,
            success=True,
            message="no variable parameters",
            cost=0.5 * chi2,
            chi2=chi2,
            reduced_chi2=np.nan,
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

    result = _run_least_squares(residual_fn, x0=x0, bounds=bounds, kwargs=opt.kwargs)
    params = unpack_parameters(result.x, names, fixed)
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


def sample_problem_parameters(
    problem: FitProblem,
    config: SamplerConfig | None = None,
) -> SamplingResult:
    """Placeholder entry point for uncertainty sampling.

    The problem/config containers are stable enough for example scripts to show
    where MCMC or another posterior sampler will plug in. A concrete sampler is
    intentionally deferred until the project chooses a backend and likelihood
    conventions.
    """

    del problem, config
    raise NotImplementedError("parameter sampling backends are not implemented yet")


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

    for dataset in problem.datasets:
        prepared = dataset.prepared().valid(require_positive_sigma=require_positive_sigma)
        if prepared.size == 0:
            raise ValueError(f"dataset {dataset.name!r} has no valid points after preprocessing")

        dataset_params = _apply_parameter_bindings(params, dataset.parameter_bindings)
        model_values = _evaluate_dataset_model(problem, dataset, prepared, dataset_params)
        if model_values.shape != prepared.intensity.shape:
            raise ValueError(
                f"model/resolution returned shape {model_values.shape} for dataset {dataset.name!r}; "
                f"expected {prepared.intensity.shape}"
            )

        residual = (prepared.intensity - model_values) / prepared.sigma
        weighted_residual = np.sqrt(dataset.weight) * residual
        residual_blocks.append(weighted_residual)
        model_blocks.append(model_values)

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


def _evaluate_dataset_model(
    problem: FitProblem,
    dataset: FitDataset,
    prepared: PointData4D,
    params: dict[str, float],
) -> FloatArray:
    resolution = dataset.resolution

    def predict(data: PointData4D, trial_params: dict[str, float]) -> FloatArray:
        return problem.predict(data, trial_params)

    if hasattr(resolution, "evaluate_model"):
        values = resolution.evaluate_model(prepared, predict, params)  # type: ignore[union-attr]
        return np.asarray(values, dtype=float)

    raw_model = np.asarray(problem.predict(prepared, params), dtype=float)
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
) -> _LeastSquaresResult:
    optimizer_kwargs = {} if kwargs is None else dict(kwargs)
    if _scipy_least_squares is not None:
        scipy_result = _scipy_least_squares(
            residual_fn,
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
    return _numpy_least_squares(residual_fn, x0=x0, bounds=bounds, **optimizer_kwargs)


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
