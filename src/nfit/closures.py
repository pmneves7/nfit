"""Self-consistency closures for the Heisenberg RPA model.

Implements the sum-rule closures described in ``docs/theory_notes.md``, each
solving one scalar equation per (temperature, field) so the local propagator
becomes self-consistent instead of freely fitted:

- **Onsager reaction field** (spherical model): ``J(Q) -> J(Q) - lambda(T)``
  with ``lambda`` solved so the per-site fluctuation amplitude ``<m^2>``
  equals a target (fixed by the user or exposed as the fit parameter
  ``m2_total``).
- **Moriya SCR**: ``chi0_eff^-1(T) = chi0^-1 + u <m^2>(T)`` solved
  self-consistently; the mode-coupling ``u`` is the fit parameter
  ``mode_coupling_u`` and the model's ``chi0`` is reinterpreted as the bare
  reference susceptibility.
- **Takahashi TAC**: ``chi0_eff(T)`` solved so zero-point plus thermal
  amplitude equals ``total_amplitude`` (fixed or fitted); the model's
  ``chi0`` only seeds the root search.

The moment integrals come from :mod:`nfit.sum_rules`. Two "moment model"
backends supply ``<m^2>`` as a function of the closure variable: Tier A
(scalar and tensor eigendecomposition paths — closed-form energy integrals on
the BZ-grid eigenvalues of ``J(Q)``, which shift rigidly under the Onsager
``lambda``) and Tier B (applied field: the gyrotropic local propagator breaks
the eigenbasis, so the site-traced ``chi''`` is solved numerically on a
(BZ x omega) grid). All solvers raise ``ValueError`` when the closure
equation has no solution in the stable domain (e.g. a moment target below
the zero-point floor), which the fit evaluator converts to its standard
unphysical-parameter sentinel.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq, newton

from .sum_rules import (
    mode_amplitude_per_site,
    tier_b_omega_grid,
    trace_moment_quadrature,
)

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

CLOSURE_MODES = ("none", "onsager", "scr", "tac")

# brentq tolerances: much tighter than optimizer finite-difference steps so
# solver noise never contaminates FD gradients.
_ROOT_XTOL = 1e-12
_ROOT_RTOL = 1e-12
_MAX_EXPANSIONS = 200


@dataclass(frozen=True)
class ClosureSpec:
    """Validated closure configuration (``model.config["closure"]``)."""

    mode: str
    energy_cutoff_mev: float = 100.0
    bz_grid: int = 16
    omega_points: int = 200
    moment_mode: str = "fixed"
    moment_target: float = 1.0

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> ClosureSpec | None:
        """Parse a config mapping; ``None`` for an absent or "none" closure."""

        if not config:
            return None
        block = config.get("closure")
        if not block:
            return None
        mode = str(block.get("mode", "none")).lower()
        if mode == "none":
            return None
        if mode not in CLOSURE_MODES:
            raise ValueError(
                f"unknown closure mode {mode!r}; expected one of {CLOSURE_MODES}"
            )
        moment_mode = str(block.get("moment_mode", "fixed")).lower()
        if moment_mode not in ("fixed", "fitted"):
            raise ValueError(
                f"closure moment_mode must be 'fixed' or 'fitted', got {moment_mode!r}"
            )
        cutoff = float(block.get("energy_cutoff_mev", 100.0))
        if cutoff <= 0:
            raise ValueError("closure energy_cutoff_mev must be positive")
        bz = int(block.get("bz_grid", 16))
        if bz < 1:
            raise ValueError("closure bz_grid must be >= 1")
        return cls(
            mode=mode,
            energy_cutoff_mev=cutoff,
            bz_grid=bz,
            omega_points=max(int(block.get("omega_points", 200)), 16),
            moment_mode=moment_mode,
            moment_target=float(block.get("moment_target", 1.0)),
        )

    def parameter_names(self) -> tuple[str, ...]:
        """Dynamic fit-parameter names this closure exposes."""

        if self.mode == "onsager":
            return ("m2_total",) if self.moment_mode == "fitted" else ()
        if self.mode == "scr":
            return ("mode_coupling_u",)
        if self.mode == "tac":
            return ("total_amplitude",) if self.moment_mode == "fitted" else ()
        return ()


@dataclass(frozen=True)
class ClosureResult:
    """Solved closure state at one (temperature, field) point."""

    lambda_shift: float
    chi0_eff: float
    m2_total: float
    m2_zero_point: float
    m2_thermal: float


class TierAMoments:
    """Moment model from the BZ-grid eigenvalues of ``J(Q)`` (field off).

    ``eigenvalues`` is ``(n_q, n_modes)``: the scalar path passes the ``N``
    sublattice eigenvalues (each carrying 3 spin components), the tensor path
    the ``3N`` Cartesian ones (1 component each). The Onsager shift acts as a
    rigid eigenvalue shift, so one eigendecomposition serves every solver
    probe.
    """

    def __init__(
        self,
        eigenvalues: FloatArray,
        n_sites: int,
        *,
        isotropic_components: int = 3,
    ) -> None:
        self.eigenvalues = np.asarray(eigenvalues, dtype=float)
        self.n_sites = int(n_sites)
        self.isotropic_components = int(isotropic_components)
        self._lambda_max = float(np.max(self.eigenvalues))

    def lambda_floor(self, chi0: float) -> float:
        """Stability edge: ``lambda_shift`` must exceed ``max lam - 1/chi0``."""

        return self._lambda_max - 1.0 / float(chi0)

    def chi0_ceiling(self, lambda_shift: float) -> float:
        """Largest stable ``chi0`` at a given shift (``inf`` when unbounded)."""

        top = self._lambda_max - float(lambda_shift)
        return np.inf if top <= 0.0 else 1.0 / top

    def moment(
        self,
        *,
        chi0: float,
        gamma0: float,
        temperature_K: float,
        cutoff_mev: float,
        lambda_shift: float = 0.0,
    ) -> tuple[float, float]:
        """Per-site ``(zero_point, thermal)`` amplitude; raises when unstable."""

        return mode_amplitude_per_site(
            self.eigenvalues,
            chi0=chi0,
            gamma0=gamma0,
            temperature_K=temperature_K,
            cutoff_mev=cutoff_mev,
            n_sites=self.n_sites,
            lambda_shift=lambda_shift,
            isotropic_components=self.isotropic_components,
            parts=True,
        )


class TierBMoments:
    """Moment model with an applied field (gyrotropic local propagator).

    ``exchange`` is the assembled ``(n_q, 3N, 3N)`` interaction matrix on the
    BZ grid; ``propagator_builder(omega, chi0, gamma0)`` returns the per-site
    ``X0(omega)`` tensor ``(n_omega, 3, 3)`` (the Zeeman Larmor propagator).
    The site-traced dissipative response is solved by batched LU per energy
    on the grid and integrated numerically -- there is no eigenbasis shortcut,
    so every solver probe pays a full (BZ x omega) sweep; see
    docs/performance.md.
    """

    def __init__(
        self,
        exchange: ComplexArray,
        n_sites: int,
        propagator_builder: Callable[[FloatArray, float, float], ComplexArray],
        *,
        omega_points: int = 200,
        static_response_bound: float = 1.0,
    ) -> None:
        self.exchange = np.ascontiguousarray(exchange)
        self.n_sites = int(n_sites)
        self.propagator_builder = propagator_builder
        self.omega_points = int(omega_points)
        # Hermitian-part eigenvalues of J bound the stability edge estimate.
        self._lambda_max = float(np.max(np.linalg.eigvalsh(self.exchange)))
        # X0(0) can exceed chi0 (e.g. chi_perp_ratio > 1); the bound rescales
        # the Tier-A edge estimate used to seed the bracketing search.
        self._static_bound = float(static_response_bound)

    def lambda_floor(self, chi0: float) -> float:
        """Estimated stability edge (seed for the bracket search, not exact)."""

        return self._lambda_max - 1.0 / (float(chi0) * max(self._static_bound, 1e-12))

    def chi0_ceiling(self, lambda_shift: float) -> float:
        top = (self._lambda_max - float(lambda_shift)) * max(self._static_bound, 1e-12)
        return np.inf if top <= 0.0 else 1.0 / top

    def _site_trace_chipp(
        self, omega: FloatArray, chi0: float, gamma0: float, lambda_shift: float
    ) -> FloatArray:
        """Per-site ``(1/N) Im Tr [(1 - X0 J')^{-1} X0]`` on (omega, Q)."""

        from .tensor_rpa import check_tensor_exchange_stability

        static_local = np.asarray(
            self.propagator_builder(np.zeros(1), chi0, gamma0)[0], dtype=complex
        )
        check_tensor_exchange_stability(
            self.exchange,
            self.n_sites,
            static_local,
            lambda_shift=lambda_shift,
        )

        dim = 3 * self.n_sites
        n_q = self.exchange.shape[0]
        shifted = self.exchange - lambda_shift * np.eye(dim)[None]
        j_reshaped = shifted.reshape(n_q, self.n_sites, 3, dim)
        propagators = self.propagator_builder(omega, chi0, gamma0)
        identity = np.eye(dim)
        trace = np.empty((omega.shape[0], n_q), dtype=float)
        for index in range(omega.shape[0]):
            x0 = propagators[index]  # (3, 3)
            x0_j = np.einsum("ag,qngB->qnaB", x0, j_reshaped).reshape(n_q, dim, dim)
            a_matrix = identity[None] - x0_j
            rhs = np.kron(np.eye(self.n_sites), x0)  # block-diagonal X0
            try:
                solution = np.linalg.solve(a_matrix, rhs[None])
            except np.linalg.LinAlgError as exc:
                raise ValueError(
                    "Tier-B closure: singular RPA denominator on the BZ grid "
                    "(parameters at or beyond the instability)"
                ) from exc
            trace[index] = (
                np.trace(solution, axis1=-2, axis2=-1).imag / self.n_sites
            )
        if not np.all(np.isfinite(trace)) or np.max(np.abs(trace)) > 1e12:
            raise ValueError(
                "Tier-B closure: diverging spectral weight on the BZ grid "
                "(parameters at or beyond the instability)"
            )
        return trace

    def moment(
        self,
        *,
        chi0: float,
        gamma0: float,
        temperature_K: float,
        cutoff_mev: float,
        lambda_shift: float = 0.0,
    ) -> tuple[float, float]:
        if chi0 <= 0 or gamma0 <= 0:
            raise ValueError("chi0 and gamma0 must be positive")
        omega = tier_b_omega_grid(cutoff_mev, self.omega_points)
        trace = self._site_trace_chipp(omega, chi0, gamma0, lambda_shift)
        if np.min(trace) < -1e-9 * max(np.max(np.abs(trace)), 1.0):
            raise ValueError(
                "Tier-B closure: negative spectral weight (parameters beyond "
                "the instability)"
            )
        zero_point = float(
            np.mean(np.trapezoid(trace, omega, axis=0)) / np.pi
        )
        total = float(np.mean(trace_moment_quadrature(trace, omega, temperature_K)))
        return zero_point, total - zero_point


def _stable_probe(
    moment_at: Callable[[float], tuple[float, float]],
    start: float,
    step: float,
    direction: float,
) -> tuple[float, tuple[float, float]]:
    """Walk from ``start`` along ``direction`` until the model is stable."""

    x = start
    for _ in range(_MAX_EXPANSIONS):
        try:
            return x, moment_at(x)
        except ValueError:
            x += direction * step
            step *= 1.5
    raise ValueError("closure solver found no stable parameter region")


def _bisect_to_boundary(
    moment_at: Callable[[float], tuple[float, float]],
    good: float,
    bad: float,
    target: float,
    *,
    increasing: bool,
) -> tuple[float, float]:
    """Bracket ``total(x) = target`` between ``good`` (stable) and ``bad``.

    Approaching the instability the amplitude diverges, so if the target is
    reachable at all a bracketing stable point exists between ``good`` and
    the (unstable) ``bad``. Returns ``(x, total(x))`` on the target's far
    side; raises when the interval collapses without reaching it.
    """

    total_good = sum(moment_at(good))
    for _ in range(_MAX_EXPANSIONS):
        mid = 0.5 * (good + bad)
        if abs(bad - good) <= 1e-12 * max(abs(good), abs(bad), 1.0):
            raise ValueError(
                "closure target unreachable: the sum rule cannot be satisfied "
                f"before the instability (best amplitude {total_good:.6g}, "
                f"target {target:.6g})"
            )
        try:
            zero_point, thermal = moment_at(mid)
        except ValueError:
            bad = mid
            continue
        total = zero_point + thermal
        good, total_good = mid, total
        crossed = total >= target if increasing else total <= target
        if crossed:
            return mid, total
    raise ValueError("closure solver failed to bracket the sum-rule target")


def _restore_temperature_order(
    order: NDArray[np.intp], sorted_results: list[ClosureResult]
) -> list[ClosureResult]:
    """Return continuation results in the caller's original order."""

    results: list[ClosureResult | None] = [None] * len(sorted_results)
    for original_index, result in zip(order, sorted_results, strict=True):
        results[int(original_index)] = result
    return [result for result in results if result is not None]


def solve_onsager(
    model: TierAMoments | TierBMoments,
    *,
    chi0: float,
    gamma0: float,
    temperature_K: float,
    cutoff_mev: float,
    target: float,
) -> ClosureResult:
    """Solve the Onsager shift ``lambda`` so the total amplitude hits ``target``.

    The amplitude decreases monotonically with ``lambda`` (modes move away
    from the instability), diverging at the stability edge and vanishing as
    ``lambda -> inf``, so the root is unique when reachable.
    """

    if target <= 0:
        raise ValueError("closure moment target must be positive")
    if chi0 <= 0 or gamma0 <= 0:
        raise ValueError("chi0 and gamma0 must be positive")

    def moment_at(lam: float) -> tuple[float, float]:
        return model.moment(
            chi0=chi0,
            gamma0=gamma0,
            temperature_K=temperature_K,
            cutoff_mev=cutoff_mev,
            lambda_shift=lam,
        )

    scale = max(abs(model.lambda_floor(chi0)), 1.0 / chi0, 1e-3)
    floor = model.lambda_floor(chi0)
    lam_lo, parts_lo = _stable_probe(moment_at, floor + 1e-9 * scale, 1e-6 * scale, +1.0)
    total_lo = sum(parts_lo)

    if total_lo < target:
        # Need a shift closer to the instability than the first stable probe;
        # the amplitude diverges there, so bracket against the unstable side.
        lam_lo, total_lo = _bisect_to_boundary(
            moment_at, lam_lo, floor, target, increasing=True
        )
        if total_lo == target:
            zero_point, thermal = moment_at(lam_lo)
            return ClosureResult(lam_lo, chi0, target, zero_point, thermal)

    # Expand upward until the amplitude falls below the target.
    lam_hi = lam_lo + scale
    total_hi = None
    for _ in range(_MAX_EXPANSIONS):
        total_hi = sum(moment_at(lam_hi))
        if total_hi < target:
            break
        lam_lo = lam_hi
        lam_hi = lam_hi + (lam_hi - floor)
    else:
        raise ValueError(
            "closure target unreachable: amplitude stays above the target for "
            "all stable Onsager shifts"
        )

    lam = brentq(
        lambda x: sum(moment_at(x)) - target,
        lam_lo,
        lam_hi,
        xtol=_ROOT_XTOL * scale,
        rtol=_ROOT_RTOL,
    )
    zero_point, thermal = moment_at(lam)
    return ClosureResult(float(lam), float(chi0), zero_point + thermal, zero_point, thermal)


def solve_onsager_temperatures(
    model: TierAMoments,
    *,
    chi0: float,
    gamma0: float,
    temperature_K: FloatArray,
    cutoff_mev: float,
    target: float,
) -> list[ClosureResult]:
    """Solve a Tier-A Onsager sweep by continuation in temperature."""

    temperatures = np.atleast_1d(np.asarray(temperature_K, dtype=float))
    if temperatures.size == 0:
        return []
    if target <= 0 or chi0 <= 0 or gamma0 <= 0:
        raise ValueError("Onsager closure requires positive target, chi0, and gamma0")
    order = np.argsort(temperatures)
    sorted_temperatures = temperatures[order]
    floor = model.lambda_floor(chi0)
    scale = max(abs(floor), 1.0 / chi0, 1e-3)
    sorted_results: list[ClosureResult] = []
    previous_roots: list[float] = []
    for index, temperature_value in enumerate(sorted_temperatures):
        temperature = float(temperature_value)
        if index == 0:
            result = solve_onsager(
                model,
                chi0=chi0,
                gamma0=gamma0,
                temperature_K=temperature,
                cutoff_mev=cutoff_mev,
                target=target,
            )
        else:
            predicted = previous_roots[-1]
            if index >= 2:
                delta_t = sorted_temperatures[index - 1] - sorted_temperatures[index - 2]
                if delta_t != 0.0:
                    predicted += (
                        (previous_roots[-1] - previous_roots[-2])
                        * (temperature - sorted_temperatures[index - 1])
                        / delta_t
                    )
            predicted = max(float(predicted), floor + 1e-10 * scale)

            def residual(lambda_shift: float, temperature_K: float) -> float:
                return (
                    sum(
                        model.moment(
                            chi0=chi0,
                            gamma0=gamma0,
                            temperature_K=temperature_K,
                            cutoff_mev=cutoff_mev,
                            lambda_shift=lambda_shift,
                        )
                    )
                    - target
                )

            try:
                root = float(
                    newton(
                        residual,
                        predicted,
                        x1=predicted + max(abs(predicted) * 1e-4, 1e-8 * scale),
                        args=(temperature,),
                        tol=_ROOT_XTOL * scale,
                        maxiter=12,
                    )
                )
                zero_point, thermal = model.moment(
                    chi0=chi0,
                    gamma0=gamma0,
                    temperature_K=temperature,
                    cutoff_mev=cutoff_mev,
                    lambda_shift=root,
                )
                if (
                    root <= floor
                    or abs(zero_point + thermal - target)
                    > 1e-9 * max(target, 1.0)
                ):
                    raise ValueError("continuation residual exceeds tolerance")
                result = ClosureResult(
                    root, chi0, zero_point + thermal, zero_point, thermal
                )
            except (RuntimeError, ValueError, OverflowError, ZeroDivisionError):
                result = solve_onsager(
                    model,
                    chi0=chi0,
                    gamma0=gamma0,
                    temperature_K=temperature,
                    cutoff_mev=cutoff_mev,
                    target=target,
                )
        sorted_results.append(result)
        previous_roots.append(result.lambda_shift)
    return _restore_temperature_order(order, sorted_results)


def _solve_chi0_equation(
    model: TierAMoments | TierBMoments,
    equation: Callable[[float, float, float], float],
    *,
    gamma0: float,
    temperature_K: float,
    cutoff_mev: float,
    guess: float,
    description: str,
) -> ClosureResult:
    """Root-find ``equation(chi0, zero_point, thermal) = 0`` over stable chi0.

    ``equation`` must be strictly decreasing in ``chi0`` (both SCR and TAC
    closures are). The stable domain is ``0 < chi0 < ceiling`` where the
    amplitude diverges, sending both closure equations to ``-inf``.
    """

    def moment_at(chi0: float) -> tuple[float, float]:
        return model.moment(
            chi0=chi0,
            gamma0=gamma0,
            temperature_K=temperature_K,
            cutoff_mev=cutoff_mev,
            lambda_shift=0.0,
        )

    def value(chi0: float) -> float:
        zero_point, thermal = moment_at(chi0)
        return equation(chi0, zero_point, thermal)

    x_lo = max(guess * 1e-8, 1e-300)
    v_lo = value(x_lo)
    if v_lo < 0.0:
        raise ValueError(
            f"{description}: the closure equation has no stable solution "
            "(negative even for vanishing chi0_eff)"
        )
    # Expand upward until the equation turns negative, backing off when the
    # probe crosses the instability.
    x_hi = max(guess, x_lo * 10.0)
    v_hi = None
    for _ in range(_MAX_EXPANSIONS):
        try:
            v_hi = value(x_hi)
        except ValueError:
            # Past the ceiling: the equation is -inf there; bisect back to a
            # stable point, which must be negative-valued or we keep going.
            lo_probe, hi_probe = x_lo, x_hi
            for _ in range(_MAX_EXPANSIONS):
                mid = 0.5 * (lo_probe + hi_probe)
                try:
                    v_mid = value(mid)
                except ValueError:
                    hi_probe = mid
                    continue
                if v_mid < 0.0:
                    x_hi, v_hi = mid, v_mid
                    break
                lo_probe = mid
                x_lo, v_lo = mid, v_mid
            if v_hi is not None and v_hi < 0.0:
                break
            raise ValueError(
                f"{description}: no sign change before the instability"
            ) from None
        if v_hi < 0.0:
            break
        x_lo, v_lo = x_hi, v_hi
        x_hi *= 4.0
    else:
        raise ValueError(f"{description}: the closure equation never crosses zero")

    chi0_eff = brentq(value, x_lo, x_hi, xtol=_ROOT_XTOL * guess, rtol=_ROOT_RTOL)
    zero_point, thermal = moment_at(chi0_eff)
    return ClosureResult(
        0.0, float(chi0_eff), zero_point + thermal, zero_point, thermal
    )


def solve_scr(
    model: TierAMoments | TierBMoments,
    *,
    chi0_bare: float,
    gamma0: float,
    temperature_K: float,
    cutoff_mev: float,
    mode_coupling_u: float,
) -> ClosureResult:
    """Moriya SCR: ``1/chi0_eff = 1/chi0_bare + u * <m^2>(chi0_eff)``.

    ``u = 0`` reduces exactly to the bare RPA (``chi0_eff = chi0_bare``);
    ``u > 0`` suppresses the response as thermal fluctuations grow, which is
    the mechanism behind the Curie-Weiss law of itinerant magnets.
    """

    if chi0_bare <= 0 or gamma0 <= 0:
        raise ValueError("chi0 and gamma0 must be positive")
    if mode_coupling_u < 0:
        raise ValueError("mode_coupling_u must be non-negative")

    def equation(chi0: float, zero_point: float, thermal: float) -> float:
        return 1.0 / chi0 - 1.0 / chi0_bare - mode_coupling_u * (zero_point + thermal)

    return _solve_chi0_equation(
        model,
        equation,
        gamma0=gamma0,
        temperature_K=temperature_K,
        cutoff_mev=cutoff_mev,
        guess=chi0_bare,
        description="SCR closure",
    )


def solve_scr_temperatures(
    model: TierAMoments,
    *,
    chi0_bare: float,
    gamma0: float,
    temperature_K: FloatArray,
    cutoff_mev: float,
    mode_coupling_u: float,
) -> list[ClosureResult]:
    """Solve a Tier-A SCR sweep by continuation in temperature."""

    temperatures = np.atleast_1d(np.asarray(temperature_K, dtype=float))
    if temperatures.size == 0:
        return []
    if chi0_bare <= 0 or gamma0 <= 0 or mode_coupling_u < 0:
        raise ValueError("SCR closure requires positive chi0/gamma0 and non-negative u")
    order = np.argsort(temperatures)
    sorted_temperatures = temperatures[order]
    sorted_results: list[ClosureResult] = []
    previous_roots: list[float] = []
    for index, temperature_value in enumerate(sorted_temperatures):
        temperature = float(temperature_value)
        if mode_coupling_u == 0.0:
            zero_point, thermal = model.moment(
                chi0=chi0_bare,
                gamma0=gamma0,
                temperature_K=temperature,
                cutoff_mev=cutoff_mev,
            )
            result = ClosureResult(
                0.0,
                chi0_bare,
                zero_point + thermal,
                zero_point,
                thermal,
            )
        elif index == 0:
            result = solve_scr(
                model,
                chi0_bare=chi0_bare,
                gamma0=gamma0,
                temperature_K=temperature,
                cutoff_mev=cutoff_mev,
                mode_coupling_u=mode_coupling_u,
            )
        else:
            predicted = previous_roots[-1]
            if index >= 2:
                delta_t = sorted_temperatures[index - 1] - sorted_temperatures[index - 2]
                if delta_t != 0.0:
                    predicted += (
                        (previous_roots[-1] - previous_roots[-2])
                        * (temperature - sorted_temperatures[index - 1])
                        / delta_t
                    )
            predicted = max(float(predicted), 1e-14)

            def residual(chi0_eff: float, temperature_K: float) -> float:
                zero_point, thermal = model.moment(
                    chi0=chi0_eff,
                    gamma0=gamma0,
                    temperature_K=temperature_K,
                    cutoff_mev=cutoff_mev,
                )
                return (
                    1.0 / chi0_eff
                    - 1.0 / chi0_bare
                    - mode_coupling_u * (zero_point + thermal)
                )

            try:
                root = float(
                    newton(
                        residual,
                        predicted,
                        x1=max(predicted * (1.0 - 1e-4), 1e-14),
                        args=(temperature,),
                        tol=_ROOT_XTOL * max(predicted, 1.0),
                        maxiter=12,
                    )
                )
                zero_point, thermal = model.moment(
                    chi0=root,
                    gamma0=gamma0,
                    temperature_K=temperature,
                    cutoff_mev=cutoff_mev,
                )
                equation = (
                    1.0 / root
                    - 1.0 / chi0_bare
                    - mode_coupling_u * (zero_point + thermal)
                )
                if root <= 0.0 or abs(equation) > 1e-9:
                    raise ValueError("continuation residual exceeds tolerance")
                result = ClosureResult(
                    0.0, root, zero_point + thermal, zero_point, thermal
                )
            except (RuntimeError, ValueError, OverflowError, ZeroDivisionError):
                result = solve_scr(
                    model,
                    chi0_bare=chi0_bare,
                    gamma0=gamma0,
                    temperature_K=temperature,
                    cutoff_mev=cutoff_mev,
                    mode_coupling_u=mode_coupling_u,
                )
        sorted_results.append(result)
        previous_roots.append(result.chi0_eff)
    return _restore_temperature_order(order, sorted_results)


def solve_tac(
    model: TierAMoments | TierBMoments,
    *,
    gamma0: float,
    temperature_K: float,
    cutoff_mev: float,
    total_amplitude: float,
    guess: float,
) -> ClosureResult:
    """Takahashi TAC: solve ``chi0_eff`` so ZP + thermal amplitude is conserved.

    The total amplitude increases monotonically with ``chi0_eff`` from zero,
    so the budget pins a unique ``chi0_eff(T)``; raising temperature moves
    weight from zero-point to thermal at fixed total, suppressing
    ``chi0_eff``. The model's fitted ``chi0`` only seeds the root search.
    """

    if total_amplitude <= 0:
        raise ValueError("closure total_amplitude must be positive")
    if gamma0 <= 0:
        raise ValueError("gamma0 must be positive")

    def equation(chi0: float, zero_point: float, thermal: float) -> float:
        return total_amplitude - (zero_point + thermal)

    return _solve_chi0_equation(
        model,
        equation,
        gamma0=gamma0,
        temperature_K=temperature_K,
        cutoff_mev=cutoff_mev,
        guess=max(guess, 1e-12),
        description="TAC closure",
    )


def solve_tac_temperatures(
    model: TierAMoments,
    *,
    gamma0: float,
    temperature_K: FloatArray,
    cutoff_mev: float,
    total_amplitude: float,
    guess: float,
) -> list[ClosureResult]:
    """Solve a Tier-A TAC sweep by continuation in sorted temperature.

    The exact scalar equation is retained, but each root starts from a linear
    extrapolation of the preceding solutions. Dense susceptibility curves then
    need only a few moment evaluations per temperature instead of rebuilding a
    wide bracket for every point. Any failed or inaccurate secant step falls
    back to :func:`solve_tac`.
    """

    temperatures = np.atleast_1d(np.asarray(temperature_K, dtype=float))
    if temperatures.size == 0:
        return []
    if total_amplitude <= 0:
        raise ValueError("closure total_amplitude must be positive")
    if gamma0 <= 0:
        raise ValueError("gamma0 must be positive")

    order = np.argsort(temperatures)
    sorted_temperatures = temperatures[order]
    sorted_results: list[ClosureResult] = []
    previous_roots: list[float] = []
    for index, temperature in enumerate(sorted_temperatures):
        temperature = float(temperature)
        if index == 0:
            result = solve_tac(
                model,
                gamma0=gamma0,
                temperature_K=temperature,
                cutoff_mev=cutoff_mev,
                total_amplitude=total_amplitude,
                guess=guess,
            )
        else:
            predicted = previous_roots[-1]
            if index >= 2:
                delta_t = sorted_temperatures[index - 1] - sorted_temperatures[index - 2]
                if delta_t != 0.0:
                    predicted += (
                        (previous_roots[-1] - previous_roots[-2])
                        * (temperature - sorted_temperatures[index - 1])
                        / delta_t
                    )
            predicted = max(float(predicted), 1e-14)

            def residual(
                chi0_eff: float, temperature_K: float = temperature
            ) -> float:
                return (
                    sum(
                        model.moment(
                            chi0=chi0_eff,
                            gamma0=gamma0,
                            temperature_K=temperature_K,
                            cutoff_mev=cutoff_mev,
                            lambda_shift=0.0,
                        )
                    )
                    - total_amplitude
                )

            try:
                root = float(
                    newton(
                        residual,
                        predicted,
                        x1=max(predicted * (1.0 - 1e-4), 1e-14),
                        tol=_ROOT_XTOL * max(predicted, 1.0),
                        maxiter=12,
                    )
                )
                zero_point, thermal = model.moment(
                    chi0=root,
                    gamma0=gamma0,
                    temperature_K=temperature,
                    cutoff_mev=cutoff_mev,
                    lambda_shift=0.0,
                )
                if (
                    root <= 0.0
                    or abs(zero_point + thermal - total_amplitude)
                    > 1e-9 * max(total_amplitude, 1.0)
                ):
                    raise ValueError("continuation residual exceeds tolerance")
                result = ClosureResult(
                    0.0,
                    root,
                    zero_point + thermal,
                    zero_point,
                    thermal,
                )
            except (RuntimeError, ValueError, OverflowError, ZeroDivisionError):
                result = solve_tac(
                    model,
                    gamma0=gamma0,
                    temperature_K=temperature,
                    cutoff_mev=cutoff_mev,
                    total_amplitude=total_amplitude,
                    guess=previous_roots[-1],
                )
        sorted_results.append(result)
        previous_roots.append(result.chi0_eff)

    return _restore_temperature_order(order, sorted_results)


def solve_closure(
    spec: ClosureSpec,
    model: TierAMoments | TierBMoments,
    *,
    chi0: float,
    gamma0: float,
    temperature_K: float,
    params: Mapping[str, float],
) -> ClosureResult:
    """Dispatch to the configured closure, reading its extra fit parameters."""

    if spec.mode == "onsager":
        target = (
            float(params.get("m2_total", spec.moment_target))
            if spec.moment_mode == "fitted"
            else spec.moment_target
        )
        return solve_onsager(
            model,
            chi0=chi0,
            gamma0=gamma0,
            temperature_K=temperature_K,
            cutoff_mev=spec.energy_cutoff_mev,
            target=target,
        )
    if spec.mode == "scr":
        return solve_scr(
            model,
            chi0_bare=chi0,
            gamma0=gamma0,
            temperature_K=temperature_K,
            cutoff_mev=spec.energy_cutoff_mev,
            mode_coupling_u=float(params.get("mode_coupling_u", 0.0)),
        )
    if spec.mode == "tac":
        target = (
            float(params.get("total_amplitude", spec.moment_target))
            if spec.moment_mode == "fitted"
            else spec.moment_target
        )
        return solve_tac(
            model,
            gamma0=gamma0,
            temperature_K=temperature_K,
            cutoff_mev=spec.energy_cutoff_mev,
            total_amplitude=target,
            guess=chi0,
        )
    raise ValueError(f"unknown closure mode {spec.mode!r}")
