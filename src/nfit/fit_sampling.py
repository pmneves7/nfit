"""GUI-independent MCMC storage, sample windows, and ensemble execution."""

from __future__ import annotations

import base64
import time
import warnings
import zlib
from collections.abc import Callable
from dataclasses import dataclass, field
from multiprocessing.pool import ThreadPool
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SamplerConfig:
    """Settings for untempered posterior sampling with optional emcee."""

    method: str = "emcee"
    n_walkers: int | None = None
    n_steps: int | None = None
    burn_in: int = 0
    thin: int = 1
    random_seed: int | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class SamplingResult:
    """Samples and raw chains, explicitly marked as posterior or exploration."""

    samples: FloatArray
    variable_names: list[str]
    log_probability: FloatArray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    chain: FloatArray | None = None
    log_probability_chain: FloatArray | None = None


class SamplingCancelled(RuntimeError):
    """A stopped ensemble run with inspectable partial samples."""

    def __init__(self, message: str, result: SamplingResult) -> None:
        super().__init__(message)
        self.result = result


def sampling_result_is_posterior(result: SamplingResult) -> bool:
    """Accept legacy untempered posteriors, excluding exploration from inference."""
    return (
        result.metadata.get("purpose", "posterior") == "posterior"
        and result.metadata.get("temperature", 1.0) == 1.0
    )


def _encode_samples(value: Any) -> dict[str, Any]:
    array = np.ascontiguousarray(value, dtype=np.float64)
    return {
        "__ndarray__": True,
        "dtype": "float64",
        "shape": list(array.shape),
        "data": base64.b64encode(zlib.compress(array.tobytes())).decode("ascii"),
    }


def _decode_samples(payload: dict[str, Any]) -> FloatArray:
    dtype = np.dtype(str(payload.get("dtype", "float32")))
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("unsupported sampling array dtype")
    return (
        np.frombuffer(zlib.decompress(base64.b64decode(payload["data"])), dtype=dtype)
        .reshape(payload["shape"])
        .astype(float)
    )


def sampling_result_to_dict(result: SamplingResult) -> dict[str, Any]:
    """Serialize scientific samples without reducing their floating precision."""
    payload = {
        "variable_names": list(result.variable_names),
        "samples": _encode_samples(result.samples),
        "metadata": dict(result.metadata),
    }
    for name in ("log_probability", "chain", "log_probability_chain"):
        value = getattr(result, name)
        if value is not None:
            payload[name] = _encode_samples(value)
    return payload


def sampling_result_from_dict(payload: Any) -> SamplingResult | None:
    """Read float64 samples and historical float32 project sample records."""
    if not isinstance(payload, dict):
        return None
    try:
        result = SamplingResult(
            _decode_samples(payload["samples"]),
            [str(name) for name in payload.get("variable_names", [])],
            metadata=dict(payload.get("metadata", {})),
        )
    except (KeyError, TypeError, ValueError, zlib.error):
        return None
    for name in ("log_probability", "chain", "log_probability_chain"):
        if isinstance(payload.get(name), dict):
            try:
                setattr(result, name, _decode_samples(payload[name]))
            except (KeyError, TypeError, ValueError, zlib.error):
                pass
    return result


def sampling_diagnostics(chain: FloatArray, *, burn_in: int = 0) -> dict[str, Any]:
    """Estimate correlation on the retained raw chain; never certify mode mixing."""
    raw = np.asarray(chain, dtype=float)
    burn = max(0, int(burn_in))
    retained = raw[burn:]
    tau: list[float] = []
    reliable = False
    if retained.ndim == 3 and retained.shape[0] > 1:
        try:
            import emcee

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                estimate = emcee.autocorr.integrated_time(retained, quiet=False)
            tau = [float(value) for value in estimate]
            reliable = bool(np.all(np.isfinite(estimate) & (estimate > 0)))
        except Exception as exc:
            tau = [float(value) for value in np.asarray(getattr(exc, "tau", []))]
    finite = [value for value in tau if np.isfinite(value) and value > 0]
    target = burn + int(np.ceil(50 * max(finite))) if finite else None
    return {
        "autocorrelation_time": tau,
        "autocorrelation_reliable": reliable,
        "autocorrelation_recommended_steps": target,
        "autocorrelation_recommended_additional_steps": max(0, target - raw.shape[0])
        if target is not None
        else None,
        "effective_sample_size": [
            float(retained.shape[0] * retained.shape[1] / value)
            if np.isfinite(value) and value > 0
            else None
            for value in tau
        ],
        "autocorrelation_note": "Correlation estimates are heuristic and do not establish exploration of remote modes.",
    }


def sampling_result_with_window(result: SamplingResult, burn_in: int, thin: int) -> SamplingResult:
    """Select raw steps ``chain[burn_in::thin]`` consistently for every entry point."""
    if result.chain is None:
        raise ValueError("stored samples do not include a raw emcee chain")
    chain = np.asarray(result.chain, dtype=float)
    if chain.ndim != 3 or chain.shape[0] == 0:
        raise ValueError("stored sampling chain is empty or malformed")
    burn = max(0, int(burn_in))
    step = max(1, int(thin))
    samples = chain[burn::step].reshape((-1, chain.shape[2]))
    log_probability = None
    if result.log_probability_chain is not None:
        log_chain = np.asarray(result.log_probability_chain, dtype=float)
        if log_chain.shape == chain.shape[:2]:
            log_probability = log_chain[burn::step].reshape(-1)
    metadata = {
        **result.metadata,
        "burn_in": burn,
        "thin": step,
        "samples": int(samples.shape[0]),
        "n_steps": int(chain.shape[0]),
        "n_walkers": int(chain.shape[1]),
    }
    metadata.update(sampling_diagnostics(chain, burn_in=burn))
    return SamplingResult(
        samples,
        list(result.variable_names),
        log_probability,
        metadata,
        chain,
        result.log_probability_chain,
    )


def combined_sampling_result(
    original: SamplingResult, appended: SamplingResult, *, burn_in: int, thin: int
) -> SamplingResult:
    """Append matching targets and recompute diagnostics on the complete raw chain."""
    if original.chain is None or appended.chain is None:
        raise ValueError("appending samples requires raw emcee chains")
    if original.variable_names != appended.variable_names:
        raise ValueError("appended sampling variables do not match the stored chain")
    for key, default in (("purpose", "posterior"), ("temperature", 1.0)):
        if original.metadata.get(key, default) != appended.metadata.get(key, default):
            raise ValueError("cannot combine samples from different sampling targets")
    chain = np.concatenate([original.chain, appended.chain], axis=0)
    log_chain = None
    if original.log_probability_chain is not None and appended.log_probability_chain is not None:
        log_chain = np.concatenate(
            [original.log_probability_chain, appended.log_probability_chain], axis=0
        )
    metadata = {**original.metadata, **appended.metadata, "appended_steps": len(appended.chain)}
    old_accept = original.metadata.get("acceptance_fraction")
    new_accept = appended.metadata.get("acceptance_fraction")
    if old_accept is not None and new_accept is not None:
        accept = (
            np.asarray(old_accept) * len(original.chain)
            + np.asarray(new_accept) * len(appended.chain)
        ) / len(chain)
        metadata.update(_acceptance_metadata(accept))
    result = SamplingResult(
        np.empty((0, chain.shape[2])),
        list(original.variable_names),
        metadata=metadata,
        chain=chain,
        log_probability_chain=log_chain,
    )
    return sampling_result_with_window(result, burn_in, thin)


def best_sampling_sample(
    result: SamplingResult, *, include_burn_in: bool = False
) -> tuple[dict[str, float], float, dict[str, int]] | None:
    """Find a finite stored target score, retaining its step/walker location."""
    names = result.variable_names
    if not names:
        return None
    if include_burn_in and result.chain is not None and result.log_probability_chain is not None:
        values = np.asarray(result.chain)
        scores = np.asarray(result.log_probability_chain)
        if values.ndim == 3 and values.shape[2] == len(names) and scores.shape == values.shape[:2]:
            finite = np.isfinite(scores) & np.all(np.isfinite(values), axis=2)
            if np.any(finite):
                index = np.unravel_index(
                    int(np.argmax(np.where(finite, scores, -np.inf))), scores.shape
                )
                return (
                    dict(zip(names, map(float, values[index]), strict=True)),
                    float(scores[index]),
                    {"step": int(index[0]), "walker": int(index[1])},
                )
    values = np.asarray(result.samples)
    scores = np.asarray(result.log_probability)
    if values.ndim != 2 or values.shape[1] != len(names) or scores.shape != (len(values),):
        return None
    finite = np.isfinite(scores) & np.all(np.isfinite(values), axis=1)
    if not np.any(finite):
        return None
    index = int(np.argmax(np.where(finite, scores, -np.inf)))
    return (
        dict(zip(names, map(float, values[index]), strict=True)),
        float(scores[index]),
        {"sample": index},
    )


def _acceptance_metadata(acceptance: FloatArray) -> dict[str, Any]:
    return {
        "acceptance_fraction": np.asarray(acceptance).tolist(),
        "acceptance_fraction_mean": float(np.mean(acceptance)),
        "acceptance_fraction_min": float(np.min(acceptance)),
        "acceptance_fraction_max": float(np.max(acceptance)),
    }


class InvalidInitialEnsemble(ValueError):
    """No starting walker has a finite scientific target."""


def _encode_random_state(state: tuple) -> dict[str, Any]:
    algorithm, keys, position, has_gauss, cached_gaussian = state
    return {
        "algorithm": algorithm,
        "keys": np.asarray(keys).tolist(),
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def _decode_random_state(payload: dict[str, Any]) -> tuple:
    try:
        keys = np.asarray(payload["keys"])
        position = int(payload["position"])
        has_gauss = int(payload["has_gauss"])
        cached_gaussian = float(payload["cached_gaussian"])
        if (
            payload["algorithm"] != "MT19937"
            or keys.shape != (624,)
            or not np.issubdtype(keys.dtype, np.integer)
            or np.any(keys < 0)
            or np.any(keys > np.iinfo(np.uint32).max)
            or not 0 <= position <= 624
            or has_gauss not in (0, 1)
            or not np.isfinite(cached_gaussian)
        ):
            raise ValueError("invalid MT19937 random state")
        state = ("MT19937", keys.astype(np.uint32), position, has_gauss, cached_gaussian)
        np.random.RandomState().set_state(state)
        return state
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("stored sampler random state is malformed; run a fresh chain") from exc


def sample_parameter_vectors(
    log_probability: Callable[[FloatArray], float],
    initial_walkers: FloatArray,
    variable_names: list[str],
    config: SamplerConfig,
    *,
    workers: int = 1,
    worker_request: Any = 1,
    purpose: str = "posterior",
    temperature: float = 1.0,
    initial_random_state: dict[str, Any] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> SamplingResult:
    """Run emcee for a caller-owned scientific target without importing a facade."""
    try:
        import emcee
    except ImportError as exc:
        raise ImportError(
            "emcee sampling requires the 'emcee' package. Install or update the nfit environment before sampling."
        ) from exc
    n_walkers, n_dim = initial_walkers.shape
    n_steps = 1000 if config.n_steps is None else int(config.n_steps)
    if n_steps <= 0:
        raise ValueError("emcee n_steps must be positive")
    backend_kwargs = dict(config.kwargs)
    backend_kwargs.pop("workers", None)
    backend_kwargs.pop("parallel_workers", None)
    pool = ThreadPool(workers) if workers > 1 else None
    sampler: Any = None

    def result_from_sampler(completed: bool) -> SamplingResult:
        chain = np.asarray(sampler.get_chain(), dtype=float)
        metadata = {
            "method": "emcee",
            "purpose": purpose,
            "temperature": temperature,
            "requested_n_steps": n_steps,
            "completed": completed,
            "cancelled": not completed,
            "random_seed": config.random_seed,
            "random_state": _encode_random_state(sampler.random_state),
            "workers": workers,
            "worker_request": worker_request,
        }
        metadata.update(_acceptance_metadata(sampler.acceptance_fraction))
        result = sampling_result_with_window(
            SamplingResult(
                np.empty((0, n_dim)),
                variable_names,
                metadata=metadata,
                chain=chain,
                log_probability_chain=np.asarray(sampler.get_log_prob(), dtype=float),
            ),
            config.burn_in,
            config.thin,
        )
        tau = result.metadata["autocorrelation_time"]
        valid = [value for value in tau if np.isfinite(value) and value > 0]
        if valid:
            print(
                f"emcee integrated autocorrelation time (steps): {tau}; recommended total chain length: at least {result.metadata['autocorrelation_recommended_steps']} steps; estimated additional steps: {result.metadata['autocorrelation_recommended_additional_steps']}."
            )
        else:
            print(
                "emcee integrated autocorrelation time is unavailable for this chain; run more steps before using it to estimate a chain length."
            )
        return result

    try:
        initial_values = (
            pool.map(log_probability, initial_walkers)
            if pool is not None
            else map(log_probability, initial_walkers)
        )
        initial_log_prob = np.asarray(list(initial_values), float)
        if not np.any(np.isfinite(initial_log_prob)):
            raise InvalidInitialEnsemble(
                "No initial walker has a finite target probability; check the model configuration, bounds, and starting parameters."
            )
        if pool is not None:
            backend_kwargs["pool"] = pool
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_probability, **backend_kwargs)
        if initial_random_state is not None:
            sampler.random_state = _decode_random_state(initial_random_state)
        elif config.random_seed is not None:
            sampler.random_state = np.random.RandomState(config.random_seed).get_state()
        start = time.perf_counter()
        initial_state = emcee.State(
            initial_walkers, log_prob=initial_log_prob, random_state=sampler.random_state
        )
        for iteration, state in enumerate(sampler.sample(initial_state, iterations=n_steps), 1):
            if progress_callback is not None:
                elapsed = time.perf_counter() - start
                try:
                    progress_callback(
                        {
                            "stage": "emcee",
                            "iteration": iteration,
                            "total": n_steps,
                            "elapsed_seconds": elapsed,
                            "seconds_per_step": elapsed / iteration,
                            "parameters": dict(
                                zip(
                                    variable_names,
                                    map(float, np.mean(state.coords, axis=0)),
                                    strict=True,
                                )
                            ),
                            "message": f"emcee {purpose} step {iteration:,}/{n_steps:,}",
                        }
                    )
                except Exception as exc:
                    raise SamplingCancelled(
                        f"emcee {purpose} sampling cancelled; partial samples were saved.",
                        result_from_sampler(False),
                    ) from exc
        return result_from_sampler(True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
