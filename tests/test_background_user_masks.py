"""Background subtraction semantics for live (nfit) masks."""

import numpy as np
import pytest

from nfit.backgrounds import subtract_aligned_background, subtract_powder_background
from nfit.mdhisto import MDHistoAxis, MDHistoData


def _powder(
    signal: float | np.ndarray = 4.0,
    errors: float | np.ndarray = 2.0,
    *,
    mask: np.ndarray | None = None,
    nfit_mask: np.ndarray | None = None,
) -> MDHistoData:
    shape = (2, 2)
    signal_array = np.full(shape, signal, dtype=float) if np.isscalar(signal) else np.asarray(signal, dtype=float)
    errors_array = np.full(shape, errors, dtype=float) if np.isscalar(errors) else np.asarray(errors, dtype=float)
    return MDHistoData(
        axes=(
            MDHistoAxis("|Q|", np.array([0.5, 1.5, 2.5]), "1/angstrom", "momentum"),
            MDHistoAxis("DeltaE", np.array([-1.5, -0.5, 0.5]), "meV", "energy"),
        ),
        signal=signal_array,
        errors=errors_array,
        mask=np.zeros(shape, dtype=bool) if mask is None else mask,
        num_events=np.ones(shape),
        metadata={
            "signal_semantics": "density",
            "nfit_mask": np.zeros(shape, dtype=bool) if nfit_mask is None else nfit_mask,
        },
    )


@pytest.mark.parametrize("interpolation", ["linear", "nearest"])
def test_powder_user_mask_is_zero_background_not_missing(interpolation):
    data = _powder(signal=10.0, errors=1.0)
    user_mask = np.zeros((2, 2), dtype=bool)
    user_mask[0, 1] = True
    background = _powder(signal=4.0, errors=3.0, mask=user_mask, nfit_mask=user_mask)
    source_signal = background.signal.copy()
    source_errors = background.errors.copy()

    result = subtract_powder_background(data, background, interpolation=interpolation)

    # A live user mask contributes exactly zero signal and variance.
    assert result.signal[0, 1] == 10.0
    assert result.errors[0, 1] == 1.0
    assert not result.mask[0, 1]
    # Unmasked source bins retain ordinary subtraction and propagation.
    assert result.signal[1, 1] == 6.0
    np.testing.assert_allclose(result.errors[1, 1], np.sqrt(10.0))
    np.testing.assert_array_equal(result.mask, False)
    np.testing.assert_array_equal(background.signal, source_signal)
    np.testing.assert_array_equal(background.errors, source_errors)


def test_powder_native_missing_bin_remains_invalid():
    data = _powder(signal=10.0, errors=1.0)
    native_mask = np.zeros((2, 2), dtype=bool)
    native_mask[0, 1] = True
    background = _powder(signal=4.0, errors=3.0, mask=native_mask)

    result = subtract_powder_background(data, background, interpolation="nearest")

    assert result.mask[0, 1]
    assert np.isnan(result.signal[0, 1])
    assert np.isnan(result.errors[0, 1])


def test_aligned_user_mask_is_zero_background_not_missing():
    data = _powder(signal=10.0, errors=1.0)
    user_mask = np.zeros((2, 2), dtype=bool)
    user_mask[1, 0] = True
    background = _powder(signal=4.0, errors=3.0, mask=user_mask, nfit_mask=user_mask)

    result = subtract_aligned_background(data, background)

    assert result.signal[1, 0] == 10.0
    assert result.errors[1, 0] == 1.0
    assert not result.mask[1, 0]
    assert result.signal[0, 0] == 6.0
    np.testing.assert_allclose(result.errors[0, 0], np.sqrt(10.0))


def test_aligned_native_missing_bin_remains_invalid():
    data = _powder(signal=10.0, errors=1.0)
    native_mask = np.zeros((2, 2), dtype=bool)
    native_mask[1, 0] = True
    background = _powder(signal=4.0, errors=3.0, mask=native_mask)

    result = subtract_aligned_background(data, background)

    assert result.mask[1, 0]
    assert np.isnan(result.signal[1, 0])
    assert np.isnan(result.errors[1, 0])
