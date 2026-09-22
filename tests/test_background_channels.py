import numpy as np

from nfit.background_channels import (
    BACKGROUND_EXCEPTIONS_KEY,
    available_background_channels,
    background_channel,
    record_background_original_exceptions,
    scale_background_channels,
)
from nfit.backgrounds import subtract_aligned_background
from nfit.mdhisto import MDHistoAxis, MDHistoChannel, MDHistoData


def _data(signal=10.0, errors=1.0, *, mask=None):
    shape = (2, 2)
    return MDHistoData(
        axes=(
            MDHistoAxis("|Q|", np.array([0.5, 1.5, 2.5]), "1/angstrom", "momentum"),
            MDHistoAxis("DeltaE", np.array([-1.5, -0.5, 0.5]), "meV", "energy"),
        ),
        signal=np.full(shape, signal, dtype=float),
        errors=np.full(shape, errors, dtype=float),
        mask=np.zeros(shape, dtype=bool) if mask is None else mask,
        num_events=np.ones(shape),
        metadata={
            "signal_unit": "mbarn/meV",
            "signal_quantity_type": "scattering_intensity",
        },
    )


def test_background_channel_accumulates_and_reconstructs_original():
    original = _data()
    first = subtract_aligned_background(original, _data(signal=2.0, errors=0.5), scale=2.0)
    result = subtract_aligned_background(first, _data(signal=1.0, errors=0.25), scale=3.0)

    assert available_background_channels(result) == ("background", "unsubtracted")
    background = background_channel(result, "background", (slice(None), 1))
    np.testing.assert_allclose(background.values, 7.0)
    np.testing.assert_allclose(background.errors, np.sqrt(1.0 + 0.75**2))
    unsubtracted = background_channel(result, "unsubtracted", (slice(None), 1))
    np.testing.assert_allclose(unsubtracted.values, 10.0)
    np.testing.assert_allclose(unsubtracted.errors, 1.0)
    np.testing.assert_array_equal(unsubtracted.mask, False)
    assert not background.values.flags.writeable


def test_first_unit_scale_aligned_background_reuses_immutable_arrays():
    original = _data()
    background = _data(signal=2.0, errors=0.5)

    result = subtract_aligned_background(original, background)
    channel = result.auxiliary_channels["background"]

    assert np.shares_memory(channel.values, background.signal)
    assert np.shares_memory(channel.errors, background.errors)


def test_invalid_background_uses_sparse_exact_original_override():
    original = _data(signal=11.0, errors=1.5)
    missing = np.zeros((2, 2), dtype=bool)
    missing[1, 0] = True
    result = subtract_aligned_background(
        original,
        _data(signal=3.0, errors=2.0, mask=missing),
    )

    record = result.metadata[BACKGROUND_EXCEPTIONS_KEY]
    np.testing.assert_array_equal(record["indices"], [2])
    assert all(not array.flags.writeable for array in record.values())
    restored = background_channel(result, "unsubtracted", (1, slice(None)))
    np.testing.assert_allclose(restored.values, [11.0, 11.0])
    np.testing.assert_allclose(restored.errors, [1.5, 1.5])
    np.testing.assert_array_equal(restored.mask, False)


def test_exact_override_restores_correlated_error_and_scaling():
    original = _data(signal=8.0, errors=3.0)
    result = subtract_aligned_background(original, original)
    cancelled = result.with_updates(
        signal=np.zeros(result.shape),
        errors=np.zeros(result.shape),
    )
    where = np.zeros(result.shape, dtype=bool)
    where[0, 1] = True
    recorded = record_background_original_exceptions(cancelled, original, where)

    restored = background_channel(recorded, "unsubtracted", (0, 1))
    assert restored.values == 8.0
    assert restored.errors == 3.0
    scaled = scale_background_channels(recorded, 2.0, unit="counts")
    restored_scaled = background_channel(scaled, "unsubtracted", (0, 1))
    assert restored_scaled.values == 16.0
    assert restored_scaled.errors == 6.0
    assert scaled.auxiliary_channels["background"].unit == "counts"


def test_zero_scale_returns_same_object_without_provenance_allocation():
    original = _data()
    result = subtract_aligned_background(original, _data(signal=2.0), scale=0.0)

    assert result is original
    assert available_background_channels(result) == ()


def test_unmarked_auxiliary_background_is_not_subtraction_provenance():
    original = _data()
    custom = original.with_updates(
        auxiliary_channels={
            "background": MDHistoChannel(np.ones(original.shape), label="Imported")
        }
    )

    assert available_background_channels(custom) == ()
    assert scale_background_channels(custom, 2.0) is custom
