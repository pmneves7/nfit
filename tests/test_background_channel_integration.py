"""Persistence and physical transformations of visualization-only backgrounds."""

import ast
from pathlib import Path

import numpy as np
import pytest

from nfit.analysis.artifacts import read_dataset_artifact, write_dataset_artifact
from nfit.background_channels import (
    BACKGROUND_EXCEPTIONS_KEY,
    background_channel,
    record_background_original_exceptions,
)
from nfit.backgrounds import subtract_aligned_background
from nfit.mdhisto import MDHistoAxis, MDHistoData
from nfit.pipeline import DatasetEntry
from nfit.project_dataset_io import _load_nfit_dataset_file, save_dataset_file
from nfit.project_view_data import _apply_dataset_scale, _apply_kinematic_normalization_to_view
from nfit.rebin_cache import CompressedBinning
from nfit.spectral_channels import with_paired_spectral_channels


def sample_pair():
    axes = (MDHistoAxis("H", [0, 1, 2], "r.l.u.", "momentum"),
            MDHistoAxis("DeltaE", [1, 2, 3], "meV", "energy"))
    sample = MDHistoData(axes, [[2, 5], [7, 10]], np.ones((2, 2)),
                         np.zeros((2, 2), bool), np.ones((2, 2)),
                         metadata={"signal_semantics": "density"})
    background = sample.with_updates(signal=np.array([[2, 3], [4, 5.]]),
                                     errors=np.full((2, 2), 2.0),
                                     mask=np.array([[False, True], [False, False]]))
    return sample, background


@pytest.mark.parametrize("codec", ["artifact", "dataset", "compressed", "mapped"])
def test_background_recovery_roundtrip(tmp_path, codec):
    sample, background = sample_pair()
    result = subtract_aligned_background(sample, background)
    path = tmp_path / "background.npz"
    if codec == "dataset":
        save_dataset_file(DatasetEntry("subtracted", result), path, use_view=False)
        restored, _ = _load_nfit_dataset_file(path)
    elif codec == "compressed":
        compressed = CompressedBinning.from_data(result, max_bytes=1_000_000)
        assert compressed is not None
        restored = compressed.restore()
    else:
        write_dataset_artifact(result, path)
        restored = read_dataset_artifact(path, memory_map=codec == "mapped")
    original = background_channel(restored, "unsubtracted")
    np.testing.assert_allclose(original.values, sample.signal)
    np.testing.assert_allclose(original.errors, sample.errors)
    np.testing.assert_array_equal(original.mask, sample.mask)
    np.testing.assert_allclose(restored.signal, result.signal)
    np.testing.assert_allclose(restored.errors, result.errors)
    assert "unsubtracted" not in restored.auxiliary_channels
    assert not restored.metadata[BACKGROUND_EXCEPTIONS_KEY]["indices"].flags.writeable


@pytest.mark.parametrize("mode", ["scale", "kinematic", "spectral_cross", "spectral_chi"])
def test_visualization_components_follow_physical_transform(mode):
    sample, background = sample_pair()
    result = subtract_aligned_background(sample, background)
    entry = DatasetEntry("sample", result, scale_factor=-2,
                         parameters={"kf_ki_included": False, "incident_energy_meV": 10})
    def transform(data):
        if mode == "scale":
            return _apply_dataset_scale(entry, data)
        if mode == "kinematic":
            return _apply_kinematic_normalization_to_view(entry, data)
        return with_paired_spectral_channels(
            data, {"enabled": True, "source_representation": "cross_section",
                   "signal_per_mbarn": 3,
                   "fit_representation": "chi_double_prime" if mode == "spectral_chi" else "cross_section"},
            temperature_K=10,
        )
    transformed = transform(result)
    expected = transform(sample)
    original = background_channel(transformed, "unsubtracted")
    np.testing.assert_allclose(original.values, expected.signal)
    np.testing.assert_allclose(original.errors, expected.errors)
    np.testing.assert_allclose(transformed.signal, transform(result.with_updates(auxiliary_channels={})).signal)


def test_background_services_are_gui_independent():
    root = Path(__file__).parents[1] / "src" / "nfit"
    for filename in ("background_channels.py", "background_channel_io.py"):
        tree = ast.parse((root / filename).read_text())
        imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        imports += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
        assert not any("PySide" in name or "project_gui" in name or "qt_slice" in name for name in imports)


def test_volume_diagnostics_select_before_reconstruction(monkeypatch):
    from nfit import qt_volume_viewer as volume

    shape = (2, 3, 4, 5)
    data = MDHistoData(
        tuple(MDHistoAxis(f"x{i}", np.arange(n + 1), "", "unknown")
              for i, n in enumerate(shape)),
        np.full(shape, 10.), np.ones(shape), np.zeros(shape, bool), np.ones(shape),
    )
    result = subtract_aligned_background(data, data.with_updates(signal=np.full(shape, 3.)))
    real = volume.background_channel
    calls = []
    def read_selection(data, name, selection):
        calls.append(selection)
        assert selection[3] == 2
        return real(data, name, selection)
    monkeypatch.setattr(volume, "background_channel", read_selection)
    view = volume.extract_volume_arrays(
        result, axes=(0, 1, 2), color_channel="unsubtracted", opacity_channel="unsubtracted",
        hidden_selections={3: 2},
    )
    assert view.color.shape == (2, 3, 4)
    np.testing.assert_allclose(view.color, 10.)
    assert view.opacity is view.color
    assert len(calls) == 1
    assert "unsubtracted" in volume.volume_channel_names(result)
    assert "unsubtracted" not in volume.volume_channel_names(data)


def test_composite_preserves_correlated_original_errors():
    from nfit.pipeline import DataGroup
    from nfit.project_composites import _composite_mdhisto_data
    from nfit.project_rebinning import _default_rebin_axes

    sample, _ = sample_pair()
    difference = subtract_aligned_background(sample, sample).with_updates(
        signal=np.zeros(sample.shape), errors=np.zeros(sample.shape),
    )
    difference = record_background_original_exceptions(difference, sample, ~sample.mask)
    axes = _default_rebin_axes(sample)
    for axis, original_axis in zip(axes, sample.axes, strict=True):
        axis.update(mode="edges", bin_edges=[original_axis.values[0], original_axis.values[-1]],
                    auto_lower=False, auto_upper=False, auto_step_size=False)
    config = {"axes": axes, "fractional": False, "mean_weighting": "uniform", "minimum_coverage": 0}
    output = _composite_mdhisto_data(
        DataGroup("parent"), config, datasets=[DatasetEntry("child", difference)],
    )
    np.testing.assert_allclose(output.signal, 0.)
    np.testing.assert_allclose(output.errors, 0.)
    recovered = background_channel(output, "unsubtracted")
    np.testing.assert_allclose(recovered.values, 6.)
    np.testing.assert_allclose(recovered.errors, .5)


@pytest.mark.parametrize("template_has_background", [False, True])
def test_metadata_stack_mixes_backgrounds_and_keeps_recovery_indices(template_has_background):
    from nfit.metadata_dimensions import MetadataDimension, stack_metadata_histograms

    sample, background = sample_pair()
    result = subtract_aligned_background(sample, background)
    stack = stack_metadata_histograms(
        result if template_has_background else sample,
        {(1,): result, (0,): sample},
        [MetadataDimension("Temperature", "parameters/temperature", "K")],
        [np.array([5., 10.])],
    )
    for index in (0, 1):
        restored = background_channel(stack, "unsubtracted", (slice(None), slice(None), index))
        np.testing.assert_allclose(restored.values, sample.signal)
        np.testing.assert_allclose(restored.errors, sample.errors)
        np.testing.assert_array_equal(restored.mask, sample.mask)
    assert not stack.metadata[BACKGROUND_EXCEPTIONS_KEY]["indices"].flags.writeable


def test_noncontiguous_reconstruction_restores_exceptions():
    sample, _ = sample_pair()
    original = sample.with_updates(
        signal=np.asfortranarray(sample.signal), errors=np.asfortranarray(sample.errors),
    )
    background = original.with_updates(
        signal=np.full(original.shape, 1e20, order="F"),
        errors=np.full(original.shape, 1e20, order="F"),
    )
    difference = subtract_aligned_background(original, background)
    restored = background_channel(difference, "unsubtracted")
    np.testing.assert_allclose(restored.values, original.signal)
    np.testing.assert_allclose(restored.errors, original.errors)


def test_fit_bundle_ignores_visualization_channels():
    from nfit.pipeline import DataGroup
    from nfit.project_gui import fit_data_bundle

    shape = (2, 1, 1, 2)
    axes = tuple(MDHistoAxis(name, np.arange(n + 1), unit, kind)
                 for name, n, unit, kind in zip(
                     ("H", "K", "L", "DeltaE"), shape,
                     ("rlu", "rlu", "rlu", "meV"),
                     ("momentum", "momentum", "momentum", "energy"), strict=True))
    data = MDHistoData(axes, np.full(shape, 10.), np.ones(shape),
                       np.zeros(shape, bool), np.ones(shape))
    result = subtract_aligned_background(data, data.with_updates(signal=np.full(shape, 3.)))
    full = DatasetEntry("with diagnostics", result, kind="mdhisto")
    primary = DatasetEntry("primary only", result.with_updates(auxiliary_channels={}), kind="mdhisto")
    group = DataGroup("group", datasets=[full, primary])
    with_channels = fit_data_bundle(group, full)
    without_channels = fit_data_bundle(group, primary)
    assert with_channels is not None and without_channels is not None
    np.testing.assert_array_equal(with_channels.points.intensity, without_channels.points.intensity)
    np.testing.assert_array_equal(with_channels.points.sigma, without_channels.points.sigma)
    assert with_channels.points.size == np.prod(shape)


def test_derived_subtraction_retains_background_channels():
    from nfit.analysis.data_reduction import combine_aligned_histograms

    sample, background = sample_pair()
    difference = combine_aligned_histograms(sample, background, right_scale=2.)
    restored = background_channel(difference, "unsubtracted")
    np.testing.assert_allclose(restored.values, sample.signal)
    np.testing.assert_allclose(restored.errors, sample.errors)
    np.testing.assert_allclose(difference.auxiliary_channels["background"].values, 2 * background.signal)
