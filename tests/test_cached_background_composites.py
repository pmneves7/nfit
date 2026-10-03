"""Analytic bank combinations preserve actual coefficients and shared dummy events."""

import copy

import numpy as np
import pytest

from nfit import (
    DataGroup,
    DatasetEntry,
    project_measured_background_mdevent,
    replay_cached_background_profile,
)
from nfit.backgrounds import subtract_aligned_background
from nfit.cached_background_replay import CACHED_BACKGROUND_REPLAY
from nfit.histogram_statistics import (
    EVENT_STATISTICS_KEY,
    EVENT_STATISTICS_METADATA,
    event_statistics_channels,
)
from nfit.mdhisto import MDHistoData
from nfit.measurement_dependencies import SourceReplayRequired
from nfit.project_composites import (
    _composite_mdhisto_data,
    _scale_composite_with_replay,
    composite_dataset_data,
    data_group_composite_config,
)
from nfit.project_dataset_io import _load_nfit_dataset_file, save_dataset_file
from tests.test_mdevent_background_statistics import _directional_fixture, _target


def sample(template, numerator, exposure, identity):
    c = np.asarray(numerator).reshape(template.shape)
    n = np.asarray(exposure).reshape(template.shape)
    return MDHistoData(template.axes, c/n, np.sqrt(c)/n, np.zeros(template.shape, bool),
        np.ones(template.shape), metadata={"signal_semantics": "density", "zero_event_bins_are_measured": True,
            "normalization_denominator": n, EVENT_STATISTICS_KEY: dict(EVENT_STATISTICS_METADATA),
            "source_lineage": {"version": 1, "source_ids": [identity]}},
        auxiliary_channels=event_statistics_channels(c, c, n))


def config(data, **changes):
    axes = [dict(name=axis.name, lower=axis.values[0], upper=axis.values[-1],
        mode="edges", bin_edges=axis.values.tolist(), vector=np.eye(4)[dim].tolist(),
        auto_lower=False, auto_upper=False, fractional=True)
        for dim, axis in enumerate(data.axes)]
    return {"enabled": True, "auto_rebin": False, "mean_weighting": "uniform",
            "fractional": True, "minimum_coverage": 0, "axes": axes, **changes}


def combine(entries, **changes):
    group = DataGroup("Banks", datasets=entries)
    return _composite_mdhisto_data(group, config(entries[0].data, **changes), datasets=entries)


def profile(data):
    return replay_cached_background_profile(data, selected=np.ones(data.shape, bool),
        indices=np.zeros(data.shape, int), edges=[0., 1.]).data


@pytest.fixture
def banks(tmp_path):
    angles, source = _directional_fixture(tmp_path)
    template = _target(angles, 2)
    background = project_measured_background_mdevent(angles, source, template)
    first = sample(template, [8., 0.], [1., 3.], "sample:A")
    second = sample(template, [0., 12.], [2., 2.], "sample:B")
    return first, second, background


def test_bank_mean_uses_fit_weight_exposure_and_signed_scale_and_shared_background(banks):
    first, second, background = banks
    entries = [DatasetEntry("A", subtract_aligned_background(first, background), fit_weight=2),
               DatasetEntry("B", subtract_aligned_background(second, background), fit_weight=.5, scale_factor=-2)]
    combined = combine(entries)
    result = profile(combined)
    assert result.signal.item() == pytest.approx(-1.2)
    # Independent sample numerator variance44; one reused dummy primitive has
    # net coefficient-2, so its variance32. Total exposure10.
    assert result.errors.item()**2 == pytest.approx(.76)
    assert result.auxiliary_channels["normalization_denominator"].values.item() == pytest.approx(10)
    assert combined.metadata[CACHED_BACKGROUND_REPLAY]["independent_source_ids"] == ["sample:A", "sample:B"]


def test_nested_bank_mean_keeps_component_weights_and_plain_sample_statistics(banks):
    first, second, background = banks
    bank = combine([DatasetEntry("A", subtract_aligned_background(first, background), fit_weight=2),
                    DatasetEntry("B", subtract_aligned_background(second, background), fit_weight=.5, scale_factor=-2)])
    third = sample(first, [3., 5.], [1., 1.], "sample:C")
    result = profile(combine([DatasetEntry("Banks", bank, fit_weight=3, scale_factor=.25),
                              DatasetEntry("Plain", third)]))
    assert result.signal.item() == pytest.approx(-1/32)
    assert result.errors.item()**2 == pytest.approx(50.75/32**2)
    assert result.auxiliary_channels["normalization_denominator"].values.item() == pytest.approx(32)


def test_public_composite_and_coverage_threshold_preserve_recipe(banks):
    first, second, background = banks
    group = DataGroup("Banks", datasets=[DatasetEntry("A", subtract_aligned_background(first, background)),
        DatasetEntry("B", second)])
    data_group_composite_config(group).update(config(first))
    result = composite_dataset_data(group)
    assert CACHED_BACKGROUND_REPLAY in result.metadata
    assert profile(result).signal.item() == pytest.approx(.5)


def test_known_composite_calibration_preserves_signed_gains_and_exposure(banks):
    first, _, background = banks
    original = subtract_aligned_background(first, background)
    scaled = _scale_composite_with_replay(original, -3.)
    result = profile(scaled)
    assert result.signal.item() == pytest.approx(6.)
    assert result.errors.item()**2 == pytest.approx(22.5)
    assert result.auxiliary_channels["normalization_denominator"].values.item() == pytest.approx(4)


def test_masks_apply_to_each_bank_coefficient_before_merging_sources(banks):
    first, second, background = banks
    first = subtract_aligned_background(first, background).with_updates(mask=np.array([False, True]).reshape(first.shape))
    second = subtract_aligned_background(second, background)
    result = profile(combine([DatasetEntry("A", first), DatasetEntry("B", second)]))
    # A contributes C8 and exposure1; B contributes C12 and exposure4.
    # Background gain = -(1+4)/2=-2.5; independent variance20.
    assert result.signal.item() == pytest.approx(0.)
    assert result.errors.item()**2 == pytest.approx((20+8*2.5**2)/25)


@pytest.mark.parametrize("kind", ["inverse_variance", "regrid", "shared_sample", "missing_recipe"])
def test_unsupported_composite_statistics_refuse_exact_recipe(banks, kind):
    first, second, background = banks
    a = subtract_aligned_background(first, background)
    b = subtract_aligned_background(second, background)
    changes = {}
    if kind == "inverse_variance":
        changes["mean_weighting"] = "inverse_variance"
    elif kind == "regrid":
        axes = config(first)["axes"]
        axes[0]["bin_edges"] = [-2., .5]
        changes["axes"] = axes
    elif kind == "shared_sample":
        b = subtract_aligned_background(first, background)
    elif kind == "missing_recipe":
        metadata = dict(b.metadata)
        metadata.pop(CACHED_BACKGROUND_REPLAY)
        b = b.with_updates(metadata=metadata)
    result = combine([DatasetEntry("A", a), DatasetEntry("B", b)], **changes)
    assert CACHED_BACKGROUND_REPLAY not in result.metadata
    assert result.metadata["background_replay_unavailable_reason"]
    with pytest.raises(SourceReplayRequired):
        profile(result)


def test_composite_recipe_roundtrip_and_term_weight_changes_invalidate_queries(banks, tmp_path):
    first, second, background = banks
    combined = combine([DatasetEntry("A", subtract_aligned_background(first, background)),
                        DatasetEntry("B", subtract_aligned_background(second, background))])
    path = tmp_path/"banks.npz"
    save_dataset_file(DatasetEntry("Banks", combined), path, use_view=False)
    restored, _ = _load_nfit_dataset_file(path)
    np.testing.assert_allclose(profile(restored).errors, profile(combined).errors)
    metadata = copy.deepcopy(restored.metadata)
    term = metadata[CACHED_BACKGROUND_REPLAY]["terms"][0]
    term["scale"] *= 2
    changed = restored.with_updates(metadata=metadata)
    assert not profile(changed).metadata["cached_background_profile"]["query_cache_hit"]


def test_composite_build_does_not_open_event_sources(banks, monkeypatch):
    import h5py

    first, second, background = banks
    a = subtract_aligned_background(first, background)
    def no_events(*args, **kwargs):
        raise AssertionError("Composite construction opened original event data")
    monkeypatch.setattr(h5py, "File", no_events)
    combined = combine([DatasetEntry("A", a), DatasetEntry("B", second)])
    assert CACHED_BACKGROUND_REPLAY in combined.metadata


def test_sample_background_identity_overlap_refuses_composite_exact_recipe(banks):
    first, second, background = banks
    a = subtract_aligned_background(first, background)
    metadata = copy.deepcopy(a.metadata)
    metadata[CACHED_BACKGROUND_REPLAY]["terms"][0]["sources"][0]["source_lineage_ids"] = ["sample:B"]
    a = a.with_updates(metadata=metadata)
    combined = combine([DatasetEntry("A", a), DatasetEntry("B", second)])
    assert CACHED_BACKGROUND_REPLAY not in combined.metadata
    assert "also appears in a background" in combined.metadata["background_replay_unavailable_reason"]


def test_composite_background_operand_preserves_its_term_specific_normalizers(banks):
    first, _, background = banks
    composite_background = combine([DatasetEntry("A", background), DatasetEntry("B", background, fit_weight=2)])
    result = profile(subtract_aligned_background(first, composite_background))
    assert result.signal.item() == pytest.approx(-2.)
    assert result.errors.item()**2 == pytest.approx(2.5)


def test_real_project_hierarchy_propagates_bank_sources_without_flattening(banks):
    from nfit import DatasetGroup
    from nfit.project_composites import _composite_scope

    first, second, background = banks
    children = [DatasetGroup("A", datasets=[DatasetEntry("A", subtract_aligned_background(first, background))]),
                DatasetGroup("B", datasets=[DatasetEntry("B", subtract_aligned_background(second, background))])]
    root = DataGroup("Temperature", subgroups=children)
    data_group_composite_config(root).update(config(first))
    for child in children:
        data_group_composite_config(_composite_scope(root, child)).update(config(first))
    result = composite_dataset_data(root, apply_spectral_channels=False)
    assert CACHED_BACKGROUND_REPLAY in result.metadata
    assert profile(result).signal.item() == pytest.approx(-1.5)
    assert profile(result).errors.item()**2 == pytest.approx(148/64)


def test_unpropagated_sample_dependencies_refuse_exact_composite(banks):
    from nfit.measurement_dependencies import SourceDependencies

    first, second, background = banks
    factors = SourceDependencies(shape=second.shape, source_ids=("a", "b"),
        source_variances=second.errors.ravel()**2, observation_indices=np.array([0, 1]),
        source_indices=np.array([0, 1]), coefficients=np.ones(2))
    second = second.with_updates(source_dependencies=factors)
    result = combine([DatasetEntry("A", subtract_aligned_background(first, background)),
                      DatasetEntry("B", second)])
    assert CACHED_BACKGROUND_REPLAY not in result.metadata
    assert "sample source dependencies" in result.metadata["background_replay_unavailable_reason"]


def test_term_specific_weight_payload_participates_in_query_cache_signature(banks):
    from nfit.mdhisto import MDHistoChannel

    first, second, background = banks
    result = combine([DatasetEntry("A", subtract_aligned_background(first, background)),
                      DatasetEntry("B", subtract_aligned_background(second, background))])
    assert profile(result).errors.item()**2 == pytest.approx(148/64)
    assert profile(result).metadata["cached_background_profile"]["query_cache_hit"]
    payload = copy.deepcopy(result.metadata)
    payload[CACHED_BACKGROUND_REPLAY]["coefficient_revision"] = 2
    name = payload[CACHED_BACKGROUND_REPLAY]["terms"][0]["weight_channel"]
    channels = dict(result.auxiliary_channels)
    channels[name] = MDHistoChannel(2*channels[name].values)
    changed = result.with_updates(auxiliary_channels=channels, metadata=payload)
    queried = profile(changed)
    assert not queried.metadata["cached_background_profile"]["query_cache_hit"]
    assert queried.errors.item()**2 == pytest.approx(308/64)


def test_composite_recipe_budget_preserves_preview_and_explains_exact_failure(banks, monkeypatch):
    import nfit.cached_background_replay as replay

    first, second, background = banks
    a = subtract_aligned_background(first, background)
    monkeypatch.setattr(replay, "MAX_RECIPE_BYTES", 1)
    result = combine([DatasetEntry("A", a), DatasetEntry("B", second)])
    assert np.isfinite(result.signal).all()
    assert CACHED_BACKGROUND_REPLAY not in result.metadata
    with pytest.raises(SourceReplayRequired, match="storage budget"):
        profile(result)


def test_metadata_stack_keeps_preview_and_marks_background_replay_unavailable(banks):
    from nfit import MetadataDimension

    first, second, background = banks
    entries = [DatasetEntry("A", subtract_aligned_background(first, background), parameters={"temperature": 5.}),
               DatasetEntry("B", second, parameters={"temperature": 10.})]
    settings = config(first)
    settings["axes"].append(dict(name="Temperature", units="K", lower=2.5, upper=12.5,
        mode="edges", bin_edges=[2.5, 7.5, 12.5], auto_lower=False, auto_upper=False, fractional=False))
    result = _composite_mdhisto_data(DataGroup("Banks", datasets=entries), settings, datasets=entries,
        metadata_dimensions=[MetadataDimension("Temperature", "parameters/temperature", "K")])
    assert len(result.axes) == 5
    assert CACHED_BACKGROUND_REPLAY not in result.metadata
    assert "source grids or physical bases differ" in result.metadata["background_replay_unavailable_reason"]
