import numpy as np
import pytest

from nfit import (
    bin_raw_dgs_group,
    bin_raw_dgs_powder_group,
    inspect_raw_dgs_run,
    raw_dgs,
    raw_dgs_dataset_group,
)
from nfit.raw_dgs import (
    TOF_US_PER_M_SQRT_MEV,
    _energy_transfer_bounds,
    _evaluate_mantid_t0_formula,
    _monitor_ei_t0,
    _retained_proton_charge_uah,
)


def _write_raw_dgs(path, *, with_he3=False):
    h5py = pytest.importorskip("h5py")
    xml = """<instrument xmlns="http://www.mantidproject.org/IDF/1.0">
      <component type="moderator"><location z="-10"/></component>
      <component type="panel" idlist="panel"><location/></component>
      <type name="moderator" is="Source"/><type name="pixel" is="detector"/>
      <type name="panel"><component type="pixel"><location x="1" y="0" z="2"/></component></type>
      <idlist idname="panel"><id start="42" end="42"/></idlist>
    </instrument>"""
    if with_he3:
        xml = xml.replace(
            '<type name="moderator" is="Source"/><type name="pixel" is="detector"/>',
            """<type name="moderator" is="Source"/>
      <type name="pixel" is="detector"><cylinder id="tube"><axis x="0" y="1" z="0"/><radius val="0.01"/></cylinder></type>
      <component-link name="panel"><parameter name="tube_pressure"><value val="10.0"/></parameter><parameter name="tube_thickness"><value val="0.0008"/></parameter><parameter name="tube_temperature"><value val="290.0"/></parameter></component-link>""",
        )
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry")
        entry.create_dataset("run_number", data=[b"42"])
        instrument = entry.create_group("instrument")
        instrument.create_group("instrument_xml").create_dataset(
            "data", data=np.frombuffer(xml.encode(), dtype="u1")
        )
        bank = entry.create_group("bank1_events")
        bank.create_dataset("event_id", data=[42, 42])
        # One useful event and one prompt event, which must be discarded.
        bank.create_dataset("event_time_offset", data=[9000.0, 1.0])
        logs = entry.create_group("DASlogs")
        for name, value in (
            ("BL17:Det:TH:BL:Ei", 20.0),
            ("proton_charge", 2.0),
            ("omega", 0.0),
            ("phi", 0.0),
            ("chi", 0.0),
        ):
            logs.create_group(name).create_dataset("average_value", data=[value])


@pytest.fixture
def geometry_cache():
    raw_dgs._detector_geometry_from_xml.cache_clear()
    yield
    raw_dgs._detector_geometry_from_xml.cache_clear()


def _rewrite_instrument_xml(path, old, new):
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "r+") as handle:
        group = handle["entry/instrument/instrument_xml"]
        xml = group["data"][()].tobytes().decode().replace(old, new)
        del group["data"]
        group.create_dataset("data", data=np.frombuffer(xml.encode(), dtype="u1"))


@pytest.mark.parametrize("instrument", ["ARCS", "CNCS", "HYSPEC", "SEQUOIA", ""])
def test_dgs_source_probe_keeps_direct_geometry_runs(tmp_path, instrument):
    path = tmp_path / "events.nxs.h5"
    _write_raw_dgs(path)
    _rewrite_instrument_xml(path, '<instrument xmlns=', f'<instrument name="{instrument}" xmlns=')
    assert raw_dgs.is_raw_dgs_nexus_file(path)


@pytest.mark.parametrize("instrument", ["WAND", "HB2C", "CORELLI", "MACS", "CG2", "D33"])
def test_event_banks_and_energy_log_do_not_route_other_measurements_through_dgs(tmp_path, instrument):
    path = tmp_path / "other_measurement.nxs.h5"
    _write_raw_dgs(path)  # Deliberately retain a positive energy log as well.
    _rewrite_instrument_xml(path, '<instrument xmlns=', f'<instrument name="{instrument}" xmlns=')
    assert not raw_dgs.is_raw_dgs_nexus_file(path)


def test_unnamed_event_banks_without_fixed_incident_energy_are_not_dgs(tmp_path):
    path = tmp_path / "unclassified.nxs.h5"
    _write_raw_dgs(path)
    with pytest.importorskip("h5py").File(path, "r+") as handle:
        del handle["entry/DASlogs/BL17:Det:TH:BL:Ei"]
    assert not raw_dgs.is_raw_dgs_nexus_file(path)


def test_hyspec_crop_cache_and_reduction_settings_use_the_same_raw_events(tmp_path, monkeypatch):
    import json

    from nfit.raw_dgs_hyspec import resolved_hyspec_preprocessing
    from nfit.reduction_recipes import set_reduction_settings

    h5py = pytest.importorskip("h5py")
    path = tmp_path / "HYS_42.nxs.h5"
    _write_raw_dgs(path)
    _rewrite_instrument_xml(path, '<instrument xmlns=', '<instrument name="HYSPEC" xmlns=')
    _rewrite_instrument_xml(path, 'z="-10"', 'z="-40.803"')
    _rewrite_instrument_xml(path, 'x="1" y="0" z="2"', 'x="2.7" y="0" z="3.6"')
    with h5py.File(path, "r+") as handle:
        entry = handle["entry"]
        for name, values in (("msd", [1803.]), ("psda", [0.])):
            entry["DASlogs"].create_group(name).create_dataset("value", data=values)
        setup = resolved_hyspec_preprocessing(entry, {}, 20., instrument_name="HYSPEC")
        upper = setup["raw_tof_bounds_microseconds"][1]
        entry["bank1_events/event_time_offset"][:] = [upper - 100., upper + 100.]
    group = raw_dgs_dataset_group([path])
    set_reduction_settings(group, {"t0_override": 0., "bad_pulse_threshold": 0., "energy_max_fraction": .99})
    options = {"vectors": np.eye(4), "num_bins": [2, 2, 2, 2],
               "lower": [-10., -10., -10., -19.], "upper": [10., 10., 10., 19.8]}
    cropped = bin_raw_dgs_group(group, **options)
    assert cropped.num_events.sum() == 1
    dataset = group.datasets[0]
    with dataset._raw_dgs_reduction_cache.open() as archive:
        header = json.loads(str(archive["header_json"].item()))
    assert header["hyspec_preprocessing"] == setup
    assert dataset.metadata["resolved_reduction"]["automatic_values"]["hyspec_tank_offset_degrees"] == 0.
    with monkeypatch.context() as context:
        context.setattr(raw_dgs, "inspect_raw_dgs_run", lambda *args, **kwargs: pytest.fail("cache reopened run"))
        cached = bin_raw_dgs_group(group, **options)
    np.testing.assert_array_equal(cached.signal, cropped.signal)
    assert cached.metadata["reduced_event_cache"]["hits"] == 1
    set_reduction_settings(group, {"hyspec_tof_crop": False})
    uncropped = bin_raw_dgs_group(group, **options)
    assert uncropped.num_events.sum() == 2
    assert uncropped.metadata["reduced_event_cache"]["misses"] == 1


def test_raw_detector_geometry_reuses_only_identical_xml(tmp_path, geometry_cache):
    first_path, second_path = (tmp_path / name for name in ("SEQ_42.nxs.h5", "SEQ_43.nxs.h5"))
    _write_raw_dgs(first_path, with_he3=True)
    _write_raw_dgs(second_path, with_he3=True)
    first = raw_dgs._detector_geometry(first_path)
    assert raw_dgs._detector_geometry(second_path) is first
    assert raw_dgs._detector_geometry_from_xml.cache_info().misses == 1
    for array in (first.detector_ids, first.positions, first.he3_exponents):
        assert not array.flags.writeable

    # A different instrument may reuse detector IDs. Its full definition,
    # including positions and efficiency parameters, must remain independent.
    _rewrite_instrument_xml(second_path, '<instrument xmlns=', '<instrument name="HYSPEC" xmlns=')
    _rewrite_instrument_xml(second_path, 'x="1" y="0" z="2"', 'x="2" y="0" z="2"')
    _rewrite_instrument_xml(second_path, 'val="10.0"', 'val="5.0"')
    second = raw_dgs._detector_geometry(second_path)
    assert second is not first
    np.testing.assert_array_equal(second.detector_ids, first.detector_ids)
    np.testing.assert_allclose(second.positions, [[2.0, 0.0, 2.0]])
    np.testing.assert_allclose(second.he3_exponents, first.he3_exponents / 2.0)
    assert raw_dgs._detector_geometry(first_path) is first


def test_raw_detector_geometry_detects_definition_edits(tmp_path, geometry_cache):
    path = tmp_path / "SEQ_42.nxs.h5"
    _write_raw_dgs(path)
    before = raw_dgs._detector_geometry(path)
    _rewrite_instrument_xml(path, 'x="1" y="0" z="2"', 'x="3" y="0" z="2"')
    after = raw_dgs._detector_geometry(path)
    assert after is not before
    np.testing.assert_allclose(before.positions, [[1.0, 0.0, 2.0]])
    np.testing.assert_allclose(after.positions, [[3.0, 0.0, 2.0]])


def test_raw_detector_geometry_cache_is_bounded(tmp_path, geometry_cache):
    path = tmp_path / "SEQ_42.nxs.h5"
    _write_raw_dgs(path)
    with pytest.importorskip("h5py").File(path, "r") as handle:
        xml = handle["entry/instrument/instrument_xml/data"][()].tobytes()
    for index in range(10):
        raw_dgs._detector_geometry_from_xml(xml.replace(b'x="1"', f'x="{index}"'.encode()))
    assert raw_dgs._detector_geometry_from_xml.cache_info().currsize == 8


def test_detector_geometry_lookup_preserves_rows_and_owns_immutable_arrays():
    positions = np.arange(9.0).reshape(3, 3)
    geometry = raw_dgs._DetectorGeometry(np.array([7, 2, 5]), positions, np.array([.7, .2, .5]))
    positions.fill(-1.0)
    ids = np.array([2, 7, 99, 2])
    found_positions, exponents, valid = geometry.event_geometry_for_ids(ids)
    np.testing.assert_array_equal(valid, [True, True, False, True])
    np.testing.assert_array_equal(found_positions, [[3, 4, 5], [0, 1, 2], [0, 0, 0], [3, 4, 5]])
    np.testing.assert_array_equal(exponents, [.2, .7, 0, .2])
    order = geometry._sorted_ids_and_order
    geometry.event_geometry_for_ids(ids)
    assert geometry._sorted_ids_and_order is order
    assert not any(array.flags.writeable for array in (*order, geometry.positions))


@pytest.mark.parametrize("powder", [False, True])
def test_raw_reduction_reads_metadata_once_per_run_but_refreshes_next_operation(tmp_path, monkeypatch, powder):
    source = tmp_path / "SEQ_42.nxs.h5"
    _write_raw_dgs(source)
    group = raw_dgs_dataset_group([source])
    original = raw_dgs.inspect_raw_dgs_run
    calls = []

    def inspect(path, **kwargs):
        calls.append(path)
        return original(path, **kwargs)

    monkeypatch.setattr(raw_dgs, "inspect_raw_dgs_run", inspect)
    options = (
        dict(lower=[0, -100], upper=[20, 40], num_bins=[1, 1], coordinate_mode="powder")
        if powder else dict(lower=[-10, -10, -10, -100], upper=[10, 10, 10, 40], num_bins=[1]*4)
    )
    first = bin_raw_dgs_group(group, **options)
    assert len(calls) == 1
    with pytest.importorskip("h5py").File(source, "r+") as handle:
        handle["entry/DASlogs/BL17:Det:TH:BL:Ei/average_value"][0] = 40.0
    second = bin_raw_dgs_group(group, **options)
    assert len(calls) == 2
    assert first.metadata["raw_dgs_energy_windows_meV"][0]["incident_energy_meV"] == 20.0
    assert second.metadata["raw_dgs_energy_windows_meV"][0]["incident_energy_meV"] == 40.0


def test_raw_dgs_metadata_and_streamed_hkle_binning(tmp_path):
    source = tmp_path / "SEQ_42.nxs.h5"
    _write_raw_dgs(source)
    info = inspect_raw_dgs_run(source)
    group = raw_dgs_dataset_group([source])

    assert info.run_number == "42"
    assert info.event_count == 2
    assert info.incident_energy == 20.0
    assert group.metadata["raw_dgs"]["energy_min_fraction"] == -0.95
    assert group.metadata["raw_dgs"]["energy_max_fraction"] == 0.95
    assert group.metadata["raw_dgs"]["q_modulus_bounds"][0] == 0.0
    assert group.metadata["raw_dgs"]["q_modulus_bounds"][1] > 0.0
    result = bin_raw_dgs_group(
        group,
        lower=[-10, -10, -10, -100],
        upper=[10, 10, 10, 20],
        num_bins=[1, 1, 1, 1],
        max_batch_bytes=128,
    )

    # The prompt event is rejected; the remaining event carries ki/kf and is
    # divided by the detector-trajectory normalization.
    assert result.num_events.item() == 1.0
    assert np.isfinite(result.signal.item()) and result.signal.item() > 0.0
    assert np.isfinite(result.errors.item()) and result.errors.item() > 0.0
    assert result.metadata["ki_kf_normalization"] is True
    assert result.metadata["raw_dgs_energy_windows_meV"][0]["minimum_meV"] == -19.0
    assert result.metadata["raw_dgs_energy_windows_meV"][0]["maximum_meV"] == 19.0
    assert not result.mask.item()


def test_raw_dgs_run_scale_and_weight_reuse_unscaled_events(tmp_path):
    source = tmp_path / "SEQ_42.nxs.h5"
    _write_raw_dgs(source)
    group = raw_dgs_dataset_group([source])
    options = dict(lower=[-10, -10, -10, -100], upper=[10, 10, 10, 20], num_bins=[1]*4)
    baseline = bin_raw_dgs_group(group, **options)
    run = group.datasets[0]
    run.scale_factor = 3
    run.fit_weight = 4
    scaled = bin_raw_dgs_group(group, **options)
    np.testing.assert_allclose(scaled.signal, baseline.signal * 3)
    np.testing.assert_allclose(scaled.errors, baseline.errors * 3)
    np.testing.assert_allclose(scaled.metadata["normalization_denominator"], baseline.metadata["normalization_denominator"] * 4)
    assert scaled.metadata["reduced_event_cache"] == {"hits": 1, "misses": 0}
    run.scale_factor = 1
    run.fit_weight = 1
    restored = bin_raw_dgs_group(group, **options)
    np.testing.assert_allclose(restored.signal, baseline.signal)
    np.testing.assert_allclose(restored.errors, baseline.errors)
    run.fit_weight = 0
    with pytest.raises(ValueError, match="positive fit weight"):
        bin_raw_dgs_group(group, **options)



def test_raw_dgs_relative_run_weights_and_disabled_runs(tmp_path):
    paths = [tmp_path / f"SEQ_{number}.nxs.h5" for number in (42, 43)]
    for path in paths:
        _write_raw_dgs(path)
    group = raw_dgs_dataset_group(paths)
    options = dict(lower=[-10, -10, -10, -100], upper=[10, 10, 10, 20], num_bins=[1] * 4)
    baseline = bin_raw_dgs_group(group, datasets=[group.datasets[0]], **options)
    group.datasets[1].scale_factor = 3
    group.datasets[1].fit_weight = 2
    result = bin_raw_dgs_group(group, **options)
    np.testing.assert_allclose(result.signal, baseline.signal * 7 / 3)
    np.testing.assert_allclose(result.errors, baseline.errors * np.sqrt(37) / 3)
    group.datasets[1].enabled = False
    result = bin_raw_dgs_group(group, **options)
    np.testing.assert_allclose(result.signal, baseline.signal)
    np.testing.assert_allclose(result.errors, baseline.errors)
    group.datasets[0].fit_weight = -1
    with pytest.raises(ValueError, match="nonnegative"):
        bin_raw_dgs_group(group, **options)

def test_raw_dgs_accumulates_repeated_events_in_one_sparse_bin(tmp_path):
    h5py = pytest.importorskip("h5py")
    source = tmp_path / "SEQ_42.nxs.h5"
    _write_raw_dgs(source)
    group = raw_dgs_dataset_group([source])
    options = dict(
        lower=[-10, -10, -10, -100],
        upper=[10, 10, 10, 20],
        num_bins=[1, 1, 1, 1],
    )
    one = bin_raw_dgs_group(group, **options)
    with h5py.File(source, "r+") as handle:
        handle["entry/bank1_events/event_time_offset"][1] = 9000.0
    two = bin_raw_dgs_group(group, **options)

    assert two.num_events.item() == 2.0
    np.testing.assert_allclose(two.signal, 2.0 * one.signal)
    np.testing.assert_allclose(two.errors, np.sqrt(2.0) * one.errors)


def test_raw_dgs_supports_one_nonuniform_axis_and_minimum_samples(tmp_path):
    source = tmp_path / "SEQ_42.nxs.h5"
    _write_raw_dgs(source)
    group = raw_dgs_dataset_group([source])

    result = bin_raw_dgs_group(
        group,
        lower=[-10, -10, -10, -100],
        upper=[10, 10, 10, 20],
        num_bins=[1, 1, 1, 1],
        bin_edges=[None, None, None, [-100.0, -10.0, 0.0, 20.0]],
        minimum_samples=2,
    )

    assert result.shape == (1, 1, 1, 3)
    np.testing.assert_allclose(result.axes[3].values, [-100.0, -10.0, 0.0, 20.0])
    assert np.all(result.mask)
    assert result.metadata["rebin"]["minimum_samples"] == 2.0


def test_raw_dgs_trajectory_normalization_keeps_each_runs_detector_geometry(monkeypatch, tmp_path):
    first_source = tmp_path / "SEQ_42.nxs.h5"
    second_source = tmp_path / "SEQ_43.nxs.h5"
    _write_raw_dgs(first_source)
    _write_raw_dgs(second_source)
    group = raw_dgs_dataset_group([first_source, second_source])
    geometries = {
        first_source: raw_dgs._DetectorGeometry(
            detector_ids=np.array([42]),
            positions=np.array([[1.0, 0.0, 0.0]]),
            he3_exponents=np.zeros(1),
        ),
        second_source: raw_dgs._DetectorGeometry(
            detector_ids=np.array([42]),
            positions=np.array([[0.0, 1.0, 0.0]]),
            he3_exponents=np.zeros(1),
        ),
    }
    directions = []
    energy_bounds = []

    monkeypatch.setattr(
        raw_dgs,
        "_detector_geometry",
        lambda path: geometries[path],
    )
    monkeypatch.setattr(raw_dgs, "_MDEVENT_NUMBA", None)
    monkeypatch.setattr(
        raw_dgs,
        "_accumulate_detector_trajectory",
        lambda result, edges, inverse, direction, ei, bounds, weight, **kwargs: (
            directions.append(direction.copy()),
            energy_bounds.append(bounds),
        ),
    )

    raw_dgs._trajectory_normalization(
        group,
        group.datasets,
        [np.array([-10.0, 10.0])] * 4,
        (1, 1, 1, 1),
        np.eye(4),
        None,
        None,
    )

    np.testing.assert_allclose(
        directions,
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )
    assert energy_bounds == [(-19.0, 19.0), (-19.0, 19.0)]


def test_raw_dgs_mismatched_detector_geometries_still_use_compiled_kernels(
    monkeypatch, tmp_path
):
    first_source = tmp_path / "SEQ_42.nxs.h5"
    second_source = tmp_path / "SEQ_43.nxs.h5"
    _write_raw_dgs(first_source)
    _write_raw_dgs(second_source)
    group = raw_dgs_dataset_group([first_source, second_source])
    geometries = {
        first_source: raw_dgs._DetectorGeometry(
            detector_ids=np.array([42]),
            positions=np.array([[1.0, 0.0, 0.0]]),
            he3_exponents=np.zeros(1),
        ),
        second_source: raw_dgs._DetectorGeometry(
            detector_ids=np.array([42]),
            positions=np.array([[0.0, 1.0, 0.0]]),
            he3_exponents=np.zeros(1),
        ),
    }
    calls = {"hkle": 0, "powder": 0}

    class Kernels:
        @staticmethod
        def run_trajectory_normalization(*args, workers):
            calls["hkle"] += 1
            return np.zeros(int(np.prod(args[-2])))

        @staticmethod
        def run_powder_trajectory_normalization(*args, workers):
            calls["powder"] += 1
            return np.zeros(int(np.prod(args[-2])))

    monkeypatch.setattr(raw_dgs, "_detector_geometry", lambda path: geometries[path])
    monkeypatch.setattr(raw_dgs, "_MDEVENT_NUMBA", Kernels)
    monkeypatch.setattr(raw_dgs, "_trajectory_worker_count", lambda _size: 1)
    monkeypatch.setattr(
        raw_dgs,
        "_accumulate_detector_trajectory",
        lambda *_args: pytest.fail("scalar HKLE normalization was used"),
    )
    monkeypatch.setattr(
        raw_dgs,
        "_accumulate_powder_detector_trajectory",
        lambda *_args: pytest.fail("scalar powder normalization was used"),
    )
    bounds = {dataset.id: (-19.0, 19.0) for dataset in group.datasets}

    raw_dgs._trajectory_normalization(
        group,
        group.datasets,
        [np.array([-10.0, 10.0])] * 4,
        (1, 1, 1, 1),
        np.eye(4),
        None,
        None,
        energy_bounds_by_dataset_id=bounds,
    )
    raw_dgs._powder_trajectory_normalization(
        group,
        group.datasets,
        [np.array([0.0, 10.0]), np.array([-19.0, 19.0])],
        (1, 1),
        None,
        None,
        bounds,
    )

    assert calls == {"hkle": 2, "powder": 2}


def test_raw_dgs_custom_energy_limits_apply_to_events_and_normalization(
    monkeypatch, tmp_path
):
    source = tmp_path / "SEQ_42.nxs.h5"
    _write_raw_dgs(source)
    group = raw_dgs_dataset_group([source])
    group.metadata["raw_dgs"]["energy_min_fraction"] = -0.25
    group.metadata["raw_dgs"]["energy_max_fraction"] = 0.9
    captured = {}

    def normalization(*args, **_kwargs):
        captured["bounds"] = args[-1]
        return np.ones(args[3])

    monkeypatch.setattr(raw_dgs, "_trajectory_normalization", normalization)
    result = bin_raw_dgs_group(
        group,
        lower=[-10, -10, -10, -100],
        upper=[10, 10, 10, 20],
        num_bins=[1, 1, 1, 1],
    )

    assert result.num_events.item() == 0.0
    assert captured["bounds"] == {group.datasets[0].id: (-5.0, 18.0)}


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"energy_min_fraction": 0.5, "energy_max_fraction": 0.5}, "below"),
        ({"energy_min_fraction": -0.95, "energy_max_fraction": 1.0}, "below 1 Ei"),
        ({"energy_min_fraction": np.nan, "energy_max_fraction": 0.95}, "finite"),
    ],
)
def test_raw_dgs_energy_limits_are_validated(config, message):
    with pytest.raises(ValueError, match=message):
        _energy_transfer_bounds(config, 20.0)


def test_raw_dgs_detector_mask_removes_events(tmp_path):
    source = tmp_path / "SEQ_42.nxs.h5"
    mask = tmp_path / "mask.nxs"
    _write_raw_dgs(source)
    h5py = pytest.importorskip("h5py")
    with h5py.File(mask, "w") as handle:
        entry = handle.create_group("mantid_workspace_1")
        detector = entry.create_group("instrument").create_group("detector")
        detector.create_dataset("detector_list", data=[42])
        workspace = entry.create_group("workspace")
        workspace.create_dataset("values", data=[[0.0]])
        workspace.create_dataset("errors", data=[[0.0]])
    group = raw_dgs_dataset_group([source], mask_path=mask)
    result = bin_raw_dgs_group(
        group, lower=[-10, -10, -10, -100], upper=[10, 10, 10, 20], num_bins=[1, 1, 1, 1]
    )
    assert result.mask.item()
    assert result.num_events.item() == 0.0


def test_raw_dgs_uses_processed_vanadium_as_trajectory_weight(tmp_path):
    source = tmp_path / "SEQ_42.nxs.h5"
    vanadium = tmp_path / "van.nxs"
    _write_raw_dgs(source)
    h5py = pytest.importorskip("h5py")
    with h5py.File(vanadium, "w") as handle:
        entry = handle.create_group("mantid_workspace_1")
        detector = entry.create_group("instrument").create_group("detector")
        detector.create_dataset("detector_list", data=[42])
        workspace = entry.create_group("workspace")
        workspace.create_dataset("values", data=[[2.0]])
        workspace.create_dataset("errors", data=[[0.0]])
    unnormalized_group = raw_dgs_dataset_group([source])
    group = raw_dgs_dataset_group([source], normalization_path=vanadium)

    unnormalized = bin_raw_dgs_group(
        unnormalized_group,
        lower=[-10, -10, -10, -100],
        upper=[10, 10, 10, 20],
        num_bins=[1, 1, 1, 1],
    )
    normalized = bin_raw_dgs_group(
        group,
        lower=[-10, -10, -10, -100],
        upper=[10, 10, 10, 20],
        num_bins=[1, 1, 1, 1],
    )
    assert normalized.signal.item() == pytest.approx(
        unnormalized.signal.item() / 2.0
    )
    assert normalized.errors.item() == pytest.approx(
        unnormalized.errors.item() / 2.0
    )
    assert normalized.metadata["normalization_denominator"].item() == pytest.approx(
        2.0 * unnormalized.metadata["normalization_denominator"].item()
    )


def test_raw_dgs_powder_binning_is_radial_and_uses_vanadium(tmp_path):
    source = tmp_path / "SEQ_42.nxs.h5"
    vanadium = tmp_path / "van.nxs"
    _write_raw_dgs(source)
    h5py = pytest.importorskip("h5py")
    with h5py.File(vanadium, "w") as handle:
        entry = handle.create_group("mantid_workspace_1")
        detector = entry.create_group("instrument").create_group("detector")
        detector.create_dataset("detector_list", data=[42])
        workspace = entry.create_group("workspace")
        workspace.create_dataset("values", data=[[2.0]])
        workspace.create_dataset("errors", data=[[0.0]])
    group = raw_dgs_dataset_group([source], normalization_path=vanadium)

    result = bin_raw_dgs_powder_group(
        group,
        lower=[0.0, -100.0],
        upper=[20.0, 20.0],
        num_bins=[1, 1],
    )

    assert result.shape == (1, 1)
    assert result.axes[0].name == "|Q|"
    assert result.axes[0].units == "1/angstrom"
    assert result.num_events.item() == 1.0
    assert np.isfinite(result.signal.item()) and result.signal.item() > 0.0
    assert result.metadata["signal_semantics_source"] == (
        "nfit_raw_tof_powder_reduction"
    )
    assert result.metadata["powder_reduction"]["coordinates"] == "|Q|,DeltaE"


def test_raw_dgs_powder_numba_matches_python_trajectory_path(monkeypatch, tmp_path):
    if raw_dgs._MDEVENT_NUMBA is None:
        pytest.skip("Numba trajectory extension is unavailable")
    source = tmp_path / "SEQ_42.nxs.h5"
    _write_raw_dgs(source)
    group = raw_dgs_dataset_group([source])
    settings = {
        "lower": [0.0, -19.0],
        "upper": [10.0, 19.0],
        "num_bins": [4, 3],
    }

    accelerated = bin_raw_dgs_powder_group(group, **settings)
    monkeypatch.setattr(raw_dgs, "_MDEVENT_NUMBA", None)
    python = bin_raw_dgs_powder_group(group, **settings)

    np.testing.assert_allclose(
        accelerated.metadata["normalization_denominator"],
        python.metadata["normalization_denominator"],
    )
    np.testing.assert_allclose(accelerated.signal, python.signal, equal_nan=True)
    np.testing.assert_allclose(accelerated.errors, python.errors, equal_nan=True)


def test_raw_dgs_uses_mantid_ki_over_kf_event_weight(monkeypatch, tmp_path):
    source = tmp_path / "SEQ_42.nxs.h5"
    _write_raw_dgs(source)
    group = raw_dgs_dataset_group([source])
    monkeypatch.setattr(raw_dgs, "_trajectory_normalization", lambda *args, **kwargs: np.ones(args[3]))

    result = bin_raw_dgs_group(
        group,
        lower=[-10, -10, -10, -100],
        upper=[10, 10, 10, 20],
        num_bins=[1, 1, 1, 1],
    )

    incident_tof = TOF_US_PER_M_SQRT_MEV * 10.0 / np.sqrt(20.0)
    final_tof = 9000.0 - incident_tof
    final_energy = (TOF_US_PER_M_SQRT_MEV * np.sqrt(5.0) / final_tof) ** 2
    expected = np.sqrt(20.0 / final_energy)
    assert result.signal.item() == pytest.approx(expected)
    assert result.errors.item() == pytest.approx(expected)


def test_raw_dgs_applies_mantid_he3_tube_efficiency(monkeypatch, tmp_path):
    source = tmp_path / "SEQ_42.nxs.h5"
    _write_raw_dgs(source, with_he3=True)
    group = raw_dgs_dataset_group([source])
    monkeypatch.setattr(raw_dgs, "_trajectory_normalization", lambda *args, **kwargs: np.ones(args[3]))

    result = bin_raw_dgs_group(
        group,
        lower=[-10, -10, -10, -100],
        upper=[10, 10, 10, 20],
        num_bins=[1, 1, 1, 1],
    )

    incident_tof = TOF_US_PER_M_SQRT_MEV * 10.0 / np.sqrt(20.0)
    final_tof = 9000.0 - incident_tof
    final_energy = (TOF_US_PER_M_SQRT_MEV * np.sqrt(5.0) / final_tof) ** 2
    kf = np.sqrt(final_energy / raw_dgs.ENERGY_TO_K2)
    exponent = (
        raw_dgs.HE3_EFFICIENCY_EXPONENTIAL_CONSTANT * (10.0 / 290.0) * (2.0 * (0.01 - 0.0008))
    )
    expected = np.sqrt(20.0 / final_energy) / (-np.expm1(-exponent * 2.0 * np.pi / kf))
    assert result.signal.item() == pytest.approx(expected)
    assert result.errors.item() == pytest.approx(expected)
    assert result.metadata["he3_detector_efficiency_correction"] is True


def test_raw_dgs_integrates_retained_pulse_charge_in_microampere_hours(tmp_path):
    source = tmp_path / "SEQ_42.nxs.h5"
    _write_raw_dgs(source)
    h5py = pytest.importorskip("h5py")
    with h5py.File(source, "a") as handle:
        log = handle["entry/DASlogs/proton_charge"]
        log.create_dataset("value", data=[100.0, 100.0, 10.0])

    with h5py.File(source, "r") as handle:
        charge = _retained_proton_charge_uah(handle["entry"], 95.0)
    assert charge == pytest.approx(200.0 / 3.6e9)


def test_raw_dgs_empty_observation_retains_zero_variance_and_exposure(monkeypatch, tmp_path):
    source = tmp_path / "SEQ_42.nxs.h5"
    _write_raw_dgs(source)
    group = raw_dgs_dataset_group([source])
    monkeypatch.setattr(raw_dgs, "_trajectory_normalization", lambda *args, **kwargs: np.full(args[3], 2.0))

    result = bin_raw_dgs_group(
        group,
        lower=[-10, -10, -10, 19],
        upper=[10, 10, 10, 20],
        num_bins=[1, 1, 1, 1],
    )

    assert result.num_events.item() == 0.0
    assert result.signal.item() == 0.0
    assert result.errors.item() == 0.0
    assert result.auxiliary_channels["event_signal_numerator"].values.item() == 0.0
    assert result.auxiliary_channels["event_variance_numerator"].values.item() == 0.0
    assert result.auxiliary_channels["normalization_denominator"].values.item() == 2.0
    assert not result.mask.item()


def test_monitor_fit_discovers_non_sequoia_monitor_names(tmp_path):
    h5py = pytest.importorskip("h5py")
    source = tmp_path / "CNCS_1.nxs.h5"
    energy, t0 = 20.0, 7.0
    distances = (8.0, 22.0)
    xml = """<instrument xmlns="http://www.mantidproject.org/IDF/1.0">
      <component type="moderator"><location z="-15"/></component>
      <component type="monitor"><location name="upstream_monitor" z="-7"/></component>
      <component type="monitor"><location name="downstream_monitor" z="7"/></component>
    </instrument>"""
    with h5py.File(source, "w") as handle:
        entry = handle.create_group("entry")
        entry.create_group("instrument").create_group("instrument_xml").create_dataset(
            "data",
            data=np.frombuffer(xml.encode(), dtype="u1"),
        )
        for name, distance in zip(
            ("upstream_monitor", "downstream_monitor"), distances, strict=True
        ):
            centre = t0 + TOF_US_PER_M_SQRT_MEV * distance / np.sqrt(energy)
            entry.create_group(name).create_dataset(
                "event_time_offset", data=np.repeat(centre, 200)
            )
    with h5py.File(source, "r") as handle:
        fitted_energy, fitted_t0 = _monitor_ei_t0(handle["entry"], energy)
    assert fitted_energy == pytest.approx(energy, rel=2e-3)
    assert fitted_t0 == pytest.approx(t0, abs=2.0)


def test_monitor_fit_uses_idf_order_and_log_driven_positions(tmp_path):
    h5py = pytest.importorskip("h5py")
    source = tmp_path / "HYSPEC_1.nxs.h5"
    energy, t0, msd = 12.0, 5.0, 6000.0
    # The locations intentionally have no names, like some Mantid IDFs. Their
    # z positions are resolved from the supplied run log.
    xml = """<instrument xmlns="http://www.mantidproject.org/IDF/1.0">
      <component type="moderator"><location z="-15"/></component>
      <component type="monitor"><location><parameter name="z"><logfile id="msd" eq="-0.001*value-3.0"/></parameter></location>
      <location><parameter name="z"><logfile id="msd" eq="-0.001*value+2.0"/></parameter></location></component>
    </instrument>"""
    distances = (6.0, 11.0)
    with h5py.File(source, "w") as handle:
        entry = handle.create_group("entry")
        entry.create_group("instrument").create_group("instrument_xml").create_dataset(
            "data",
            data=np.frombuffer(xml.encode(), dtype="u1"),
        )
        entry.create_group("DASlogs").create_group("msd").create_dataset(
            "average_value", data=[msd]
        )
        for name, distance in zip(
            ("monitor1", "monitor2"), distances, strict=True
        ):
            centre = t0 + TOF_US_PER_M_SQRT_MEV * distance / np.sqrt(energy)
            monitor = entry.create_group(name)
            monitor.attrs["NX_class"] = "NXmonitor"
            monitor.create_dataset("event_time_offset", data=np.repeat(centre, 200))
    with h5py.File(source, "r") as handle:
        fitted_energy, fitted_t0 = _monitor_ei_t0(handle["entry"], energy)
    assert fitted_energy == pytest.approx(energy, rel=2e-3)
    assert fitted_t0 == pytest.approx(t0, abs=2.0)


def test_mantid_cncs_t0_formula_uses_requested_incident_energy():
    energy = 12.0
    formula = "288.595706061-110.833514059/sqrt(incidentEnergy)+89.6080314589/incidentEnergy-42.6999684563*sqrt(incidentEnergy)+1.89672170078*incidentEnergy"
    expected = (
        288.595706061
        - 110.833514059 / np.sqrt(energy)
        + 89.6080314589 / energy
        - 42.6999684563 * np.sqrt(energy)
        + 1.89672170078 * energy
    )
    assert _evaluate_mantid_t0_formula(formula, energy) == pytest.approx(expected)


@pytest.mark.parametrize("energy", [3.8, 15., 31., 60.])
def test_mantid_hyspec_t0_power_syntax_and_automatic_provenance(tmp_path, energy):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "HYS_42.nxs.h5"
    _write_raw_dgs(path)
    _rewrite_instrument_xml(path, '<instrument xmlns=', '<instrument name="HYSPEC" xmlns=')
    with h5py.File(path, "r+") as handle:
        handle["entry/DASlogs/BL17:Det:TH:BL:Ei/average_value"][:] = energy
    expected = 4. + 107. / (1. + (energy / 31.)**3)
    assert _evaluate_mantid_t0_formula(raw_dgs.MANTID_T0_FORMULAS["HYSPEC"], energy) == expected
    info = inspect_raw_dgs_run(path)
    assert info.incident_energy == energy
    assert info.t0 == expected
    assert info.calibration_source == "instrument_t0_formula"
    assert info.calibration_warning is None
    if energy == 15.:
        # Recorded Mantid HYSPEC value, independent of a Mantid runtime.
        assert info.t0 == 100.11159018271724


@pytest.mark.parametrize("formula", ["unknown(incidentEnergy)", "1/0", "sqrt(-1)", "1e999", "("])
def test_failed_instrument_t0_formula_is_warned_and_recorded(tmp_path, formula):
    from xml.sax.saxutils import escape

    path = tmp_path / "bad_formula.nxs.h5"
    _write_raw_dgs(path)
    _rewrite_instrument_xml(path, '</instrument>',
        '<component-link name="instrument"><parameter name="t0_formula">'
        f'<value val="{escape(formula)}"/></parameter></component-link></instrument>')
    with pytest.warns(RuntimeWarning, match="instrument T0 formula could not be evaluated"):
        info = inspect_raw_dgs_run(path)
    assert info.calibration_source == "requested_energy_failed_t0_formula"
    assert "Set explicit Ei/T0 overrides" in info.calibration_warning
    assert info.incident_energy == 20.
    assert info.t0 == 0.


def test_raw_dgs_streamed_symmetry_matches_separate_operations(tmp_path):
    source = tmp_path / 'SEQ_42.nxs.h5'
    _write_raw_dgs(source)
    group = raw_dgs_dataset_group([source])
    options = dict(lower=[-10, -10, -10, -100], upper=[10, 10, 10, 20],
                   num_bins=[2, 2, 2, 3], max_batch_bytes=128)
    operations = [np.eye(3), -np.eye(3), np.diag([-1.0, 1.0, 1.0])]
    combined = bin_raw_dgs_group(group, symmetry_operations=operations, **options)
    separate = [bin_raw_dgs_group(group, symmetry_operations=[op], **options) for op in operations]
    np.testing.assert_array_equal(combined.num_events, sum(data.num_events for data in separate))
    denominator = sum(data.metadata['normalization_denominator'] for data in separate)
    np.testing.assert_allclose(combined.metadata['normalization_denominator'], denominator,
                               rtol=8*np.finfo(float).eps, atol=0)
    # Reconstruct summed counts, rather than averaging normalized intensities.
    counts = sum(np.nan_to_num(data.signal) * data.metadata['normalization_denominator']
                 for data in separate)
    valid = ~combined.mask
    np.testing.assert_allclose(combined.signal[valid], (counts / np.where(denominator > 0, denominator, 1))[valid],
                               rtol=8*np.finfo(float).eps, atol=0)


@pytest.mark.parametrize('mixed_geometry', [False, True])
def test_raw_trajectory_accumulator_reuses_worker_grids_across_batches(
    tmp_path, monkeypatch, mixed_geometry,
):
    from types import SimpleNamespace

    backend = raw_dgs._MDEVENT_NUMBA
    if backend is None:
        pytest.skip('Numba trajectory backend is unavailable')
    sources = [tmp_path / f'SEQ_{number}.nxs.h5' for number in (42, 43, 44)]
    for path in sources:
        _write_raw_dgs(path)
    group = raw_dgs_dataset_group(sources)
    one = raw_dgs._DetectorGeometry(np.array([42]), np.array([[1., 0., 2.]]), np.zeros(1))
    two = raw_dgs._DetectorGeometry(
        np.array([42, 43]), np.array([[2., 0., 1.], [0., 1., 2.]]), np.zeros(2),
    )
    monkeypatch.setattr(raw_dgs, '_detector_geometry', lambda path: two if mixed_geometry and path == sources[1] else one)
    monkeypatch.setattr(raw_dgs, '_trajectory_worker_count', lambda *_: 2)
    monkeypatch.setattr(raw_dgs, 'MDEVENT_TRAJECTORY_BATCH_TASKS', 1)
    options = (group, group.datasets, [np.linspace(-4, 4, 6)]*3+[np.linspace(-19, 19, 6)], (5,)*4, np.eye(4), None, None)
    # The compatibility path allocates and reduces a separate worker grid per batch.
    monkeypatch.setattr(raw_dgs, '_MDEVENT_NUMBA', SimpleNamespace(run_trajectory_normalization=backend.run_trajectory_normalization))
    expected = raw_dgs._trajectory_normalization(*options)
    created, accumulated = [], []

    def factory(*args, **kwargs):
        accumulator = backend.trajectory_normalization_accumulator(*args, **kwargs)
        created.append(accumulator)
        accumulate = accumulator.accumulate

        def record(*payload):
            accumulated.append(payload[0].copy())
            return accumulate(*payload)

        accumulator.accumulate = record
        return accumulator

    monkeypatch.setattr(raw_dgs, '_MDEVENT_NUMBA', SimpleNamespace(
        trajectory_normalization_accumulator=factory,
        run_trajectory_normalization=lambda *_args, **_kwargs: pytest.fail('one-shot normalization was used'),
    ))
    actual = raw_dgs._trajectory_normalization(*options)
    assert len(created) == 1
    assert len(accumulated) == 3
    assert [len(theta) for theta in accumulated] == ([1, 2, 1] if mixed_geometry else [1, 1, 1])
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("pause", [False, True])
def test_pulse_intervals_remove_deadtime_from_events_and_charge(tmp_path, pause):
    from nfit.raw_dgs_pulses import select_pulses

    h5py = pytest.importorskip("h5py")
    source = tmp_path / "pulses.nxs.h5"
    with h5py.File(source, "w") as f:
        entry = f.create_group("entry")
        logs = entry.create_group("DASlogs")
        charge = logs.create_group("proton_charge")
        charge.create_dataset("value", data=[100, 100, 0, 100, 100, 100])
        t = charge.create_dataset("time", data=np.arange(6.0))
        t.attrs.update(offset="2026-01-01T00:00:00Z", units="second")
        if pause:
            log = logs.create_group("pause")
            log.create_dataset("value", data=[0, 1, 0])
            t = log.create_dataset("time", data=[0, 3, 4])
            t.attrs.update(start="2026-01-01T00:00:00Z", units="second")
        bank = entry.create_group("bank1_events")
        t = bank.create_dataset("event_time_zero", data=np.arange(6.0) * 1e6)
        t.attrs.update(offset="2026-01-01T00:00:00Z", units="microsecond")
        selection = select_pulses(entry, 95)
        expected = [True, False, False, not pause, True, False]
        # Centred zero-tolerance intervals exclude the last good pulse in
        # each uninterrupted charge interval, plus the half-open run endpoint.
        np.testing.assert_array_equal(selection.charge_keep, expected)
        np.testing.assert_array_equal(selection.bank_keep(bank), expected)
        assert _retained_proton_charge_uah(entry, 95) == pytest.approx(sum(expected) * 100 / 3.6e9)


def test_monitor_half_height_width_and_histogram_rebin_conserve_counts():
    from nfit.raw_dgs_monitors import _mantid_getei_peak_region, _rebin_monitor_histogram

    x = np.arange(-50., 51.)
    y = 10000 * np.exp(-x**2 / (2 * 5**2)) + 10
    _, _, width = _mantid_getei_peak_region(x, y, np.sqrt(y))
    assert width == pytest.approx(2 * np.sqrt(2 * np.log(2)) * 5, rel=0.003)
    counts, errors = _rebin_monitor_histogram(np.array([4., 16.]),
        np.array([0., 1., 2.]), np.array([0., .5, 1.5, 2.]))
    np.testing.assert_allclose(counts, [2, 10, 8])
    np.testing.assert_allclose(errors**2, counts)


@pytest.mark.parametrize("unordered_charge", [False, True])
@pytest.mark.parametrize("unordered_pause", [False, True])
def test_backdated_pulse_logs_preserve_charge_pairing_and_event_intervals(
    tmp_path, unordered_charge, unordered_pause
):
    from nfit.raw_dgs_pulses import select_pulses

    h5py = pytest.importorskip("h5py")
    # Different good charges expose accidental pairing of sorted timestamps
    # with the file-order charge values. Equal timestamps retain file order.
    charge_times = np.array([0., 1., 2., 2., 3., 4., 5., 6.])
    charge_values = np.array([100., 99., 0., 98., 97., 101., 102., 100.])
    charge_order = np.array([0, 1, 4, 5, 2, 3, 6, 7]) if unordered_charge else np.arange(8)
    pause_order = np.array([0, 2, 1]) if unordered_pause else np.arange(3)
    with h5py.File(tmp_path / "backdated.nxs.h5", "w") as f:
        entry = f.create_group("entry")
        logs = entry.create_group("DASlogs")
        charge = logs.create_group("proton_charge")
        times = charge.create_dataset("time", data=charge_times)
        times.attrs.update(offset="2026-01-01T00:00:00Z", units="second")
        charge.create_dataset("value", data=charge_values)
        pause = logs.create_group("pause")
        times = pause.create_dataset("time", data=[0., 3., 4.])
        times.attrs.update(offset="2026-01-01T00:00:00Z", units="second")
        pause.create_dataset("value", data=[0., 1., 0.])
        bank = entry.create_group("bank1_events")
        times = bank.create_dataset("event_time_zero", data=np.arange(7.))
        times.attrs.update(offset="2026-01-01T00:00:00Z", units="second")
        ordered = select_pulses(entry, 95)
        expected_charge = _retained_proton_charge_uah(entry, 95)
        charge["time"][:] = charge_times[charge_order]
        charge["value"][:] = charge_values[charge_order]
        pause["time"][:] = np.array([0., 3., 4.])[pause_order]
        pause["value"][:] = np.array([0., 1., 0.])[pause_order]
        actual = select_pulses(entry, 95)
        np.testing.assert_array_equal(actual.intervals, ordered.intervals)
        np.testing.assert_array_equal(actual.charge_keep, ordered.charge_keep[charge_order])
        np.testing.assert_array_equal(actual.bank_keep(bank), ordered.bank_keep(bank))
        assert _retained_proton_charge_uah(entry, 95) == pytest.approx(expected_charge)


def test_monitor_derivative_error_cancels_shared_central_measurement():
    from nfit.raw_dgs_monitors import _peak_derivative_uncertainty

    # On a uniform grid, a centred difference is (next-previous)/(2*step).
    # A noisy central measurement does not contribute, even in an empty tail.
    assert _peak_derivative_uncertainty(0., np.sqrt(19. / .711234), 0., .711234, .711234) == 0.
    assert _peak_derivative_uncertainty(3., 1e12, 4., 2., 2.) == pytest.approx(1.25)
    # Unequal spacing gives a central coefficient: derivative coefficients are
    # (-1/2, 1/4, 1/4) for distances 1 and 2. Independent variances add.
    expected = np.sqrt((.5 * 2)**2 + (.25 * 3)**2 + (.25 * 4)**2)
    assert _peak_derivative_uncertainty(2., 3., 4., 1., 2.) == pytest.approx(expected)


@pytest.mark.parametrize("peak, reason", [
    (None, "fewer than two monitor peaks"),
    (100., "coincident flight times"),
])
def test_monitor_failed_peak_is_warned_and_recorded(tmp_path, monkeypatch, peak, reason):
    h5py = pytest.importorskip("h5py")
    source = tmp_path / "monitor_failure.nxs.h5"
    _write_raw_dgs(source)
    with h5py.File(source, "r+") as handle:
        group = handle["entry/instrument/instrument_xml"]
        xml = group["data"][()].tobytes().decode().replace(
            '<component type="moderator">',
            '<component type="monitor"><location name="monitor1" z="-2"/>'
            '<location name="monitor2" z="2"/></component><component type="moderator">',
        )
        del group["data"]
        group.create_dataset("data", data=np.frombuffer(xml.encode(), dtype="u1"))
        for name in ("monitor1", "monitor2"):
            handle["entry"].create_group(name).create_dataset("event_time_offset", data=[100.])
    monkeypatch.setattr(raw_dgs, "_mantid_getei_v2_peak", lambda *args, **kwargs: peak)
    with pytest.warns(RuntimeWarning, match="Set explicit Ei/T0 overrides"):
        info = inspect_raw_dgs_run(source)
    assert info.calibration_source == "requested_energy_failed_monitor_calibration"
    assert info.calibration_warning and reason in info.calibration_warning
    assert info.incident_energy == 20.
    assert info.t0 == 0.


def test_monitor_unavailable_calibration_has_explicit_provenance(tmp_path):
    source = tmp_path / "no_monitors.nxs.h5"
    _write_raw_dgs(source)
    info = inspect_raw_dgs_run(source)
    assert info.calibration_source == "requested_energy_no_usable_monitors"
    assert info.calibration_warning is None
    group = raw_dgs_dataset_group([source])
    assert group.datasets[0].metadata["calibration_source"] == info.calibration_source
    result = bin_raw_dgs_group(group, lower=[-10, -10, -10, -100],
        upper=[10, 10, 10, 20], num_bins=[2, 2, 2, 3])
    assert result.metadata["raw_dgs_calibration"][group.datasets[0].id]["source"] == info.calibration_source


def test_final_grid_measurement_replay_reuses_cache_and_preserves_policy(tmp_path):
    from nfit import replay_measurement_histogram
    source = tmp_path / "SEQ_42.nxs.h5"
    _write_raw_dgs(source)
    group = raw_dgs_dataset_group([source])
    kwargs = dict(lower=[-10, -10, -10, -100], upper=[10, 10, 10, 20], num_bins=[1]*4, max_batch_bytes=128)
    direct = bin_raw_dgs_group(group, **kwargs)
    replay = replay_measurement_histogram(group, **kwargs)
    np.testing.assert_array_equal(replay.signal, direct.signal)
    np.testing.assert_array_equal(replay.errors, direct.errors)
    assert replay.metadata["reduced_event_cache"]["hits"] == 1
    assert replay.metadata["dgs_reduction_policies"] == direct.metadata["dgs_reduction_policies"]
    assert replay.metadata["measurement_replay"]["normalizer"] == "known"
    with pytest.raises(ValueError, match="Repeated source"):
        replay_measurement_histogram(group, datasets=[group.datasets[0]]*2, **kwargs)
