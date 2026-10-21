"""Reference tests for the DAVE-style angle-integrated calibration."""
import json

import h5py
import numpy as np
import pytest

from nfit.macs_vanadium import calculate_macs_vanadium


def _scan(path, *, kidney=0, fixed=5, final=True, weak=0.5):
    x = np.linspace(80, 100, 51)
    monitor = np.linspace(1e5, 3e5, len(x))
    y = 0.001 + 0.02 * np.exp(-0.5 * ((x - 90) / 1.3)**2)
    efficiencies = np.ones(20)
    efficiencies[0] = weak
    with h5py.File(path, 'w') as f:
        e = f.create_group('entry')
        e.create_dataset('DAS_logs/experiment/instrument', data=np.bytes_('MACS'))
        e.create_dataset('DAS_logs/specDetector/counts', data=monitor[:, None] * y[:, None] * efficiencies)
        e.create_dataset('DAS_logs/counter/liveMonitor', data=monitor)
        e.create_dataset('DAS_logs/kidneyMotor/softPosition', data=kidney)
        e.create_dataset('DAS_logs/ei/energy', data=fixed if final else np.linspace(4, 6, len(x)))
        e.create_dataset('DAS_logs/ef/energy', data=np.linspace(4, 6, len(x)) if final else fixed)
        if final:
            for j in range(1, 21):
                e.create_dataset(f'DAS_logs/anaTwoTheta{j:02d}/softPosition', data=x)
        else:
            e.create_dataset('DAS_logs/monoTwoTheta/softPosition', data=x)
    return path


def test_inverse_area_monitor_normalization_and_multi_kidney(tmp_path):
    a = _scan(tmp_path/'a.nxs', kidney=0)
    b = _scan(tmp_path/'b.nxs', kidney=40)
    result = calculate_macs_vanadium([a, b])
    assert result['missing_channels'] == []
    factors = np.asarray(result['factors'])
    assert factors[0] / factors[1] == pytest.approx(2)
    assert np.mean(factors) == pytest.approx(1)
    fit = result['diagnostics'][0]['channels'][0]
    assert fit['sigma_deg'] == pytest.approx(1.3)
    assert fit['area'] == pytest.approx(np.sqrt(2*np.pi)*.01*1.3)
    json.dumps(result, allow_nan=False)


def test_missing_channels_are_explicit_and_mask_is_inclusive(tmp_path):
    result = calculate_macs_vanadium([_scan(tmp_path/'a.nxs', kidney=1)], 15)
    # Detector 9: A4 = 1 - 76 + 8*8 = -11; detector 12: A4 = 13.
    assert result['factors'][9] is None
    assert result['usable_channels'][9] is False
    boundary = calculate_macs_vanadium([_scan(tmp_path/'b.nxs', kidney=3)], 15)
    assert boundary['factors'][11] is None  # A4 = 15 deg


def test_scan_compatibility_and_validation(tmp_path):
    a = _scan(tmp_path/'a.nxs')
    b = _scan(tmp_path/'b.nxs', fixed=6)
    with pytest.raises(ValueError, match='same fixed energy'):
        calculate_macs_vanadium([a, b])
    c = _scan(tmp_path/'c.nxs', final=False)
    with pytest.raises(ValueError, match='same energy axis'):
        calculate_macs_vanadium([a, c])
    for mask in (-1, np.nan, 181):
        with pytest.raises(ValueError, match='mask'):
            calculate_macs_vanadium([a], mask)
    with pytest.raises(ValueError, match='At least one'):
        calculate_macs_vanadium([])


def test_cache_is_independent_and_invalidates_changed_file(tmp_path):
    path = _scan(tmp_path/'a.nxs')
    result = calculate_macs_vanadium([path])
    result['factors'][0] = 100
    assert calculate_macs_vanadium([path])['factors'][0] != 100
    _scan(path, weak=.25)
    updated = calculate_macs_vanadium([path])
    assert updated['factors'][0] / updated['factors'][1] == pytest.approx(4)


def test_flat_and_uncovered_scans_fail_without_unity_fallback(tmp_path):
    path = _scan(tmp_path/'a.nxs')
    with h5py.File(path, 'a') as f:
        mon = f['entry/DAS_logs/counter/liveMonitor'][()]
        f['entry/DAS_logs/specDetector/counts'][...] = mon[:, None] * np.ones((len(mon), 20))
    with pytest.raises(ValueError, match='No usable'):
        calculate_macs_vanadium([path])


def test_services_are_gui_independent_and_geometry_reexports_are_identical():
    import ast
    import inspect

    from nfit import macs, macs_geometry, macs_vanadium

    tree = ast.parse(inspect.getsource(macs_vanadium))
    imported = [node.module or '' for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any('qt' in name.lower() or name == 'macs' or name.startswith('project_') for name in imported)
    assert macs.MACS_DETECTOR_OFFSETS_DEG is macs_geometry.MACS_DETECTOR_OFFSETS_DEG


def test_lightweight_provenance_includes_missing_sources(tmp_path):
    from nfit.macs_vanadium import vanadium_source_provenance

    path = _scan(tmp_path/'a.nxs')
    records = vanadium_source_provenance([path, tmp_path/'missing.nxs'])
    assert records[0] == calculate_macs_vanadium([path])['source_files'][0]
    assert records[1] == {'path': str(tmp_path/'missing.nxs'), 'available': False}


def test_fixed_motor_jitter_and_mean_kidney_mask(tmp_path):
    path = _scan(tmp_path/'jitter.nxs', kidney=3)
    with h5py.File(path, 'a') as f:
        e = f['entry']
        e.create_dataset('DAS_logs/monoTwoTheta/primaryNode', data=np.linspace(89, 89.018, 51))
        del e['DAS_logs/kidneyMotor/softPosition']
        e.create_dataset('DAS_logs/kidneyMotor/softPosition', data=np.linspace(2, 4, 51))
    result = calculate_macs_vanadium([path])
    assert result['sweep'] == 'final'
    assert result['factors'][11] is None  # Mean A4 = 15 deg: entire channel excluded.
    assert result['diagnostics'][0]['channels'][11]['points'] == 0
    assert result['diagnostics'][0]['channels'][11]['status'] == 'direct_beam_excluded'
    assert result['diagnostics'][0]['channels'][11]['mean_a4_deg'] == 15


def test_zero_count_uncertainties_use_common_monitor_reference(monkeypatch):
    from nfit import macs_vanadium

    captured = []

    def fit(function, x, y, **kwargs):
        captured.append(kwargs['sigma'])
        return np.array([2., 0., .1, 0.]), np.eye(4)

    monkeypatch.setattr(macs_vanadium, 'curve_fit', fit)
    x = np.linspace(-1, 1, 7)
    counts = np.array([0., 1., 2., 10., 2., 1., 0.])
    monitor = np.arange(1, 8) * 100.
    result = macs_vanadium._fit(x, counts, monitor, monitor_reference=200.)
    assert result['status'] == 'ok'
    np.testing.assert_allclose(captured[0], [1/200, 1/200, np.sqrt(2)/300,
                                          np.sqrt(10)/400, np.sqrt(2)/500, 1/600, 1/200])


def test_monitor_reference_is_first_valid_monitor_of_first_file(tmp_path):
    first = _scan(tmp_path/'first.nxs')
    second = _scan(tmp_path/'second.nxs', kidney=40)
    with h5py.File(first, 'a') as f:
        f['entry/DAS_logs/counter/liveMonitor'][0] = 0
    record = calculate_macs_vanadium([first, second])
    assert record['monitor_reference'] == 104000.


def test_integrated_area_not_peak_height_and_average_areas_before_inversion(tmp_path):
    first = _scan(tmp_path/'first.nxs', kidney=0, weak=.5)
    second = _scan(tmp_path/'second.nxs', kidney=40, weak=.25)
    combined = calculate_macs_vanadium([first, second])
    # Mean channel-1 response is (.5 + .25)/2 of channel 2.
    assert combined['factors'][0] / combined['factors'][1] == pytest.approx(8/3)
    for path in (first, second):
        with h5py.File(path, 'a') as f:
            e = f['entry/DAS_logs']
            x = e['anaTwoTheta01/softPosition'][()]
            monitor = e['counter/liveMonitor'][()]
            # Twice the peak height and half the width give the same area.
            e['specDetector/counts'][:, 0] = monitor * (
                .001 + .04 * np.exp(-.5 * ((x - 90) / .65)**2))
    combined = calculate_macs_vanadium([first, second])
    assert combined['factors'][0] == pytest.approx(combined['factors'][1])
