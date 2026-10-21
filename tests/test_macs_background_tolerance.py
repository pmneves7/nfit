"""Per-link MACS energy matching survives caches, saved projects and scripts."""
import numpy as np
import pytest

from nfit import NfitProject, composite_dataset_data, load_project, save_project
from nfit.project_composites import (
    _composite_background_signature,
    _composite_cache_signature,
    _composite_scope,
)
from nfit.workflow import composite_workflow_script
from tests.test_macs_background_integration import macs_groups as _macs_groups


@pytest.fixture
def macs_groups(tmp_path):
    return _macs_groups.__wrapped__(tmp_path)


@pytest.mark.parametrize('projection', ['measured_events', 'macs_isotropic'])
def test_explicit_energy_tolerance_round_trip(macs_groups, tmp_path, projection):
    root, sample, dummy, link, config = macs_groups
    dummy.datasets[0].metadata['reduction_overrides'] = {'incident_energy_override': 3.535}
    link.projection = projection
    link.interpolation = 'nearest'
    link.metadata['q_bin_width'] = .3
    for axis in config['axes'][:3]:
        axis.update(lower=-4., upper=4., num_bins=31)
    config['axes'][3].update(lower=-.3, upper=-.1)
    scope = _composite_scope(root, sample)
    result_signature = _composite_cache_signature(scope, config_override=config)
    background_signature = _composite_background_signature(scope, link, config)
    with pytest.raises(ValueError, match='within 0.02 meV'):
        composite_dataset_data(root, node=sample, config_override=config)
    link.metadata['energy_tolerance_meV'] = .05
    assert _composite_cache_signature(scope, config_override=config) != result_signature
    assert _composite_background_signature(scope, link, config) != background_signature
    expected = composite_dataset_data(root, node=sample, config_override=config)
    assert np.any(~expected.mask)
    provenance = expected.metadata['background_subtractions'][-1]['projection']
    assert provenance['energy_tolerance_meV'] == .05
    project = NfitProject([root])
    namespace = {'__name__': 'standalone'}
    exec(compile(composite_workflow_script(project, root.name, node_id=sample.id),
                 'macs_tolerance.py', 'exec'), namespace)
    replayed = namespace['run']()
    path = tmp_path / 'tolerance.nfit'
    save_project(project, path)
    restored_root = load_project(path).data_groups[0]
    restored = composite_dataset_data(restored_root, node=restored_root.subgroups[0], config_override=config)
    assert restored_root.subgroups[0].backgrounds[0].metadata['energy_tolerance_meV'] == .05
    for actual in (replayed, restored):
        np.testing.assert_allclose(actual.signal, expected.signal, equal_nan=True)
        np.testing.assert_allclose(actual.errors, expected.errors, equal_nan=True)
        np.testing.assert_array_equal(actual.mask, expected.mask)
    link.metadata['energy_tolerance_meV'] = .03
    with pytest.raises(ValueError, match='within 0.03 meV'):
        composite_dataset_data(root, node=sample, config_override=config)


@pytest.mark.parametrize('value', [-.01, np.nan, np.inf])
def test_invalid_energy_tolerance_rejected(macs_groups, value):
    root, sample, _, link, config = macs_groups
    link.metadata['energy_tolerance_meV'] = value
    with pytest.raises(ValueError, match='finite and nonnegative'):
        composite_dataset_data(root, node=sample, config_override=config)


def test_energy_tolerance_gui_control(macs_groups, monkeypatch):
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    widgets = pytest.importorskip('PySide6.QtWidgets')
    from nfit.project_gui import NfitProjectExplorer

    root, sample, _, link, _ = macs_groups
    link.metadata['energy_tolerance_meV'] = .05
    explorer = NfitProjectExplorer(NfitProject([root]))
    calls = []
    monkeypatch.setattr(explorer, '_update_background', lambda *args, **kwargs: calls.append(kwargs))
    explorer._set_background_details(root, sample, link)
    control = explorer.details_widget.findChild(widgets.QDoubleSpinBox, 'background_energy_tolerance')
    assert control.value() == .05
    assert 'without shifting' in control.toolTip()
    control.setValue(.06)
    control.editingFinished.emit()
    assert calls[-1]['metadata']['energy_tolerance_meV'] == .06
    explorer.window.close()
