"""Channel labels are presentation only; data identifiers stay stable."""
import pytest


def test_grouped_channels_preserve_identifiers_and_selection(monkeypatch):
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    QtWidgets = pytest.importorskip('PySide6.QtWidgets')
    from nfit.qt_channel_menu import ChannelComboBox, populate_channel_combo

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    combo = ChannelComboBox()
    calls = []
    combo.currentIndexChanged.connect(calls.append)
    channels = ['signal', 'errors', 'nfit_mask', 'background', 'unsubtracted',
                'fit', 'num_events', 'custom channel']
    populate_channel_combo(combo, channels, 'background')
    assert combo.currentData() == 'background'
    assert combo.currentText() == 'Background'
    assert calls == []
    assert [combo.itemData(i) for i in range(combo.count()) if combo.itemData(i)] == [
        'signal', 'unsubtracted', 'background', 'errors', 'custom channel',
        'fit', 'num_events', 'nfit_mask',
    ]
    for i in range(combo.count()):
        if combo.itemData(i) is None:
            assert not combo.model().item(i).isEnabled()
    combo.setCurrentText('errors')  # Existing scripted widget callers remain valid.
    assert combo.currentData() == 'errors'
    assert combo.currentText() == 'Standard uncertainty'
    populate_channel_combo(combo, ['signal', 'errors'], 'signal')
    assert combo.findData('background') == -1
    assert app is not None
    combo.close()


def test_point_channels_keep_user_labels(monkeypatch):
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    QtWidgets = pytest.importorskip('PySide6.QtWidgets')
    from nfit.qt_channel_menu import ChannelComboBox, populate_channel_combo

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    combo = ChannelComboBox()
    populate_channel_combo(combo, ['Moment', 'fit'], 'Moment', point_list=True)
    assert [combo.itemText(i) for i in range(combo.count())] == ['Moment', 'fit']
    combo.setCurrentIndex(1)
    assert combo.currentData() == 'fit'
    assert app is not None
    combo.close()
