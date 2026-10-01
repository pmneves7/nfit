"""Channel labels are presentation only; data identifiers stay stable."""

import pytest


def test_grouped_channels_preserve_identifiers_and_selection(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from nfit.qt_channel_menu import ChannelComboBox, populate_channel_combo

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    combo = ChannelComboBox()
    calls = []
    combo.currentIndexChanged.connect(calls.append)
    channels = [
        "signal",
        "errors",
        "nfit_mask",
        "background",
        "unsubtracted",
        "fit",
        "num_events",
        "custom channel",
    ]
    populate_channel_combo(combo, channels, "background")
    assert combo.currentData() == "background"
    assert combo.currentText() == "Background"
    assert calls == []
    assert [combo.itemData(i) for i in range(combo.count()) if combo.itemData(i)] == [
        "signal",
        "unsubtracted",
        "background",
        "errors",
        "custom channel",
        "fit",
        "num_events",
        "nfit_mask",
    ]
    for i in range(combo.count()):
        if combo.itemData(i) is None:
            assert not combo.model().item(i).isEnabled()
    combo.setCurrentText("errors")  # Existing scripted widget callers remain valid.
    assert combo.currentData() == "errors"
    assert combo.currentText() == "Standard uncertainty"
    populate_channel_combo(combo, ["signal", "errors"], "signal")
    assert combo.findData("background") == -1
    assert app is not None
    combo.close()


def test_point_channels_keep_user_labels(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from nfit.qt_channel_menu import ChannelComboBox, populate_channel_combo

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    combo = ChannelComboBox()
    populate_channel_combo(combo, ["Moment", "fit"], "Moment", point_list=True)
    assert [combo.itemText(i) for i in range(combo.count())] == ["Moment", "fit"]
    combo.setCurrentIndex(1)
    assert combo.currentData() == "fit"
    assert app is not None
    combo.close()


def test_event_statistics_channels_have_specific_labels_and_tooltips(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from PySide6 import QtCore

    from nfit.histogram_statistics import EVENT_SIGNAL_NUMERATOR, EVENT_VARIANCE_NUMERATOR
    from nfit.qt_channel_menu import ChannelComboBox, populate_channel_combo

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    combo = ChannelComboBox()
    channels = [
        "signal",
        "errors",
        "num_events",
        "normalization_denominator",
        EVENT_SIGNAL_NUMERATOR,
        EVENT_VARIANCE_NUMERATOR,
    ]
    populate_channel_combo(combo, channels, "errors", event_statistics=True)
    assert combo.currentText() == "Observed event standard error"
    for name, text in (
        ("num_events", "Event contributions"),
        ("normalization_denominator", "Exposure"),
        (EVENT_VARIANCE_NUMERATOR, "Event numerator variance"),
    ):
        assert combo.itemText(combo.findData(name)) == text
    tooltip = combo.itemData(combo.findData("errors"), QtCore.Qt.ItemDataRole.ToolTipRole)
    assert "not a confidence interval" in tooltip
    tooltip = combo.itemData(combo.findData("num_events"), QtCore.Qt.ItemDataRole.ToolTipRole)
    assert "Symmetry copies" in tooltip
    assert any(combo.itemText(i) == "Event statistics" for i in range(combo.count()))
    assert app is not None
    combo.close()


def test_event_contract_does_not_change_point_list_menu(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from nfit.qt_channel_menu import ChannelComboBox, populate_channel_combo

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    combo = ChannelComboBox()
    populate_channel_combo(
        combo,
        ["num_events", "errors"],
        "errors",
        point_list=True,
        event_statistics=True,
        labels={"errors": "Override"},
    )
    assert [combo.itemText(i) for i in range(combo.count())] == ["num_events", "errors"]
    assert app is not None
    combo.close()


def test_viewer_menu_passes_count_and_histogram_cell_semantics(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from types import SimpleNamespace

    from PySide6 import QtCore

    from nfit.plotting_core import MDHistoSliceViewer
    from nfit.qt_channel_menu import ChannelComboBox
    from nfit.qt_slice_viewer import QtMDHistoSliceViewer
    from tests.test_histogram_statistics import histogram

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    data = histogram([[4.0, 0.0]], [[4.0, 0.0]], [[1.0, 100.0]])
    combo = ChannelComboBox()
    for semantics, label in (
        (None, "Event contributions"),
        ("contributing_histogram_cells", "Contributing histogram cells"),
    ):
        current = data.with_updates(metadata=dict(data.metadata, num_events_semantics=semantics))
        model = MDHistoSliceViewer(current, x_dim=1, y_dim=0)
        context = SimpleNamespace(model=model, data=current, channel_combo=combo)
        QtMDHistoSliceViewer._sync_channel_combo(context)
        assert combo.itemText(combo.findData("num_events")) == label
        assert combo.itemText(combo.findData("normalization_denominator")) == "Exposure"
        assert combo.itemText(combo.findData("errors")) == "Observed event standard error"
        if semantics:
            tooltip = combo.itemData(
                combo.findData("num_events"), QtCore.Qt.ItemDataRole.ToolTipRole
            )
            assert "source histogram cells" in tooltip
    assert app is not None
    combo.close()
