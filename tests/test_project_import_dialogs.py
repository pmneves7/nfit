from PySide6 import QtCore, QtWidgets

from nfit import project_import_dialogs


def test_import_choice_is_visible_and_raised_before_modal_loop(monkeypatch):
    monkeypatch.setattr(project_import_dialogs.sys, "platform", "linux")
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    parent = QtWidgets.QWidget()
    parent.show()
    inspected = []

    def accept_after_inspection(dialog):
        choice = dialog.findChild(QtWidgets.QComboBox, "import_data_type")
        buttons = dialog.findChild(QtWidgets.QDialogButtonBox)
        assert dialog.isVisible()
        assert dialog.windowModality() == QtCore.Qt.WindowModality.WindowModal
        assert dialog.windowFlags() & QtCore.Qt.WindowType.WindowStaysOnTopHint
        assert choice is not None
        assert choice.toolTip()
        assert buttons is not None
        for button in (
            QtWidgets.QDialogButtonBox.StandardButton.Ok,
            QtWidgets.QDialogButtonBox.StandardButton.Cancel,
        ):
            assert buttons.button(button).toolTip()
        choice.setCurrentIndex(1)
        inspected.append(True)
        return QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr(QtWidgets.QDialog, "exec", accept_after_inspection)
    selected = project_import_dialogs.prompt_import_choice(
        parent,
        title="Data type",
        prompt="What type of data is this?",
        choices=[("Powder", "powder"), ("Single crystal", "single_crystal")],
        object_name="import_data_type",
    )

    assert selected == "single_crystal"
    assert inspected == [True]
    parent.close()
    application.processEvents()
