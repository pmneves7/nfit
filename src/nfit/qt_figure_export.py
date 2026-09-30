"""File selection and error presentation for Matplotlib figure exports."""

from pathlib import Path

from PySide6 import QtWidgets

from .figure_export import figure_export_filters, save_figure
from .file_dialogs import get_save_file_name


def prompt_save_figure(parent, figure) -> Path | None:
    """Save the displayed figure at the public export API's default resolution."""
    filters = figure_export_filters(figure)
    path, selected_filter = get_save_file_name(
        parent, "Save figure", "nfit_figure", ";;".join(filters),
        require_selected_filter=True,
    )
    if not path:
        return None
    destination = Path(path).expanduser()
    if not destination.suffix:
        destination = destination.with_suffix(filters.get(selected_filter, ".png"))
        # The chooser confirmed the entered name, not the appended filename.
        if destination.exists() and QtWidgets.QMessageBox.question(
            parent, "Replace figure?", f"Replace the existing file {destination}?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        ) != QtWidgets.QMessageBox.StandardButton.Yes:
            return None
    try:
        return save_figure(figure, destination)
    except (OSError, ValueError, RuntimeError) as exc:
        QtWidgets.QMessageBox.warning(parent, "Save figure", f"Could not save figure:\n{exc}")
        return None
