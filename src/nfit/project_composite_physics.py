"""Qt presentation of the shared INS controls for a composite recipe."""

from .composite_spectral import common_source_value, composite_spectral_config
from .pipeline import DatasetEntry
from .project_composites import _composite_candidates
from .spectral_channels import SPECTRAL_CHANNEL_CONFIG_KEY, updated_spectral_channel_config


def composite_physics_panel(window, group, recipe):
    from PySide6 import QtWidgets

    from .project_dataset_panels import _dataset_spectral_channels_group_box

    sources = _composite_candidates(group)
    if sources and any(d.data_type not in {"", "single_crystal_inelastic", "powder_inelastic"} for d in sources):
        return None
    config = composite_spectral_config(recipe, sources)
    temperature = common_source_value(sources, ("temperature",), point_temperature=True)
    entry = DatasetEntry("Composite", None, parameters={"temperature": temperature})
    page = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(page)
    editor = QtWidgets.QLineEdit(str(config.get("temperature_K") or ""))
    editor.setObjectName("composite_temperature")
    editor.setPlaceholderText(f"Automatic: {temperature:g} K" if temperature else "Automatic from temperature axis or agreeing sources")
    editor.setToolTip(
        "Optional nominal temperature in kelvin for a single-temperature composite. "
        "A temperature metadata dimension in K takes precedence. Leave blank for automatic resolution."
    )

    def changed(_dataset, _group, key, value):
        config.update(updated_spectral_channel_config(config, key, value))
        recipe[SPECTRAL_CHANNEL_CONFIG_KEY] = dict(config)
        window._after_group_composite_changed(group)
        tabs = window.window.findChild(QtWidgets.QTabWidget, "group_composite_tabs")
        if tabs is not None:
            for index in range(tabs.count()):
                if tabs.tabText(index) == "Physics":
                    tabs.setCurrentIndex(index)
                    break

    editor.editingFinished.connect(lambda: changed(
        None, None, "temperature_K", float(editor.text()) if editor.text().strip() else None
    ))
    row = QtWidgets.QFormLayout()
    row.addRow("Nominal temperature (K)", editor)
    layout.addLayout(row)
    layout.addWidget(_dataset_spectral_channels_group_box(
        window, entry, group, config_override=config, setting_changed=changed,
        status_override=(
            "Uses each temperature coordinate when a metadata dimension in K is present; "
            "otherwise requires agreeing source temperatures or an explicit nominal temperature. "
            "Fixed Ei/Ef must describe all sources when k_f/k_i is included."
        ),
    ))
    layout.addStretch(1)
    return page
