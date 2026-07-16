from __future__ import annotations

from nfit.fit_scripts import fit_state_script


def test_fit_state_script_is_backend_only_and_editable(tmp_path):
    script = fit_state_script(
        project_path=tmp_path / "sample.nfit",
        group_name="Workspace",
        fit_id="a" * 32,
    )
    assert "PySide" not in script
    assert "RUN_FIT = False" in script
    assert "restore_data_group_state" in script
    compile(script, "fit_state.py", "exec")
