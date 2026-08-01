from pathlib import Path

import tomllib


def _project_metadata() -> dict:
    root = Path(__file__).resolve().parents[1]
    return tomllib.loads((root / "pyproject.toml").read_text())["project"]


def test_distribution_name_and_gui_entry_point_are_pip_ready():
    project = _project_metadata()

    assert project["name"] == "nfit"
    assert project["scripts"]["nfit"] == "nfit.project_gui:main"
    assert "pyside6>=6.6" in project["dependencies"]
    assert "ase>=3.23" in project["dependencies"]


def test_release_metadata_has_author_license_and_urls():
    project = _project_metadata()

    assert project["version"] == "0.62.0"
    assert project["authors"] == [{"name": "Paul M. Neves", "email": "pneves1@jhu.edu"}]
    assert project["license"] == {"text": "MIT"}
    assert project["urls"]["Repository"] == "https://github.com/pmneves7/nfit"
