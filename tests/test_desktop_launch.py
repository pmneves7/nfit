"""System desktop programs must not use the frozen application's libraries."""

import ast
import os
from pathlib import Path

import pytest

from nfit import desktop_launch


@pytest.mark.parametrize("original", [None, "", "/system/lib"])
def test_frozen_child_environment_restores_library_path_without_mutating_parent(
    monkeypatch, tmp_path, original
):
    bundle = tmp_path / "_internal"
    monkeypatch.setattr(desktop_launch.sys, "_MEIPASS", str(bundle), raising=False)
    monkeypatch.setenv("LD_LIBRARY_PATH", str(bundle))
    monkeypatch.setenv("LIBPATH", str(bundle))
    monkeypatch.delenv("LD_LIBRARY_PATH_ORIG", raising=False)
    if original is not None:
        monkeypatch.setenv("LD_LIBRARY_PATH_ORIG", original)
    monkeypatch.setenv("DISPLAY", ":7")
    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "unix:path=/session/bus")
    before = dict(os.environ)

    child = desktop_launch.system_process_environment()

    assert child.get("LD_LIBRARY_PATH") == original
    assert "LIBPATH" not in child
    assert child["DISPLAY"] == ":7"
    assert child["DBUS_SESSION_BUS_ADDRESS"] == "unix:path=/session/bus"
    assert dict(os.environ) == before


def test_child_environment_removes_only_bundle_owned_discovery_paths(monkeypatch, tmp_path):
    bundle = tmp_path / "_internal"
    monkeypatch.setattr(desktop_launch.sys, "_MEIPASS", str(bundle), raising=False)
    keys = ("PATH", "DYLD_LIBRARY_PATH", "QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH",
            "GIO_EXTRA_MODULES", "GI_TYPELIB_PATH", "GTK_PATH")
    external = f"{bundle}-other/bin"
    for key in keys:
        monkeypatch.setenv(key, os.pathsep.join((str(bundle / "plugins"), external, "/usr/bin")))
    monkeypatch.setenv("GIO_MODULE_DIR", str(bundle / "gio"))
    monkeypatch.setenv("GDK_PIXBUF_MODULE_FILE", "/user/pixbuf/loaders.cache")
    child = desktop_launch.system_process_environment()
    for key in keys:
        assert child[key] == os.pathsep.join((external, "/usr/bin"))
    assert "GIO_MODULE_DIR" not in child
    assert child["GDK_PIXBUF_MODULE_FILE"] == "/user/pixbuf/loaders.cache"


def test_unfrozen_environment_preserves_user_library_paths(monkeypatch):
    monkeypatch.delattr(desktop_launch.sys, "_MEIPASS", raising=False)
    monkeypatch.setattr(desktop_launch.sys, "frozen", False, raising=False)
    monkeypatch.delenv("LD_LIBRARY_PATH_ORIG", raising=False)
    monkeypatch.delenv("LIBPATH_ORIG", raising=False)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/user/lib")
    child = desktop_launch.system_process_environment()
    assert child == dict(os.environ)
    assert child is not os.environ


@pytest.mark.parametrize("platform,opener", [("linux", "xdg-open"), ("darwin", "open")])
def test_local_document_uses_default_opener_sanitized_env_and_no_shell(
    monkeypatch, tmp_path, platform, opener
):
    document = tmp_path / "space ; $(unsafe) # résumé.html"
    document.write_text("help")
    monkeypatch.setattr(desktop_launch.sys, "platform", platform)
    monkeypatch.setattr(desktop_launch.sys, "_MEIPASS", str(tmp_path / "bundle"), raising=False)
    monkeypatch.setenv("PATH", f"{tmp_path / 'bundle' / 'bin'}{os.pathsep}/usr/bin")
    monkeypatch.setenv("LD_LIBRARY_PATH", str(tmp_path / "bundle"))
    monkeypatch.setenv("LD_LIBRARY_PATH_ORIG", "/system")
    searches, calls = [], []

    def which(name, *, path):
        searches.append((name, path))
        return f"/usr/bin/{name}"

    process = object()
    monkeypatch.setattr(desktop_launch.shutil, "which", which)
    monkeypatch.setattr(desktop_launch.subprocess, "Popen", lambda argv, **kw: calls.append((argv, kw)) or process)
    assert desktop_launch.open_local_document(document) is process
    assert searches == [(opener, "/usr/bin")]
    argv, options = calls[0]
    assert argv == [f"/usr/bin/{opener}", document.as_uri()]
    assert options["env"]["LD_LIBRARY_PATH"] == "/system"
    assert not options["shell"]
    assert options["stdin"] == options["stdout"] == options["stderr"] == desktop_launch.subprocess.DEVNULL
    assert options["start_new_session"]


def test_missing_default_opener_reports_clear_error_without_guessing_browser(monkeypatch, tmp_path):
    document = tmp_path / "index.html"
    document.write_text("help")
    monkeypatch.setattr(desktop_launch.sys, "platform", "linux")
    searches = []
    monkeypatch.setattr(desktop_launch.shutil, "which", lambda name, **_kw: searches.append(name))
    with pytest.raises(FileNotFoundError, match="xdg-open.*not installed"):
        desktop_launch.open_local_document(document)
    assert searches == ["xdg-open"]


def test_windows_uses_native_document_association(monkeypatch, tmp_path):
    document = tmp_path / "index.html"
    document.write_text("help")
    monkeypatch.setattr(desktop_launch.sys, "platform", "win32")
    opened = []
    monkeypatch.setattr(desktop_launch.os, "startfile", lambda path: opened.append(path), raising=False)
    assert desktop_launch.open_local_document(document) is None
    assert opened == [str(document)]


def test_desktop_launch_service_has_no_gui_imports():
    tree = ast.parse(Path(desktop_launch.__file__).read_text())
    modules = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    modules += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    assert not any("PySide" in name or "qt_" in name or "project_gui" in name for name in modules)
