from __future__ import annotations

import ast
import hashlib
import io
import json
from pathlib import Path
from urllib.request import Request

import pytest

from nfit import app_updates as updates
from nfit.app_distribution import local_help_index
from tools.distribution import build as distribution_build


def github_release(**asset_changes):
    asset = {
        "id": 42,
        "name": "nfit-0.87.0-macos-arm64.pkg",
        "state": "uploaded",
        "digest": f"sha256:{hashlib.sha256(b'installer').hexdigest()}",
        "size": 9,
    }
    asset.update(asset_changes)
    return json.dumps(
        {"tag_name": "v0.87.0", "body": "A tested release", "assets": [asset]}
    ).encode()


def release(payload=None):
    return updates.parse_github_release(
        github_release() if payload is None else payload,
        repository="pmneves7/nfit",
        current_version="0.86.2",
        target="macos-arm64",
    )


def test_help_finds_bundled_pages_before_checkout(tmp_path):
    source = tmp_path / "src/nfit/project_gui.py"
    checkout = tmp_path / "docs/_build/html/index.html"
    bundled = source.parent / "resources/help/index.html"
    assert local_help_index(source_file=source) is None
    checkout.parent.mkdir(parents=True)
    checkout.write_text("checkout")
    assert local_help_index(source_file=source) == checkout
    bundled.parent.mkdir(parents=True)
    bundled.write_text("bundled")
    assert local_help_index(source_file=source) == bundled


def test_local_mathjax_is_pinned_and_used():
    root = Path(__file__).resolve().parents[1]
    data = (root / "docs/_static/mathjax/tex-svg-full.js").read_bytes()
    assert hashlib.sha256(data).hexdigest() == (
        "a4354ff94fd868aea0cc6eaaa79a57fda0588646fc46ee3700a349ee0a11cbe6"
    )
    config = {}
    exec((root / "docs/conf.py").read_text(), config)
    assert config["mathjax_path"] == "mathjax/tex-svg-full.js"


def test_linux_bundle_uses_matching_conda_openssl_pair(tmp_path):
    prefix = tmp_path / "conda"
    libraries = prefix / "lib"
    libraries.mkdir(parents=True)
    bundle = tmp_path / "bundle"
    internal = bundle / "_internal"
    internal.mkdir(parents=True)
    for name in ("libcrypto.so.3", "libssl.so.3"):
        (libraries / name).write_bytes(f"conda:{name}".encode())
        (internal / name).write_bytes(f"system:{name}".encode())

    installed = distribution_build._install_linux_openssl_libraries(
        bundle, prefix=prefix
    )

    assert installed == (
        internal / "libcrypto.so.3",
        internal / "libssl.so.3",
    )
    assert installed[0].read_bytes() == b"conda:libcrypto.so.3"
    assert installed[1].read_bytes() == b"conda:libssl.so.3"


def test_beta_workflow_uses_node24_actions():
    workflow = (
        Path(__file__).resolve().parents[1]
        / ".github/workflows/build-beta-installers.yml"
    ).read_text(encoding="utf-8")
    assert "mamba-org/setup-micromamba@v3" in workflow
    assert "actions/upload-artifact@v7" in workflow
    assert "actions/download-artifact@v7" in workflow
    assert "setup-micromamba@v2" not in workflow
    assert "upload-artifact@v4" not in workflow
    assert "download-artifact@v4" not in workflow


def test_beta_workflow_exports_the_release_version_without_nested_shell_quotes():
    workflow = (
        Path(__file__).resolve().parents[1]
        / ".github/workflows/build-beta-installers.yml"
    ).read_text(encoding="utf-8")
    assert 'value="$(python -c \'import tomllib;' in workflow
    assert 'echo "value=$value" >> "$GITHUB_OUTPUT"' in workflow
    assert 'echo "value=$(python -c' not in workflow


def test_beta_workflow_publishes_a_release_visible_to_repository_readers():
    workflow = (
        Path(__file__).resolve().parents[1]
        / ".github/workflows/build-beta-installers.yml"
    ).read_text(encoding="utf-8")
    assert "Publish private beta release" in workflow
    assert "gh release create" in workflow
    assert "--draft" not in workflow
    assert "--prerelease" not in workflow


def test_linux_smoke_runner_installs_debian_runtime_dependencies():
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github/workflows/build-beta-installers.yml").read_text(
        encoding="utf-8"
    )
    build_script = (root / "tools/distribution/build.py").read_text(
        encoding="utf-8"
    )
    for package in (
        "libegl1",
        "libgl1",
        "libopengl0",
        "libxkbcommon0",
        "libxcb-cursor0",
    ):
        assert package in workflow
        assert package in build_script


def test_windows_installer_exposes_a_start_menu_uninstaller():
    installer = (
        Path(__file__).resolve().parents[1] / "tools/distribution/windows.iss"
    ).read_text(encoding="utf-8")
    assert 'Name: "{autoprograms}\\Uninstall nfit"' in installer
    assert 'Filename: "{uninstallexe}"' in installer


def test_release_selection_is_version_and_platform_aware():
    result = release()
    assert result.installer_url == (
        "https://api.github.com/repos/pmneves7/nfit/releases/assets/42"
    )
    assert result.filename == "nfit-0.87.0-macos-arm64.pkg"
    assert (
        updates.parse_github_release(
            github_release(),
            repository="pmneves7/nfit",
            current_version="0.87.0",
            target="macos-arm64",
        )
        is None
    )
    assert (
        updates.parse_github_release(
            github_release(),
            repository="pmneves7/nfit",
            current_version="0.86.2",
            target="windows-x86_64",
        )
        is None
    )
    assert release(github_release().replace(b"v0.87.0", b"v0.88.0rc1")) is None


@pytest.mark.parametrize(
    "asset",
    [
        {"digest": "missing"},
        {"size": -1},
        {"size": True},
        {"size": updates.INSTALLER_LIMIT + 1},
        {"id": -1},
        {"id": True},
        {"state": "new"},
    ],
)
def test_github_release_rejects_invalid_installer_metadata(asset):
    with pytest.raises(updates.UpdateError):
        release(github_release(**asset))


@pytest.mark.parametrize(
    "repository",
    ["nfit", "owner/repo/extra", "owner repo/nfit", "https://github.com/a/b"],
)
def test_repository_names_are_strict(repository):
    with pytest.raises(updates.UpdateError):
        updates.validate_repository(repository)


def test_github_redirects_require_https_and_strip_the_token():
    handler = updates._GitHubRedirect()
    request = Request(
        "https://api.github.com/repos/pmneves7/nfit/releases/assets/42",
        headers={"Authorization": "Bearer test", "User-Agent": "nfit-updater"},
    )
    redirected = handler.redirect_request(
        request,
        None,
        302,
        "redirect",
        {},
        "https://release-assets.githubusercontent.com/file",
    )
    assert redirected is not None
    assert not redirected.has_header("Authorization")
    with pytest.raises(updates.UpdateError):
        handler.redirect_request(
            request, None, 302, "redirect", {}, "http://example.org/file"
        )


def test_github_requests_use_the_certifi_ca_bundle(monkeypatch):
    calls = []

    class FakeOpener:
        def open(self, request, timeout):
            calls.append((request, timeout))
            return io.BytesIO(b"{}")

    class FakeContext:
        def load_verify_locations(self, *, cafile):
            calls.append(("cafile", cafile))

    context = FakeContext()
    monkeypatch.setattr(updates.ssl, "create_default_context", lambda: context)
    monkeypatch.setattr(updates, "build_opener", lambda *handlers: FakeOpener())

    with updates._open_github(
        "https://api.github.com/repos/pmneves7/nfit/releases/latest", "token"
    ) as response:
        assert response.read() == b"{}"

    assert calls[0] == ("cafile", updates.certifi.where())
    assert calls[1][1] == 30


def test_verified_download_is_the_only_returned_installer(monkeypatch, tmp_path):
    monkeypatch.setattr(
        updates, "_open_github", lambda *args, **kwargs: io.BytesIO(b"installer")
    )
    progress = []
    result = updates.download_installer(
        release(), tmp_path, progress=lambda done, total: progress.append((done, total))
    )
    assert result.read_bytes() == b"installer"
    assert progress == [(9, 9)]
    assert not (result.parent / "download.partial").exists()


@pytest.mark.parametrize(
    "payload", [b"changed!!", b"truncated", b"installer-extra", b"short"]
)
def test_bad_download_is_removed_without_returning_a_path(
    monkeypatch, tmp_path, payload
):
    monkeypatch.setattr(
        updates, "_open_github", lambda *args, **kwargs: io.BytesIO(payload)
    )
    with pytest.raises(updates.UpdateError):
        updates.download_installer(release(), tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_download_cancel_removes_partial_files(monkeypatch, tmp_path):
    monkeypatch.setattr(
        updates, "_open_github", lambda *args, **kwargs: io.BytesIO(b"installer")
    )
    with pytest.raises(updates.DownloadCancelled):
        updates.download_installer(release(), tmp_path, cancelled=lambda: True)
    assert list(tmp_path.iterdir()) == []


def test_release_services_do_not_import_gui_modules():
    root = Path(__file__).resolve().parents[1] / "src/nfit"
    for name in ["app_distribution.py", "app_updates.py"]:
        tree = ast.parse((root / name).read_text(encoding="utf-8"))
        imports = [
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        ]
        imports += [
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        ]
        assert not any(
            "PySide" in name or "project_gui" in name or name.endswith("_gui")
            for name in imports
        )


def test_peak_process_memory_is_available_on_current_platform():
    from nfit.performance import peak_process_memory_mib

    assert peak_process_memory_mib() > 0
