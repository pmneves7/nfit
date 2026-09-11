"""Build a self-contained nfit app and native installer on the target OS.

Run with the nfit interpreter. This builds local files only; it never publishes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import plistlib
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOLS = Path(__file__).resolve().parent


def run(*command: str, **kwargs) -> None:
    subprocess.run(list(map(str, command)), check=True, **kwargs)


def _install_linux_openssl_libraries(
    bundle: Path, *, prefix: Path | None = None
) -> tuple[Path, Path]:
    """Replace system OpenSSL copies with the pair used by the Conda build.

    PyInstaller can discover Ubuntu's older libcrypto while collecting h5py,
    even though another collected library such as libs2n was linked against the
    newer OpenSSL in the active Conda environment. Keeping libcrypto and libssl
    from one prefix prevents that ABI mismatch in the standalone application.
    """
    prefix = Path(sys.prefix) if prefix is None else Path(prefix)
    destination = bundle / "_internal"
    installed = []
    for name in ("libcrypto.so.3", "libssl.so.3"):
        source = prefix / "lib" / name
        if not source.is_file():
            raise FileNotFoundError(
                f"The active build environment is missing {source}"
            )
        target = destination / name
        target.unlink(missing_ok=True)
        shutil.copy2(source.resolve(), target)
        installed.append(target)
    return installed[0], installed[1]


def build(
    *,
    github_repository: str = "pmneves7/nfit",
    github_token_env: str = "NFIT_GITHUB_READ_TOKEN",
    bundle_only: bool = False,
) -> Path:
    if github_repository and not re.fullmatch(
        r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", github_repository
    ):
        raise ValueError("The GitHub repository must have the form owner/name")
    github_token = os.environ.pop(github_token_env, "")
    if any(character.isspace() for character in github_token):
        raise ValueError(f"{github_token_env} contains whitespace")
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    work = ROOT / "build/distribution"
    output = ROOT / "dist/installers"
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    run(sys.executable, "-m", "sphinx", "-W", "--keep-going", "-b", "html", ROOT / "docs", work / "help")
    # During private beta testing a read-only, repository-scoped token is
    # intentionally bundled. It is omitted automatically for a public release.
    config = work / "distribution.json"
    config.write_text(
        json.dumps(
            {
                "github_repository": github_repository,
                "github_token": github_token,
            }
        ),
        encoding="utf-8",
    )
    from PIL import Image

    icon = work / ("nfit.icns" if sys.platform == "darwin" else "nfit.ico")
    with Image.open(ROOT / "src/nfit/resources/nfit-icon.png") as image:
        image.save(icon)
    command = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir",
               "--name", "nfit", "--distpath", str(work / "bundle"), "--workpath", str(work / "pyinstaller"),
               "--specpath", str(work), "--paths", str(ROOT / "src"),
               "--additional-hooks-dir", str(TOOLS / "hooks"),
               "--runtime-hook", str(TOOLS / "runtime_hook.py"),
               "--add-data", f"{work / 'help'}:nfit/resources/help",
               "--add-data", f"{ROOT / 'docs/_static/nfit-logo.svg'}:nfit/resources",
               "--add-data", f"{config}:nfit/resources",
               "--collect-data", "ase", "--collect-data", "seekpath",
               "--hidden-import", "vtkmodules.vtkRenderingOpenGL2",
               "--hidden-import", "vtkmodules.vtkInteractionStyle",
               "--hidden-import", "vtkmodules.vtkRenderingFreeType",
               "--exclude-module", "PyQt5", "--exclude-module", "PyQt6",
               "--exclude-module", "PySide2", "--exclude-module", "IPython",
               "--exclude-module", "notebook", "--exclude-module", "sphinx",
               "--exclude-module", "pytest", "--exclude-module", "tkinter"]
    if sys.platform in ("darwin", "win32"):
        command += ["--windowed", "--icon", str(icon)]
    if sys.platform == "darwin":
        command += ["--osx-bundle-identifier", "org.nfit.desktop", "--osx-entitlements-file", str(TOOLS / "entitlements.plist")]
        if identity := os.environ.get("NFIT_CODESIGN_IDENTITY"):
            command += ["--codesign-identity", identity]
    command.append(str(TOOLS / "entry.py"))
    run(*command, cwd=ROOT)
    bundle = work / "bundle" / ("nfit.app" if sys.platform == "darwin" else "nfit")
    if sys.platform == "linux":
        _install_linux_openssl_libraries(bundle)
    if sys.platform == "darwin":
        info_path = bundle / "Contents/Info.plist"
        info = plistlib.loads(info_path.read_bytes())
        info["CFBundleShortVersionString"] = version
        info["CFBundleVersion"] = version
        info_path.write_bytes(plistlib.dumps(info))
        identity = os.environ.get("NFIT_CODESIGN_IDENTITY", "-")
        signing = [
            "codesign",
            "--force",
            "--deep",
            "--sign",
            identity,
            "--entitlements",
            str(TOOLS / "entitlements.plist"),
        ]
        if identity != "-":
            signing += ["--options", "runtime", "--timestamp"]
        run(*signing, bundle)
    executable = bundle / ("Contents/MacOS/nfit" if sys.platform == "darwin" else "nfit.exe" if sys.platform == "win32" else "nfit")
    smoke = work / "smoke.json"
    smoke.unlink(missing_ok=True)
    run(executable, "--smoke-test", "--smoke-output", smoke)
    assert json.loads(smoke.read_text())["status"] == "ok"
    if bundle_only:
        return bundle
    machine = {"aarch64": "arm64", "amd64": "x86_64"}.get(platform.machine().lower(), platform.machine().lower())
    system = {"darwin": "macos", "win32": "windows", "linux": "linux"}[sys.platform]
    key = f"{system}-{machine}"
    stem = f"nfit-{version}-{key}"
    if sys.platform == "darwin":
        installer = output / f"{stem}.pkg"
        installer.unlink(missing_ok=True)
        package_root = work / "pkg-root"
        if package_root.exists():
            shutil.rmtree(package_root)
        package_root.mkdir()
        run("ditto", bundle, package_root / "nfit.app")
        component = work / "component.plist"
        run("pkgbuild", "--analyze", "--root", str(package_root), component)
        component_data = plistlib.loads(component.read_bytes())
        for item in component_data:
            item["BundleIsRelocatable"] = False
        component.write_bytes(plistlib.dumps(component_data))
        command = ["pkgbuild", "--root", str(package_root), "--install-location", "/Applications",
                   "--component-plist", str(component), "--identifier", "org.nfit.desktop", "--version", version]
        if identity := os.environ.get("NFIT_INSTALLER_SIGN_IDENTITY"):
            command += ["--sign", identity]
        run(*command, installer)
        if profile := os.environ.get("NFIT_NOTARY_PROFILE"):
            run("xcrun", "notarytool", "submit", installer, "--keychain-profile", profile, "--wait")
            run("xcrun", "stapler", "staple", installer)
    elif sys.platform == "win32":
        installer = output / f"{stem}.exe"
        installer.unlink(missing_ok=True)
        compiler = shutil.which("ISCC") or r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
        run(compiler, f"/DAppVersion={version}", f"/DBundleDir={bundle}", f"/DIconFile={icon}",
            f"/DOutputDir={output}", f"/DInstallerName={stem}", TOOLS / "windows.iss")
    else:
        installer = output / f"{stem}.deb"
        installer.unlink(missing_ok=True)
        staging = work / "deb"
        if staging.exists():
            shutil.rmtree(staging)
        shutil.copytree(bundle, staging / "opt/nfit", symlinks=True)
        (staging / "DEBIAN").mkdir()
        architecture = {"x86_64": "amd64", "arm64": "arm64"}[machine]
        (staging / "DEBIAN/control").write_text(
            f"Package: nfit\nVersion: {version}\nArchitecture: {architecture}\n"
            "Maintainer: Paul M. Neves <pneves1@jhu.edu>\n"
            "Depends: libc6 (>= 2.35), libgl1, libegl1, libopengl0, "
            "libxkbcommon0, libxcb-cursor0\n"
            "Section: science\nPriority: optional\nDescription: Magnetic-scattering analysis and fitting\n")
        desktop = staging / "usr/share/applications/nfit.desktop"
        desktop.parent.mkdir(parents=True)
        desktop.write_text("[Desktop Entry]\nType=Application\nName=nfit\nComment=Magnetic-scattering analysis\nExec=/opt/nfit/nfit\nIcon=nfit\nTerminal=false\nCategories=Education;Science;Physics;\n")
        png = staging / "usr/share/icons/hicolor/256x256/apps/nfit.png"
        png.parent.mkdir(parents=True)
        with Image.open(ROOT / "src/nfit/resources/nfit-icon.png") as image:
            image.resize((256, 256)).save(png)
        run("dpkg-deb", "--root-owner-group", "--build", staging, installer)
        shutil.make_archive(str(output / stem), "gztar", root_dir=bundle.parent, base_dir=bundle.name)
    with installer.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    asset = {"version": version, "platform": key, "file": installer.name, "size": installer.stat().st_size, "sha256": digest}
    (output / f"{stem}.json").write_text(json.dumps(asset, indent=2) + "\n", encoding="utf-8")
    return installer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--github-repository",
        default="pmneves7/nfit",
        help="Repository whose latest GitHub Release supplies application updates.",
    )
    parser.add_argument(
        "--github-token-env",
        default="NFIT_GITHUB_READ_TOKEN",
        help="Environment variable containing the private-beta read token.",
    )
    parser.add_argument("--bundle-only", action="store_true")
    args = parser.parse_args()
    print(
        build(
            github_repository=args.github_repository,
            github_token_env=args.github_token_env,
            bundle_only=args.bundle_only,
        )
    )
