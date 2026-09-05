"""Build a local macOS app using this interpreter and the current checkout."""

from __future__ import annotations

import argparse
import json
import plistlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def build_launcher(destination: Path) -> Path:
    if sys.platform != "darwin":
        raise RuntimeError("The local app launcher requires macOS.")
    root = Path(__file__).resolve().parents[2]
    destination = destination.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    library = Path(sys.prefix) / "lib" / (
        f"libpython{sys.version_info.major}.{sys.version_info.minor}.dylib"
    )
    if not library.is_file():
        raise FileNotFoundError(library)

    from PIL import Image

    with tempfile.TemporaryDirectory(prefix="nfit-launcher-") as temporary:
        work = Path(temporary)
        bundle = work / "nfit.app"
        macos = bundle / "Contents" / "MacOS"
        resources = bundle / "Contents" / "Resources"
        macos.mkdir(parents=True)
        resources.mkdir()
        with Image.open(root / "src/nfit/resources/nfit-icon.png") as icon:
            icon.save(resources / "nfit.icns", format="ICNS")
        log = Path.home() / "Library" / "Logs" / "nfit.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        constants = {
            "NFIT_SOURCE": str(root), "NFIT_PREFIX": sys.prefix,
            "NFIT_PYTHON": sys.executable, "NFIT_LIBRARY": str(library),
            "NFIT_LOG": str(log),
        }
        (work / "launcher_config.h").write_text("".join(
            f"#define {name} {json.dumps(value)}\n" for name, value in constants.items()
        ))
        subprocess.run(["/usr/bin/clang", "-Wall", "-Wextra", "-Werror", "-O2",
                        "-I", str(work), str(root / "tools/macos/launcher.c"),
                        "-o", str(macos / "nfit")], check=True)
        info = {
            "CFBundleName": "nfit", "CFBundleDisplayName": "nfit",
            "CFBundleIdentifier": "org.nfit.local",
            "CFBundleExecutable": "nfit", "CFBundlePackageType": "APPL",
            "CFBundleIconFile": "nfit.icns", "NSHighResolutionCapable": True,
            "NSPrincipalClass": "NSApplication",
        }
        (bundle / "Contents" / "Info.plist").write_bytes(plistlib.dumps(info))
        subprocess.run([str(macos / "nfit"), "--smoke-test"], check=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(bundle, destination)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[2] / "build/nfit.app")
    print(build_launcher(parser.parse_args().output))
