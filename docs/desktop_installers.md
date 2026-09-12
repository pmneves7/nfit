# Desktop installers and updates

Native installers let users run nfit without installing Python, Conda, or the
source repository. Each installer contains the scientific Python environment,
the nfit icon and full startup logo, and a local copy of this documentation.

## Installing a beta build

Open the repository's [GitHub Releases](https://github.com/pmneves7/nfit/releases/latest)
page while signed into an account with repository access, then download only
the installer for your computer. The version in the filename changes with each
beta release. A OneDrive link supplied by the project owner can also provide
the initial installer.

### macOS

Choose `nfit-<version>-macos-arm64.pkg` for an Apple-silicon Mac (M1 or newer)
or `nfit-<version>-macos-x86_64.pkg` for an Intel Mac.

1. Double-click the downloaded `.pkg` file.
2. Follow Apple Installer and wait for its progress bar to finish.
3. If macOS blocks the unsigned beta, try opening nfit once, then open
   **System Settings > Privacy & Security**, select **Open Anyway**, and confirm
   with your Mac login password.
4. Open **nfit** from Applications or Launchpad.

### Windows

Use `nfit-<version>-windows-x86_64.exe` on a 64-bit Windows 10 or Windows 11
computer.

1. Double-click the downloaded `.exe` file.
2. If Microsoft Defender SmartScreen shows **Windows protected your PC**,
   select **More info > Run anyway**.
3. Follow the nfit Setup wizard. You can create a desktop shortcut during
   setup.
4. Wait for the progress bar to finish, then open nfit from the Start menu or
   desktop shortcut.

A managed workplace or university computer may prohibit unsigned applications.
If **Run anyway** is unavailable, contact the computer administrator.

### Linux

Use `nfit-<version>-linux-x86_64.deb` on a 64-bit Ubuntu 22.04 or newer system,
or on a compatible Debian-based distribution.

For a graphical installation, double-click the `.deb` file, open it with App
Center, Software Install, or the distribution's package manager, and select
**Install**. To install from a terminal in the download directory, run:

```bash
sudo apt install ./nfit-<version>-linux-x86_64.deb
```

Using `apt` installs the required Linux desktop libraries at the same time.
After installation, open nfit from the application menu or run
`/opt/nfit/nfit`.

Without administrator access, download
`nfit-<version>-linux-x86_64.tar.gz` from the same release and run:

```bash
mkdir -p "$HOME/.local/opt"
tar -xzf nfit-<version>-linux-x86_64.tar.gz -C "$HOME/.local/opt"
"$HOME/.local/opt/nfit/nfit"
```

This uses the same standalone application but does not install missing system
libraries or create an application-menu entry. Replace the extracted `nfit`
directory with the archive from a newer release to update it.

## Uninstalling nfit

On macOS, quit nfit and move **nfit** from the Applications folder to the
Trash. Empty the Trash when convenient. This removes the application; saved
project files elsewhere on the computer are left in place.

On Windows, quit nfit, open **Settings > Apps > Installed apps**, find
**nfit**, and select **Uninstall**. The same uninstaller is available as
**Uninstall nfit** in the nfit folder of the Start menu.

On Ubuntu or another Debian-based Linux system, run:

```bash
sudo apt remove nfit
```

For an installation extracted without administrator access, remove the
directory used during extraction, normally `$HOME/.local/opt/nfit`. If you
created a desktop-menu entry manually, also remove
`$HOME/.local/share/applications/nfit.desktop`.

These procedures leave user-created `.nfit` project files untouched.

The beta installers are not yet signed. macOS may require **Open Anyway** in
**System Settings > Privacy & Security** after the first launch attempt.
Windows may require **More info > Run anyway** in Microsoft Defender
SmartScreen. Signing and notarization can be enabled later without changing the
application or update design.

When nfit opens, a splash screen displays the full logo, version, authorship,
documentation link, and an animated loading indicator. The documentation link
and the application's **Help** button open the bundled pages, so the manual and
its mathematical notation work without internet access.

## Requirements and bundled dependencies

The native installers contain Python 3.14, nfit, Qt, VTK, and the scientific
Python libraries used by the application. A tester does not need to install
Python, Conda, Git, or the source repository. Internet access is needed to
download the installer and check for updates; nfit and its bundled manual work
offline after installation.

| Platform | Requirement |
| --- | --- |
| macOS | Apple silicon or Intel Mac; use the package matching the processor |
| Windows | 64-bit Windows 10 or Windows 11 |
| Linux | 64-bit Ubuntu 22.04 or newer, or a compatible Debian-based distribution |

The Debian package declares `libc6 (>= 2.35)`, `libgl1`, `libegl1`,
`libopengl0`, `libxkbcommon0`, and `libxcb-cursor0`; `apt install` resolves
these automatically. It also recommends Zenity for a reliable native open-file
chooser, especially in remote Linux desktop sessions. Tar installations use
Zenity or Yad when either is already available and otherwise use nfit's Qt
fallback for opening files.

For a source installation, nfit requires Python 3.12 or newer. The supplied
`environment.yml` is the tested developer environment and currently selects
Python 3.14, NumPy 2.5 or newer, SciPy 1.18 or newer, Matplotlib 3.11.1 or
newer, h5py 3.16 or newer, PySide 6.11.2 or newer, PyVista 0.48.4 or newer,
PyVistaQt 0.13.1 or newer, VTK 9.6.2 or newer, and Numba 0.67 or newer. The
complete authoritative runtime and optional dependency lists are in
`pyproject.toml`; `environment.yml` also contains the test, documentation, and
packaging tools used by contributors.

## Application updates

Installed copies check the latest published GitHub Release shortly after
startup. A failed background check does not interrupt offline use. When a newer
installer is available, nfit asks before downloading it, shows progress, checks
its size and GitHub-provided SHA-256 digest, and asks again before closing nfit
and installing it. On macOS and Windows, nfit opens the native installer. A
Linux tar installation atomically replaces its user-local bundle and relaunches
nfit; a system-wide Linux installation opens the `.deb` package. Canceling a
download removes the partial file. Update checks can also be run or disabled
from the **File** menu.

During private beta testing, the installer contains a fine-grained GitHub token
limited to read-only access to the `pmneves7/nfit` repository. Published beta
releases remain private to GitHub accounts and tokens with repository read
access. Anyone with that access can download the installers from the
repository's **Releases** page, and installed copies can discover updates with
the bundled credential. When the repository becomes public, build without that
token and the same updater uses anonymous GitHub access.

## Building beta installers

Create a fine-grained personal access token for `pmneves7/nfit` with only the
repository **Contents: Read-only** permission. Store it as the repository Actions
secret `NFIT_GITHUB_READ_TOKEN`; never commit it to source. Run **Build beta
installers** from the GitHub Actions page. Separate GitHub-hosted runners build
and smoke-test Apple silicon macOS, Intel macOS, Windows x86-64, and Linux
x86-64 installers, then publish them as a beta GitHub Release.

The release is immediately visible to anyone with repository read access and
is discoverable by installed beta copies. The repository remains private, so
publishing the release does not make it or its installers public on the wider
internet. An initial installer can still be given to a tester through a
OneDrive **Anyone with the link** download.

Published beta releases are listed under **Releases** on the repository page
for every signed-in user with repository read access. Drafts are visible only
to collaborators who can manage releases and are not offered by nfit's update
checker.

To build on the current operating system instead, set
`NFIT_GITHUB_READ_TOKEN` in the environment and run:

```bash
/Users/pmneves/anaconda3/envs/nfit/bin/python tools/distribution/build.py
```

The output is written to `dist/installers`. The build always constructs the
offline help and runs the standalone application smoke test before creating an
installer. It does not upload or publish anything.

## Authorship and AI assistance

nfit was authored and is maintained by Paul M. Neves (Johns Hopkins
University, pneves1@jhu.edu). AI and large-language-model coding tools have
assisted with portions of the code, tests, and documentation. These tools are
part of the development process; authorship and responsibility for the project
remain with Paul M. Neves.
