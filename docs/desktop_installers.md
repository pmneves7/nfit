# Desktop installers and updates

Native installers let users run nfit without installing Python, Conda, or the
source repository. Each installer contains the scientific Python environment,
the nfit icon and full startup logo, and a local copy of this documentation.

## Installing a beta build

The macOS package opens in Apple Installer, and the Windows executable opens a
standard setup wizard. Both display installation progress. On Ubuntu and other
compatible Debian-based Linux systems, opening the `.deb` file in the software
installer displays equivalent progress. The Linux package can also be installed
from a terminal with `sudo apt install ./nfit-<version>-linux-x86_64.deb`.

The beta installers are not yet signed. macOS may require **Open Anyway** in
**System Settings > Privacy & Security** after the first launch attempt.
Windows may require **More info > Run anyway** in Microsoft Defender
SmartScreen. Signing and notarization can be enabled later without changing the
application or update design.

When nfit opens, a splash screen displays the full logo, version, authorship,
documentation link, and an animated loading indicator. The documentation link
and the application's **Help** button open the bundled pages, so the manual and
its mathematical notation work without internet access.

## Application updates

Installed copies check the latest published GitHub Release shortly after
startup. A failed background check does not interrupt offline use. When a newer
installer is available, nfit asks before downloading it, shows progress, checks
its size and GitHub-provided SHA-256 digest, and asks again before closing nfit
and opening the native installer. Canceling a download removes the partial
file. Update checks can also be run or disabled from the **File** menu.

During private beta testing, the installer contains a fine-grained GitHub token
limited to read-only access to the `pmneves7/nfit` repository. This deliberately
gives anyone who receives the installer equivalent read access; distributing
the installer is the invitation mechanism. When the repository becomes public,
build without that token and the same updater uses anonymous GitHub access.

## Building beta installers

Create a fine-grained personal access token for `pmneves7/nfit` with only the
repository **Contents: Read-only** permission. Store it as the repository Actions
secret `NFIT_GITHUB_READ_TOKEN`; never commit it to source. Run **Build beta
installers** from the GitHub Actions page. Separate GitHub-hosted runners build
and smoke-test Apple silicon macOS, Intel macOS, Windows x86-64, and Linux
x86-64 installers, then assemble them into a draft GitHub Release.

Review the draft and publish it when it is ready for installed beta copies to
discover. The initial installer can be given to each tester through a OneDrive
**Anyone with the link** download or by inviting the tester to the repository.
Later versions download directly through the bundled beta credential.

To build on the current operating system instead, set
`NFIT_GITHUB_READ_TOKEN` in the environment and run:

```bash
/Users/pmneves/anaconda3/envs/nfit/bin/python tools/distribution/build.py
```

The output is written to `dist/installers`. The build always constructs the
offline help and runs the standalone application smoke test before creating an
installer. It does not upload or publish anything.
