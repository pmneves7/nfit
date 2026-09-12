# Getting started

Most beta testers should use a native application installer. It includes the
Python environment and offline manual, and does not require Git, Python, or
Conda. Follow [Desktop installers and updates](desktop_installers.md) for macOS,
Windows, or Linux. The setup below is for contributors and users who want to run
nfit from source.

## First-time source setup

These instructions are for someone who has not previously installed Git,
Python, or Conda. They work on macOS, Windows, and Linux. Allow a few
gigabytes of free disk space and a reliable internet connection; the first
environment setup can take several minutes.

### 1. Get a copy of the repository

Then choose one of these ways to obtain the code:

**Recommended: clone with Git.** This makes it easy to receive updates and
share your changes.

1. Install Git from [git-scm.com/downloads](https://git-scm.com/downloads).
   On macOS, the installer may instead offer to install the Apple command-line
   developer tools; accept that offer. On Windows, accept the default options
   in the Git installer.
2. Open a terminal:
   - macOS: open **Terminal** from Applications > Utilities.
   - Windows: open **Git Bash** (installed with Git) or **Miniforge Prompt**.
   - Linux: open your usual terminal application.
3. In the terminal, choose a folder for code and clone the repository. Replace
   the public repository:

   ```bash
   mkdir -p ~/code
   cd ~/code
   git clone https://github.com/pmneves7/nfit.git
   cd nfit
   ```

**No Git or GitHub account: download a ZIP.** On the repository page, select
**Code > Download ZIP**, unzip it, and open a terminal in the resulting
`nfit` folder. You can run the software this way, but you will not be able to
pull updates or contribute through GitHub until you set up an account and Git.

### 2. Install Miniforge (Python and Conda)

`nfit` uses Conda to install a compatible Python and scientific libraries.
You do **not** need to install Python separately.

1. Download and install [Miniforge](https://github.com/conda-forge/miniforge#install).
   Choose the installer for your operating system and processor. The default
   installer choices are appropriate for most people.
2. Close and reopen the terminal after installation. On Windows, use the
   **Miniforge Prompt** from the Start menu; it is already configured for the
   `conda` command.
3. Confirm that Conda is available:

   ```bash
   conda --version
   ```

   If this reports that `conda` is not found, reopen the terminal. On macOS or
   Linux, run `conda init`, close the terminal, and open it again. If it still
   is not found, return to the Miniforge installer instructions for your
   platform.

### 3. Create the nfit environment

In a terminal, change into the repository folder—the folder containing
`environment.yml`—and create the environment:

```bash
cd ~/code/nfit
conda env create -f environment.yml
conda activate nfit
```

If you downloaded a ZIP or chose another location, replace `~/code/nfit` with
that folder's path. The `conda env create` command downloads Python 3.14 and
all libraries required by this project. When it completes, your prompt should
start with `(nfit)`.

The package requires Python 3.12 or newer. The development environment uses
standard CPython 3.14, NumPy 2.5, SciPy 1.18, and Numba 0.67 or newer.
Conda resolves Numba and its matching llvmlite together. The environment also
installs Qt, PyVista, and VTK from conda-forge so that compiled dependencies
are managed by the same package manager. The optional GPU backend requires
a separate compatible CuPy installation on supported hardware.

### 4. Run nfit

With the `(nfit)` environment active and from the repository folder, launch
the graphical project explorer:

```bash
nfit
```

To make sure the installation is healthy before opening the GUI, run the test
suite instead:

```bash
python -m pytest -q
```

You can also run the small script-only analysis example:

```bash
python examples/data_playground.py
```

### Everyday use and updates

Each time you want to use the project, open a terminal and run:

```bash
cd ~/code/nfit
conda activate nfit
nfit
```

For a Git clone, get the latest shared changes with:

```bash
git pull
conda env update -f environment.yml --prune
```

Run the environment update after pulling changes, especially when
`environment.yml` changed. If `git pull` says that local changes would be
overwritten, do not force it; ask the collaborator who manages the project for
help preserving your work.

Before upgrading an existing Python 3.11 environment, keep a rollback copy:

```bash
conda create -n nfit-backup --clone nfit
conda env update -f environment.yml
conda activate nfit
python -m pip check
python -m pytest -q
```

Close running nfit windows before upgrading. Restart nfit afterward so that
the application loads the new Python and Qt libraries. Keep the backup until
imports, fitting, and the 2D and 3D viewers work with your projects.
On macOS, rebuild any local app launcher after changing Python's minor version;
the launcher embeds the Python shared-library path.

### Reproduce the tested macOS environment

For an Apple Silicon Mac, `environments/nfit-osx-arm64.explicit.txt` records
the exact conda packages and checksums used for validation. From the repository
root, create a separate environment without resolving newer versions:

```bash
conda create -n nfit-repro --file environments/nfit-osx-arm64.explicit.txt
conda activate nfit-repro
python -m pip install --no-deps cmcrameri==1.10 -e .
python -m pip check
python -m pytest -q
```

The explicit file is specific to macOS ARM64. Use `environment.yml` on other
platforms or when intentionally updating dependencies. cmcrameri is installed
from PyPI because the conda-forge 1.10 build has incorrect version metadata;
the compiled numerical and graphics libraries come from conda-forge.

### Quick troubleshooting

- **`conda` or `nfit` is not found:** close and reopen the terminal, then run
  `conda activate nfit`. On Windows, use Miniforge Prompt.
- **The GUI does not appear:** start it from a terminal with `nfit` so any
  error message remains visible. Verify the environment with
  `python -m pytest -q`.
- **Environment creation fails partway through:** check your internet
  connection, then run the same `conda env create -f environment.yml` command
  again. Conda reuses packages it already downloaded.
- **You are unsure which folder is active:** `pwd` shows it on macOS/Linux;
  `cd` with no arguments shows it in Windows Command Prompt. The active nfit
  folder should contain `environment.yml`.

## Developer setup

The first-time steps above are also the recommended developer setup. After
pulling changes, update the existing environment with:

```bash
conda env update -f environment.yml --prune
conda activate nfit
```

To refresh the editable installation explicitly:

```bash
python -m pip install -e ".[dev,docs]"
```

### Validate the checkout

```bash
python -m pytest -q
python -m pytest --cov=nfit --cov-branch --cov-report=term-missing
python -m ruff check
```

The coverage run reports untested lines and branches. It is diagnostic: the
goal is meaningful tests for important behavior, not maximizing a percentage.
The configured 75% floor is a regression guard below the current baseline, not
a target that replaces scientific review.

Launch the graphical project explorer:

```bash
nfit
```

The GUI opens the project explorer, where users can create workspaces, import
datasets, add masks and models, inspect metadata, run fits, and open the data
viewer. See [GUI workflows](gui_workflows.md) for the user-facing workflow.

Run the script-only analysis example:

```bash
python examples/data_playground.py
```

Build the documentation and fail on warnings:

```bash
python -m sphinx -W -b html docs docs/_build/html
```

### Validate a release

```bash
python -m build
python -m twine check dist/*
python -m pip install dist/nfit-*.whl
nfit
```

## macOS local app launcher

A local `nfit.app` opens the current checkout using the Python environment that
built it. It does not bundle Python or require rebuilding after ordinary code
updates. Rebuild it after changing Python's minor version. Keep that environment
and checkout at their existing paths.

From the repository, with the nfit environment active, run:

```bash
python tools/macos/build_launcher.py
```

The builder requires the Apple command-line developer tools and checks that the
app can load nfit and its Qt icon. It creates `build/nfit.app`; move the app to
`~/Applications`, then double-click it. In the Dock, right-click its icon and
choose **Options → Keep in Dock**. Startup output is saved in
`~/Library/Logs/nfit.log`.

Use `--output /path/to/nfit.app` to choose another destination. The builder
refuses to overwrite an existing app. Rebuild if the checkout or environment
moves. This launcher is for local use; it is not a standalone distribution for
other computers.
