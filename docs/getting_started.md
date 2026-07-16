# Getting started

## First-time collaborator setup

These instructions are for someone who has not previously installed GitHub,
Git, Python, or Conda. They work on macOS, Windows, and Linux. Allow a few
gigabytes of free disk space and a reliable internet connection; the first
environment setup can take several minutes.

### 1. Get a copy of the repository

If the project owner invited you to the private GitHub repository, first
accept the invitation sent by GitHub. You need a free GitHub account to accept
an invitation and to contribute changes. Create one at
[github.com](https://github.com/) if needed.

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
   the example URL with the **Code > HTTPS** URL copied from the repository's
   GitHub page:

   ```bash
   mkdir -p ~/code
   cd ~/code
   git clone https://github.com/OWNER/nfit.git
   cd nfit
   ```

   GitHub may ask you to sign in or create a personal access token. Follow the
   browser prompt; GitHub no longer accepts an account password at the terminal
   prompt for HTTPS Git operations.

**No GitHub account or Git: download a ZIP.** On the repository page, select
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
that folder's path. The `conda env create` command downloads Python 3.11 and
all libraries required by this project. When it completes, your prompt should
start with `(nfit)`.

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

You can also run the small synthetic example:

```bash
python examples/synthetic_single_q_fit.py
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

## Recommended conda setup

Use a dedicated conda environment for `nfit`. This avoids mixing compiled
scientific packages from the project with your base Python installation.

```bash
cd ~/code/nfit
conda env create -f environment.yml
conda activate nfit
```

If the environment already exists, update it after changes to `environment.yml`:

```bash
conda env update -f environment.yml --prune
conda activate nfit
```

The environment uses conda-forge for NumPy, SciPy, Matplotlib, pytest, Sphinx,
MyST Markdown, MyST-NB, and the documentation theme. It also installs `nfit`
in editable mode with:

```bash
/Users/pmneves/anaconda3/envs/nfit/bin/python -m pip install -e ".[dev,docs]"
```

## Validate the install

Run the test suite:

```bash
/Users/pmneves/anaconda3/envs/nfit/bin/python -m pytest -q
```

Lint the Analysis Window implementation:

```bash
/Users/pmneves/anaconda3/envs/nfit/bin/python -m ruff check
```

Launch the graphical project explorer:

```bash
nfit
```

The GUI opens the project explorer, where users can create workspaces, import
datasets, add masks and models, inspect metadata, run fits, and open the data
viewer. See [GUI workflows](gui_workflows.md) for the current user-facing
workflow and tooltip/documentation expectations.

Run the first reference example:

```bash
/Users/pmneves/anaconda3/envs/nfit/bin/python examples/synthetic_single_q_fit.py
```

Build the documentation:

```bash
sphinx-build -b html docs docs/_build/html
```

The long-term distribution goal is:

```bash
pip install nfit
```

After the package is published, `pip install nfit` should install the runtime
dependencies and expose the GUI launcher:

```bash
nfit
```

Before publishing, check release artifacts locally:

```bash
/Users/pmneves/anaconda3/envs/nfit/bin/python -m build
/Users/pmneves/anaconda3/envs/nfit/bin/python -m twine check dist/*
/Users/pmneves/anaconda3/envs/nfit/bin/python -m pip install dist/nfit-*.whl
nfit
```
