---
applyTo: ".github/workflows/**"
---

# CI Workflow Changes

Use these rules for files under `.github/workflows/`. Apply the shared guidance
from `.github/copilot-instructions.md` first, then this workflow-specific
guidance.

## Workflow Overview

This repository has six GitHub Actions workflows in `.github/workflows/`. All
workflows except `api-docs.yml` use concurrency groups to cancel in-progress
runs when superseded.

### `quick-checks.yml`

Runs pre-commit hooks on every pull request and on `workflow_dispatch`. The
hooks (configured in
`.pre-commit-config.yaml`) are: Black, pylint, isort (with special no-sections
config for sample files), yamlfix, shellcheck, GitHub workflow/action schema
validation (`check-github-workflows`, `check-github-actions`), and a local
`samples-generation` hook that regenerates Markdown samples when
`khiops/samples/samples.py` or `khiops/samples/samples_sklearn.py` change.

### `tests.yml`

The main test suite. Triggers on PRs that touch `khiops/**/*.py`,
`tests/**/*.py`, `tests/resources/**` (excluding `tests/resources/**/*.md`), or
the workflow file itself. Also supports `workflow_dispatch`.

Three job groups:

- **`run`** (Linux matrix): Runs across Python 3.11-3.14 on `ubuntu-24.04` in
  custom Docker containers
  (`ghcr.io/khiopsml/khiops-python/khiopspydev-ubuntu24.04`). Each
  Python version uses a dedicated Conda environment with native Khiops.
  Conda and Pip are invoked through `conda run -n` without activation.
  Coverage is collected with `coverage` and reported as XML. Test results use
  JUnit XML via `unittest-xml-reporting`.
- **`check-khiops-integration-on-linux`**: Runs integration tests on
  `ubuntu-24.04` in multiple Linux containers (ubuntu24.04, rocky8, rocky9,
  debian13). The cloned sources are tested, not an installed package. The
  interpreter is stored in `PYTHON` (via `$GITHUB_ENV`) and used directly in
  all subsequent steps, hence no virtual environment is ever activated. On
  Ubuntu and Debian, where the system Python is externally managed (PEP 668),
  `PYTHON` is the interpreter of a `khiops-venv` virtual environment, whose
  `bin` directory is also added to `$PATH` in the steps that need it (the Khiops
  runner looks for `khiops_env` in the `PATH`). On Rocky, the system-wide Python
  3.11 is used.
  New Debian or Ubuntu containers must be added to the venv condition.
- **`check-khiops-integration-on-windows`**: Installs Khiops Desktop via NSIS
  installer on Windows 2022 with Python 3.14 (`setup-python`). Runs
  integration tests and samples from the cloned sources with the
  `setup-python` interpreter (system-wide dependencies; an "improper setup"
  status is expected). It then creates `khiops-windows-venv` (never
  activated; its `python.exe` is invoked directly), installs khiops-python
  from the sources there, and requires the installation status to succeed.

**Expensive tests** (remote file access with S3/GCS/Azure): Skipped by default
on feature branches. Enabled on `main`/`main-v10` branches or via the
`run-expensive-tests` workflow dispatch input. These require GCP Workload
Identity Federation, a local fake S3 server, and Azure storage credentials.

**Environment variables**: `KHIOPS_SAMPLES_DIR` points to a checkout of
`khiopsml/khiops-samples`. `KHIOPS_PROC_NUMBER=4` forces MPI multi-process
execution. MPI oversubscribe flags are set for Open MPI 4.x and 5+.

### `pip.yml`

Builds an **sdist** package (no wheel) with Python 3.14, then tests it on brand
new environments (not the dev Docker images): `ubuntu-24.04` (bare runner, and
`debian:trixie` and `rockylinux/rockylinux:10` containers), `windows-2025` and
`macos-15`. Python is installed in the containers when missing. The package is
always installed in a virtual environment (`khiops-venv`): system-wide
installation is not supported, because the system-wide Python can be too old
and can contain EOL Python versions or outdated package versions. The virtual
environment is never activated: its interpreter is invoked directly through
`VENV_PYTHON`, and its executables directory is added to `$PATH` in the step
that needs it, so that `khiops_env` and the `kh-*` scripts are found. Triggers
on:

- Tag pushes (any tag) — automatically publishes to GitHub Releases
- PRs touching `pyproject.toml`, `LICENSE.md`, or the workflow file
- `workflow_dispatch` with optional `pypi-target` choice (`None`, `testpypi`,
  `pypi`)

Publishing to TestPyPI/PyPI uses OIDC Trusted Publishing and requires the
corresponding GitHub environment (`testpypi` or `pypi`). Only runs for the
`KhiopsML` org on tag pushes.

### `api-docs.yml`

Builds Zensical documentation on a `ubuntu-24.04` runner directly, with
a Python 3.14 virtual environment created by `uv venv` (`astral-sh/setup-uv`,
with `UV_PYTHON_PREFERENCE: only-managed` so that no system-level Python is
used). The virtual environment is not activated: `VIRTUAL_ENV` and its `bin`
directory are exported through `$GITHUB_ENV` and prepended to `$PATH` where
needed, respectively. Triggers on:

- PRs touching `doc/site/**/*.md`, `doc/util/create-doc`, `doc/util/clean-doc`,
  `doc/util/*.py`, `zensical.toml`, `khiops/**/*.py`, `pyproject.toml`,
  `doc/util/requirements.txt`,
  `scripts/extract_dependencies_from_pyproject_toml.py`, or the workflow file
- `workflow_dispatch` with optional tutorial and samples revision inputs

Installs the dependencies (khiops-core and khiops-drivers-* from TestPyPI) and
khiops-python with `uv pip`, then runs
`doc/util/create-doc -t -d -g <revision>`. The built HTML is uploaded as
a `api-docs` artifact for review.

Note: the production API docs are built by the
[khiops-doc](https://github.com/KhiopsML/khiops-doc) CI, which clones this
repository and builds the docs using mkdocstrings.

### `dev-docker.yml`

Builds development Docker images for multiple OS targets (ubuntu24.04, rocky8,
rocky9, debian13) with configurable image tag (default `dev`), server revision
(default `main`), Python versions (3.11-3.14).
Triggers on PRs touching `packaging/docker/khiopspydev/Dockerfile.*` or the
workflow file, and on `workflow_dispatch`. Images are pushed to
`ghcr.io/khiopsml/khiops-python/khiopspydev-*` only when manually requested via
`push: true`. The `set-latest` flag only works on the `main` or `main-v10`
branches.

### `test-conda-forge-package.yml`

Manual-only workflow that tests the released `khiops` Conda package on the
`conda-forge` channel across Python 3.11 and 3.14 on `ubuntu-24.04`,
`windows-2025` and `macos-15`, using `conda-incubator/setup-miniconda` with
`miniforge-version: latest`. Tests both normal Conda environments and
"Conda-based environments" (where `CONDA_PREFIX` is unset to simulate non-Conda
invocation).

## Editing Rules

- Workflow YAML files are validated by pre-commit hooks
  (`check-github-workflows`, `check-github-actions`) and formatted by `yamlfix`.
- The dev Docker images are the test environment for `tests.yml`.
  If you need new system dependencies in CI, they go into the
  Dockerfiles under `packaging/docker/khiopspydev/`.
- Do not activate virtual environments in workflows. Store the interpreter
  path in an environment variable (`PYTHON`, `VENV_PYTHON`) and add the venv
  executables directory to `PATH` inside the step that needs it. Always quote
  the interpreter (`"$PYTHON"`). Inline commands like `run: "$PYTHON" -m ...`
  are invalid YAML: use a `|-` block. Never use `$GITHUB_PATH`  for the virtual
  environment: in container jobs it breaks the `PATH`  of subsequent steps (e.g.
  `bash`  is not found on Rocky).
- Prefer `ubuntu-24.04` over `ubuntu-latest` for runners.
- Test dependencies are in `test-requirements.txt` (`coverage`, `wrapt`).
  Package dependencies are extracted from `pyproject.toml` at CI time via
  `scripts/extract_dependencies_from_pyproject_toml.py`.
