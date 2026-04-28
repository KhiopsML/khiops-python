# Copilot Instructions for khiops-python

Use this file as the shared repository guide. When you work in a path covered by a
scoped instruction file, apply both this document and the matching file in
`.github/instructions/`.

## Scoped Instruction Files

- `.github/instructions/python-changes.instructions.md` — Python source and test
  changes (`**/*.py`)
- `.github/instructions/docker-changes.instructions.md` — development Docker image
  changes (`packaging/docker/khiopspydev/**`)
- `.github/instructions/doc-changes.instructions.md` — documentation source changes
  (`doc/**`)
- `.github/instructions/ci-workflows.instructions.md` — GitHub Actions workflow
  changes (`.github/workflows/**`)

## Architecture

Khiops Python is a Python interface to the **Khiops AutoML suite** for building
supervised models (classifiers, regressors, encoders) and unsupervised models
(coclusterings). It provides two ways to use Khiops from Python:

- **`khiops.core`** — The low-level API that drives the Khiops binaries via
  dictionary files (`.kdic`, `.kdicj`) and tabular data files. The code which implements this API must depend only on Python built-in modules.
  - `core.api` — public functions such as `train_predictor` and
    `train_recoder`
  - `core.dictionary` — data classes for Khiops dictionary files (in the
    `.kdic` and JSON `.kdicj` formats)
  - `core.analysis_results` — data classes for Khiops JSON analysis reports
    (`.khj`)
  - `core.coclustering_results` — data classes for Khiops coclustering report
    files (`.khcj`)
  - `core.internals.runner` — backend abstraction for local, Docker, and other
    execution modes, configurable with `get_runner()` and `set_runner()`
  - `core.internals.filesystems` — filesystem abstraction for local, S3, GCS and
    Azure access
  - `core.internals.task`, `core.internals.tasks` — task definitions for
    Khiops operations
- **`khiops.sklearn`** — Scikit-Learn compatible estimators built on top of
  `khiops.core`. The code which implements these estimators may depend on Pandas and Scikit-learn only.
  ```
  KhiopsEstimator(ABC, BaseEstimator)
      ├── KhiopsCoclustering(ClusterMixin)
      └── KhiopsSupervisedEstimator
          ├── KhiopsPredictor
          │   ├── KhiopsClassifier(ClassifierMixin)
          │   └── KhiopsRegressor(RegressorMixin)
          └── KhiopsEncoder(TransformerMixin)
  ```
  - `sklearn.dataset` — normalizes DataFrames, file paths, and multi-table
    dictionaries into Khiops-compatible datasets
- **`khiops.extras`** — Optional integrations such as the Docker runner
- **`khiops.tools`** — Miscellaneous utility tools and CLI entry points
- **`khiops.samples`** — Sample scripts, also used to generate parts of the
  documentation via `doc/convert-samples-hook`

Keep changes inside these layer boundaries.

## Shared Conventions

### Dependency Rules

- Do not add new external dependencies without discussion. Minimize external
  package dependencies to reduce installation problems.
- Development and documentation generation dependencies (e.g., `black`,
  `isort`, `sphinx`, `wrapt`, `furo`) can be more permissive, but still avoid
  unnecessary additions.
- Test dependencies are listed in `test-requirements.txt` (`coverage`, `wrapt`).
  Package dependencies are extracted from `pyproject.toml` at CI time via
  `scripts/extract_dependencies_from_pyproject_toml.py`.

### Python Support Policy

- CI tests run against Python 3.10–3.14.

### Versioning

The project uses `MAJOR.MINOR.PATCH.INCREMENT[-PRE_RELEASE]`, where
`MAJOR.MINOR.PATCH` tracks the compatible Khiops native version and `INCREMENT`
tracks the Python package's own evolution.

For Pip and Conda packages, the dash before the pre-release atom is removed to
comply with
[Python version specifiers](https://packaging.python.org/en/latest/specifications/version-specifiers/#version-specifiers)
(e.g., `11.0.0.2a1` instead of `11.0.0.2-a.1`).

## License

BSD 3-Clause-Clear. See `LICENSE.md`.
