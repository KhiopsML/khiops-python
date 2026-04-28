---
applyTo: "**/*.py"
---

# Python Code Changes

Use these rules for any Python source or test change. Apply the shared guidance
from `.github/copilot-instructions.md` first, then these Python-specific rules.

## Architecture and Dependencies

- `khiops.core` is the low-level API that drives the Khiops binary and must only
  depend on Python built-in modules.
- `khiops.sklearn` builds on top of `khiops.core` and may directly depend on
  pandas and scikit-learn only.
- Keep changes inside these layer boundaries and do not introduce new external
  dependencies without asking.
- Extra, optional, dependencies are required for the remote-access filesystem
  API in `core.internals.filesystems`, viz. for accessing S3, GCS, and Azure
  remote storages.
- Test dependencies are listed in `test-requirements.txt` (`coverage`, `wrapt`).
  Package dependencies are declared in `pyproject.toml`.

## Style and Formatting

- Format with **Black** (88 char line length) and sort imports with **isort**
  (Black profile). Configuration is in `pyproject.toml`.
- **isort exception**: `khiops/samples/samples.py` and
  `khiops/samples/samples_sklearn.py` are handled by a separate `isort-samples`
  pre-commit hook with `--no-sections` (no import section grouping).
- Wrap long literal strings to stay under 88 chars.
- Pylint **hard failures** (`fail-on` in `pyproject.toml`): all errors (code E),
  `line-too-long`, `unused-variable`, and `unused-import`. The overall pylint
  score must stay at or above `9.9` (`fail-under`). Other pylint warnings are
  lower priority.
- Pylint is **not run** on `doc/convert_samples.py` and `doc/conf.py` (excluded
  in `.pre-commit-config.yaml`).
- All code and comments must be in English.
- `pylint: disable=invalid-name` is used in `khiops/sklearn/estimators.py` to
  permit scikit-learn's `X`, `y` naming convention. Do not add this suppression
  elsewhere.

## Paragraph-Oriented Programming

Structure code as **paragraphs**: a comment header describing the intent,
followed by the code body, separated by blank lines.

```python
def value_count(values):
    """Prints the counts of each unique value in an array"""

    # Initialize the counts dictionary
    counts = {}

    # Count the unique occurrences in values
    for value in values:
        if value in counts:
            counts[value] += 1
        else:
            counts[value] = 1

    # Print the counts
    for value, count in counts.items():
        print(f"{value}: {count}")
```

Exceptions where a paragraph header is not needed:
- Return statements
- Loop variable assignments (e.g., in `while` loops)
- Very short and obvious methods where the docstring suffices

Keep the number of paragraphs minimal. Commenting every line technically
conforms but defeats the purpose.

## Pre-Commit Hooks

The `.pre-commit-config.yaml` runs the following hooks on Python files:

| Hook | Scope | Notes |
|---|---|---|
| **black** | All `.py` files | Code formatting |
| **pylint** | All `.py` except `doc/convert_samples.py`, `doc/conf.py` | Linting |
| **isort** | All `.py` except samples scripts | Import sorting (Black profile) |
| **isort-samples** | `khiops/samples/samples.py`, `samples_sklearn.py` | Import sorting with `--no-sections` |
| **samples-generation** | Triggered by changes to samples scripts | Runs `doc/convert-samples-hook` to regenerate reST pages and notebooks |
