---
applyTo: "doc/**"
---

# Documentation Changes

Use these rules for files under `doc/`. Apply the shared guidance from
`.github/copilot-instructions.md` first, then this documentation-specific
guidance.

## Folder Structure

```
doc/
├── conf.py                  # Sphinx configuration
├── index.rst                # Top-level doc page
├── create-doc               # Full build script (tutorials + Sphinx)
├── clean-doc                # Clean script (supports --clean-tutorial)
├── convert-samples-hook     # Pre-commit hook: regenerates sample reST + notebooks
├── convert_samples.py       # Converts samples.py / samples_sklearn.py to reST or .ipynb
├── convert_tutorials.py     # Converts tutorial Jupyter notebooks to reST
├── requirements.txt         # Python doc-build dependencies
├── multi_table_primer.rst   # Multi-table learning guide
├── notes.rst                # API notes (common params, input types, sampling)
├── core/index.rst           # khiops.core API reference (autosummary)
├── sklearn/index.rst        # khiops.sklearn API reference (autosummary)
├── internal/index.rst       # Internal modules reference
├── tools/index.rst          # khiops.tools reference
├── samples/                 # Generated reST sample pages (via convert-samples-hook)
├── tutorials/               # Generated reST tutorials (via create-doc -t)
├── _static/                 # CSS and images (branding, logo)
└── _templates/autosummary/  # Custom autosummary templates (class, function, method, module)
```

## Build and Validation

```bash
cd doc

# Install doc dependencies (do NOT create a virtualenv inside doc/ — Sphinx will process its .rst files)
pip install -U -r requirements.txt

# Also requires:
# - A system-wide pandoc installation (used by nbconvert for notebook→reST conversion)
# - The 'black' Python package (used by convert_samples.py to format code snippets)

# Regenerate reST samples and notebooks from samples.py / samples_sklearn.py.
# This hook also runs automatically via pre-commit when those files are modified.
./convert-samples-hook

# Full build: download tutorials, convert notebooks to reST, run Sphinx
./create-doc -d -t

# Incremental build (Sphinx only, after reST files are already generated):
sphinx-build -M html . _build/

# Clean generated docs (add --clean-tutorial to also remove tutorials/ and khiops-python-tutorial/)
./clean-doc
```

The `create-doc` script requires `tar`, `python`, `make`, `zip`, and `git` (if
downloading tutorials). Output goes to `doc/_build/html/`.

The `create-doc` script accepts the following options:

- `-d` — Download the khiops-python-tutorial repository (implies `-t`)
- `-t` — Transform tutorial Jupyter notebooks into reST
- `-r REPO_URL` — Set the tutorial repository URL
- `-g GIT_REF` — Set the tutorial repository Git reference (branch or tag)
- `-l DIR` — Set the local directory of the tutorial repository

## CI Workflow

The **API Docs** workflow (`.github/workflows/api-docs.yml`) triggers on:

- **Tag pushes** — builds docs and uploads a zip archive to GitHub Releases as
  a prerelease (with `allowUpdates: true`)
- **PRs** touching `doc/**.rst`, `doc/create-doc`, `doc/clean-doc`, `doc/*.py`,
  `khiops/**.py`, or the workflow file itself
- **`workflow_dispatch`** with optional inputs:
  - `khiops-python-tutorial-revision` (default: `11.0.0.0`)
  - `khiops-samples-revision` (default: `11.0.0`)
  - `image-tag` (default: `latest`) — the dev Docker image tag

The workflow uses a concurrency group (`pages`) so only one deployment runs at a
time — queued runs are skipped but in-progress runs are never cancelled.

**Build job** — runs inside the
`ghcr.io/khiopsml/khiops-python/khiopspydev-ubuntu22.04:<image-tag>` Docker
image:

1. Installs the khiops-python package itself (`pip install .`)
2. Downloads sample datasets via `kh-download-datasets`
3. Installs doc Python requirements from `doc/requirements.txt`
4. Runs `./create-doc -t -d -g <tutorial-revision>`
5. Uploads the built HTML as a `api-docs` artifact

**Release job** (tag pushes only) — downloads the artifact, zips it, and
uploads the zip to GitHub Releases.

## Sphinx Setup

- **Engine**: Sphinx with the [Furo](https://pradyunsg.me/furo/) theme
  (Orange-branded colors and Helvetica Neue font)
- **Docstring format**: [NumPy style](https://numpydoc.readthedocs.io/en/latest/format.html) parsed by the `numpydoc` extension
  (`numpydoc_show_class_members = False`)
- **Extensions**: `autodoc`, `autosummary`, `intersphinx`, `numpydoc`,
  `sphinx_copybutton`
- **Intersphinx targets**: Python, pandas, scikit-learn, NumPy, SciPy
- **Custom templates**: `_templates/autosummary/` provides templates for
  `class.rst`, `function.rst`, `method.rst`, `module.rst`
- **Strict mode**: `nitpicky = True` — broken references are errors
- **Default role**: `obj` (configured as `default_role = "obj"` in `conf.py`) —
  allows cross-referencing without explicit `:class:`/`:func:` qualifiers in
  most cases
- **Warning suppression**: `conf.py` defines a `suppress_sklearn_warnings`
  callback that silences known false-positive missing-reference warnings for
  sklearn variables (`X`, `y`) and tutorial literals
- Sphinx warnings **should not be ignored** — they almost always indicate
  rendering errors

## Docstring Conventions

### Parameters and Attributes (NumPy format)

**Always put a space before the colon** or the rendering will break:

```
# Mandatory parameter
some_param : str
    Description ending in a period.

# Optional parameter
some_param : str, optional
    Description ending in a period.

# Optional with default
some_param : int, default 10
    Description ending in a period.
```

### Punctuation Rules

- Docstring title: **no punctuation**. Put details in the long description.
- Parameter/attribute header: only a colon, no trailing period.
- Parameter/attribute description: **must end in a period**.

```python
# Correct
def train(data):
    """Trains a model

    Trains a supervised model on the provided dataset.
    """
```

### Verbatim Markup

Use for: Python constants (`True`, `None`), file names/extensions, parameter names.
Do **not** use for: string values (use double quotes), numeric values.

### Container Types

Keep concise — use `list of <type>` for simple cases. For complex containers, put
`list` or `dict` and describe contents in the description body.

### Type Referencing

Use type referencing (backtick cross-references) only for complex types and
Exceptions. Do not use it for built-in types like `str` or `int`.

```
# No — str and int link to the Python docs unnecessarily:
some_string : `str`
some_int : `int`

# Yes — Khiops internal class:
dictionary : `.Dictionary`

# Yes — Pandas project class (via intersphinx):
df : `pandas.DataFrame`

# Yes — Exception:
Raises
------
`ValueError`
    When something wrong happens.
```

### Cross-References

```rst
`~khiops.core.api.train_predictor`     # shows "train_predictor" (short form)
`khiops.core.api.train_predictor`      # shows full path
`.train_predictor`                      # wildcard — works if unambiguous
`train_predictor`                       # within the same module
```

Use explicit `:func:`, `:class:` domains only for complex types and Exceptions.
The `default_role = "obj"` setting handles most cases.

## reStructuredText Pitfalls

The docstrings use **reST, not Markdown**. Key differences:

- **Lists** require an empty line before the first item
- **Code blocks** use `::` (with empty line + indentation) or `.. code-block:: python`
- **Links**: `` `Link text <https://example.com>`_ `` instead of `[text](url)`
