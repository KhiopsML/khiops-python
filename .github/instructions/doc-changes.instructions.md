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
├── util/                        # Build tooling (outside Zensical's docs_dir)
│   ├── create-doc               # Full build script (tutorials + Zensical)
│   ├── clean-doc                # Clean script (supports --clean-tutorial)
│   ├── convert-samples-hook     # Pre-commit hook: regenerates sample Markdown + notebooks
│   ├── convert_samples.py       # Converts samples.py / samples_sklearn.py to Markdown or .ipynb
│   ├── convert_tutorials.py     # Converts tutorial Jupyter notebooks to Markdown
│   ├── requirements.txt         # Python doc-build dependencies
│   └── README.md                # Documentation guide
├── site/                        # docs_dir (Zensical content only)
│   ├── index.md                 # Top-level doc page
│   ├── multi_table_primer.md    # Multi-table learning guide
│   ├── notes.md                 # API notes (common params, input types, sampling)
│   ├── core/index.md            # khiops.core API reference (mkdocstrings)
│   ├── sklearn/index.md         # khiops.sklearn API reference (mkdocstrings)
│   ├── internal/index.md        # Internal modules reference
│   ├── tools/index.md           # khiops.tools reference
│   ├── samples/                 # Generated Markdown sample pages (via convert-samples-hook)
│   ├── tutorials/               # Generated Markdown tutorials (via create-doc -t)
│   ├── _static/                 # CSS and images (branding, logo)
│   └── _templates/              # mkdocstrings Jinja templates
└── build/                       # Zensical output (site_dir)
    └── html/
```

The Zensical configuration file `zensical.toml` is at the repository root.

## Build and Validation

```bash
# Install doc dependencies (do NOT create a virtualenv inside doc/ — Zensical will process its .md files)
pip install -U -r doc/util/requirements.txt

# Also requires:
# - The 'black' Python package (used by convert_samples.py to format code snippets)

# Regenerate Markdown samples and notebooks from samples.py / samples_sklearn.py.
# This hook also runs automatically via pre-commit when those files are modified.
doc/util/convert-samples-hook

# Full build: download tutorials, convert notebooks to Markdown, run Zensical
doc/util/create-doc -d -t

# Incremental build (Zensical only, after Markdown files are already generated):
zensical build

# Serve locally for development:
zensical serve

# Clean generated docs (add --clean-tutorial to also remove tutorials/ and khiops-python-tutorial/)
doc/util/clean-doc
```

The `create-doc` script requires `python`, `zip`, and `git` (if
downloading tutorials). Output goes to `doc/build/html/`.

The `create-doc` script accepts the following options:

- `-d` — Download the khiops-python-tutorial repository (implies `-t`)
- `-t` — Transform tutorial Jupyter notebooks into Markdown
- `-r REPO_URL` — Set the tutorial repository URL
- `-g GIT_REF` — Set the tutorial repository Git reference (branch or tag)
- `-l DIR` — Set the local directory of the tutorial repository
- `-p` — Prepare only: download tutorials, convert notebooks, create ZIPs,
  and copy samples, but skip the final Zensical build. Used by the
  [khiops-doc](https://github.com/KhiopsML/khiops-doc) CI.

## CI Workflow

The **API Docs** workflow (`.github/workflows/api-docs.yml`) validates
documentation builds. It triggers on:

- **PRs** touching `doc/site/**.md`, `doc/util/create-doc`, `doc/util/clean-doc`, `doc/util/*.py`,
  `zensical.toml`, `khiops/**.py`, or the workflow file itself
- **`workflow_dispatch`** with optional inputs:
  - `khiops-python-tutorial-revision` (default: `11.0.0.0`)
  - `khiops-samples-revision` (default: `11.0.0`)
  - `image-tag` (default: `latest`) — the dev Docker image tag

**Build job** — runs inside the
`ghcr.io/khiopsml/khiops-python/khiopspydev-ubuntu22.04:<image-tag>` Docker
image:

1. Installs the khiops-python package itself (`pip install .`)
2. Downloads sample datasets via `kh-download-datasets`
3. Installs doc Python requirements from `doc/util/requirements.txt`
4. Runs `doc/util/create-doc -t -d -g <tutorial-revision>`
5. Uploads the built HTML as a `api-docs` artifact

Note: the production API docs are built by the
[khiops-doc](https://github.com/KhiopsML/khiops-doc) CI, which clones this
repository at the version tag and builds the docs natively using mkdocstrings.

## Zensical Setup

- **Engine**: Zensical with the [Material](https://squidfunk.github.io/mkdocs-material/)
  theme (Orange-branded colors and Helvetica Neue font)
- **Docstring format**: [NumPy style](https://numpydoc.readthedocs.io/en/latest/format.html)
  parsed by the `mkdocstrings[python]` plugin
- **Plugins**: `mkdocstrings[python]`, `autorefs`, `search`
- **Intersphinx-equivalent**: mkdocstrings `import` option loads inventory files
  from Python, pandas, scikit-learn, NumPy, SciPy
- **Cross-references**: Use `[display text][fully.qualified.name]` or
  `[fully.qualified.name][]` syntax for linking to documented objects
- **Custom CSS**: `doc/site/_static/css/custom.css` provides Orange branding via CSS
  custom properties (Material theme variables)

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

Use cross-references only for complex types and Exceptions. Do not use them for
built-in types like `str` or `int`.

```
# No — str and int do not need cross-references:
some_string : str
some_int : int

# Yes — Khiops internal class:
dictionary : `Dictionary`

# Yes — Pandas project class (via intersphinx inventory):
df : `pandas.DataFrame`

# Yes — Exception:
Raises
------
`ValueError`
    When something wrong happens.
```

### Cross-References in Markdown

```markdown
[train_predictor][khiops.core.api.train_predictor]   # shows "train_predictor"
[khiops.core.api.train_predictor][]                  # shows full path
```

Use mkdocstrings `:::` directives for API documentation blocks:

```markdown
::: khiops.core.api
    options:
      heading_level: 3
```
