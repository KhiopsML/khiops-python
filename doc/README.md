# Khiops Python Library Documentation
The documentation of the documentation.

Below you'll find the tools and practices related to the documentation of the
Khiops Python library.

## Build the documentation
```bash
# Working dir = khiops-python (repository root)

# You'll need the python packages in the doc/util/requirements.txt file
# Warning: If you create a virtualenv, do not place it within the doc/util
directory. The installed packages may contain .md files and Zensical will
process them!

# Execute this if there were non committed updates to samples.py or samples_sklearn.py:
# doc/util/convert-samples-hook

# To clean the html documentation
# doc/util/clean-doc

# Create the HTML documentation:
# - Downloads the khiops-python-tutorial resources
# - Generates the Markdown version of the tutorials
# - Executes Zensical (output: doc/build/html)
doc/util/create-doc -d -t

# To only execute Zensical on updated Markdown resources
# zensical build

# To serve locally for development (with live reload)
# zensical serve
```

## Zensical
We use [Zensical](https://www.zensical.org/) with the
[Material](https://squidfunk.github.io/mkdocs-material/) theme to generate the
documentation and the
[NumPy docstring format](https://numpydoc.readthedocs.io/en/latest/format.html).

The Zensical configuration file `zensical.toml` lives at the repository root.

The following MkDocs plugins and extensions are used in Zensical through
Zensical's MkDocs compatibility layer:
- `mkdocstrings[python]`: Automatically creates API documentation from Python docstrings
  (NumPy format). Replaces Sphinx's `autodoc`, `autosummary`, and `numpydoc`.
- `autorefs`: Enables cross-references to documented objects across pages.
- `search`: Built-in search functionality.
- `pymdownx.superfences`, `pymdownx.highlight`: Fenced code blocks with syntax highlighting
  and a copy button.
- `admonition`, `pymdownx.details`: Note/warning/tip admonition blocks.

Cross-references to external projects (Python, pandas, scikit-learn, NumPy, SciPy) are
handled via `objects.inv` inventory files configured in the `mkdocstrings` handler's
`import` option.

## Khiops Python Docstring Patterns

When documenting Khiops Python, respect the following docstring patterns.

### Parameter and Attributes
Mandatory parameters and attributes must be written as follows
```
<name> : <type>
    <description>
```
and optional parameters
```
<name> : <type>, default <value>
    <description>

# If the default value is None
<name> : <type>, optional
    <description>
```
Do not forget **to put a space before a colon**. If you do not the documentation will not be
rendered as expected.
```
# No:
some_file_path: str
  A path to a file.

# Yes:
some_file_path : str
  A path to a file.
```

### Punctuation
The title of a docstring should not be punctuated in any way. This is to enforce a simple
description. If there are relevant details put it in the long the description of the docstring.
```python
# No:
def some_method(some_parameter):
    """Does something, allowing the next thing."""

# Yes:
def some_method(some_parameter):
   """Does something

   The thing done allows another thing afterwards
   """
```
The header of the documentation of a parameter attribute should contain only a colon
```
# No:
some_parameter : str, optional.

# Yes:
some_parameter : str, optional
```

The description of a parameter or attribute should end in a period.
```
# No:
some_parameter : str, optional
    The main parameter

# Yes:
some_parameter : str, optional
    The main parameter.
```

### Verbatim
Use verbatim (backticks `` ` ``) in mid-sentence for:
- Common Python constants (`True`, `None`)
- File names and extensions
- Parameter names

Do not use verbatim for
- String values (use double quotes instead)
- Int or float values

```
# No:
some_string : "AValue" or "AnotherValue"

some_boolean : optional, default "True"

dictionary_file : str
    With extension ".kdic"

some_parameter : int
    When greater than `0` affects "other_parameter"

# Yes:
some_string : "AValue" or "AnotherValue"

some_boolean : bool, default `True`

dictionary_file : str
    A file with extension `.kdic`

some_parameter : int
    When greater than 0 affects `other_parameter`
```

### Container Types
The description of container types such as `list` should be kept concise:
- If the contained type is simple just put `list of <type>`
- If the contained type is complex put `list` and describe the contents in the description, types
  optional.

```
# No:
a_bunch_of_stuff : list of tuple(str, list of str)
    A bunch of tuple stuff.

# Yes:
a_bunch_of_stuff : list of tuple
    A bunch of 2-tuple stuff. Each 2-tuple contains a:
        - key : str
        - the stuff : list of str
```
For container types such as `dict` describe the keys and the values types in the description.

### Type referencing
Use type referencing only for complex types and Exceptions
```python
# No:
# int and str do not need cross-references
some_string : str
    a string
some_int : int
    an int

# Yes:
# Khiops internal class
dictionary : `Dictionary`
  A Khiops dictionary.

# Pandas project class (via intersphinx inventory)
df : `pandas.DataFrame`
  A dataframe.

# Exception
Raises
------
`ValueError`
   When something wrong happens.
```

## Cross-References in Markdown

The documentation pages use Markdown, not reST. Cross-references use the
mkdocstrings/autorefs syntax:

```markdown
# Link showing "train_predictor"
[train_predictor][khiops.core.api.train_predictor]

# Link showing the full path
[khiops.core.api.train_predictor][]
```

For API documentation blocks, use the `:::` directive:

```markdown
::: khiops.core.api
    options:
      heading_level: 3
```

### Admonitions
Notes and warnings use the `!!!` syntax:

```markdown
!!! note
    This is a note.

!!! warning
    This is a warning.
```

See the [Zensical admonitions docs](https://zensical.org/docs/authoring/admonitions/) for more details.
