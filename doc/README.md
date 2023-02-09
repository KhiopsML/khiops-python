# pyKhiops Documentation²
The documentation of the documentation.

Below you'll find the tools and practices related to the pyKhiops documentation.

## Build the documentation
```bash
# Working dir = pykhiops/doc

# You'll need the python packages in the requirements.txt file in this directory
# Warning: If you create a virtualenv, do not place it within the pykhiops/doc directory.
#          The installed packages contain reST files and Sphinx will process them!
# pip install -U -r requirements.txt

# You'll also need a system-wide installation of pandoc (https://pandoc.org)

# Execute this if there were non commited updates to samples.py or samples_sklearn.py:
# ./convert-samples-hook

# To clean the html documentation
# ./clean.sh


# Create the HTML documentation (
# - Downloads the pykhiops-tutorial resources
# - Generates the reST version of the tutorials
# - Executes Sphinx (output: ./_build/html)
./create.sh

# To only execute Sphinx on updated reST resources
# make html
```

## Sphinx
We use [Sphinx](https://www.sphinx-doc.org/en/master/) to generate the documentation and
the [Numpy docstring format](https://numpydoc.readthedocs.io/en/latest/format.html).

Within Sphinx we use the following extensions:
- `numpydoc`: Parses the numpydoc docstrings **and** creates compact reST output.
- `sphinx.ext.autdoc`: Automatically creates the documentation from docstrings.
- `sphinx.ext.autosummary`: Automatically creates summaries from the module structure. It depends
  on `autodoc`.
- `sphinx.ext.intersphinx`: Creates links to other Sphinx-generated sites (eg. Python doc, Pandas
  doc).
- `sphinx_copybutton`: Puts a copy button in all code snippets within the documentation.

Any of these extensions can be the culprit of bogus or invalid output. Note that the Sphinx
documentation for some of these extensions is not complete and even answers in StackOverflow are
not up to date.

Warnings emitted by Sphinx **should not be ignored** as there are most likely rendering errors.

[reStructuredText](https://docutils.sourceforge.io/rst.html) or *reST* is the input format of
Sphinx. One important thing that while very similar **it is not Markdown**, reST is not identical
to it. See below [reSTructuredText Common Problems](#restructuredtext-common-problems).

## pyKhiops Docstring Patterns

When documenting pyKhiops respect the following docstring patterns.

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
Use verbatim (double backticks ` `` `) in mid-sentence for:
- Common Python constants (`True`, `None`)
- File names and extensions
- Parameter names

Do not use verbatim for
- String values (use double quotes instead)
- Int or float values

```
# No:
some_string : ``AValue`` or ``AnotherValue``

some_boolean : optional, default "True"

dictionary_file : str
    With extension ".kdic"

some_parameter : int
    When greater than ``0`` affects "other_parameter"

# Yes:
some_string : "AValue" or "AnotherValue"

some_boolean : bool, default ``True``

dictionary_file : str
    A file with extension ``.kdic``

some_parameter : int
    When greater than 0 affects ``other_parameter``
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
# int and str point to the Python docs (via intersphinx)
some_string : `str`
    a string
some_int : `int`
    an int

# Yes:
# pyKhiops internal class
dictionary : `.Dictionary`
  A Khiops dictionary.

# Pandas project class (via intersphinx)
df : `pandas.DataFrame`
  A dataframe.

# Python project class (via intersphinx)
Raises
------
`ValueError`
   When something wrong happens.
```

## reStructuredText Common Problems

There are three common cases where the differences may pose problems: lists, monospaced blocks and
links.

### Lists
In reST lists *must* have an empty line before and when nesting. So the following Markdown list:
```
These are some letters:
- A
- B
- C
- D
The end
```
must be written in the following way:
```
These are some letters:

- A
- B
- C
- D
The end
```
Now, because Sphinx makes many transformations (docstring -> reST -> HTML) *sometimes* it is
possible to get away without the extra spaces or by indenting the lists. But this is context
dependent and one must check if the output is the desired one (no warnings is a good sign).

### Monospaced Blocks
Consider the following monospaced block in Markdown
````
The following is a monospaced text:
```
Some
|- monospaced
|- text
```
Nice figure
````
In reST there are at least two ways
```
The following is a monospaced text:
::

    Some
    |- monospaced
    |- text

Nice figure
```
or the more compact
```
The following is a monospaced text::

    Some
    |- monospaced
    |- text

Nice figure
```
Note that is necessary
- an empty line after the `::` operator and after the monospaced text
- an indentation of the monospaced text.

A third way allows to specify programming languages
```
Some python code
.. code-block:: python

    import pprint
    pprint.print("hola")

That was a nice snippet.
```
Note again that indentation and empty lines are necessary.
### Links and Cross References
The external URL link in Markdown
```
[Python website](https://www.python.org)
```
can be expressed in reST as
```
`Python website <https://www.python.org>`_
```
or
```
`Python website`_

.. _`Python website`: https://www.python.org
```

For internal URL links the Sphinx semantics called *domains* help to reference diverse
elements of the module. For example the `train_predictor` function of the core API
belongs to the `:func:` domain so we can reference it as:
```
:func:`pykhiops.core.api.train_predictor`
```
This will show the long link `pykhiops.core.api.train_predictor`. Adding a `~` before
the path makes the link show only the last component `train_predictor`.
```
:func:`~pykhiops.core.api.train_predictor`
```
Additionally, we configured Sphinx with
```python
# In conf.py file
default_object = 'obj'
```
which allows to not use the *domain* most of the time. So the link can be further shortened to
```
`~pykhiops.core.api.train_predictor`
```
and if we were referencing within the `pykhiops.core.api` module one can simply write
```
`train_predictor`
```
Outside the module there is a compact way to reference it with a _wildcard_
```
`.train_predictor`
```
but be careful about name collisions.

See the Sphinx documentation for more information about referencing.

