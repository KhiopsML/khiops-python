Notes
=====

pykhiops.core Functions Input Types
-----------------------------------

The types accepted in most methods and classes of `pykhiops.core` are flexible:

- `str` can be replaced by `bytes`

  - This adds flexibility for file paths and automatically created variable names (data-dependent).

- `list` can be replaced by any class implementing the `collections.abc.Sequence` interface except
  `str` and `bytes`.
- `dict` can be replaced by any class implementing the `collections.abc.Mapping` interface.


Khiops JSON Files
-----------------

Generalities
~~~~~~~~~~~~

The structure of the Khiops JSON files is self-documented:

- Most of the information is available as key-value pairs, where the keys resemble the labels used
  in Khiops' classic report files (tab-separated plain-text files with extension ``.xls``) or
  dictionary files.
- In order to be human-readable the files are *beautified* with a comfortable spacing and
  indentation.

Structure and Performance
~~~~~~~~~~~~~~~~~~~~~~~~~

The Khiops JSON files may be large (tens of MB) when analyzing datasets with many columns, or when
specifying the creation of thousands of variables in the multi-table case. To handle these
situations, the report attributes in the JSON file are sorted by increasing size, thus easing the
use of streaming parsers.

Furthermore, memory-scalable parsing techniques can be implemented. For example, the heavier parts
of the file can be separated and split into chunks. Then, these chunks can be indexed using the
information found at the top of the report, allowing the on-demand access to the detailed parts of
the report.

Khiops Report Files Structure (.khj)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At the top level the order is as follows:

#. Modeling report
#. Evaluation report(s)
#. Preparation report(s)

The preparation reports are at the end because they can be very large when many
variables are analyzed.

Each report field is organized in three sections:

- Summary: General (short) information about the report
- A list of report items:

    - Variable statistics (preparation), trained predictor (modeling) and predictor
      performance (evaluation)
    - Each item has a "Rank"

        - Example: The second most informative variable has the categorical rank "R02"

    - Each item is described by a few summary attributes

- A dictionary of detailed report items. The keys of this dictionary are the
  previously mentioned "Rank" attributes. Note that:

    - Not all report items are detailed
    - The detailed information may be large (example: data grid).
