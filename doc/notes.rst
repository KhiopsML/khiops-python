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


Database Sampling Modes for the pykhiops.core API
-------------------------------------------------

Introduction
~~~~~~~~~~~~

Several ``pykhiops.core`` API functions can operate on dataset *samples* instead of the full
datasets.  The dataset sampling behavior is fully customizable by the user. Basically, one can
specify that the function operates on the specified sample, or on its *complement*. In some cases,
this *complement* can be used for assessing the result of the function operation (viz. the
statistical model).

Common Behaviour
~~~~~~~~~~~~~~~~

Several ``pykhiops.core`` API functions allow the user to control the dataset sampling behaviour via
two parameters:

- ``sample_percentage``: a real number beween 0.0 and 100.O which specifies the relative size of the
  data that constitutes the sample

- ``sampling_mode``: a string which specifies the function operation behaviour with respect to the
  sample:

  - "Include sample": the function operates on a randomly-chosen ``sample_percentage`` percent of
    the individuals in the dataset

  - "Exclude sample": the function operates on the *complement* of the randomly-chosen
    ``sample_percentage`` percent of the individuals in the dataset, that is, on the remaining
    ``100.0 - sample_percentage`` percent of the individuals in the dataset

Some Particulars
~~~~~~~~~~~~~~~~

A specific ``pykhiops.core`` API function, ``train_predictor``, can use both the dataset sample and
its complement, but for different purposes: for training the model *and* for computing the model's
performance metrics. This behaviour is triggered by the ``use_complement_as_test`` boolean flag: if
set, then the complement of the data being used for training the model is used for evaluating the
model's performance metrics. In turn, the data being used for training the model is controlled by
the ``sample_percentage`` and ``sampling_mode``, as shown above.

More specifically, for example, if:

- ``sample_percentage`` is set to 20.0
- ``sampling_mode`` is set to "Exclude sample"
- ``use_complement_as_test`` is set to ``True``

then the data sample will consist of a set of randomly-chosen 20.0 % of the individuals in the
dataset. However, the model will be trained on the complement of the data sample, viz. on the
remaining 80.0 % of the dataset. Moreover, the model will be evaluated on the complement of the
*data that has been used for training the model*, that is, on the initially-chosen 20.0 of the
dataset.
