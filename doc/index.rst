
Khiops Python API Docs
======================

Welcome to the Khiops Python API documentation page.

Installation
------------
Khiops is better installed with `conda package manager <https://docs.conda.io/en/latest/>`_

.. code-block:: shell

    # Windows
    conda install khiops -c khiops

    # Linux/macOS
    conda install khiops -c conda-forge -c khiops

More details and other install methods are documented at the `Khiops website
<https://www.khiops.org/setup>`_.


Main Submodules
---------------
This package contains the following main submodules.

``sklearn`` submodule
~~~~~~~~~~~~~~~~~~~~~
The :doc:`sklearn/index` module is a `Scikit-learn <https://scikit-learn.org>`_ based interface to
Khiops. Use it if you are just started using Khiops and are familiar with the Scikit-learn workflow
based on dataframes and estimator classes.

``core`` submodule
~~~~~~~~~~~~~~~~~~
The :doc:`core/index` module is a pure Python library exposing all Khiops functionalities. Use it if
you are familiar with the Khiops workflow based on plain-text tabular data files and dictionary
files (``.kdic``).

.. toctree::
  :caption: API Reference
  :hidden:

  sklearn <sklearn/index>
  core <core/index>
  tools <tools/index>
  internals <internal/index>

.. toctree::
  :caption: Tutorials and Code Samples
  :hidden:

  Tutorials <tutorials/index>
  samples/samples_sklearn
  Samples core <samples/samples>

.. toctree::
  :caption: Other Topics
  :hidden:

  multi_table_primer
  notes

