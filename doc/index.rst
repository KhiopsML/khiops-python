pyKhiops Documentation
======================

Welcome to the pyKhiops library documentation. With it is it possible to script the  `Khiops Auto ML
suite <https://www.khiops.com>`_ functionalities. This package contains the following main
submodules.

pyKhiops ``core``
-----------------
The :doc:`core/index` module is a pure Python library exposing all Khiops functionalities.  Use it
if you are already familiar with the Khiops workflow based on plain-text tabular data files and
dictionary (``kdic``) files.

pyKhiops ``sklearn``
--------------------
The :doc:`sklearn/index` module is a `Scikit-learn <https://scikit-learn.org>`_ based interface to
Khiops. Use it if you are just starting to use Khiops and are familiar with the Scikit-learn
workflow based on dataframes and estimator classes.


.. toctree::
  :caption: Main Modules
  :hidden:

  core/index
  sklearn/index

.. toctree::
   :caption: Complementary Materials
   :hidden:

   Samples core <samples/samples>
   samples/samples_sklearn
   multi_table_tasks
   core_input_types
   json_files

.. toctree::
  :caption: Internal modules
  :hidden:

  internal/index
