=============
khiops.core
=============
.. automodule:: khiops.core

.. note::
  Input types in this module are flexible. See :doc:`../notes`.

.. note::
  For convenience, the public members of the above modules are imported to the ``core`` namespace.
  For example the function `~.api.train_predictor` can be used as follows::

    from khiops import core as kh
    kh.train_predictor(...)

Main Modules
============
.. autosummary::
  :toctree: generated
  :recursive:
  :nosignatures:

  api
  dictionary
  analysis_results
  coclustering_results
  exceptions
  helpers
