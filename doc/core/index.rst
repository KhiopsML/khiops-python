=============
pykhiops.core
=============
.. automodule:: pykhiops.core

.. note::
  Input types in this module are flexible. See :doc:`../notes`.

.. note::
  For convenience, the public members of the above modules are imported to the ``core`` namespace.
  For example the function `~.api.train_predictor` can be used as follows::

    from pykhiops import core as pk
    pk.train_predictor(...)

Main Modules
============
.. autosummary::
  :toctree: generated
  :recursive:
  :nosignatures:

  pykhiops.core.api
  pykhiops.core.dictionary
  pykhiops.core.analysis_results
  pykhiops.core.coclustering_results
  pykhiops.core.exceptions
