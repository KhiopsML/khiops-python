pykhiops.core
=============

.. currentmodule:: pykhiops

.. automodule:: pykhiops.core

.. autosummary::
  :caption: core
  :recursive:
  :nosignatures:
  :toctree:

  pykhiops.core.api
  pykhiops.core.helpers
  pykhiops.core.dictionary
  pykhiops.core.analysis_results
  pykhiops.core.coclustering_results
  pykhiops.core.common


.. note::
  Input types on this module are flexible. See :doc:`../core_input_types`

.. note::
  For convenience, the public members of the above modules are imported to the ``core`` namespace.
  For example the function `~.api.train_predictor` can be used as follows::

    from pykhiops import core as pk
    pk.train_predictor(...)

Related Docs
------------
- :doc:`../samples/samples`
- :doc:`../multi_table_tasks`
- :doc:`../json_files`
