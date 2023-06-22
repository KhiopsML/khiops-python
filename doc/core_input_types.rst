Note on Input Types
===================

The types accepted in most methods and classes of `pykhiops.core` are flexible:

- `str` can be replaced by `bytes`

  - This adds flexibility for file paths and automatically created variable names (data-dependent).

- `list` can be replaced by any class implementing the `collections.abc.Sequence` interface except
  `str` and `bytes`.
- `dict` can be replaced by any class implementing the `collections.abc.Mapping` interface.

