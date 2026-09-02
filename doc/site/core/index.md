# khiops.core

::: khiops.core
    options:
      members: false
      show_root_heading: false

!!! note

    Input types in this module are flexible. See [Notes](../notes.md#core-api-input-types).

!!! note

    For convenience, the public members of the above modules are imported to the `core` namespace.
    For example the function [train_predictor][khiops.core.api.train_predictor] can be used as follows:

    ```python
    from khiops import core as kh
    kh.train_predictor(...)
    ```

## Modules

- [api](api.md) - Main functions for training models and deploying predictors
- [dictionary](dictionary.md) - Data classes for Khiops dictionary files
- [analysis_results](analysis_results.md) - Data classes for Khiops report files
- [coclustering_results](coclustering_results.md) - Data classes for coclustering report files
- [exceptions](exceptions.md) - Exception classes
- [helpers](helpers.md) - Helper functions
