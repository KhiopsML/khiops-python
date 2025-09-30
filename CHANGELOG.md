# Changelog

**Notes:**
- Versioning: The versioning scheme depends on the Khiops version supported (first 3 digits) and
  a Khiops Python Library correlative (4th digit).
  - Example: 10.2.1.4 is the 5th version that supports khiops 10.2.1.
- Internals: Changes in *Internals* sections are unlikely to be of interest for data scientists.

## Unreleased -

### Added
- (`core`) Dictionary API support for dictionary, variable and variable block
  comments, and dictionary and variable block internal comments.
- (`core`) Dictionary `Rule` class and supporting API for serializing `Rule` instances.
- (`core`) New way to add a variable to a dictionary using a complete specification.
- (`sklearn`) `Text` Khiops type support at the estimator level.

### Fixed
- (General) Inconsistency between the `tools.download_datasets` function and the
  current samples directory according to `core.api.get_samples_dir()`.
  

## 11.0.0.0-b.0 - 2025-07-10

### Added
- (`core`) API support for predictor interpretation and reinforcement.
- (`core`) API support for instance-variable coclustering model training.
- (`core`) Support for text types in prediction and coclustering models.
- (`core`) Analysis and coclustering report JSON serialization support.
- (`sklearn`) Automatic removal of newline characters in strings on Pandas
  dataframe columns. This is to ensure the proper working of the Khiops engine.

### Changed
- (`core`) Syntax for additional data tables specification, which uses the data
  paths.
- (`core`) API specification of the results path: full paths to report files are
  now used instead of result directories.
- (`sklearn`) Specification of the hierarchical multi-table schemata, which now
  uses data paths as in the Core API.
- (`general`) Various other changes and updates for Khiops 11.0.0-b.0
  compatibility.

### Deprecated
- (`core`) The results directory parameter of the Core API functions. The full
  path to the reports must now be specified instead.
- (`core`) The "``"-based secondary table path specification. The "/"-based data
  paths must now be used instead.
- (`sklearn`) The specification syntax for hierarchical multi-table datasets.
  The "/"-based data paths must now be used instead, as in the Core API.

### Removed
- (`general`) All functions, attributes and features that had been deprecated in
  the 10.3.2.0 version.

## 10.3.2.0 - 2025-07-03

### Fixed
- (`sklearn`) Documentation display for the `train_test_split_dataset` sklearn
helper function.

## 10.3.1.0 - 2025-04-16

### Added
- (`sklearn`) Support for boolean and float targets in `KhiopsClassifier`.

### Fixed
- (`sklearn`) Crash when there were no informative trees in predictors.

### Deprecated
- (`core`) The `build_multi_table_dictionary_domain` helper function.

## 10.3.0.0 - 2025-02-10

### Fixed
- (`core`) Dictionary file `.json` extension check in the `khiops.dictionary.read_dictionary_file`
function.

### Changed
- (`sklearn`) The `train_test_split_dataset` helper has been moved from `khiops.utils` to
`khiops.sklearn`.
- (`sklearn`) The `transform_pairs` parameter of the `KhiopsEncoder` sklearn estimator has been
renamed to `transform_type_pairs`.

### Removed
- (`sklearn`) The `is_fitted_` estimator attribute. The Scikit-learn `check_is_fitted` function
can be used to test the fitted state of the estimators.
- (`sklearn`) The `n_pairs` parameter of the `KhiopsRegressor` sklearn estimator. It was never
supported.

## 10.2.4.0 - 2024-12-19

### Added
- (General) Support for Python 3.13.
- (General) `visualize_report` helper function to open reports with the Khiops Visualization and
  Khiops Co-Visualization app.

### Fixed
- (General) Initialization failing in Conda-based environments.

### Changed
- (`core`) Support for system parameters has been moved from the `KhiopsLocalRunner` to the `core` API.
- (`core`) System parameter `max_memory_mb` has been renamed to `memory_limit_mb`.
- (`core`) System parameter `khiops_temp_dir` has been renamed to `temp_dir`.

### Removed
- (General) Khiops Python 9 compatibility.

## 10.2.3.0 - 2024-11-13

### Added

- (`sklearn`) `train_test_split_dataset` helper function to ease the splitting in train/test for
  multi-table datasets.
- (`sklearn`) Complete support for `core` API functions parameters in the `sklearn` estimators.

### Changed

- (General) The Conda package only depends on the `conda-forge` and `khiops` channels.
- *Internals*:
  - Improve and simplify the integration with the `khiops-core` package via its `khiops_env`
  script.

## 10.2.2.4 - 2024-08-05

### Added
- (`sklearn`) Sklearn's attributes for supervised estimators.

## 10.2.2.3 - 2024-08-02

### Fixed
- (`core`) API functions handling of unknown parameters: they now fail.
- *Internals*:
  - Detection of the path to the MPI command: the real path to the executable is
    now used.

## 10.2.2.2 - 2024-07-19

### Fixed
- (`core`) Documentation of the `specific_pairs` parameter for the `train_predictor` and
  `train_recoder` core API functions.

### Deprecated
- (`core`) The following parameters of the `train_predictor` core API functions:
  - `max_groups`
  - `max_intervals`
  - `min_group_frequency`
  - `min_interval_frequency`
  - `results_prefix`
  - `snb_predictor`
  - `univariate_predictor_number`
  - `discretization_method` for supervised learning
  - `grouping_method` for supervised learning

## 10.2.2.1 - 2024-07-05

### Changed
- *Internals*:
  - The OpenMPI backend now executes with the `--allow-run-as-root` option.

## 10.2.2.0 - 2024-07-03

### Added
- (`sklearn`) Support for sparse arrays in sklearn estimators.

### Changed
- *Internals*:
  - MPI backend from MPICH to OpenMPI for native + Pip-based Linux installations.

### Fixed
- `core`
  - Metric name search in estimator analysis report.

## 10.2.1.0 - 2024-03-26

### Added
- (`sklearn`) 1:1 relations to multi-table datasets.
- (`sklearn`) Estimators' `fit` methods now accept single-column pandas dataframes as `y` target.

### Changed
- (`core`) Improve user error and warning messaging.

### Fixed
- (General) Reinstate Rocky Linux 8 support.


## 10.2.0.0 - 2024-02-15
Note: This release marks the open sourcing of Khiops:
- The `khiops` package replaces the old `pykhiops` package. We recommend to uninstall
  `pykhiops` before installing `khiops`. More information at the [Khiops site][khiops].
- The `khiops` package uses a new four digit versioning convention.
- The `khiops` conda package is available for many environments. See the [Khiops site][khiops]
  for more information.

### Added
- General:
  - `khiops-python` is now available with conda `khiops` package. This package bundles
    `khiops-python` and the Khiops binaries so no system-wide Khiops installation is needed. More
    information at the [Khiops website][khiops].
  - Support for python 3.12.
- `sklearn`
  - Estimator classes can now be trained from Numpy arrays in single-table mode.
- `core`
  - `stdout_file_path` and `stderr_file_path` parameters for `khiops.core.api` functions. These
    parameters allow to save the stdout/stderr output of the Khiops execution.

### Changed
- `sklearn`
  - Estimator classes now comply with scikit-learn standards.
- `core`
  - The JSON initialization of `AnalysisResults`, `CoclusteringResults` and its component classes
    is coherent with the empty initialization.

### Fixed
- `core`
  - Wrong default discretization and grouping methods in `train_predictor` and `train_recoder`.
  - `KhiopsDockerRunner` checking the existence `shared_dir` on remote paths.

## 10.1.3 - 2023-06-14

### Added
- `sklearn`:
  - Direct support for coclustering simplification, via the `KhiopsCoclustering.simplify` method.
- *Internals*:
  - The `TaskRegistry.set_task_end_version` method for specifying the ending Khiops version for a
    task.

### Fixed
- `sklearn`:
  - Verbose mode support is now complete for coclustering.
- *Internals*:
  - User-provided scenario prologue is now taken into account into the tasks.

### Changed
- General:
  - License has been updated to BSD-3 Clear.
- `sklearn`:
  -  ``auto_sort`` replaces ``internal_sort`` to control input table sorting in estimators.
  - The multi-table documentation has been streamlined to be more precise and clearer.

### Deprecated
- `sklearn`:
  - The `max_part_numbers` parameter of `KhiopsCoclustering` `fit` method. The `KhiopsClustering`
    `simplify` method now contains the simplification feature.
  - The `internal_sort` estimator parameter. The `auto_sort` estimator parameter replaces it.
- `core`:
  - The `build_multi_table_dictionary` API function. The `build_multi_table_dictionary_domain`
    helper function provides the same functionality.
- *Internals*:
  - The `build_multi_table_dictionary` task. This task will not be supported after Khiops 11.

## 10.1.2 - 2023-03-14

### Added
- `sklearn`:
  - Support for snowflake database schemas.
- `core`:
  - Support for Khiops on MacOS.

### Fixed
- core:
  - Khiops coclustering is not executed with MPI anymore.
  - Bug when the JSON reports had colliding character ranges but no particular colliding character.

### Changed
- *Internals*:
  - The transformation of the `core.api` function parameters to scenario files has now an additional
    layer mediated by the `KhiopsTask` class. These objects have all the necessary information about
    a particular Khiops tasks (ex. `train_predictor`) to transform its parameters to an scenario
    file. Furthermore, this allows to export the task signatures to API description languages such
    as Protocol Buffers.
  - The `core.filesystem` now exposes its API as a set of functions instead of _resource_ objects.
    They are still available but the API should be prioritized for its use.


### Removed
- General:
  - Support for Python 3.6, pyKhiops 10.1.1 was the last version to support it.

## 10.1.1 - 2022-11-10

### Added
- General:
  - Jupyter notebooks tutorials to the documentation site.
  - `pk-status` script to check the pyKhiops installation.

### Fixed
- General:
  - Code samples scripts not being installed: They are located in `<pykhiops_install_dir>/samples`.
- `sklearn`
  - `KhiopsCoclustering` raising an exception instead of a warning when no informative coclustering
    was found.
  - `internal_sort` parameter being ignored in `KhiopsCoclustering`.
- `core`
  - `detect_format` failing when the Khiops log had extra output.

## 10.1.0 - 2022-09-20

### Added
- `sklearn`:
  - Estimators now accept dataframes with numerical column indexes.
  - `KhiopsClassifier` now accepts integer target vectors.
  - `classes_` estimator attribute for `KhiopsClassifier` (available once fitted).
  - `feature_names_out_` estimator attribute for `KhiopsEncoder` (available once fitted).
  - `export_report_file` and `export_dictionary_file` to export Khiops report and dictionary files
    once the estimators are fitted.
  - `internal_sort` parameter for estimators that may be used to not sort the tables on the
    internal procedures of pyKhiops (default is `True`). Disabling it may give speed gains in large
    datasets.
  - `verbose` flag for debugging estimators: It shows internal information and doesn't erase
    temporary files.
- `core`:
  - `get_khiops_version` API function.
  - New rule `LocalTimestamp` rule for AutoML feature generation (requires Khiops 10.1).
  - `max_total_parts` parameter to `simplify_coclustering` core API function (requires Khiops 10.1).
- *Internals*:
  - Khiops samples directory in Linux now defaults to `/opt/khiops/samples` which is where it is
    installed by default.

### Changed
- `sklearn`:
  - **Breaking**: Estimators return NumPy arrays instead of dataframes in `predict`,
    `predict_proba`, `transform`, `fit_predict` and `fit_transform` methods.
- `core`:
  - `train_recoder` API function does not build trees by default anymore.
  - When pyKhiops reads a legacy Khiops JSON report/dictionary with Unicode decoding errors it now
    only warns and loads it anyway with the `errors="replace"` setting. Before it raised an
    exception.
- General:
  - Simpler multi-table samples in the documentation.

### Deprecated
- `sklearn`:
  - Datasets based on file paths. From pyKhiops 11 only in-memory datasets will be accepted. File
    based treatments can be treated with the `core` API.
  - `max_part_number` as instance parameter of `KhiopsCoclustering`. It is now a `fit` parameter.
    It will be eliminated in pyKhiops 11.
- `core`:
  - `get_khiops_info` and `get_khiops_coclustering` API functions. From Khiops 10.1 there is no
    need of license key so these methods have no use anymore. They are kept deprecated for
    backwards compatibility only. It will be eliminated in pyKhiops 11.

### Removed
- *Internals*:
  - `legacy_mode` in `PyKhiopsRunner`. It its place there is generic versioning scheme to handle
    features and Khiops scenarios.

### Fixed
- `sklearn`:
  - Bug in dataframe-based datasets with numerical key columns

## 10.0.5 - 2022-07-04

### Added
- `sklearn`:
  - A new way to specify multi-table inputs for estimators via a `dict`. From now on it is the
    standard way to specify multi-table datasets and the others are deprecated. See the
    documentation for more details.
  - New examples of use of `sklearn` in the script `samples_sklearn.py`. Available also in the
    documentation.
- `core`:
  - It now fully supports remote filesystems provided for which the extra dependencies are
    installed (it is still necessary to install Khiops remote filesystem plugins).
- Other:
  - Most methods that accept containers now additionally accept classes implementing their abstract
    interface (eg. `collections.abc.Sequence`, `collections.abc.Mapping`).
- *Internals*:
  - The default value of `samples_dir` of the `PyKhiopsLocalRunner` class can now be set via the
    environment variable `KHIOPS_SAMPLES_DIR`.
  - New classes `Dataset` and `DatasetTable` to `sklearn.tables` to handle sklearn table
    transformations for Khiops.

### Changed
- General:
  - Improved documentation completeness and layout.
- `sklearn`
  - Estimators do not depend anymore on local files. This fixes many issues including those related
    to serialization.
  - `KhiopsRegressor` now warns when `n_trees > 0`.
- `core`
  - Functions `deploy_coclustering` and `deploy_model_for_metrics` are moved from `core.api` to
    `core.helpers`. The latter module will contain non-basic functionality, whereas `core.api` will
    contain only the official Khiops API.

### Deprecated
- `sklearn`:
  - `tuple` and `list` multi-table input modes in estimators.
  - `key` parameter of estimators.
  - `variables` parameter of `KhiopsCoclustering` estimator.

### Removed
- `sklearn`:
  - **Breaking** `computation_dir` parameter in `sklearn` estimators. Khiops output files can still
    be saved with the parameter `output_dir`.
- Other:
  - **Breaking** Support for Python 2: 10.0.4 was the last version to support it.

### Fixed
- `sklearn`:
  - Data-race when using many estimators in parallel.
  - Bug in `KhiopsCoclustering` when the trained coclustering did not cluster the key variable.
  - Bug in `KhiopsEncoder` that happened because a bad handling of OS-dependent line separators.
- Other:
  - Bug with `KHIOPS_HOME` environment variable not properly being taken into account when
    initializing the runner.

## 10.0.4 - 2022-01-13

### Added
- Class `PyKhiopsDockerRunner` in package `pykhiops.extras` allowing to run pyKhiops
  with a remote Khiops Docker image as backend.

### Fixed
- Bug with `PyKhiopsRunner`'s `scenario_prologue` failing to execute.
- Bug in `pykhiops.sklearn` estimators not taking into account the target variable as
  `Used`.
- Bug in CentOS not taking into account environment variables and failing to execute.

## 10.0.3 - 2021-11-22

### Added
- `extract_clusters` core API function to extract a dimension's clusters into a file.
- `deploy_predictor_for_metrics` core API function to evaluate performance metrics with
  third-party.
- `detect_data_table_format` core API function to obtain (heuristically) the
  `header_line` and `field_separator` from a data file (requires Khiops >= 10.0.1)
  libraries.
- `train_predictor` and `evaluate_predictor` now accept a `main_target_value` parameter
- Various ease-of-access methods:
  - `AnalysisResults`: `get_reports`
  - `EvaluationReport`: `get_snb_performance`, `get_snb_lift_curve` and
    `get_snb_rec_curve`
  - `PredictorPerformance`: `get_metric` and `get_metric_names`
- New examples to `samples.py`.

Internals:
- Support for remote filesystems `s3` and `gcs` in `sklearn` module (installation with extra
  dependencies required).
- New `scenario` module containing classes to write templatized-scenarios and that also handle
  character encoding (see *Fixed* below).
- Support for the new `subTool` key of Khiops JSON files.
- Command-line options for `samples.py` to specify which samples to run.

### Changed
- `str` parameters of core API functions may now also be `bytes` and `bytearray`
- Changed all `core` module docstrings to the "NumPy" style.
- `write` and `writeln` methods of classes in the `dictionary`, `analysis_results` and
  `coclustering_results` now require a `PyKhiopsOutputFile` object as argument.
- Query methods such as `get_dictionary` from `DictionaryDomain` now raise `KeyError`
  instead of returning `None` if the query fails.
- Core API functions that use a `field_separator` parameter now accept the string "\\t".
- `KhiopsClassifier` and `KhiopsRegressor` now warn of incorrect types of target variable.
- *Internals*:
  - `PyKhiopsLocalRunner` now calls directly the `MODL` executables instead of the Khiops launch
    scripts (only for Khiops >= 10.0.1).
  - Specific pair parameter is not handled anymore with a temporary file.
  - Improved temporary file services in `PyKhiopsRunner`.

### Removed
- Field separator constructor parameter for estimator classes of `sklearn`

### Fixed
- Dictionary files created with pyKhiops are now guaranteed to be free of character
  encoding errors unless the new JSON field `khiops_encoding` is non-existent or set to
  `colliding_ansi_utf8` in which case a warning is emitted
- Khiops execution problems due to the character encoding of certain parameters
- Khiops error reporting problems due to to character encoding
- `train_coclustering` now returns the path of the JSON coclustering report (`.khcj`)
- `get_dimensions` not working at all in `CoclusteringReport`
- Some Python 2 incompatibilities in Linux

## 10.0.2 - 2021-06-28

### Added
- `get_samples_dir` core API function (works only with a local runner).
- `train_predictor`, `evaluate_model`, `train_recoder`, `train_coclustering` and
  `deploy_coclustering` now have return values (paths of relevant output files).

### Changed
- `transfer_database` core API function renamed to `deploy_model`.
- `build_transferred_dictionary` core API function renamed to
  `build_deployed_dictionary`.
- In general the "model deployment" concept replaces that of "database transfer" in all
  code and in particular in the samples scripts.
- It is not necessary to specify a relative path as `./path` for the
  `results_dir` argument.
- Messages enabled with the `trace` parameter go again to `stdout`.

## 10.0.1 - 2021-06-24

### Added
- `sklearn` sub-module updated for pyKhiops 10.
- `sklearn` samples notebooks.
- `deploy_coclustering` core API function.
- `build_multitable_dictionary` core API function.

### Changed
- The information messages of `sklearn` are now deactivated by default (they
  can be reactivated manually).

### Removed
- `sklearn` dependency on `overrides` package.

### Fixed
- Small transformation bug in `convert-pk10`.

## 10.0.0 - 2021-06-20
### Added
- `detect_format` parameters to API methods that read databases. It is enabled by
  default and Khiops will try to automatically detect the format of input data tables.
  See the docstrings for the new behavior of `header_line` and `field_separator`.
- `specific_pairs` option replacing `only_pairs_with`. It allows the methods
  `train_predictor` and `train_recoder` more options to generate pairs of variables
  (`only_pairs_with` kept in legacy mode).
- `PykhiopsRunner` class to extend pyKhiops to different backends. `PyKhiopsLocalRunner`
  implements the current functionality and is the default runner.

### Changed
- `dictionary_domain` parameter removed from all relevant API methods. Now methods
  accepting a dictionary file path as argument also accept a `DictionaryDomain` object.
- Renamed various parameters. Until the next major release pykhiops will warn when
  these old parameters are used.
- All optional parameters of API methods are now proper named parameters (no kwargs).
- All errors are now handled with custom `PyKhiops*` exceptions.
- Updated default values to those of Khiops 10. Notably `max_trees == 10` by default.
- `tools/convert-pk10.py` script no longer exists. Now when installing pykhiops a
  `convert-pk10` will be automatically be installed the user's local python scripts
  directory. Optionally, the function `pykhiops.tools.convert_pk10` provides the same
  functionality.
- `samples.py` script is now in snake case and improved.
- Simplified `samples.ipynb`.
- Messages in `trace` mode now go to `stderr`.

### Removed
- Naive Bayes classifier option from `train_predictor`.

### Fixed
- `simplify_coclustering`: `results_prefix` now works.
- `subprocess.Popen` returning 1 in Linux even when the Khiops process ended correctly.
  This made the legacy mode detection fail.
- API functions failing when `stderr` was not empty even though the Khiops process ended
  correctly. Now it just emits a warning.

## 9.6.0b1- 2021-05-19

### Added
- Compatibility for Khiops 10
- Legacy support for Khiops 9
- Partial compatibility for Khiops 10 JSON reports (no tree report)
- Script `tools/convert_pk10.py` to migrate from pyKhiops 9 to 10. See _Changed_ below
- Extraction of dictionary data paths: See `core.DictionaryDomain.extract_data_paths`
- Robust JSON loading: tries `utf-8` encoding, then the system's default.
- Licence file

### Changed
- Now all variable/method names follow the PEP8 convention: **All methods are now in
  snake_case**

### Removed
- In `core.train_predictor`: `fill_test_database_settings` and `map` (kept in legacy
  mode).

## 9.0.1 - 2020-09-30

### Added
- Sources (first commit)

[khiops]: https://khiops.org
