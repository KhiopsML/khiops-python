######################################################################################
# Copyright (c) 2024 Orange. All rights reserved.                                    #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""API for the execution of the Khiops AutoML suite

The methods in this module allow to execute all Khiops and Khiops Coclustering tasks.

See also:
    - :ref:`core-api-common-params`
    - :ref:`core-api-env-vars`
    - :ref:`core-api-input-types`
    - :ref:`core-api-sampling-mode`
"""
import io
import os
import warnings

import khiops.core.internals.filesystems as fs
from khiops.core.dictionary import DictionaryDomain, read_dictionary_file
from khiops.core.exceptions import KhiopsRuntimeError
from khiops.core.helpers import build_multi_table_dictionary_domain
from khiops.core.internals.common import (
    CommandLineOptions,
    create_unambiguous_khiops_path,
    deprecation_message,
    is_string_like,
    removal_message,
    renaming_message,
    type_error_message,
)
from khiops.core.internals.io import KhiopsOutputWriter
from khiops.core.internals.runner import (
    _get_tool_info_khiops9,
    _get_tool_info_khiops10,
    get_runner,
)
from khiops.core.internals.task import get_task_registry
from khiops.core.internals.version import KhiopsVersion

# List of all available construction rules in the Khiops tool
all_construction_rules = [
    "Day",
    "DecimalTime",
    "DecimalWeekDay",
    "DecimalYear",
    "DecimalYearTS",
    "GetDate",
    "GetTime",
    "GetValue",
    "GetValueC",
    "LocalTimestamp",
    "TableCount",
    "TableCountDistinct",
    "TableMax",
    "TableMean",
    "TableMedian",
    "TableMin",
    "TableMode",
    "TableSelection",
    "TableStdDev",
    "TableSum",
    "WeekDay",
    "YearDay",
]

##########################
# Private module methods #
##########################


def _check_dictionary_file_path_or_domain(dictionary_file_path_or_domain):
    """Checks if the argument is a string or DictionaryDomain or raise TypeError"""
    if not is_string_like(dictionary_file_path_or_domain) and not isinstance(
        dictionary_file_path_or_domain, DictionaryDomain
    ):
        raise TypeError(
            type_error_message(
                "dictionary_file_path_or_domain",
                dictionary_file_path_or_domain,
                str,
                DictionaryDomain,
            )
        )


def _get_or_create_execution_dictionary_file(dictionary_file_path_or_domain, trace):
    """Access the dictionary path or creates one from a DictionaryDomain object"""
    # Check the type of dictionary_file_path_or_domain
    _check_dictionary_file_path_or_domain(dictionary_file_path_or_domain)

    # If the argument is a DictionaryDomain export it to a temporary file
    if isinstance(dictionary_file_path_or_domain, DictionaryDomain):
        execution_dictionary_file_path = get_runner().create_temp_file(
            "_dictionary_", ".kdic"
        )
        dictionary_file_path_or_domain.export_khiops_dictionary_file(
            execution_dictionary_file_path
        )
        if trace:
            print(f"Khiops execution dictionary file: {execution_dictionary_file_path}")
    else:
        execution_dictionary_file_path = dictionary_file_path_or_domain

    return execution_dictionary_file_path


def _run_task(task_name, task_args):
    """Generic task run method

    Parameters
    ----------
    task_name : str
        Name of the task.
    task_args : dict
        Arguments of the task.
    """
    # Save the `runner.run` arguments other than the task parameters and options
    trace = task_args["trace"]
    stdout_file_path = task_args["stdout_file_path"]
    stderr_file_path = task_args["stderr_file_path"]

    command_line_options, task_called_with_domain = _preprocess_arguments(task_args)

    # Obtain the api function from the registry
    task = get_task_registry().get_task(task_name, get_khiops_version())

    # Execute the Khiops task and cleanup when necessary
    try:
        get_runner().run(
            task,
            task_args,
            command_line_options=command_line_options,
            trace=trace,
            stdout_file_path=stdout_file_path,
            stderr_file_path=stderr_file_path,
        )
    finally:
        if task_called_with_domain and not trace:
            fs.remove(task_args["dictionary_file_path"])


def _preprocess_arguments(args):
    """Preprocessing of Khiops arguments

    Parameters
    ----------
    args : dict
        The Khiops arguments.

    Returns
    -------
    tuple
        A 2-tuple containing:
        - A `~.CommandLineOptions` instance
        - A `bool` that is ``True`` if the value of the `dictionary_file_or_domain`
          `args` key is a `~.DictionaryDomain` instance.

    .. note:: This function *mutates* the input `args` dictionary.
    """
    # Execute the preprocess of common task arguments
    task_is_called_with_domain = _preprocess_task_arguments(args)

    # Create a command line options object
    command_line_options = CommandLineOptions(
        batch_mode=args["batch_mode"] if "batch_mode" in args else True,
        log_file_path=(args["log_file_path"] if "log_file_path" in args else ""),
        output_scenario_path=(
            args["output_scenario_path"] if "output_scenario_path" in args else ""
        ),
        task_file_path=(args["task_file_path"] if "task_file_path" in args else ""),
    )

    # Clean the args to leave only the task arguments
    _clean_task_args(args)

    return command_line_options, task_is_called_with_domain


def _preprocess_task_arguments(task_args):
    """Preprocessing of task arguments common to various tasks

    Parameters
    ----------
    task_args : dict
        The task arguments.

    Returns
    -------
    bool
        ``True`` if the task was called with an input `.DictionaryDomain`.
    """
    # Process the input dictionary domain if any
    # build_frequency_variables and detect_format are processed differently below
    task_called_with_domain = False
    if "dictionary_file_path_or_domain" in task_args:
        task_called_with_domain = isinstance(
            task_args["dictionary_file_path_or_domain"], DictionaryDomain
        )
        task_args["dictionary_file_path"] = _get_or_create_execution_dictionary_file(
            task_args["dictionary_file_path_or_domain"], task_args["trace"]
        )

    # Set the discretization/grouping default values
    if "discretization_method" in task_args:
        # Set the default values if the discretization method is not specified
        if task_args["discretization_method"] is None:
            if task_args["target_variable"]:
                task_args["discretization_method"] = "MODL"
            else:
                task_args["discretization_method"] = "EqualWidth"
        # Otherwise raise an error if the values are not in the range to avoid a khiops
        # segmentation fault. This won't be necessary from version 11 on.
        else:
            if not task_args["target_variable"] and task_args[
                "discretization_method"
            ] not in ("EqualWidth", "EqualFrequency", "None"):
                raise ValueError(
                    "'discretization_method' must be either "
                    "'EqualWidth', 'EqualFrequency' or 'None'."
                )

    if "grouping_method" in task_args:
        # Set the default values if the grouping method is not specified
        if task_args["grouping_method"] is None:
            if task_args["target_variable"]:
                task_args["grouping_method"] = "MODL"
            else:
                task_args["grouping_method"] = "BasicGrouping"
        # Otherwise raise an error if the values are not in the range to avoid a khiops
        # segmentation fault. This won't be necessary from version 11 on.
        else:
            if not task_args["target_variable"] and task_args[
                "grouping_method"
            ] not in ("BasicGrouping", "None"):
                raise ValueError(
                    "'grouping_method' must be either 'BasicGrouping' or 'None'."
                )

    # Transform the use_complement_as_test bool parameter to its string counterpart
    if "use_complement_as_test" in task_args:
        if task_args["use_complement_as_test"]:
            if get_khiops_version() < KhiopsVersion("10.0.0"):
                task_args["fill_test_database_settings"] = True
            else:
                task_args["test_database_mode"] = "Complementary"
        else:
            if get_khiops_version() < KhiopsVersion("10"):
                task_args["fill_test_database_settings"] = False
            else:
                task_args["test_database_mode"] = "None"
        del task_args["use_complement_as_test"]

    # Preprocess the database format parameters
    if "detect_format" in task_args:
        assert "header_line" in task_args
        assert "field_separator" in task_args
        detect_format, header_line, field_separator = _preprocess_format_spec(
            task_args["detect_format"],
            task_args["header_line"],
            task_args["field_separator"],
        )
        task_args["detect_format"] = detect_format
        task_args["header_line"] = header_line
        task_args["field_separator"] = field_separator
    if "output_header_line" in task_args:
        assert "output_field_separator" in task_args
        _, header_line, field_separator = _preprocess_format_spec(
            False, task_args["output_header_line"], task_args["output_field_separator"]
        )
        task_args["output_header_line"] = header_line
        task_args["output_field_separator"] = field_separator

    # Preprocess the selection_value parameter
    if "selection_value" in task_args:
        if isinstance(task_args["selection_value"], (int, float)):
            task_args["selection_value"] = str(task_args["selection_value"])

    # Warn the simple deprecations for Khiops 11
    simple_khiops_11_deprecations = [
        ("max_groups", "the upcoming 'max_parts' parameter", 0),
        ("max_intervals", "the upcoming 'max_parts' parameter", 0),
        ("min_group_frequency", None, 0),
        ("min_interval_frequency", None, 0),
        ("results_prefix", None, ""),
        ("snb_predictor", None, True),
        ("univariate_predictor_number", None, 0),
    ]
    for param, replacement_param, param_default_value in simple_khiops_11_deprecations:
        if param in task_args and task_args[param] != param_default_value:
            warnings.warn(
                deprecation_message(
                    f"'{param}'", "11.0.0", replacement=replacement_param, quote=False
                )
            )

    # Warn the grouping/interval supervised method deprecation values for Khiops 11
    if "target_variable" in task_args and task_args["target_variable"] != "":
        if "grouping_method" in task_args and task_args["grouping_method"] != "MODL":
            warnings.warn(
                deprecation_message(
                    "'grouping_method' on supervised learning", "11.0.0", quote=False
                )
            )
        if (
            "discretization_method" in task_args
            and task_args["discretization_method"] != "MODL"
        ):
            warnings.warn(
                deprecation_message(
                    "'discretization_method' on supervised learning",
                    "11.0.0",
                    quote=False,
                )
            )

    # Flatten kwargs
    if "kwargs" in task_args:
        task_args.update(task_args["kwargs"])
        del task_args["kwargs"]

    return task_called_with_domain


def _preprocess_format_spec(detect_format, header_line, field_separator):
    r"""Preprocess the user format spec to be used in a task

    More precisely:
        - Sets ``detect_format`` to ``False`` if either ``header_line`` or
          ``field_separator`` are set
        - If either ``header_line`` or ``field_separator`` is ``None``, then they are
          set to their default values
        - It transforms the field separator "\\t" to the empty string ""
    """
    # Ignore detect_format if header_line or field_separator are set
    if header_line is not None or field_separator is not None:
        detect_format = False

    # Set the default values of header_line and field_separator
    if header_line is None:
        header_line = True
    if field_separator is None:
        field_separator = ""

    # Fail on separators with more than one char
    if len(field_separator) > 1:
        raise ValueError("'field_separator' must have length at most 1")

    # Transform tab field_separator to empty string
    if field_separator == "\t":
        field_separator = ""

    return detect_format, header_line, field_separator


def _clean_task_args(task_args):
    """Cleans the task arguments

    More precisely it removes:
        - Command line arguments (they already are in another object).
        - Parameters removed from the API and warns about it.
        - Renamed API parameters and warns about it.
    """
    # Remove non-task parameters
    command_line_arg_names = [
        "batch_mode",
        "log_file_path",
        "output_scenario_path",
        "task_file_path",
    ]
    other_arg_names = [
        "dictionary_file_path_or_domain",
        "trace",
        "stdout_file_path",
        "stderr_file_path",
    ]
    for arg_name in command_line_arg_names + other_arg_names:
        if arg_name in task_args:
            del task_args[arg_name]

    # Remove removed parameters
    removed_parameters = [
        ("dictionary_domain", "dictionary_file_path_or_domain", "10"),
        ("fill_test_database_settings", "use_complement_as_test", "10"),
        ("map_predictor", None, "10"),
        ("nb_predictor", None, "10"),
        ("only_pairs_with", "specific_pairs", "10"),
    ]
    for arg_name, replacement_arg_name, removal_version in removed_parameters:
        if arg_name in task_args and get_runner().khiops_version >= KhiopsVersion(
            removal_version
        ):
            del task_args[arg_name]
            warnings.warn(
                removal_message(
                    arg_name,
                    removal_version,
                    replacement=replacement_arg_name,
                ),
                stacklevel=4,
            )
    # Remove renamed parameters
    renamed_parameters = [
        ("max_evaluated_variable_number", "max_evaluated_variables", "10"),
        ("max_selected_variable_number", "max_selected_variables", "10"),
        ("constructed_number", "max_constructed_variables", "10"),
        ("tree_number", "max_trees", "10"),
        ("pair_number", "max_pairs", "10"),
        ("max_interval_number", "max_intervals", "10"),
        ("max_group_number", "max_groups", "10"),
        ("max_variable_number", "max_variables", "10"),
        ("recode_categorical_variables", "categorical_recoding_method", "10"),
        ("recode_numerical_variables", "numerical_recoding_method", "10"),
        ("recode_bivariate_variables", "pairs_recoding_method", "10"),
        ("max_cell_number", "max_cells", "10"),
    ]
    for arg_name, new_arg_name, rename_version in renamed_parameters:
        if arg_name in task_args and get_runner().khiops_version >= KhiopsVersion(
            rename_version
        ):
            del task_args[arg_name]
            warnings.warn(
                renaming_message(arg_name, new_arg_name, rename_version),
                stacklevel=4,
            )


#########
# Tasks #
#########

# WARNING: All API methods that use task objects have the following first instruction:
#
#     task_args = locals()
#
# This line must not be moved from there because the return value of locals() depends on
# the state of the program. When it is called as the first instruction of a function it
# contains only the values of its parameters.


def get_khiops_version():
    """Returns the Khiops version

    Returns
    -------
    str
        The Khiops version of the current `.KhiopsRunner` backend.
    """
    return get_runner().khiops_version


def get_samples_dir():
    """Returns the Khiops' *samples* directory path

    Returns
    -------
    str
        The path of the Khiops *samples* directory.
    """
    return get_runner().samples_dir


# Disable the unused arg rule because we use locals() to pass the arguments to _run_task
# pylint: disable=unused-argument


def export_dictionary_as_json(
    dictionary_file_path_or_domain,
    json_dictionary_file_path,
    batch_mode=True,
    log_file_path=None,
    output_scenario_path=None,
    task_file_path=None,
    trace=False,
    stdout_file_path="",
    stderr_file_path="",
):
    """Exports a Khiops dictionary file to JSON format (``.kdicj``)

    Parameters
    ----------
    dictionary_file_path_or_domain : str or `.DictionaryDomain`
        Path of a Khiops dictionary file or a DictionaryDomain object.
    ... :
        See :ref:`core-api-common-params`.

    Examples
    --------
    See the following function of the ``samples.py`` documentation script:
        - `samples.export_dictionary_files()`
    """
    # Save the task arguments
    # WARNING: Do not move this line, see the top of the "tasks" section for details
    task_args = locals()

    # Run the task
    _run_task("export_dictionary_as_json", task_args)


def build_dictionary_from_data_table(
    data_table_path,
    output_dictionary_name,
    output_dictionary_file_path,
    detect_format=True,
    header_line=None,
    field_separator=None,
    batch_mode=True,
    log_file_path=None,
    output_scenario_path=None,
    task_file_path=None,
    trace=False,
    stdout_file_path="",
    stderr_file_path="",
    **kwargs,
):
    r"""Builds a dictionary file by analyzing a data table file

    Parameters
    ----------
    data_table_path : str
        Path of the data table file.
    output_dictionary_name : str
        Name dictionary to be created.
    output_dictionary_file_path : str
        Path of the output dictionary file.
    detect_format : bool, default ``True``
        If ``True`` detects automatically whether the data table file has a header and
        its field separator. It is set to ``False`` if ``header_line`` or
        ``field_separator`` are set.
    header_line : bool, optional (default ``True``)
        If ``True`` it uses the first line of the data as column names. Sets
        ``detect_format`` to ``False`` if set. Ignored if ``detect_format``
        is ``True``.
    field_separator : str, optional (default "\\t")
        A field separator character. "" has the same effect as "\\t". Sets
        ``detect_format`` to ``False`` if set. Ignored if ``detect_format``
        is ``True``.
    ... :
        See :ref:`core-api-common-params`.
    """
    # Save the task arguments
    # WARNING: Do not move this line, see the top of the "tasks" section for details
    task_args = locals()

    # Run the ttask
    _run_task("build_dictionary_from_data_table", task_args)


def check_database(
    dictionary_file_path_or_domain,
    dictionary_name,
    data_table_path,
    detect_format=True,
    header_line=None,
    field_separator=None,
    sample_percentage=100.0,
    sampling_mode="Include sample",
    selection_variable="",
    selection_value="",
    additional_data_tables=None,
    max_messages=20,
    batch_mode=True,
    log_file_path=None,
    output_scenario_path=None,
    task_file_path=None,
    trace=False,
    stdout_file_path="",
    stderr_file_path="",
    **kwargs,
):
    r"""Checks if a data table is compatible with a dictionary file

    Parameters
    ----------
    dictionary_file_path_or_domain : str or `.DictionaryDomain`
        Path of a Khiops dictionary file or a DictionaryDomain object.
    dictionary_name : str
        Name of the dictionary of the table to be checked.
    data_table_path : str
        Path of the data table file.
    detect_format : bool, default ``True``
        If ``True`` detects automatically whether the data table file has a header and
        its field separator. It is set to ``False`` if ``header_line`` or
        ``field_separator`` are set.
    header_line : bool, optional (default ``True``)
        If ``True`` it uses the first line of the data as column names. Sets
        ``detect_format`` to ``False`` if set. Ignored if ``detect_format``
        is ``True``.
    field_separator : str, optional (default "\\t")
        A field separator character. "" has the same effect as "\\t". Sets
        ``detect_format`` to ``False`` if set. Ignored if ``detect_format``
        is ``True``.
    sample_percentage : float, default 100.0
        See the ``sampling_mode`` option below.
    sampling_mode : "Include sample" or "Exclude sample"
        If equal to "Include sample" it checks ``sample_percentage`` percent of
        the data; if equal to "Exclude sample" it checks the complement of the
        data selected with "Include sample". See also :ref:`core-api-sampling-mode`.
    selection_variable : str, default ""
        It checks only the records such that the value of ``selection_variable`` is
        equal to ``selection_value``. Ignored if equal to "".
    selection_value: str or int or float, default ""
        See ``selection_variable`` option above. Ignored if equal to "".
    additional_data_tables : dict, optional
        A dictionary containing the data paths and file paths for a multi-table
        dictionary file. For more details see :doc:`/multi_table_primer`.
    max_messages : int, default 20
        Maximum number of error messages to write in the log file.
    ... :
        See :ref:`core-api-common-params`.

    Examples
    --------
    See the following function of the ``samples.py`` documentation script:
        - `samples.check_database()`
    """
    # Save the task arguments
    # WARNING: Do not move this line, see the top of the "tasks" section for details
    task_args = locals()

    # Run the task
    _run_task("check_database", task_args)


def train_predictor(
    dictionary_file_path_or_domain,
    dictionary_name,
    data_table_path,
    target_variable,
    results_dir,
    detect_format=True,
    header_line=None,
    field_separator=None,
    sample_percentage=70.0,
    sampling_mode="Include sample",
    use_complement_as_test=True,
    selection_variable="",
    selection_value="",
    additional_data_tables=None,
    main_target_value="",
    snb_predictor=True,
    univariate_predictor_number=0,
    max_evaluated_variables=0,
    max_selected_variables=0,
    max_constructed_variables=100,
    construction_rules=None,
    max_trees=10,
    max_pairs=0,
    all_possible_pairs=True,
    specific_pairs=None,
    group_target_value=False,
    discretization_method=None,
    min_interval_frequency=0,
    max_intervals=0,
    grouping_method=None,
    min_group_frequency=0,
    max_groups=0,
    results_prefix="",
    batch_mode=True,
    log_file_path=None,
    output_scenario_path=None,
    task_file_path=None,
    trace=False,
    stdout_file_path="",
    stderr_file_path="",
    **kwargs,
):
    r"""Trains a model from a data table

    Parameters
    ----------
    dictionary_file_path_or_domain : str or `.DictionaryDomain`
        Path of a Khiops dictionary file or a DictionaryDomain object.
    dictionary_name : str
        Name of the dictionary to be analyzed.
    data_table_path : str
        Path of the data table file.
    target_variable : str
        Name of the target variable. If the specified variable is categorical it
        constructs a classifier and if it is numerical a regressor. If equal to "" it
        performs an unsupervised analysis.
    results_dir : str
        Path of the results directory.
    detect_format : bool, default ``True``
        If ``True`` detects automatically whether the data table file has a header and
        its field separator. It is set to ``False`` if ``header_line`` or
        ``field_separator`` are set.
    header_line : bool, optional (default ``True``)
        If ``True`` it uses the first line of the data as column names. Sets
        ``detect_format`` to ``False`` if set. Ignored if ``detect_format``
        is ``True``.
    field_separator : str, optional (default "\\t")
        A field separator character. "" has the same effect as "\\t". Sets
        ``detect_format`` to ``False`` if set. Ignored if ``detect_format``
        is ``True``.
    sample_percentage : float, default 70.0
        See the ``sampling_mode`` option below.
    sampling_mode : "Include sample" or "Exclude sample"
        If equal to "Include sample" it trains the predictor on ``sample_percentage``
        percent of the data and tests the model on the remainder of the data if
        ``use_complement_as_test`` is set to ``True``.  If equal to "Exclude sample" the
        train and test datasets above are exchanged. See also
        :ref:`core-api-sampling-mode`.
    use_complement_as_test : bool, default ``True``
        Uses the complement of the sampled database as test database for
        computing the model's performance metrics.
    fill_test_database_settings : bool, default ``False``
        It creates a test database as the complement of the train database.
        **Deprecated** will be removed in Khiops 11, use ``use_complement_as_test``
    selection_variable : str, default ""
        It trains with only the records such that the value of ``selection_variable`` is
        equal to ``selection_value``. Ignored if equal to "".
    selection_value: str or int or float, default ""
        See ``selection_variable`` option above. Ignored if equal to "".
    additional_data_tables : dict, optional
        A dictionary containing the data paths and file paths for a multi-table
        dictionary file. For more details see :doc:`/multi_table_primer`.
    main_target_value : str, default ""
        If this target value is specified then it guarantees the calculation of lift
        curves for it.
    snb_predictor : bool, default ``True``
        If ``True`` it trains a Selective Naive Bayes predictor.
    univariate_predictor_number : int, default 0
        Number of univariate predictors to train.
    map_predictor : bool, default ``False``
        If ``True`` trains a Maximum a Posteriori Naive Bayes predictor.
        **Deprecated** will be removed in Khiops Python 11.
    max_evaluated_variables : int, default 0
        Maximum number of variables to be evaluated in the SNB predictor training. If
        equal to 0 it evaluates all informative variables.
    max_selected_variables : int, default 0
        Maximum number of variables to be selected in the SNB predictor. If equal to
        0 it selects all the variables kept in the training.
    max_constructed_variables : int, default 100
        Maximum number of variables to construct.
    construction_rules : list of str, optional
        Allowed rules for the automatic variable construction. If not set it uses all
        possible rules.
    max_trees : int, default 10
        Maximum number of trees to construct. Not yet available in regression.
    max_pairs : int, default 0
        Maximum number of variables pairs to construct.
    specific_pairs : list of tuple, optional
        User-specified pairs as a list of 2-tuples of variable names. If a given tuple
        contains only one non-empty variable name, then it generates all the pairs
        containing it (within the limit ``max_pairs``).
    all_possible_pairs : bool, default ``True``
        If ``True`` tries to create all possible pairs within the limit ``max_pairs``.
        The pairs and variables given in ``specific_pairs`` have priority.
    only_pairs_with : str, default ""
        Constructs only pairs with the specifed variable name. If equal to the empty
        string "" it considers all variables to make pairs.
        **Deprecated** will be removed in Khiops Python 11, use ``specific_pairs``.
    group_target_value : bool, default ``False``
        Allows grouping of the target variable values in classification. It can
        substantially increase the training time.
    discretization_method : str
        Name of the discretization method. Its valid values depend on the task:
            - Supervised: "MODL" (default), "EqualWidth" or "EqualFrequency"
            - Unsupervised: "EqualWidth" (default), "EqualFrequency" or  "None"
    min_interval_frequency : int, default 0
        Minimum number of instances in an interval. If equal to 0 it is
        automatically calculated.
    max_intervals : int, default 0
        Maximum number of intervals to construct. If equal to 0 it is automatically
        calculated.
    grouping_method : str
        Name of the grouping method. Its valid values depend on the task:
            - Supervised: "MODL" (default) or "BasicGrouping"
            - Unsupervised: "BasicGrouping" (default) or "None"
    min_group_frequency : int, default 0
        Minimum number of instances for a group.
    max_groups : int, default 0
        Maximum number of groups. If equal to 0 it is automatically calculated.
    results_prefix : str, default ""
        Prefix of the result files.
    ... :
        See :ref:`core-api-common-params`.

    Returns
    -------
    tuple
        A 2-tuple containing:
            - The reports file path
            - The modeling dictionary file path in the supervised case.

    Raises
    ------
    `ValueError`
        Invalid values of an argument
    `TypeError`
        Invalid type of an argument

    Examples
    --------
    See the following functions of the ``samples.py`` documentation script:
        - `samples.train_predictor()`
        - `samples.train_predictor_file_paths()`
        - `samples.train_predictor_error_handling()`
        - `samples.train_predictor_mt()`
        - `samples.train_predictor_mt_with_specific_rules()`
        - `samples.train_predictor_with_train_percentage()`
        - `samples.train_predictor_with_trees()`
        - `samples.train_predictor_with_pairs()`
        - `samples.train_predictor_with_multiple_parameters()`
        - `samples.train_predictor_detect_format()`
        - `samples.train_predictor_with_cross_validation()`
        - `samples.multiple_train_predictor()`
    """
    # Save the task arguments
    # WARNING: Do not move this line, see the top of the "tasks" section for details
    task_args = locals()

    # Run the task
    _run_task("train_predictor", task_args)

    # Return the paths of the JSON report and modelling dictionary file
    reports_file_name = results_prefix
    if get_runner().khiops_version < KhiopsVersion("10.0.0"):
        reports_file_name += "AllReports.json"
    else:
        reports_file_name += "AllReports.khj"
    reports_file_path = fs.get_child_path(results_dir, reports_file_name)

    if target_variable != "":
        modeling_dictionary_file_path = fs.get_child_path(
            results_dir, f"{results_prefix}Modeling.kdic"
        )
    else:
        modeling_dictionary_file_path = None

    return (reports_file_path, modeling_dictionary_file_path)


def evaluate_predictor(
    dictionary_file_path_or_domain,
    train_dictionary_name,
    data_table_path,
    results_dir,
    detect_format=True,
    header_line=None,
    field_separator=None,
    sample_percentage=100.0,
    sampling_mode="Include sample",
    selection_variable="",
    selection_value="",
    additional_data_tables=None,
    main_target_value="",
    results_prefix="",
    batch_mode=True,
    log_file_path=None,
    output_scenario_path=None,
    task_file_path=None,
    trace=False,
    stdout_file_path="",
    stderr_file_path="",
    **kwargs,
):
    r"""Evaluates the predictors in a dictionary file on a database

    Parameters
    ----------
    dictionary_file_path_or_domain : str or `.DictionaryDomain`
        Path of a Khiops dictionary file or a DictionaryDomain object.
    train_dictionary_name : str
        Name of the main dictionary used while training the models.
    data_table_path : str
        Path of the evaluation data table file.
    results_dir : str
        Path of the results directory.
    detect_format : bool, default ``True``
        If ``True`` detects automatically whether the data table file has a header and
        its field separator. It is set to ``False`` if ``header_line`` or
        ``field_separator`` are set.
    header_line : bool, optional (default ``True``)
        If ``True`` it uses the first line of the data as column names. Sets
        ``detect_format`` to ``False`` if set. Ignored if ``detect_format``
        is ``True``.
    field_separator : str, optional (default "\\t")
        A field separator character. "" has the same effect as "\\t". Sets
        ``detect_format`` to ``False`` if set. Ignored if ``detect_format``
        is ``True``.
    sample_percentage : float, default 100.0
        See ``sampling_mode`` option below.
    sampling_mode : "Include sample" or "Exclude sample"
        If equal to "Include sample" it evaluates the predictor on ``sample_percentage``
        percent of the data. If equal to "Exclude sample" it evaluates the predictor on
        the complement of the data selected with "Include sample". See also
        :ref:`core-api-sampling-mode`.
    selection_variable : str, default ""
        It trains with only the records such that the value of ``selection_variable`` is
        equal to ``selection_value``. Ignored if equal "".
    selection_value: str or int or float, default ""
        See ``selection_variable`` option above. Ignored if equal to "".
    additional_data_tables : dict, optional
        A dictionary containing the data paths and file paths for a multi-table
        dictionary file. For more details see :doc:`/multi_table_primer`.

        .. note:: Use the initial dictionary name in the data paths.

    main_target_value : str, default ""
        If this target value is specified then it guarantees the calculation of lift
        curves for it.
    results_prefix : str, default ""
        Prefix of the result files.
    ... :
        See :ref:`core-api-common-params`.

    Returns
    -------
    str
        The path of the JSON evaluation report (extension ``.khj``).

    Raises
    ------
    `TypeError`
        Invalid type of an argument.

    Examples
    --------
    See the following functions of the ``samples.py`` documentation script:
        - `samples.evaluate_predictor()`
        - `samples.access_predictor_evaluation_report()`
        - `samples.train_predictor_with_cross_validation()`
    """
    # Save the task arguments
    # WARNING: Do not move this line, see the top of the "tasks" section for details
    task_args = locals()

    # Create the evaluation file path and remove the directory and prefix arguments
    task_args["evaluation_report_path"] = fs.get_child_path(
        task_args["results_dir"], f"{task_args['results_prefix']}EvaluationReport.xls"
    )
    del task_args["results_dir"]
    del task_args["results_prefix"]

    # Run the task
    _run_task("evaluate_predictor", task_args)

    # Return the path of the JSON report
    report_file_name = results_prefix
    if get_runner().khiops_version < KhiopsVersion("10.0.0"):
        report_file_name += "EvaluationReport.json"
    else:
        report_file_name += "EvaluationReport.khj"

    return fs.get_child_path(results_dir, report_file_name)


def train_recoder(
    dictionary_file_path_or_domain,
    dictionary_name,
    data_table_path,
    target_variable,
    results_dir,
    detect_format=True,
    header_line=None,
    field_separator=None,
    sample_percentage=100.0,
    sampling_mode="Include sample",
    selection_variable="",
    selection_value="",
    additional_data_tables=None,
    max_constructed_variables=100,
    construction_rules=None,
    max_trees=0,
    max_pairs=0,
    all_possible_pairs=True,
    specific_pairs=None,
    informative_variables_only=True,
    max_variables=0,
    keep_initial_categorical_variables=False,
    keep_initial_numerical_variables=False,
    categorical_recoding_method="part Id",
    numerical_recoding_method="part Id",
    pairs_recoding_method="part Id",
    group_target_value=False,
    discretization_method=None,
    min_interval_frequency=0,
    max_intervals=0,
    grouping_method=None,
    min_group_frequency=0,
    max_groups=0,
    results_prefix="",
    batch_mode=True,
    log_file_path=None,
    output_scenario_path=None,
    task_file_path=None,
    trace=False,
    stdout_file_path="",
    stderr_file_path="",
    **kwargs,
):
    r"""Trains a recoding model from a data table

    A recoding model consists in the discretization of numerical variables and the
    grouping of categorical variables.

    If the ``target_variable`` is specified these partitions are constructed in
    supervised mode, meaning that each resulting discretizations/groupings best
    separates the target variable while maintaining a simple interval/group model of the
    data. Different recoding methods can be specified via the
    ``numerical_recoding_method``, ``categorical_recoding_method`` and
    ``pairs_recoding_method`` options.

    The output files of this process contain a dictionary file (``.kdic``) that can be
    used to recode databases with the `deploy_model` function.

    Parameters
    ----------
    dictionary_file_path_or_domain : str or `.DictionaryDomain`
        Path of a Khiops dictionary file or a DictionaryDomain object.
    dictionary_name : str
        Name of the dictionary to be recoded.
    data_table_path : str
        Path of the data table file.
    target_variable : str
        Name of the target variable. If equal to "" it trains an unsupervised recoder.
    results_dir : str
        Path of the results directory.
    detect_format : bool, default ``True``
        If ``True`` detects automatically whether the data table file has a header and
        its field separator. It is set to ``False`` if ``header_line`` or
        ``field_separator`` are set.
    header_line : bool, optional (default ``True``)
        If ``True`` it uses the first line of the data as column names. Sets
        ``detect_format`` to ``False`` if set. Ignored if ``detect_format``
        is ``True``.
    field_separator : str, optional (default "\\t")
        A field separator character. "" has the same effect as "\\t". Sets
        ``detect_format`` to ``False`` if set. Ignored if ``detect_format``
        is ``True``.
    sample_percentage : float, default 100.0
        See ``sampling_mode`` option below.
    sampling_mode : "Include sample" or "Exclude sample"
        If equal to "Include sample" it trains the recoder on ``sample_percentage``
        percent of the data. If equal to "Exclude sample" it trains the recoder on the
        complement of the data selected with "Include sample". See also
        :ref:`core-api-sampling-mode`.
    selection_variable : str, default ""
        It trains with only the records such that the value of ``selection_variable`` is
        equal to ``selection_value``. Ignored if equal to "".
    selection_value: str or int or float, default ""
        See ``selection_variable`` option above. Ignored if equal to "".
    additional_data_tables : dict, optional
        A dictionary containing the data paths and file paths for a multi-table
        dictionary file. For more details see :doc:`/multi_table_primer`.
    max_constructed_variables : int, default 100
        Maximum number of variables to construct.
    construction_rules : list of str, optional
        Allowed rules for the automatic variable construction. If not set it uses all
        possible rules.
    max_trees : int, default 0
        Maximum number of trees to construct. Not yet available in regression.
    max_pairs : int, default 0
        Maximum number of variables pairs to construct.
    specific_pairs : list of tuple, optional
        User-specified pairs as a list of 2-tuples of variable names. If a given tuple
        contains only one non-empty variable name, then it generates all the pairs
        containing it (within the limit ``max_pairs``).
    all_possible_pairs : bool, default ``True``
        If ``True`` tries to create all possible pairs within the limit ``max_pairs``.
        The pairs and variables given in ``specific_pairs`` have priority.
    only_pairs_with : str, default ""
        Constructs only pairs with the specifed variable name. If equal to the empty
        string "" it considers all variables to make pairs.
        **Deprecated** will be removed in Khiops Python 11, use ``specific_pairs``.
    group_target_value : bool, default ``False``
        Allows grouping of the target variable values in classification. It can
        substantially increase the training time.
    discretization_method : str
        Name of the discretization method. Its valid values depend on the task:
            - Supervised: "MODL" (default), "EqualWidth" or "EqualFrequency".
            - Unsupervised: "EqualWidth" (default), "EqualFrequency" or "None".
    min_interval_frequency : int, default 0
        Minimum number of instances in an interval. If equal to 0 it is automatically
        calculated.
    max_intervals : int, default 0
        Maximum number of intervals to construct. If equal to 0 it is automatically
        calculated.
    informative_variables_only : bool, default ``True``
        If ``True`` keeps only informative variables.
    max_variables : int, default 0
        Maximum number of variables to keep. If equal to 0 keeps all variables.
    keep_initial_categorical_variables : bool, default ``True``
        If ``True`` keeps the initial categorical variables.
    keep_initial_numerical_variables : bool, default ``True``
        If ``True`` keeps initial numerical variables.
    categorical_recoding_method : str
        Type of recoding for categorical variables. Types available:
            - "part Id" (default): An id for the interval/group
            - "part label": A label for the interval/group
            - "0-1 binarization": A 0's and 1's coding the interval/group id
            - "conditional info": Conditional information of the interval/group
            - "none": Keeps the variable as-is
    numerical_recoding_method : str
        Type of recoding recoding for numerical variables. Types available:
            - "part Id" (default): An id for the interval/group
            - "part label": A label for the interval/group
            - "0-1 binarization": A 0's and 1's coding the interval/group id
            - "conditional info": Conditional information of the interval/group
            - "center-reduction": "(X - Mean(X)) / StdDev(X)"
            - "0-1 normalization": "(X - Min(X)) / (Max(X) - Min(X))"
            - "rank normalization": mean normalized rank (between 0 and 1) of the
              instances
            - "none": Keeps the variable as-is
    pairs_recoding_method : str
        Type of recoding for bivariate variables. Types available:
            - "part Id" (default): An id for the interval/group
            - "part label": A label for the interval/group
            - "0-1 binarization": A 0's and 1's coding the interval/group id
            - "conditional info": Conditional information of the interval/group
            - "none": Keeps the variable as-is
    grouping_method : str
        Name of the grouping method. Its vaild values depend on the task:
            - Supervised: "MODL" (default) or "BasicGrouping".
            - Unsupervised: "BasicGrouping" (default) or "None".
    min_group_frequency : int, default 0
        Minimum number of instances for a group.
    max_groups : int, default 0
        Maximum number of groups. If equal to 0 it is automatically calculated.
    results_prefix : str, default ""
        Prefix of the result files.
    ... :
        See :ref:`core-api-common-params`.

    Returns
    -------
    tuple
        A 2-tuple containing:
            - The path of the JSON file report of the process
            - The path of the dictionary containing the recoding model

    Examples
    --------
    See the following functions of the ``samples.py`` documentation script:
        - `samples.train_recoder()`
        - `samples.train_recoder_with_multiple_parameters()`
        - `samples.train_recoder_mt_flatten()`
    """
    # Save the task arguments
    # WARNING: Do not move this line, see the top of the "tasks" section for details
    task_args = locals()

    # Run the task
    _run_task("train_recoder", task_args)

    # Return the paths of the JSON report and modelling dictionary file
    reports_file_name = f"{results_prefix}AllReports"
    if get_runner().khiops_version < KhiopsVersion("10.0.0"):
        reports_file_name += ".json"
    else:
        reports_file_name += ".khj"
    reports_file_path = fs.get_child_path(results_dir, reports_file_name)
    modeling_dictionary_file_path = fs.get_child_path(
        results_dir, f"{results_prefix}Modeling.kdic"
    )

    return (reports_file_path, modeling_dictionary_file_path)


def deploy_model(
    dictionary_file_path_or_domain,
    dictionary_name,
    data_table_path,
    output_data_table_path,
    detect_format=True,
    header_line=None,
    field_separator=None,
    sample_percentage=100.0,
    sampling_mode="Include sample",
    selection_variable="",
    selection_value="",
    additional_data_tables=None,
    output_header_line=True,
    output_field_separator="\t",
    output_additional_data_tables=None,
    batch_mode=True,
    log_file_path=None,
    output_scenario_path=None,
    task_file_path=None,
    trace=False,
    stdout_file_path="",
    stderr_file_path="",
    **kwargs,
):
    r"""Deploys a model on a data table

    Parameters
    ----------
    dictionary_file_path_or_domain : str or `.DictionaryDomain`
        Path of a Khiops dictionary file or a DictionaryDomain object. This file/object
        defines the model to be deployed. Note that this model is not necessarily a
        predictor, it can be a generic table transformation.
    dictionary_name : str
        Name of the dictionary to be analyzed.
    data_table_path : str
        Path of the data table file.
    output_data_table_path : str
        Path of the output data file.
    detect_format : bool, default ``True``
        If ``True`` detects automatically whether the data table file has a header and
        its field separator. It is set to ``False`` if ``header_line`` or
        ``field_separator`` are set.
    header_line : bool, optional (default ``True``)
        If ``True`` it uses the first line of the data as column names. Sets
        ``detect_format`` to ``False`` if set. Ignored if ``detect_format``
        is ``True``.
    field_separator : str, optional (default "\\t")
        A field separator character. "" has the same effect as "\\t". Sets
        ``detect_format`` to ``False`` if set. Ignored if ``detect_format``
        is ``True``.
    sample_percentage : float, default 100.0
        See ``sampling_mode`` option below.
    sampling_mode : "Include sample" or "Exclude sample"
        If equal to "Include sample" it deploys the model on ``sample_percentage``
        percent of the data. If equal to "Exclude sample" it deploys the model on the
        complement of the data selected with "Include sample". See also
        :ref:`core-api-sampling-mode`.
    selection_variable : str, default ""
        It deploys only the records such that the value of ``selection_variable`` is
        equal to ``selection_value``. Ignored if equal to "".
    selection_value: str or int or float, default ""
        See ``selection_variable`` option above. Ignored if equal to "".
    additional_data_tables : dict, optional
        A dictionary containing the data paths and file paths for a multi-table
        dictionary file. For more details see :doc:`/multi_table_primer`.
    output_header_line : bool, default ``True``
        If ``True`` writes a header line with the column names in the output table.
    output_field_separator : str, default "\\t"
        The field separator character for the output table ("" counts as "\\t").
    output_additional_data_tables : dict, optional
        A dictionary containing the output data paths and file paths for a multi-table
        dictionary file. For more details see :doc:`/multi_table_primer`.
    results_prefix : str, default ""
        Prefix of the result files.
    ... :
        See :ref:`core-api-common-params`.

    Raises
    ------
    `TypeError`
        Invalid type of an argument.

    Examples
    --------
    See the following functions of the ``samples.py`` documentation script:
        - `samples.deploy_model()`
        - `samples.deploy_model_mt()`
        - `samples.deploy_model_mt_snowflake()`
        - `samples.deploy_model_expert()`

    """
    # Save the task arguments
    # WARNING: Do not move this line, see the top of the "tasks" section for details
    task_args = locals()

    # Run the task
    _run_task("deploy_model", task_args)


def build_deployed_dictionary(
    dictionary_file_path_or_domain,
    dictionary_name,
    output_dictionary_file_path,
    batch_mode=True,
    log_file_path=None,
    output_scenario_path=None,
    task_file_path=None,
    trace=False,
    stdout_file_path="",
    stderr_file_path="",
    **kwargs,
):
    """Builds a dictionary file to read the output table of a deployed model

    Parameters
    ----------
    dictionary_file_path_or_domain : str or `.DictionaryDomain`
        Path of a Khiops dictionary file or a DictionaryDomain object.
    dictionary_name : str
        Name of the dictionary to be analyzed.
    output_dictionary_file_path : str
        Path of the output dictionary file.
    ... :
        See :ref:`core-api-common-params`.

    Raises
    ------
    `TypeError`
        Invalid type of an argument

    Examples
    --------
    See the following functions of the ``samples.py`` documentation script:
        - `samples.build_deployed_dictionary()`
    """
    # Save the task arguments
    # WARNING: Do not move this line, see the top of the "tasks" section for details
    task_args = locals()

    # run the task
    _run_task("build_deployed_dictionary", task_args)


def sort_data_table(
    dictionary_file_path_or_domain,
    dictionary_name,
    data_table_path,
    output_data_table_path,
    sort_variables=None,
    detect_format=True,
    header_line=None,
    field_separator=None,
    output_header_line=True,
    output_field_separator="\t",
    batch_mode=True,
    log_file_path=None,
    output_scenario_path=None,
    task_file_path=None,
    trace=False,
    stdout_file_path="",
    stderr_file_path="",
    **kwargs,
):
    r"""Sorts a data table

    Parameters
    ----------
    dictionary_file_path_or_domain : str or `.DictionaryDomain`
        Path of a Khiops dictionary file or a DictionaryDomain object.
    dictionary_name : str
        Name of the dictionary to be analyzed.
    data_table_path : str
        Path of the data table file.
    output_data_table_path : str
        Path of the output data file.
    sort_variables : list of str, optional
        The names of the variables to sort. If not set sorts the table by its key.
    detect_format : bool, default ``True``
        If ``True`` detects automatically whether the data table file has a header and
        its field separator. It is set to ``False`` if ``header_line`` or
        ``field_separator`` are set.
    header_line : bool, optional (default ``True``)
        If ``True`` it uses the first line of the data as column names. Sets
        ``detect_format`` to ``False`` if set. Ignored if ``detect_format``
        is ``True``.
    field_separator : str, optional (default "\\t")
        A field separator character. "" has the same effect as "\\t". Sets
        ``detect_format`` to ``False`` if set. Ignored if ``detect_format``
        is ``True``.
    output_header_line : bool, default ``True``
        If ``True`` writes a header line with the column names in the output table.
    output_field_separator : str, default "\\t"
        The field separator character for the output table ("" counts as "\\t").
    ... :
        See :ref:`core-api-common-params`.


    Raises
    ------
    `TypeError`
        Invalid type of a argument.

    Examples
    --------
    See the following functions of the ``samples.py`` documentation script:
        - `samples.sort_data_table()`
        - `samples.sort_data_table_expert()`
    """
    # Save the task arguments
    # WARNING: Do not move this line, see the top of the "tasks" section for details
    task_args = locals()

    # Run the task
    _run_task("sort_data_table", task_args)


def extract_keys_from_data_table(
    dictionary_file_path_or_domain,
    dictionary_name,
    data_table_path,
    output_data_table_path,
    detect_format=True,
    header_line=None,
    field_separator=None,
    output_header_line=True,
    output_field_separator="\t",
    batch_mode=True,
    log_file_path=None,
    output_scenario_path=None,
    task_file_path=None,
    trace=False,
    stdout_file_path="",
    stderr_file_path="",
    **kwargs,
):
    r"""Extracts from data table unique occurrences of a key variable

    Parameters
    ----------
    dictionary_file_path_or_domain : str or `.DictionaryDomain`
        Path of a Khiops dictionary file or a DictionaryDomain object.
    dictionary_name : str
        Name of the dictionary of the data table.
    data_table_path : str
        Path of the data table file.
    output_data_table_path : str
        Path of the output data file.
    detect_format : bool, default ``True``
        If ``True`` detects automatically whether the data table file has a header and
        its field separator. It is set to ``False`` if ``header_line`` or
        ``field_separator`` are set.
    header_line : bool, optional (default ``True``)
        If ``True`` it uses the first line of the data as column names. Sets
        ``detect_format`` to ``False`` if set. Ignored if ``detect_format``
        is ``True``.
    field_separator : str, optional (default "\\t")
        A field separator character. "" has the same effect as "\\t". Sets
        ``detect_format`` to ``False`` if set. Ignored if ``detect_format``
        is ``True``.
    output_header_line : bool, default ``True``
        If ``True`` writes a header line with the column names in the output table.
    output_field_separator : str, default "\\t"
        The field separator character for the output table ("" counts as "\\t").
    ... :
        See :ref:`core-api-common-params`.

    Raises
    ------
    `TypeError`
        Invalid type of an argument.

    Examples
    --------
    See the following function of the ``samples.py`` documentation script:
        - `samples.extract_keys_from_data_table()`
    """
    # Save the task arguments
    # WARNING: Do not move this line, see the top of the "tasks" section for details
    task_args = locals()

    # Run the task
    _run_task("extract_keys_from_data_table", task_args)


def train_coclustering(
    dictionary_file_path_or_domain,
    dictionary_name,
    data_table_path,
    coclustering_variables,
    results_dir,
    detect_format=True,
    header_line=None,
    field_separator=None,
    sample_percentage=100.0,
    sampling_mode="Include sample",
    selection_variable="",
    selection_value="",
    additional_data_tables=None,
    frequency_variable="",
    min_optimization_time=0,
    results_prefix="",
    batch_mode=True,
    log_file_path=None,
    output_scenario_path=None,
    task_file_path=None,
    trace=False,
    stdout_file_path="",
    stderr_file_path="",
    **kwargs,
):
    r"""Trains a coclustering model from a data table

    Parameters
    ----------
    dictionary_file_path_or_domain : str or `.DictionaryDomain`
        Path of a Khiops dictionary file or a DictionaryDomain object.
    dictionary_name : str
        Name of the dictionary to be analyzed.
    data_table_path : str
        Path of the data table file.
    coclustering_variables : list of str
        The names of variables to use in coclustering. Min length: 2. Max length: 10.
    results_dir : str
        Path of the results directory.
    detect_format : bool, default ``True``
        If ``True`` detects automatically whether the data table file has a header and
        its field separator. It is set to ``False`` if ``header_line`` or
        ``field_separator`` are set.
    header_line : bool, optional (default ``True``)
        If ``True`` it uses the first line of the data as column names. Sets
        ``detect_format`` to ``False`` if set. Ignored if ``detect_format``
        is ``True``.
    field_separator : str, optional (default "\\t")
        A field separator character. "" has the same effect as "\\t". Sets
        ``detect_format`` to ``False`` if set. Ignored if ``detect_format``
        is ``True``.
    sample_percentage : float, default 100.0
        See ``sampling_mode`` option below.
    sampling_mode : "Include sample" or "Exclude sample"
        If equal to "Include sample" it trains the coclustering estimator on
        ``sample_percentage`` percent of the data. If equal to "Exclude sample" it
        trains the coclustering estimator on the complement of the data selected with
        "Include sample". See also :ref:`core-api-sampling-mode`.
    selection_variable : str, default ""
        It trains with only the records such that the value of ``selection_variable`` is
        equal to ``selection_value``. Ignored if equal to "".
    selection_value: str or int or float, default ""
        See ``selection_variable`` option above. Ignored if equal to "".
    additional_data_tables : dict, optional
        A dictionary containing the data paths and file paths for a multi-table
        dictionary file. For more details see :doc:`/multi_table_primer`.
    frequency_variable : str, default ""
        Name of frequency variable.
    min_optimization_time : int, default 0
        Minimum optimization time in seconds.
    results_prefix : str, default ""
        Prefix of the result files.
    ... :
        See :ref:`core-api-common-params`.

    Returns
    -------
    str
        The path of the of the resulting coclustering file.

    Raises
    ------
    `ValueError`
        Number of coclustering variables out of the range 2-10.
    `TypeError`
        Invalid type of an argument.

    Examples
    --------
    See the following function of the ``samples.py`` documentation script:
        - `samples.train_coclustering()`
    """
    # Save the task arguments
    # WARNING: Do not move this line, see the top of the "tasks" section for details
    task_args = locals()

    # Check the size of coclustering_variables
    if len(task_args["coclustering_variables"]) < 2:
        raise ValueError("coclustering_variables must have at least 2 elements")
    elif len(task_args["coclustering_variables"]) > 10:
        raise ValueError("coclustering_variables must have at most 10 elements")

    # Run the task
    _run_task("train_coclustering", task_args)

    # Return the path of the coclustering file
    return fs.get_child_path(results_dir, results_prefix + "Coclustering.khcj")


def simplify_coclustering(
    coclustering_file_path,
    simplified_coclustering_file_path,
    results_dir,
    max_preserved_information=0,
    max_cells=0,
    max_total_parts=0,
    max_part_numbers=None,
    results_prefix="",
    batch_mode=True,
    log_file_path=None,
    output_scenario_path=None,
    task_file_path=None,
    trace=False,
    stdout_file_path="",
    stderr_file_path="",
    **kwargs,
):
    """Simplifies a coclustering model

    Parameters
    ----------
    coclustering_file_path : str
        Path of the coclustering file (extension ``.khc``, or ``.khcj``).
    simplified_coclustering_file_path : str
        Path of the output coclustering file.
    results_dir : str
        Path of the results directory.
    max_preserved_information : int, default 0
        Maximum information preserve in the simplified coclustering. If equal to 0
        there is no limit.
    max_cells : int, default 0
        Maximum number of cells in the simplified coclustering. If equal to 0 there
        is no limit.
    max_total_parts : int, default 0
        Maximum number of parts totaled over all variables. If equal to 0 there is no
        limit.
    max_part_numbers : dict, optional
      Dictionary that associate variable names to their maximum number of parts to
      preserve in the simplified coclustering. If not set there is no limit.
    results_prefix : str, default ""
        Prefix of the result files.
    ... :
        See :ref:`core-api-common-params`.

    Raises
    ------
    `TypeError`
        Invalid type of an argument.

    Examples
    --------
    See the following function of the ``samples.py`` documentation script:
        - `samples.simplify_coclustering()`
    """
    # Save the task arguments
    # WARNING: Do not move this line, see the top of the "tasks" section for details
    task_args = locals()

    # Run the task
    _run_task("simplify_coclustering", task_args)


def prepare_coclustering_deployment(
    dictionary_file_path_or_domain,
    dictionary_name,
    coclustering_file_path,
    table_variable,
    deployed_variable_name,
    results_dir,
    max_preserved_information=0,
    max_cells=0,
    max_part_numbers=None,
    build_cluster_variable=True,
    build_distance_variables=False,
    build_frequency_variables=False,
    variables_prefix="",
    results_prefix="",
    batch_mode=True,
    log_file_path=None,
    output_scenario_path=None,
    task_file_path=None,
    trace=False,
    stdout_file_path="",
    stderr_file_path="",
    **kwargs,
):
    """Prepares a *individual-variable* coclustering deployment

    Parameters
    ----------
    dictionary_file_path_or_domain : str or `.DictionaryDomain`
        Path of a Khiops dictionary file or a DictionaryDomain object.
    dictionary_name : str
        Name of the dictionary to be analyzed.
    coclustering_file_path : str
        Path of the coclustering model file (extension ``.khc`` or ``.khcj``).
    table_variable : str
        Name of the table variable in the dictionary.
    deployed_variable_name : str
        Name of the coclustering variable to deploy.
    results_dir : str
        Path of the results directory.
    max_preserved_information : int, default 0
        Maximum information preserve in the simplified coclustering. If equal to 0
        there is no limit.
    max_cells : int, default 0
        Maximum number of cells in the simplified coclustering. If equal to 0 there
        is no limit.
    max_part_numbers : dict, optional
      Dictionary associating variable names to their maximum number of parts to
      preserve in the simplified coclustering. For variables not present in
      ``max_part_numbers`` there is no limit.
    build_cluster_variable : bool, default ``True``
        If ``True`` includes a cluster id variable in the deployment.
    build_distance_variables : bool, default ``False``
        If ``True`` includes a cluster distance variable in the deployment.
    build_frequency_variables : bool, default ``False``
        If ``True`` includes the frequency variables in the deployment.
    variables_prefix : str, default ""
        Prefix for the variables in the deployment dictionary.
    results_prefix : str, default ""
        Prefix of the result files.
    ... :
        See :ref:`core-api-common-params`.

    Raises
    ------
    `TypeError`
        Invalid type of an argument

    Examples
    --------
    See the following function of the ``samples.py`` documentation script:
        - `samples.deploy_model_expert()`
    """
    # Save the task arguments
    # WARNING: Do not move this line, see the top of the "tasks" section for details
    task_args = locals()

    # Run the task
    _run_task("prepare_coclustering_deployment", task_args)


def extract_clusters(
    coclustering_file_path,
    cluster_variable,
    clusters_file_path,
    max_preserved_information=0,
    max_cells=0,
    batch_mode=True,
    log_file_path=None,
    output_scenario_path=None,
    task_file_path=None,
    trace=False,
    stdout_file_path="",
    stderr_file_path="",
    **kwargs,
):
    """Extracts clusters to a tab separated (TSV) file

    Parameters
    ----------
    coclustering_file_path : str
        Path of the coclustering model file (extension ``.khc`` or ``.khcj``).
    cluster_variable : str
        Name of the variable for which the clusters are extracted.
    clusters_file_path : str
        Path of the output clusters TSV file.
    max_preserved_information : int, default 0
        Maximum information preserve in the simplified coclustering. If equal to 0 there
        is no limit.
    max_cells : int, default 0
        Maximum number of cells in the simplified coclustering. If equal to 0 there is
        no limit.
    ... :
        See :ref:`core-api-common-params`.

    Examples
    --------
    See the following function of the ``samples.py`` documentation script:
        - `samples.extract_clusters()`
    """
    # Save the task arguments
    # WARNING: Do not move this line, see the top of the "tasks" section for details
    task_args = locals()

    # Obtain the directory and name of the clusters file
    clusters_file_path = create_unambiguous_khiops_path(task_args["clusters_file_path"])
    clusters_file_name = os.path.basename(clusters_file_path)
    clusters_file_dir_path = fs.get_parent_path(clusters_file_path)
    clusters_file_dir = create_unambiguous_khiops_path(clusters_file_dir_path)
    task_args["clusters_file_name"] = clusters_file_name
    task_args["results_dir"] = clusters_file_dir

    # Delete the clusters_file_path argument
    del task_args["clusters_file_path"]

    # Run the task
    _run_task("extract_clusters", task_args)


def detect_data_table_format(
    data_table_path,
    dictionary_file_path_or_domain=None,
    dictionary_name=None,
    trace=False,
    stdout_file_path="",
    stderr_file_path="",
):
    """Detects the format of a data table

    Runs an heuristic to detect the format of a data table. The detection heuristic is
    more accurate if a dictionary with the table schema is provided.

    Parameters
    ----------
    data_table_path : str
        Path of the data table file.
    dictionary_file_path_or_domain : str or `.DictionaryDomain`, optional
        Path of a Khiops dictionary file or a DictionaryDomain object.
    dictionary_name : str, optional
        Name of the dictionary.

    Returns
    -------
    tuple
        A 2-tuple containing:
            - the ``header_line`` boolean
            - the ``field_separator`` character

        These are exactly the parameters expected in many Khiops Python API functions.

    Examples
    --------
    See the following function of the ``samples.py`` documentation script:
        - `samples.detect_data_table_format()`
    """
    # Save the task arguments
    # WARNING: Do not move this line, see the top of the "tasks" section for details
    task_args = locals()

    # Create log file to save the detect format output
    log_file_path = get_runner().create_temp_file("_detect_data_table_format_", ".log")
    task_args["log_file_path"] = log_file_path

    # Run the task without dictionary
    if dictionary_file_path_or_domain is None:
        if "dictionary_name" in task_args:
            del task_args["dictionary_name"]
            del task_args["dictionary_file_path_or_domain"]
        _run_task("detect_data_table_format", task_args)
    # Run the task with dictionary
    else:
        if task_args["dictionary_name"] is None:
            raise ValueError(
                "'dictionary_name' must be specified with "
                "'dictionary_file_path_or_domain'"
            )
        _run_task("detect_data_table_format_with_dictionary", task_args)

    # Parse the log file to obtain the header_line and field_separator parameters
    # Notes:
    # - If there is an error the run method will raise an exception; so at this stage we
    #   have a warning in the worst case.
    # - The contents of this Khiops execution are always ASCII
    log_file_contents = io.BytesIO(fs.read(log_file_path))
    with io.TextIOWrapper(log_file_contents, encoding="ascii") as log_file:
        log_file_lines = log_file.readlines()

    # Obtain the first line that contains the format spec
    header_line = None
    field_separator = None
    format_line_pattern = "File format detected: "
    for line in log_file_lines:
        if line.startswith(format_line_pattern):
            raw_format_spec = line.rstrip().replace(format_line_pattern, "")
            header_line_str, field_separator_str = raw_format_spec.split(
                " and field separator "
            )
            header_line = header_line_str == "header line"
            if field_separator_str == "tabulation":
                field_separator = "\t"
            else:
                field_separator = field_separator_str[1]
            break

    # Fail if there was no file format in the log
    if header_line is None or field_separator is None:
        raise KhiopsRuntimeError(
            "Khiops did not write the log line with the data table file format."
        )

    # Clean up the log file if necessary
    if trace:
        print(f"detect_format log file: {log_file_path}")
    else:
        fs.remove(log_file_path)

    return header_line, field_separator


# pylint: enable=unused-argument

########################
# Deprecated functions #
########################


# pylint: disable=unused-argument


def build_multi_table_dictionary(
    dictionary_file_path_or_domain,
    root_dictionary_name,
    secondary_table_variable_name,
    output_dictionary_file_path,
    overwrite_dictionary_file=False,
    batch_mode=True,
    log_file_path=None,
    output_scenario_path=None,
    task_file_path=None,
    trace=False,
):
    """Builds a multi-table dictionary from a dictionary with a key

    .. warning::
        This method is *deprecated* since Khiops 10.1.3 and will be removed in Khiops
        11. Use the `.build_multi_table_dictionary_domain` helper function to
        the same effect.

    Parameters
    ----------
    dictionary_file_path_or_domain : str or `.DictionaryDomain`
        Path of a Khiops dictionary file or a `.DictionaryDomain` object.
    root_dictionary_name : str
        Name for the new root dictionary
    secondary_table_variable_name : str
        Name, in the root dictionary, for the "table" variable of the secondary table.
    output_dictionary_file_path : str
        Path of the output dictionary path.
    overwrite_dictionary_file : bool, default ``False``
        If ``True`` it will overwrite an input dictionary file.
    ... :
        See :ref:`core-api-common-params`.

    Raises
    ------
    `ValueError`
        Invalid values of an argument
    """
    # Warn the user that this API function is deprecated and will be removed
    warnings.warn(deprecation_message("build_multi_table_dictionary", "11.0.0"))

    # Create the execution dictionary domain if it is a file
    _check_dictionary_file_path_or_domain(dictionary_file_path_or_domain)
    if isinstance(dictionary_file_path_or_domain, str):
        dictionary_domain = read_dictionary_file(dictionary_file_path_or_domain)
    else:
        dictionary_domain = dictionary_file_path_or_domain

    # Generate multi-table domain by using the eponymous helper function
    # Honor exception API:
    try:
        multi_table_domain = build_multi_table_dictionary_domain(
            dictionary_domain, root_dictionary_name, secondary_table_variable_name
        )
    except TypeError as error:
        raise ValueError from error

    # If overwrite_dictionary_file is set and the input is a path to a dictionary,
    # then the output path is set to the input path
    if overwrite_dictionary_file and isinstance(dictionary_file_path_or_domain, str):
        output_dictionary_file_path = dictionary_file_path_or_domain

    # Write multi-table domain to file
    with io.BytesIO() as stream:
        writer = KhiopsOutputWriter(stream)
        multi_table_domain.write(writer)
        fs.write(output_dictionary_file_path, stream.getvalue())


# pylint: enable=unused-argument


def get_khiops_info():
    """Returns the Khiops license information

    .. warning::
        This method is *deprecated* since Khiops 10.1 and will be removed in Khiops
        11. Use `get_khiops_version` to obtain the Khiops version of your system.

    Returns
    -------
    tuple
        A 4-tuple containing:

        - The tool version
        - The name of the machine
        - The ID of the machine
        - The number of remaining days for the license
    """
    warnings.warn(deprecation_message("get_khiops_info", "11.0.0"))
    if get_runner().khiops_version >= KhiopsVersion("10.1.0"):
        return get_khiops_version(), None, None, None
    elif get_runner().khiops_version >= KhiopsVersion("10.0.0"):
        return _get_tool_info_khiops10(get_runner(), "khiops")
    else:
        return _get_tool_info_khiops9(get_runner(), "khiops")


def get_khiops_coclustering_info():
    """Returns the Khiops Coclustering license information

    .. warning::
        This method is *deprecated* since Khiops 10.1 and will be removed in Khiops
        11. Use `get_khiops_version` to obtain the Khiops version of your system.

    Returns
    -------
    tuple
        A 4-tuple containing:

        - The tool version
        - The name of the machine
        - The ID of the machine
        - The number of remaining days for the license
    """
    warnings.warn(deprecation_message("get_khiops_coclustering_info", "11.0.0"))
    if get_runner().khiops_version >= KhiopsVersion("10.1.0"):
        return get_khiops_version(), None, None, None
    elif get_runner().khiops_version >= KhiopsVersion("10.0.0"):
        return _get_tool_info_khiops10(get_runner(), "khiops_coclustering")
    else:
        return _get_tool_info_khiops9(get_runner(), "khiops_coclustering")
