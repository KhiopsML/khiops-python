######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""API for the execution of the Khiops AutoML suite

The methods in this module allow to execute all Khiops and Khiops Coclustering tasks.

See also:
    - :ref:`core-api-common-params`
    - :ref:`core-api-input-types`
    - :ref:`core-api-sampling-mode`
    - :ref:`core-api-env-samples-dir`
"""
import io
import os
import warnings

import khiops.core.internals.filesystems as fs
from khiops.core.dictionary import DictionaryDomain
from khiops.core.exceptions import KhiopsRuntimeError
from khiops.core.internals.common import (
    CommandLineOptions,
    SystemSettings,
    deprecation_message,
    is_string_like,
    type_error_message,
)
from khiops.core.internals.runner import get_runner
from khiops.core.internals.task import get_task_registry

# Construction rules
DEFAULT_CONSTRUCTION_RULES = [
    "GetValue",
    "GetValueC",
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
]
"""List of construction rules that Khiops uses by default

.. note::
    These are all the multi-table rules.
"""  # pylint: disable=pointless-string-statement

CALENDRICAL_CONSTRUCTION_RULES = [
    "Day",
    "DecimalTime",
    "DecimalWeekDay",
    "DecimalYear",
    "DecimalYearTS",
    "GetDate",
    "GetTime",
    "LocalTimestamp",
    "WeekDay",
    "YearDay",
]
"""List of calendrical construction rules

These rules include: date, time and timestamp rules.

.. note::
    These rules are not enabled by default. The user needs to explicitly
    select each of them via the ``construction_rules`` parameter of the
    relevant Core API functions.
"""  # pylint: disable=pointless-string-statement

# List of all available construction rules in the Khiops tool
ALL_CONSTRUCTION_RULES = DEFAULT_CONSTRUCTION_RULES + CALENDRICAL_CONSTRUCTION_RULES

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

    command_line_options, system_settings, task_called_with_domain = (
        _preprocess_arguments(task_args)
    )

    # Obtain the api function from the registry
    task = get_task_registry().get_task(task_name, get_khiops_version())

    # Execute the Khiops task and cleanup when necessary
    try:
        get_runner().run(
            task,
            task_args,
            command_line_options=command_line_options,
            trace=trace,
            system_settings=system_settings,
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
        A 3-tuple containing:
            - A `~.CommandLineOptions` instance
            - A `~.SystemSettings` instance
            - A `bool` that is ``True`` if the value of the `dictionary_file_or_domain`
             `args` key is a `~.DictionaryDomain` instance.

    .. note:: This function *mutates* the input `args` dictionary.
    """
    # Execute the preprocess of common task arguments
    task_is_called_with_domain = _preprocess_task_arguments(args)

    # Create a command line options object
    command_line_options = CommandLineOptions(
        log_file_path=(args["log_file_path"] if "log_file_path" in args else ""),
        output_scenario_path=(
            args["output_scenario_path"] if "output_scenario_path" in args else ""
        ),
        task_file_path=(args["task_file_path"] if "task_file_path" in args else ""),
    )

    # Create a system settings object
    system_settings = SystemSettings()
    for arg in args:
        if arg == "max_cores":
            max_cores = args[arg]
            if max_cores is not None:
                system_settings.max_cores = int(max_cores)
        elif arg == "memory_limit_mb":
            memory_limit_mb = args[arg]
            if memory_limit_mb is not None:
                system_settings.memory_limit_mb = int(memory_limit_mb)
        elif arg == "temp_dir":
            temp_dir = args[arg]

            # temp_dir is set to a non-empty string
            if temp_dir:
                system_settings.temp_dir = temp_dir
        elif arg == "scenario_prologue":
            scenario_prologue = args[arg]

            # User-defined scenario prologue is set to a non-empty string
            if scenario_prologue:
                system_settings.scenario_prologue = scenario_prologue
    system_settings.check()

    # Clean the args to leave only the task arguments
    _clean_task_args(args)

    return command_line_options, system_settings, task_is_called_with_domain


def _deprecate_legacy_data_path(data_path_task_arg_name, task_args):
    """Detect and replace legacy data path with the current syntax

    .. note:: The function mutates task_args.
    """
    if (
        data_path_task_arg_name in task_args
        and task_args[data_path_task_arg_name] is not None
    ):
        assert "dictionary_name" in task_args or "train_dictionary_name" in task_args
        if "dictionary_name" in task_args:
            current_dictionary_name = task_args["dictionary_name"]
        else:
            current_dictionary_name = task_args["train_dictionary_name"]

        for kdic_path in task_args[data_path_task_arg_name].keys():
            if isinstance(kdic_path, str):
                deprecated_data_path_separator = "`"
                data_path_separator = "/"
                kdic_path_for_warning = kdic_path
            else:
                assert isinstance(kdic_path, bytes)
                deprecated_data_path_separator = b"`"
                data_path_separator = b"/"
                if isinstance(current_dictionary_name, str):
                    current_dictionary_name = bytes(
                        current_dictionary_name, encoding="ascii"
                    )
                kdic_path_for_warning = kdic_path.decode("ascii")

            # Path split "`" yields non-empty fragments; the first fragment
            # starts with the current dictionary name
            kdic_path_parts = kdic_path.split(deprecated_data_path_separator)
            if all(len(path_part) > 0 for path_part in kdic_path_parts):
                source_dictionary_name = kdic_path_parts[0]
                if source_dictionary_name == current_dictionary_name:
                    # Escape any "/" char in the path parts except for the
                    # current dictionary, which is is skipped from the new path
                    new_kdic_path_parts = []
                    for kdic_path_part in kdic_path_parts[1:]:
                        new_kdic_path_parts.append(
                            kdic_path_part.replace(
                                data_path_separator,
                                deprecated_data_path_separator + data_path_separator,
                            )
                        )

                    # Replace the legacy data path with the current data path
                    new_kdic_path = data_path_separator.join(new_kdic_path_parts)
                    kdic_file_path = task_args[data_path_task_arg_name].pop(kdic_path)
                    task_args[data_path_task_arg_name][new_kdic_path] = kdic_file_path
                    warnings.warn(
                        deprecation_message(
                            "'`'-based dictionary data path: "
                            f"'{kdic_path_for_warning}'",
                            "11.0.1",
                            replacement=(
                                "'/'-based dictionary data path "
                                f"convention: '{new_kdic_path}'"
                            ),
                            quote=False,
                        )
                    )


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
    # Process the output path
    # if path is dir, then generate full report path according to GUI defaults
    file_path_arg_names = {
        "analysis_report_file_path": "AnalysisResults.khj",
        "evaluation_report_file_path": "EvaluationReport.khj",
        "coclustering_report_file_path": "Coclustering.khcj",
        "coclustering_dictionary_file_path": "Coclustering.kdic",
    }
    for file_path_arg_name, default_file_name in file_path_arg_names.items():
        if file_path_arg_name in task_args:
            file_path = task_args[file_path_arg_name]

            # If path ends with path separator or exists as a dir, then consider
            # it is dir and concatenate default report file name to it
            if file_path.endswith(os.path.sep) or (
                fs.is_local_resource(file_path) and os.path.isdir(file_path)
            ):
                # Add deprecation warning
                warnings.warn(
                    deprecation_message(
                        "'results_dir'",
                        "11.0.1",
                        replacement=file_path_arg_name,
                        quote=False,
                    )
                )
                # Update the path
                if fs.is_local_resource(file_path):
                    norm_file_path = os.path.normpath(file_path)
                else:
                    norm_file_path = file_path
                full_file_path = fs.get_child_path(norm_file_path, default_file_name)
                task_args[file_path_arg_name] = full_file_path

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

    # Transform the use_complement_as_test bool parameter to its string counterpart
    if "use_complement_as_test" in task_args:
        if task_args["use_complement_as_test"]:
            task_args["test_database_mode"] = "Complementary"
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

    # Detect and replace deprecated data-path syntax on additional_data_tables
    # Mutate task_args in the process
    for data_path_task_arg_name in (
        "additional_data_tables",
        "output_additional_data_tables",
    ):
        _deprecate_legacy_data_path(data_path_task_arg_name, task_args)

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
        - System settings (they already are in another object).
        - Parameters removed from the API and warns about it.
        - Renamed API parameters and warns about it.
    """
    # Remove non-task parameters
    command_line_arg_names = [
        "log_file_path",
        "output_scenario_path",
        "task_file_path",
    ]
    system_settings_arg_names = [
        "max_cores",
        "memory_limit_mb",
        "temp_dir",
        "scenario_prologue",
    ]
    other_arg_names = [
        "dictionary_file_path_or_domain",
        "trace",
        "stdout_file_path",
        "stderr_file_path",
    ]
    for arg_name in (
        command_line_arg_names + system_settings_arg_names + other_arg_names
    ):
        if arg_name in task_args:
            del task_args[arg_name]


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
    log_file_path=None,
    output_scenario_path=None,
    task_file_path=None,
    trace=False,
    stdout_file_path="",
    stderr_file_path="",
    max_cores=None,
    memory_limit_mb=None,
    temp_dir="",
    scenario_prologue="",
):
    """Exports a Khiops dictionary file to JSON format (``.kdicj``)

    Parameters
    ----------
    dictionary_file_path_or_domain : str or `.DictionaryDomain`
        Path of a Khiops dictionary file or a DictionaryDomain object.
    json_dictionary_file_path : str
        Path (absolute path recommended) to the output dictionary file,
        in the JSON format. Note that a relative path will produce a file in
        the current working directory.
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
    log_file_path=None,
    output_scenario_path=None,
    task_file_path=None,
    trace=False,
    stdout_file_path="",
    stderr_file_path="",
    max_cores=None,
    memory_limit_mb=None,
    temp_dir="",
    scenario_prologue="",
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
        Path (absolute path recommended) of the output dictionary file. Note that
        a relative path will produce a file in the current working directory.
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
    log_file_path=None,
    output_scenario_path=None,
    task_file_path=None,
    trace=False,
    stdout_file_path="",
    stderr_file_path="",
    max_cores=None,
    memory_limit_mb=None,
    temp_dir="",
    scenario_prologue="",
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
    analysis_report_file_path,
    detect_format=True,
    header_line=None,
    field_separator=None,
    sample_percentage=70.0,
    sampling_mode="Include sample",
    use_complement_as_test=True,
    selection_variable="",
    selection_value="",
    additional_data_tables=None,
    do_data_preparation_only=False,
    main_target_value="",
    keep_selected_variables_only=True,
    max_evaluated_variables=0,
    max_selected_variables=0,
    max_constructed_variables=1000,
    construction_rules=None,
    max_text_features=10000,
    text_features="words",
    max_trees=10,
    max_pairs=0,
    all_possible_pairs=True,
    specific_pairs=None,
    group_target_value=False,
    discretization_method="MODL",
    grouping_method="MODL",
    max_parts=0,
    log_file_path=None,
    output_scenario_path=None,
    task_file_path=None,
    trace=False,
    stdout_file_path="",
    stderr_file_path="",
    max_cores=None,
    memory_limit_mb=None,
    temp_dir="",
    scenario_prologue="",
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
    analysis_report_file_path : str
        Path (absolute path recommended) to the analysis report file,
        in the JSON format. An additional dictionary file with the same name and
        extension ``.model.kdic`` is built, which contains the trained models.
        Note that a relative path will produce a report file in the current working
        directory.
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
    selection_variable : str, default ""
        It trains with only the records such that the value of ``selection_variable`` is
        equal to ``selection_value``. Ignored if equal to "".
    selection_value: str or int or float, default ""
        See ``selection_variable`` option above. Ignored if equal to "".
    additional_data_tables : dict, optional
        A dictionary containing the data paths and file paths for a multi-table
        dictionary file. For more details see :doc:`/multi_table_primer`.
    do_data_preparation_only : bool, default ``False``
        If ``True`` it only does data preparation via MODL preprocessing without
        training a Selective Naive Bayes Predictor.
    main_target_value : str, default ""
        If this target value is specified then it guarantees the calculation of lift
        curves for it.
    keep_selected_variables_only : bool, default ``True``
        Keeps only predictor-selected variables in the supervised analysis report.
    max_evaluated_variables : int, default 0
        Maximum number of variables to be evaluated in the SNB predictor training. If
        equal to 0 it evaluates all informative variables.
    max_selected_variables : int, default 0
        Maximum number of variables to be selected in the SNB predictor. If equal to
        0 it selects all the variables kept in the training.
    max_constructed_variables : int, default 1000
        Maximum number of variables to construct.
    construction_rules : list of str, optional
        Allowed rules for the automatic variable construction. If not set, Khiops
        uses the multi-table construction rules listed in
        `DEFAULT_CONSTRUCTION_RULES`.
    max_text_features : int, default 10000
        Maximum number of text features to construct.
    text_features : str, default "words"
        Type of the text features. Can be either one of:

            - "words": sequences of non-space characters
            - "ngrams": sequences of bytes
            - "tokens": user-defined

    max_trees : int, default 10
        Maximum number of trees to construct.
    max_pairs : int, default 0
        Maximum number of variable pairs to construct.
    specific_pairs : list of tuple, optional
        User-specified pairs as a list of 2-tuples of feature names. If a given tuple
        contains only one non-empty feature name, then it generates all the pairs
        containing it (within the maximum limit ``max_pairs``). These pairs have top
        priority: they are constructed first.
    all_possible_pairs : bool, default ``True``
        If ``True`` tries to create all possible pairs within the limit ``max_pairs``.
        Pairs specified with ``specific_pairs`` have top priority: they are constructed
        first.
    group_target_value : bool, default ``False``
        Allows grouping of the target variable values in classification. It can
        substantially increase the training time.
    discretization_method : str, default "MODL"
        Name of the discretization method in case of unsupervised analysis.
        Its valid values are: "MODL", "EqualWidth", "EqualFrequency" or "none".
        Ignored for supervised analysis.
    grouping_method : str, default "MODL"
        Name of the grouping method in case of unsupervised analysis.
        Its valid values are: "MODL", "BasicGrouping" or "none".
        Ignored for supervised analysis.
    max_parts : int, default 0
        Maximum number of variable parts produced by preprocessing methods. If equal
        to 0 it is automatically calculated.
        Special default values for unsupervised analysis:

            - If ``discretization_method`` is "EqualWidth" or "EqualFrequency": 10
            - If ``grouping_method`` is "BasicGrouping": 10

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
    if target_variable != "":
        current_dir = fs.parent_path(analysis_report_file_path)
        report_file_name, _ = os.path.splitext(
            os.path.basename(analysis_report_file_path)
        )
        modeling_dictionary_file_path = fs.get_child_path(
            current_dir, f"{report_file_name}.model.kdic"
        )
    else:
        modeling_dictionary_file_path = None

    return (analysis_report_file_path, modeling_dictionary_file_path)


def interpret_predictor(
    dictionary_file_path_or_domain,
    predictor_dictionary_name,
    interpretor_file_path,
    max_variable_importances=100,
    importance_ranking="Global",
    log_file_path=None,
    output_scenario_path=None,
    task_file_path=None,
    trace=False,
    stdout_file_path="",
    stderr_file_path="",
    max_cores=None,
    memory_limit_mb=None,
    temp_dir="",
    scenario_prologue="",
    **kwargs,
):
    r"""Builds an interpretation dictionary from a predictor

    Parameters
    ----------
    dictionary_file_path_or_domain : str or `.DictionaryDomain`
        Path of a Khiops dictionary file or a DictionaryDomain object.
    predictor_dictionary_name : str
        Name of the predictor dictionary used while building the interpretation model.
    interpretor_file_path : str
        Path to the interpretor dictionary file.
    max_variable_importances : int, default 100
        Maximum number of variable importances to be selected in the interpretation
        model. If the predictor contains fewer variables than this number, then
        all the variables of the predictor are considered.
    importance_ranking : str, default "Global"
        Ranking of the Shapley values produced by the interpretor. Ca be one of:
            - "Global": predictor variables are ranked by decreasing global importance.
            - "Individual": predictor variables are ranked by decreasing individual
              Shapley value.
    ... :
        See :ref:`core-api-common-params`.

    Raises
    ------
    `ValueError`
        Invalid values of an argument
    `TypeError`
        Invalid type of an argument

    Examples
    --------
    See the following functions of the ``samples.py`` documentation script:
        - `samples.interpret_predictor()`
        - `samples.deploy_model_mt_with_interpretation()`
    """
    # Save the task arguments
    # WARNING: Do not move this line, see the top of the "tasks" section for details
    task_args = locals()

    # Run the task
    _run_task("interpret_predictor", task_args)


def reinforce_predictor(
    dictionary_file_path_or_domain,
    predictor_dictionary_name,
    reinforced_predictor_file_path,
    reinforcement_target_value="",
    reinforcement_lever_variables=None,
    log_file_path=None,
    output_scenario_path=None,
    task_file_path=None,
    trace=False,
    stdout_file_path="",
    stderr_file_path="",
    max_cores=None,
    memory_limit_mb=None,
    temp_dir="",
    scenario_prologue="",
    **kwargs,
):
    r"""Builds a reinforced predictor from a predictor

    A reinforced predictor is a model which increases the importance of specified lever
    variables in order to increase the probability of occurrence of the specified target
    value.

    Parameters
    ----------
    dictionary_file_path_or_domain : str or `.DictionaryDomain`
        Path of a Khiops dictionary file or a DictionaryDomain object.
    predictor_dictionary_name : str
        Name of the predictor dictionary used while building the reinforced predictor.
    reinforced_predictor_file_path : str
        Path to the reinforced predictor dictionary file.
    reinforcement_target_value : str, default ""
        If this target value is specified, then its probability of occurrence is
        tentatively increased.
    reinforcement_lever_variables : list of str
        The names of variables to use as lever variables while building the
        reinforced predictor. Min length: 1. Max length: the total number of variables
        in the prediction model.
    ... :
        See :ref:`core-api-common-params`.

    Raises
    ------
    `ValueError`
        Invalid values of an argument
    `TypeError`
        Invalid type of an argument

    Examples
    --------
    See the following functions of the ``samples.py`` documentation script:
        - `samples.reinforce_predictor()`
        - `samples.deploy_reinforced_model_mt()`
    """
    # Save the task arguments
    # WARNING: Do not move this line, see the top of the "tasks" section for details
    task_args = locals()

    # Run the task
    _run_task("reinforce_predictor", task_args)


def evaluate_predictor(
    dictionary_file_path_or_domain,
    train_dictionary_name,
    data_table_path,
    evaluation_report_file_path,
    detect_format=True,
    header_line=None,
    field_separator=None,
    sample_percentage=100.0,
    sampling_mode="Include sample",
    selection_variable="",
    selection_value="",
    additional_data_tables=None,
    main_target_value="",
    log_file_path=None,
    output_scenario_path=None,
    task_file_path=None,
    trace=False,
    stdout_file_path="",
    stderr_file_path="",
    max_cores=None,
    memory_limit_mb=None,
    temp_dir="",
    scenario_prologue="",
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
    evaluation_report_file_path : str
        Path (absolute path recommended) to the evaluation report file,
        in the JSON format. Note that a relative path will produce a report file in
        the current working directory.
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

    # Run the task
    _run_task("evaluate_predictor", task_args)

    # Return the path of the JSON report
    return evaluation_report_file_path


def train_recoder(
    dictionary_file_path_or_domain,
    dictionary_name,
    data_table_path,
    target_variable,
    analysis_report_file_path,
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
    max_text_features=10000,
    text_features="words",
    max_trees=10,
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
    discretization_method="MODL",
    grouping_method="MODL",
    max_parts=0,
    log_file_path=None,
    output_scenario_path=None,
    task_file_path=None,
    trace=False,
    stdout_file_path="",
    stderr_file_path="",
    max_cores=None,
    memory_limit_mb=None,
    temp_dir="",
    scenario_prologue="",
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
    analysis_report_file_path : str
        Path (absolute path recommended) to the analysis report file,
        in the JSON format. An additional dictionary file with the same name and
        extension ``.model.kdic`` is built, which contains the trained recoding model.
        Note that a relative path will produce a report file in the current working
        directory.
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
        Allowed rules for the automatic variable construction. If not set, Khiops
        uses the multi-table construction rules listed in
        `DEFAULT_CONSTRUCTION_RULES`.
    max_text_features : int, default 10000
        Maximum number of text features to construct.
    text_features : str, default "words"
        Type of the text features. Can be either one of:

            - "words": sequences of non-space characters
            - "ngrams": sequences of bytes
            - "tokens": user-defined

    max_trees : int, default 10
        Maximum number of trees to construct.
    max_pairs : int, default 0
        Maximum number of variable pairs to construct.
    specific_pairs : list of tuple, optional
        User-specified pairs as a list of 2-tuples of feature names. If a given tuple
        contains only one non-empty feature name, then it generates all the pairs
        containing it (within the maximum limit ``max_pairs``). These pairs have top
        priority: they are constructed first.
    all_possible_pairs : bool, default ``True``
        If ``True`` tries to create all possible pairs within the limit ``max_pairs``.
        Pairs specified with ``specific_pairs`` have top priority: they are constructed
        first.
    group_target_value : bool, default ``False``
        Allows grouping of the target variable values in classification. It can
        substantially increase the training time.
    informative_variables_only : bool, default ``True``
        If ``True`` keeps only informative variables.
    max_variables : int, default 0
        Maximum number of variables to keep. If equal to 0 keeps all variables.
    keep_initial_categorical_variables : bool, default ``False``
        If ``True`` keeps the initial categorical variables.
    keep_initial_numerical_variables : bool, default ``False``
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

    discretization_method : str, default "MODL"
        Name of the discretization method in case of unsupervised analysis.
        Its valid values are: "MODL", "EqualWidth", "EqualFrequency" or "none".
        Ignored for supervised analysis.
    grouping_method : str, default "MODL"
        Name of the grouping method in case of unsupervised analysis.
        Its valid values are: "MODL", "BasicGrouping" or "none".
        Ignored for supervised analysis.
    max_parts : int, default 0
        Maximum number of variable parts produced by preprocessing methods. If equal
        to 0 it is automatically calculated.
        Special default values for unsupervised analysis:

            - If ``discretization_method`` is "EqualWidth" or "EqualFrequency": 10
            - If ``grouping_method`` is "BasicGrouping": 10

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
    current_dir = fs.parent_path(analysis_report_file_path)
    report_file_name, _ = os.path.splitext(os.path.basename(analysis_report_file_path))

    modeling_dictionary_file_path = fs.get_child_path(
        current_dir, f"{report_file_name}.model.kdic"
    )

    return (analysis_report_file_path, modeling_dictionary_file_path)


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
    log_file_path=None,
    output_scenario_path=None,
    task_file_path=None,
    trace=False,
    stdout_file_path="",
    stderr_file_path="",
    max_cores=None,
    memory_limit_mb=None,
    temp_dir="",
    scenario_prologue="",
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
        Path (absolute path recommended) of the output data file. Note that a
        relative path will produce a file in the current working directory.
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
        - `samples.deploy_model_mt_with_interpretation()`
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
    log_file_path=None,
    output_scenario_path=None,
    task_file_path=None,
    trace=False,
    stdout_file_path="",
    stderr_file_path="",
    max_cores=None,
    memory_limit_mb=None,
    temp_dir="",
    scenario_prologue="",
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
        Path (absolute path recommended) of the output dictionary file. Note that
        a relative path will produce a file in the current working directory.
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
    log_file_path=None,
    output_scenario_path=None,
    task_file_path=None,
    trace=False,
    stdout_file_path="",
    stderr_file_path="",
    max_cores=None,
    memory_limit_mb=None,
    temp_dir="",
    scenario_prologue="",
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
        Path (absolute path recommended) of the output data file. Note that a
        relative path will produce a file in the current working directory.
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
    log_file_path=None,
    output_scenario_path=None,
    task_file_path=None,
    trace=False,
    stdout_file_path="",
    stderr_file_path="",
    max_cores=None,
    memory_limit_mb=None,
    temp_dir="",
    scenario_prologue="",
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
        Path (absolute path recommended) of the output data file. Note that a
        relative path will produce a file in the current working directory.
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
    coclustering_report_file_path,
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
    log_file_path=None,
    output_scenario_path=None,
    task_file_path=None,
    trace=False,
    stdout_file_path="",
    stderr_file_path="",
    max_cores=None,
    memory_limit_mb=None,
    temp_dir="",
    scenario_prologue="",
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
    coclustering_report_file_path : str
        Path (absolute path recommended) to the coclustering report file,
        in the JSON format. Note that a relative path will produce a report file in
        the current working directory.
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
    return coclustering_report_file_path


def train_instance_variable_coclustering(
    dictionary_file_path_or_domain,
    dictionary_name,
    data_table_path,
    coclustering_report_file_path,
    detect_format=True,
    header_line=None,
    field_separator=None,
    sample_percentage=100.0,
    sampling_mode="Include sample",
    selection_variable="",
    selection_value="",
    additional_data_tables=None,
    min_optimization_time=0,
    log_file_path=None,
    output_scenario_path=None,
    task_file_path=None,
    trace=False,
    stdout_file_path="",
    stderr_file_path="",
    max_cores=None,
    memory_limit_mb=None,
    temp_dir="",
    scenario_prologue="",
    **kwargs,
):
    r"""Trains an instance-variable coclustering model from a data table
    .. note::

        If keys are available in the input dictionary, they are used as instance
        identifiers. Otherwise, line numbers in the instance data table are used as
        instance idenfitiers.

    Parameters
    ----------
    dictionary_file_path_or_domain : str or `.DictionaryDomain`
        Path of a Khiops dictionary file or a DictionaryDomain object.
    dictionary_name : str
        Name of the dictionary to be analyzed.
    data_table_path : str
        Path of the data table file.
    coclustering_report_file_path : str
        Path (absolute path recommended) to the coclustering report file,
        in the JSON format. Note that a relative path will produce a report file in
        the current working directory.
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
    min_optimization_time : int, default 0
        Minimum optimization time in seconds.
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
        - `samples.train_instance_variable_coclustering()`
    """
    # Save the task arguments
    # WARNING: Do not move this line, see the top of the "tasks" section for details
    task_args = locals()

    # Run the task
    _run_task("train_instance_variable_coclustering", task_args)

    # Return the path of the coclustering file
    return coclustering_report_file_path


def simplify_coclustering(
    coclustering_file_path,
    simplified_coclustering_file_path,
    results_dir=None,
    max_preserved_information=0,
    max_cells=0,
    max_total_parts=0,
    max_part_numbers=None,
    log_file_path=None,
    output_scenario_path=None,
    task_file_path=None,
    trace=False,
    stdout_file_path="",
    stderr_file_path="",
    max_cores=None,
    memory_limit_mb=None,
    temp_dir="",
    scenario_prologue="",
    **kwargs,
):
    """Simplifies a coclustering model

    Parameters
    ----------
    coclustering_file_path : str
        Path of the coclustering file (extension ``.khc``, or ``.khcj``).
    simplified_coclustering_file_path : str
        Path (absolute path recommended) of the output coclustering file. Note
        that a relative path will produce a report file in the current working
        directory.
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

    # Special processing for the case when:
    # - simplified_coclustering_file_path which does not start with the path separator
    #   (i.e. is a file name or a relative path), and
    # - results_dir which is not None.
    # Steps:
    # 1. concatenate simplified_coclustering_file_path to results_dir
    # 2. issue deprecation warning for results_dir
    # 3. remove 'results_dir' from the task arguments

    if results_dir is not None and not simplified_coclustering_file_path.startswith(
        os.path.sep
    ):
        warnings.warn(
            deprecation_message(
                results_dir,
                "11.0.1",
                replacement="simplified_coclustering_file_path",
                quote=False,
            )
        )
        task_args["simplified_coclustering_file_path"] = os.path.join(
            results_dir, simplified_coclustering_file_path
        )

    # Remove results_dir from the task arguments in all cases
    # Note: it is ignored if None or if simplified_coclustering_file_path is absolute
    del task_args["results_dir"]

    # Run the task
    _run_task("simplify_coclustering", task_args)


def prepare_coclustering_deployment(
    dictionary_file_path_or_domain,
    dictionary_name,
    coclustering_file_path,
    table_variable,
    deployed_variable_name,
    coclustering_dictionary_file_path,
    max_preserved_information=0,
    max_cells=0,
    max_total_parts=0,
    max_part_numbers=None,
    build_cluster_variable=True,
    build_distance_variables=False,
    build_frequency_variables=False,
    variables_prefix="",
    log_file_path=None,
    output_scenario_path=None,
    task_file_path=None,
    trace=False,
    stdout_file_path="",
    stderr_file_path="",
    max_cores=None,
    memory_limit_mb=None,
    temp_dir="",
    scenario_prologue="",
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
    coclustering_dictionary_file_path : str
        Path of the coclustering dictionary file for deployment.
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
    max_total_parts=0,
    max_part_numbers=None,
    log_file_path=None,
    output_scenario_path=None,
    task_file_path=None,
    trace=False,
    stdout_file_path="",
    stderr_file_path="",
    max_cores=None,
    memory_limit_mb=None,
    temp_dir="",
    scenario_prologue="",
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
        Path (absolute path recommended) of the output clusters TSV file. Note
        that a relative path will produce a file in the current working directory.
    max_preserved_information : int, default 0
        Maximum information preserve in the simplified coclustering. If equal to 0 there
        is no limit.
    max_cells : int, default 0
        Maximum number of cells in the simplified coclustering. If equal to 0 there is
        no limit.
    max_total_parts : int, default 0
        Maximum number of parts totaled over all variables. If equal to 0 there is no
        limit.
    max_part_numbers : dict, optional
      Dictionary that associate variable names to their maximum number of parts to
      preserve in the simplified coclustering. If not set there is no limit.
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

    # Run the task
    _run_task("extract_clusters", task_args)


def detect_data_table_format(
    data_table_path,
    dictionary_file_path_or_domain=None,
    dictionary_name=None,
    log_file_path=None,
    output_scenario_path=None,
    task_file_path=None,
    trace=False,
    stdout_file_path="",
    stderr_file_path="",
    max_cores=None,
    memory_limit_mb=None,
    temp_dir="",
    scenario_prologue="",
    **kwargs,
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
    ... :
        See :ref:`core-api-common-params`.

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

    # If not defined, create log file to save the detect format output
    run_log_file_path = log_file_path
    if log_file_path is None:
        run_log_file_path = get_runner().create_temp_file(
            "_detect_data_table_format_", ".log"
        )
        task_args["log_file_path"] = run_log_file_path

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
    log_file_contents = io.BytesIO(fs.read(run_log_file_path))
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
        print(f"detect_format log file: {run_log_file_path}")
    elif log_file_path is None:
        fs.remove(run_log_file_path)

    return header_line, field_separator


# pylint: enable=unused-argument
