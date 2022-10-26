######################################################################################
# Copyright (c) 2018 - 2022 Orange - All Rights Reserved                             #
# * This software is the confidential and proprietary information of Orange.         #
# * You shall not disclose such Restricted Information and shall use it only in      #
#   accordance with the terms of the license agreement you entered into with Orange  #
#   named the "Khiops - Python Library Evaluation License".                          #
# * Unauthorized copying of this file, via any medium is strictly prohibited.        #
# * See the "LICENSE.md" file for more details.                                      #
######################################################################################
"""API for the execution of the Khiops AutoML suite

The methods in this module allow to execute all Khiops and Khiops Coclustering tasks.

"""
import io
import os
import warnings
from urllib.parse import urlparse

from ..core import filesystems as fs
from ..core.analysis_results import AnalysisResults
from ..core.coclustering_results import CoclusteringResults
from ..core.common import (
    KhiopsVersion,
    PyKhiopsEnvironmentError,
    PyKhiopsRuntimeError,
    deprecation_message,
    is_dict_like,
    is_list_like,
    is_string_like,
    removal_message,
    renaming_message,
    type_error_message,
)
from ..core.dictionary import DictionaryDomain
from ..core.runner import _get_tool_info_khiops9, _get_tool_info_khiops10, get_runner
from ..core.scenario import (
    ConfigurableKhiopsScenario,
    DatabaseParameter,
    KeyValueListParameter,
    PathParameter,
    RecordListParameter,
    get_scenario_file_path,
)

# Disable pylint's too many lines: This module is big and won't be smaller anytime soon
# pylint: disable=too-many-lines

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


def _create_unambiguous_khiops_path(path):
    """Creates a path that is unambiguous for Khiops

    Khiops needs that a non absolute path starts with "." so not use the path of an
    internally saved state as reference point.

    For example: if we open the data table "/some/path/to/data.txt" and then set the
    results directory simply as "results" the effective location of the results
    directory will be "/some/path/to/results" instead of "$CWD/results". This behavior
    is a feature in Khiops but it is undesirable when using it as a library.

    This function returns a path so the library behave as expected: a path relative to
    the $CWD if it is a non absolute path.
    """
    # Check for string
    if not isinstance(path, str):
        raise TypeError(type_error_message("path", path, str))

    # Empty path returned as-is
    if not path:
        return path

    # Add a "." to a local path if necessary. It is *not* necessary when:
    # - `path` is an URI
    # - `path` is an absolute path
    # - `path` is a path starting with "."
    uri_info = urlparse(path, allow_fragments=False)
    if os.path.isabs(path) or path.startswith(".") or uri_info.scheme != "":
        return path
    else:
        return os.path.join(".", path)


def _get_or_create_execution_dictionary_file(dictionary_file_path_or_domain, trace):
    """Access the dictionary path or creates one from a DictionaryDomain object"""
    # Allow only 'str' or 'DictionaryDomain' types
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


def _resolve_format_spec(detect_format, header_line, field_separator):
    """Transforms the format spec into parameters ready to be used in an scenario"""
    if detect_format and header_line is None and field_separator is None:
        disable_detect_format = ""
        header_line = "true"
        field_separator = ""
    else:
        disable_detect_format = "// "
        if header_line is None:
            header_line = "true"
        else:
            if not isinstance(header_line, bool):
                raise TypeError(type_error_message("header_line", header_line, bool))
            header_line = str(header_line).lower()
        if field_separator is None:
            field_separator = ""
        else:
            if not isinstance(field_separator, str):
                raise TypeError(
                    type_error_message("field_separator", field_separator, str)
                )
            if len(field_separator) > 1:
                raise ValueError("field_separator must have length at most 1")
            if field_separator == "\t":
                field_separator = ""

    return disable_detect_format, header_line, field_separator


#######
# API #
#######


def get_khiops_version():
    """Returns the Khiops version

    Returns
    -------
    str
        The Khiops version of the current `.PyKhiopsRunner` backend.
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


def export_dictionary_as_json(
    dictionary_file_path_or_domain, json_dictionary_file_path
):
    """Exports a Khiops dictionary file to JSON format (``.kdicj``)

    Parameters
    ----------
    dictionary_file_path_or_domain : str or `.DictionaryDomain`
        Path of a Khiops dictionary file or a DictionaryDomain object.

    Examples
    --------
    See the following function of the ``samples.py`` documentation script:
        - `samples.export_dictionary_files()`
    """
    # Get or create the execution dictionary
    execution_dictionary_file_path = _get_or_create_execution_dictionary_file(
        dictionary_file_path_or_domain, False
    )

    # Create the scenario parameters
    scenario_params = {
        "__dictionary_file__": PathParameter(execution_dictionary_file_path),
        "__json_dictionary_file__": PathParameter(json_dictionary_file_path),
    }

    # Create scenario
    scenario = ConfigurableKhiopsScenario(
        get_scenario_file_path(
            "export_dictionary_as_json", get_runner().khiops_version
        ),
        scenario_params,
    )

    # Run Khiops
    get_runner().run("khiops", scenario)


def read_dictionary_file(dictionary_file_path):
    """Reads a Khiops dictionary file

    Parameters
    ----------
    dictionary_file : str
        Path of the file to be imported. The file can be either Khiops Dictionary
        (extension ``kdic``) or Khiops JSON Dictionary (extension ``.json`` or
        ``.kdicj``).

    Returns
    -------
    `.DictionaryDomain`
        An dictionary domain representing the information in the dictionary file.

    Raises
    ------
    `ValueError`
        When the file has an extension other than ``.kdic``, ``.kdicj`` or ``.json``.

    Examples
    --------
    See the following functions of the ``samples.py`` documentation script:
        - `samples.export_dictionary_files()`
        - `samples.train_predictor_with_cross_validation()`
        - `samples.multiple_train_predictor()`
        - `samples.deploy_model_expert()`
    """
    extension = os.path.splitext(dictionary_file_path)[1].lower()
    if extension not in [".kdic", ".kdicj", "json"]:
        raise ValueError(
            f"Input file must have extension 'kdic', 'kdicj' or 'json'."
            f"It has extension: '{extension}'."
        )
    # Import dictionary file: Translate to JSON first if it is 'kdic'
    try:
        if extension == ".kdic":
            tmp_dictionary_file_path = get_runner().create_temp_file(
                "_read_dictionary_file_", ".kdicj"
            )
            export_dictionary_as_json(dictionary_file_path, tmp_dictionary_file_path)
            json_dictionary_file_path = tmp_dictionary_file_path
        else:
            json_dictionary_file_path = dictionary_file_path
        domain = DictionaryDomain()
        domain.read_khiops_dictionary_json_file(json_dictionary_file_path)
    # Always clean up temporary files
    finally:
        if extension == ".kdic":
            fs.create_resource(tmp_dictionary_file_path).remove()

    return domain


def read_analysis_results_file(json_file_path):
    """Reads a Khiops JSON report

    Parameters
    ----------
    json_file_path : str
        Path of the JSON report file.

    Returns
    -------
    `.AnalysisResults`
        An instance of AnalysisResults containing the report's information.

    Examples
    --------
    See the following functions of the ``samples.py`` documentation script:
        - `samples.access_predictor_evaluation_report()`
        - `samples.train_predictor_with_cross_validation()`
        - `samples.multiple_train_predictor()`
    """
    results = AnalysisResults()
    results.read_khiops_json_file(json_file_path)
    return results


def read_coclustering_results_file(json_file_path):
    """Reads a Khiops Coclustering JSON report

    Parameters
    ----------
    json_file_path : str
        Path of the JSON report file.

    Returns
    -------
    `.CoclusteringResults`
        An instance of CoclusteringResults containing the report's information.
    """
    coclustering_results = CoclusteringResults()
    coclustering_results.read_khiops_coclustering_json_file(json_file_path)
    return coclustering_results


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
        its field separator. It's ignored if ``header_line`` or ``field_separator`` are
        set.
    header_line : bool, optional (default ``True`` if ``detect_format`` is ``False``)
        If ``True`` it uses the first line of the data as column names. Overrides
        ``detect_format`` if set.
    field_separator : str, optional (default "\\t" if ``detect_format`` is ``False``)
        A field separator character, overrides ``detect_format`` if set ("" counts
        as "\\t").
    ... :
        Options of the `.PyKhiopsRunner.run` method from the class `.PyKhiopsRunner`.
    """
    # Resolve the database format parameters
    disable_detect_format, header_line, field_separator = _resolve_format_spec(
        detect_format, header_line, field_separator
    )

    # Create the scenario parameters
    scenario_params = {
        "__data_table__": PathParameter(data_table_path),
        "__output_dictionary_file__": PathParameter(output_dictionary_file_path),
        "__output_dictionary__": output_dictionary_name,
        "__header_line__": header_line,
        "__field_separator__": field_separator,
        "__disable_detect_format__": disable_detect_format,
    }

    # Create the scenario
    scenario = ConfigurableKhiopsScenario(
        get_scenario_file_path(
            "build_dictionary_from_data_table", get_runner().khiops_version
        ),
        scenario_params,
    )

    # Execute Khiops
    get_runner().run(
        "khiops",
        scenario,
        batch_mode=batch_mode,
        log_file_path=log_file_path,
        output_scenario_path=output_scenario_path,
        task_file_path=task_file_path,
        trace=trace,
        **kwargs,
    )


def check_database(
    dictionary_file_path_or_domain,
    dictionary_name,
    data_table_path,
    detect_format=True,
    header_line=None,
    field_separator=None,
    sample_percentage=100,
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
        its field separator. It's ignored if ``header_line`` or ``field_separator`` are
        set.
    header_line : bool, optional (default ``True`` if ``detect_format`` is ``False``)
        If ``True`` it uses the first line of the data as column names. Overrides
        ``detect_format`` if set.
    field_separator : str, optional (default "\\t" if ``detect_format`` is ``False``)
        A field separator character, overrides ``detect_format`` if set ("" counts
        as "\\t").
    sample_percentage : int, default 100
        See ``sampling_mode`` option below.
    sampling_mode : "Include sample" or "Exclude sample"
        If equal to "Include sample" it checks ``sample_percentage`` percent of data and
        if equal to "Exclude sample" ``100 - sample_percentage`` percent of data.
    selection_variable : str, default ""
        It checks only the records such that the value of ``selection_variable`` is
        equal to ``selection_value``. Ignored if equal to "".
    selection_value : str, default ""
        See ``selection_variable`` option above. Ignored if equal to "".
    additional_data_tables : dict, optional
        A dictionary containing the data paths and file paths for a multi-table
        dictionary file. For more details see :doc:`/multi_table_tasks`.
    max_messages : int, default 20
        Maximum number of error messages to write in the log file.
    ... :
        Options of the `.PyKhiopsRunner.run` method from the class `.PyKhiopsRunner`.

    Examples
    --------
    See the following function of the ``samples.py`` documentation script:
        - `samples.check_database()`
    """
    # Handle renamed/removed parameters
    if "max_message_number" in kwargs:
        warnings.warn(
            renaming_message("max_message_number", "max_messages", "10.0"), stacklevel=2
        )
        del kwargs["max_message_number"]

    if "dictionary_domain" in kwargs:
        warnings.warn(
            removal_message(
                "dictionary_domain",
                "10.0",
                replacement="dictionary_file_path_or_domain",
            ),
            stacklevel=2,
        )
        del kwargs["dictionary_domain"]

    # Check the type of non basic keyword arguments specific to this function
    if additional_data_tables and not is_dict_like(additional_data_tables):
        raise TypeError(
            type_error_message(
                "additional_data_tables", additional_data_tables, "dict-like"
            )
        )

    # Get or create the execution dictionary
    execution_dictionary_file_path = _get_or_create_execution_dictionary_file(
        dictionary_file_path_or_domain, trace
    )

    # Resolve the database format parameters
    disable_detect_format, header_line, field_separator = _resolve_format_spec(
        detect_format, header_line, field_separator
    )

    # Create the scenario parameters
    data_tables = {dictionary_name: data_table_path}
    if additional_data_tables:
        data_tables.update(additional_data_tables)

    scenario_params = {
        "__train_database_files__": DatabaseParameter("TrainDatabase", data_tables),
        "__dictionary_file__": PathParameter(execution_dictionary_file_path),
        "__dictionary__": dictionary_name,
        "__header_line__": header_line,
        "__field_separator__": field_separator,
        "__disable_detect_format__": disable_detect_format,
        "__sample_percentage__": str(sample_percentage),
        "__sampling_mode__": sampling_mode,
        "__selection_variable__": selection_variable,
        "__selection_value__": str(selection_value),
        "__max_messages__": str(max_messages),
    }

    scenario = ConfigurableKhiopsScenario(
        get_scenario_file_path("check_database", get_runner().khiops_version),
        scenario_params,
    )

    # Execute Khiops and cleanup when necessary
    try:
        get_runner().run(
            "khiops",
            scenario,
            batch_mode=batch_mode,
            log_file_path=log_file_path,
            output_scenario_path=output_scenario_path,
            task_file_path=task_file_path,
            trace=trace,
            **kwargs,
        )
    finally:
        if isinstance(dictionary_file_path_or_domain, DictionaryDomain) and not trace:
            fs.create_resource(execution_dictionary_file_path).remove()


def train_predictor(
    dictionary_file_path_or_domain,
    dictionary_name,
    data_table_path,
    target_variable,
    results_dir,
    detect_format=True,
    header_line=None,
    field_separator=None,
    sample_percentage=70,
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
        its field separator. It's ignored if ``header_line`` or ``field_separator`` are
        set.
    header_line : bool, optional (default ``True`` if ``detect_format`` is ``False``)
        If ``True`` it uses the first line of the data as column names. Overrides
        ``detect_format`` if set.
    field_separator : str, optional (default "\\t" if ``detect_format`` is ``False``)
        A field separator character, overrides ``detect_format`` if set ("" counts
        as "\\t").
    sample_percentage : int, default 100
        See ``sampling_mode`` option below.
    sampling_mode : "Include sample" or "Exclude sample"
        If equal to "Include sample" it trains the predictor with  ``sample_percentage``
        percent of data and if equal to "Exclude sample" with ``100 -
        sample_percentage`` percent of data.
    use_complement_as_test : bool, default ``True``
        Uses the complement of the sampled database as test database.
    fill_test_database_settings : bool, default ``False``
        It creates a test database as the complement of the train database.
        **Deprecated** will be removed in pyKhiops 11, use ``use_complement_as_test``
    selection_variable : str, default ""
        It trains with only the records such that the value of ``selection_variable`` is
        equal to ``selection_value``. Ignored if equal to "".
    selection_value : str, default ""
        See ``selection_variable`` option above. Ignored if equal to "".
    additional_data_tables : dict, optional
        A dictionary containing the data paths and file paths for a multi-table
        dictionary file. For more details see :doc:`/multi_table_tasks`.
    main_target_value : str, default ""
        If this target value is specified then it guarantees the calculation of lift
        curves for it.
    snb_predictor : bool, default ``True``
        If ``True`` it trains a Selective Naive Bayes predictor.
    univariate_predictor_number : int, default 0
        Number of univariate predictors to train.
    map_predictor : bool, default ``False``
        If ``True`` trains a Maximum a Posteriori Naive Bayes predictor.
        **Deprecated** will be removed in pyKhiops 11.
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
        contains only one non-empty string generated within the maximum limit
        ``max_pairs``.
    all_possible_pairs : bool, default ``True``
        If ``True`` tries to create all possible pairs within the limit ``max_pairs``.
        The pairs and variables given in ``specific_pairs`` have priority.
    only_pairs_with : str, default ""
        Constructs only pairs with the specifed variable name. If equal to the empty
        string "" it considers all variables to make pairs.
        **Deprecated** will be removed in pyKhiops 11, use ``specific_pairs``.
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
        Options of the `.PyKhiopsRunner.run` method from the class `.PyKhiopsRunner`.

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
    # Handle removed parameters kept in legacy mode
    if get_runner().khiops_version < KhiopsVersion("10"):
        if "fill_test_database_settings" in kwargs:
            warnings.warn(
                removal_message(
                    "fill_test_database_settings",
                    "10.0",
                    replacement="use_complement_as_test",
                ),
                stacklevel=2,
            )
            del kwargs["fill_test_database_settings"]
        if "map_predictor" in kwargs:
            warnings.warn(removal_message("map_predictor", "10.0"), stacklevel=2)
            del kwargs["map_predictor"]
        if "only_pairs_with" in kwargs:
            warnings.warn(
                removal_message(
                    "only_pairs_with", "10.0", replacement="specific_pairs"
                ),
                stacklevel=2,
            )
            del kwargs["only_pairs_with"]

    # Handle removed parameters
    if "nb_predictor" in kwargs:
        warnings.warn(removal_message("nb_predictor", "10.0"), stacklevel=2)
        del kwargs["nb_predictor"]

    if "dictionary_domain" in kwargs:
        warnings.warn(
            removal_message(
                "dictionary_domain",
                "10.0",
                replacement="dictionary_file_path_or_domain",
            ),
            stacklevel=2,
        )
        del kwargs["dictionary_domain"]

    # Handle renamed parameters
    if "max_evaluated_variable_number" in kwargs:
        warnings.warn(
            renaming_message(
                "max_evaluated_variable_number", "max_evaluated_variables", "10.0"
            ),
            stacklevel=2,
        )
        del kwargs["max_evaluated_variable_number"]

    if "max_selected_variable_number" in kwargs:
        warnings.warn(
            renaming_message(
                "max_selected_variable_number", "max_selected_variables", "10.0"
            ),
            stacklevel=2,
        )
        del kwargs["max_selected_variable_number"]

    if "constructed_number" in kwargs:
        warnings.warn(
            renaming_message("constructed_number", "max_constructed_variables", "10.0"),
            stacklevel=2,
        )
        del kwargs["constructed_number"]

    if "tree_number" in kwargs:
        warnings.warn(
            renaming_message("tree_number", "max_trees", "10.0"), stacklevel=2
        )
        del kwargs["tree_number"]

    if "pair_number" in kwargs:
        warnings.warn(
            renaming_message("pair_number", "max_pairs", "10.0"), stacklevel=2
        )
        del kwargs["pair_number"]

    if "max_interval_number" in kwargs:
        warnings.warn(
            renaming_message("max_interval_number", "max_intervals", "10.0"),
            stacklevel=2,
        )
        del kwargs["max_interval_number"]

    if "max_group_number" in kwargs:
        warnings.warn(
            renaming_message("max_group_number", "max_groups", "10.0"), stacklevel=2
        )
        del kwargs["max_group_number"]

    # Get or create the execution dictionary
    execution_dictionary_file_path = _get_or_create_execution_dictionary_file(
        dictionary_file_path_or_domain, trace
    )

    # Disambiguate the results directory path if necessary
    results_dir = _create_unambiguous_khiops_path(results_dir)

    # Check the type of non basic keyword arguments specific to this function
    if additional_data_tables and not is_dict_like(additional_data_tables):
        raise TypeError(
            type_error_message(
                "additional_data_tables", additional_data_tables, "dict-like"
            )
        )
    if construction_rules and not is_list_like(construction_rules):
        raise TypeError(
            type_error_message("construction_rules", construction_rules, "list-like")
        )
    if specific_pairs:
        if not is_list_like(specific_pairs):
            raise TypeError(
                type_error_message("specific_pairs", specific_pairs, "list-like")
            )
        for pair_index, pair in enumerate(specific_pairs):
            if not isinstance(pair, tuple):
                raise TypeError(
                    "specific_pairs list elements must be 'tuple'. "
                    f"Found '{type(pair).__name__}' at position {pair_index}"
                )
            if len(pair) != 2:
                raise ValueError("specific_pairs elements must have length 2")

    # Resolve the database format parameters
    disable_detect_format, header_line, field_separator = _resolve_format_spec(
        detect_format, header_line, field_separator
    )

    # Handle discretization/grouping default values
    if discretization_method is None:
        if target_variable:
            discretization_method = "MODL"
        else:
            discretization_method = "EqualWidth"
    if grouping_method is None:
        if target_variable:
            grouping_method = "MODL"
        else:
            grouping_method = "BasicGrouping"

    # Create an empty specific pairs list if not specified
    if specific_pairs is None:
        specific_pairs = []

    # Create the scenario parameters
    data_tables = {dictionary_name: data_table_path}
    if additional_data_tables:
        data_tables.update(additional_data_tables)

    used_rules = {}
    if construction_rules:
        used_rules = {rule: "true" for rule in construction_rules}

    scenario_params = {
        "__train_database_files__": DatabaseParameter(
            name="TrainDatabase", data_tables=data_tables
        ),
        "__dictionary_file__": PathParameter(execution_dictionary_file_path),
        "__results_dir__": PathParameter(results_dir),
        "__construction_rules_spec__": KeyValueListParameter(
            name="ConstructionRules", value_field_name="Used", keyvalues=used_rules
        ),
        "__specific_pairs_spec__": RecordListParameter(
            name="SpecificAttributePairs",
            records_header=("FirstName", "SecondName"),
            records=specific_pairs,
        ),
        "__dictionary__": dictionary_name,
        "__header_line__": header_line,
        "__field_separator__": field_separator,
        "__disable_detect_format__": disable_detect_format,
        "__sample_percentage__": str(sample_percentage),
        "__sampling_mode__": sampling_mode,
        "__selection_variable__": selection_variable,
        "__selection_value__": str(selection_value),
        "__test_database_mode__": "Complementary" if use_complement_as_test else "None",
        "__disable_construction_unselect_all__": "" if construction_rules else "// ",
        "__target_variable__": target_variable,
        "__main_target_value__": main_target_value,
        "__max_constructed_variables__": str(max_constructed_variables),
        "__max_trees__": str(max_trees),
        "__max_pairs__": str(max_pairs),
        "__all_possible_pairs__": str(all_possible_pairs).lower(),
        "__results_prefix__": results_prefix,
        "__snb_predictor__": str(snb_predictor).lower(),
        "__univariate_predictor_number__": str(univariate_predictor_number),
        "__max_evaluated_variables__": str(max_evaluated_variables),
        "__max_selected_variables__": str(max_selected_variables),
        "__group_target_value__": str(group_target_value).lower(),
        "__supervised_discretization_method__": (
            discretization_method if target_variable else "MODL"
        ),
        "__unsupervised_discretization_method__": (
            "EqualWidth" if target_variable else discretization_method
        ),
        "__min_interval_frequency__": str(min_interval_frequency),
        "__max_intervals__": str(max_intervals),
        "__supervised_grouping_method__": (
            grouping_method if target_variable else "MODL"
        ),
        "__unsupervised_grouping_method__": (
            "BasicGrouping" if target_variable else grouping_method
        ),
        "__min_group_frequency__": str(min_group_frequency),
        "__max_groups__": str(max_groups),
    }

    # Set parameters available only in v9 legacy mode
    if get_runner().khiops_version < KhiopsVersion("10"):
        scenario_params["__fill_test_database_settings__"] = "//"
        if "fill_test_database_settings" in kwargs:
            if kwargs["fill_test_database_settings"]:
                scenario_params["__fill_test_database_settings__"] = ""
            del kwargs["fill_test_database_settings"]

        scenario_params["__map_predictor__"] = "false"
        if "map_predictor" in kwargs:
            if kwargs["map_predictor"]:
                scenario_params["__map_predictor__"] = "true"
            del kwargs["map_predictor"]

        scenario_params["__only_pairs_with__"] = ""
        if "only_pairs_with" in kwargs:
            scenario_params["__only_pairs_with__"] = kwargs["only_pairs_with"]
            del kwargs["only_pairs_with"]

    # Create the scenario
    scenario = ConfigurableKhiopsScenario(
        get_scenario_file_path("train_predictor", get_runner().khiops_version),
        scenario_params,
    )

    # Execute Khiops and cleanup when necessary
    try:
        get_runner().run(
            "khiops",
            scenario,
            batch_mode=batch_mode,
            log_file_path=log_file_path,
            output_scenario_path=output_scenario_path,
            task_file_path=task_file_path,
            trace=trace,
            **kwargs,
        )
    finally:
        if isinstance(dictionary_file_path_or_domain, DictionaryDomain) and not trace:
            fs.create_resource(execution_dictionary_file_path).remove()

    # Return the paths of the JSON report and modelling dictionary file
    reports_file_name = results_prefix
    if get_runner().khiops_version < KhiopsVersion("10"):
        reports_file_name += "AllReports.json"
    else:
        reports_file_name += "AllReports.khj"
    reports_file_res = fs.create_resource(results_dir).create_child(reports_file_name)

    if target_variable != "":
        modeling_dictionary_file_res = fs.create_resource(results_dir).create_child(
            f"{results_prefix}Modeling.kdic"
        )
    else:
        modeling_dictionary_file_res = None
    return (
        reports_file_res.uri,
        modeling_dictionary_file_res.uri if modeling_dictionary_file_res else None,
    )


def evaluate_predictor(
    dictionary_file_path_or_domain,
    train_dictionary_name,
    data_table_path,
    results_dir,
    detect_format=True,
    header_line=None,
    field_separator=None,
    sample_percentage=100,
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
        its field separator. It's ignored if ``header_line`` or ``field_separator`` are
        set.
    header_line : bool, optional (default ``True`` if ``detect_format`` is ``False``)
        If ``True`` it uses the first line of the data as column names. Overrides
        ``detect_format`` if set.
    field_separator : str, optional (default "\\t" if ``detect_format`` is ``False``)
        A field separator character, overrides ``detect_format`` if set ("" counts
        as "\\t").
    sample_percentage : int, default 100
        See ``sampling_mode`` option below.
    sampling_mode : "Include sample" or "Exclude sample"
        If equal to "Include sample" it trains the model with  ``sample_percentage``
        percent of data and if equal to "Exclude sample" with ``100 -
        sample_percentage`` percent of data.
    selection_variable : str, default ""
        It trains with only the records such that the value of ``selection_variable`` is
        equal to ``selection_value``. Ignored if equal "".
    selection_value : str, default ""
        See ``selection_variable`` option above. Ignored if equal to "".
    additional_data_tables : dict, optional
        A dictionary containing the data paths and file paths for a multi-table
        dictionary file. For more details see :doc:`/multi_table_tasks`.

        .. note:: Use the initial dictionary name in the data paths.

    main_target_value : str, default ""
        If this target value is specified then it guarantees the calculation of lift
        curves for it.
    results_prefix : str, default ""
        Prefix of the result files.
    ... :
        Options of the `.PyKhiopsRunner.run` method from the class `.PyKhiopsRunner`.

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
    # Handle renamed/removed parameters
    if "dictionary_domain" in kwargs:
        warnings.warn(
            removal_message(
                "dictionary_domain",
                "10.0",
                replacement="dictionary_file_path_or_domain",
            ),
            stacklevel=2,
        )
        del kwargs["dictionary_domain"]

    # Check the type of non basic keyword arguments specific to this function
    if additional_data_tables and not is_dict_like(additional_data_tables):
        raise TypeError(
            type_error_message(
                "additional_data_tables", additional_data_tables, "dict-like"
            )
        )

    # Resolve the database format parameters
    disable_detect_format, header_line, field_separator = _resolve_format_spec(
        detect_format, header_line, field_separator
    )

    # Generate output file path
    evaluation_report_res = fs.create_resource(results_dir).create_child(
        f"{results_prefix}EvaluationReport.xls"
    )

    # Get or create the execution dictionary file
    execution_dictionary_file_path = _get_or_create_execution_dictionary_file(
        dictionary_file_path_or_domain, trace
    )

    # Disambiguate the results directory path if necessary
    results_dir = _create_unambiguous_khiops_path(results_dir)

    # Create the scenario parameters
    data_tables = {train_dictionary_name: data_table_path}
    if additional_data_tables:
        data_tables.update(additional_data_tables)
    scenario_params = {
        "__evaluation_database_files__": DatabaseParameter(
            name="EvaluationDatabase", data_tables=data_tables
        ),
        "__dictionary_file__": PathParameter(execution_dictionary_file_path),
        "__evaluation_file__": PathParameter(evaluation_report_res.uri),
        "__header_line__": header_line,
        "__field_separator__": field_separator,
        "__disable_detect_format__": disable_detect_format,
        "__sample_percentage__": str(sample_percentage),
        "__sampling_mode__": sampling_mode,
        "__selection_variable__": selection_variable,
        "__selection_value__": str(selection_value),
        "__main_target_value__": main_target_value,
    }
    # Create the scenario
    scenario = ConfigurableKhiopsScenario(
        get_scenario_file_path("evaluate_predictor", get_runner().khiops_version),
        scenario_params,
    )

    # Execute Khiops and cleanup when necessary
    try:
        get_runner().run(
            "khiops",
            scenario,
            batch_mode=batch_mode,
            log_file_path=log_file_path,
            output_scenario_path=output_scenario_path,
            task_file_path=task_file_path,
            trace=trace,
            **kwargs,
        )
    finally:
        if isinstance(dictionary_file_path_or_domain, DictionaryDomain) and not trace:
            fs.create_resource(execution_dictionary_file_path).remove()

    # Return the path of the JSON report
    report_file_name = results_prefix
    if get_runner().khiops_version < KhiopsVersion("10"):
        report_file_name += "EvaluationReport.json"
    else:
        report_file_name += "EvaluationReport.khj"
    report_file_res = fs.create_resource(results_dir).create_child(report_file_name)
    return report_file_res.uri


def train_recoder(
    dictionary_file_path_or_domain,
    dictionary_name,
    data_table_path,
    target_variable,
    results_dir,
    detect_format=True,
    header_line=None,
    field_separator=None,
    sample_percentage=70,
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
        its field separator. It's ignored if ``header_line`` or ``field_separator`` are
        set.
    header_line : bool, optional (default ``True`` if ``detect_format`` is ``False``)
        If ``True`` it uses the first line of the data as column names. Overrides
        ``detect_format`` if set.
    field_separator : str, optional (default "\\t" if ``detect_format`` is ``False``)
        A field separator character, overrides ``detect_format`` if set ("" counts
        as "\\t").
    sample_percentage : int, default 100
        See ``sampling_mode`` option below.
    sampling_mode : "Include sample" or "Exclude sample"
        If equal to "Include sample" it trains the recoder with  ``sample_percentage``
        percent of data and if equal to "Exclude sample" with ``100 -
        sample_percentage`` percent of data.
    selection_variable : str, default ""
        It trains with only the records such that the value of ``selection_variable`` is
        equal to ``selection_value``. Ignored if equal to "".
    selection_value : str, default ""
        See ``selection_variable`` option above. Ignored if equal to "".
    additional_data_tables : dict, optional
        A dictionary containing the data paths and file paths for a multi-table
        dictionary file. For more details see :doc:`/multi_table_tasks`.
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
        contains only one non-empty string generated within the maximum limit
        ``max_pairs``.
    all_possible_pairs : bool, default ``True``
        If ``True`` tries to create all possible pairs within the limit ``max_pairs``.
        The pairs and variables given in ``specific_pairs`` have priority.
    only_pairs_with : str, default ""
        Constructs only pairs with the specifed variable name. If equal to the empty
        string "" it considers all variables to make pairs.
        **Deprecated** will be removed in pyKhiops 11, use ``specific_pairs``.
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
        Options of the `.PyKhiopsRunner.run` method from the class `.PyKhiopsRunner`.

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
    # Handle renamed/removed parameters
    if (
        get_runner().khiops_version < KhiopsVersion("10")
        and "only_pairs_with" in kwargs
    ):
        warnings.warn(
            removal_message("only_pairs_with", "10.0", replacement="specific_pairs"),
            stacklevel=2,
        )
        del kwargs["only_pairs_with"]

    if "dictionary_domain" in kwargs:
        warnings.warn(
            removal_message(
                "dictionary_domain",
                "10.0",
                replacement="dictionary_file_path_or_domain",
            ),
            stacklevel=2,
        )
        del kwargs["dictionary_domain"]

    # Renamed parameters warnings
    if "constructed_number" in kwargs:
        warnings.warn(
            renaming_message("constructed_number", "max_constructed_variables", "10.0"),
            stacklevel=2,
        )
        del kwargs["constructed_number"]

    if "tree_number" in kwargs:
        warnings.warn(
            renaming_message("tree_number", "max_trees", "10.0"), stacklevel=2
        )
        del kwargs["tree_number"]

    if "pair_number" in kwargs:
        warnings.warn(
            renaming_message("pair_number", "max_pairs", "10.0"), stacklevel=2
        )
        del kwargs["pair_number"]

    if "max_variable_number" in kwargs:
        warnings.warn(
            renaming_message("max_variable_number", "max_variables", "10.0"),
            stacklevel=2,
        )
        del kwargs["max_variable_number"]

    if "max_interval_number" in kwargs:
        warnings.warn(
            renaming_message("max_interval_number", "max_intervals", "10.0"),
            stacklevel=2,
        )
        del kwargs["max_interval_number"]

    if "max_group_number" in kwargs:
        warnings.warn(
            renaming_message("max_group_number", "max_groups", "10.0"), stacklevel=2
        )
        del kwargs["max_group_number"]

    if "recode_categorical_variables" in kwargs:
        warnings.warn(
            renaming_message(
                "recode_categorical_variables", "categorical_recoding_method", "10.0"
            ),
            stacklevel=2,
        )
        del kwargs["recode_categorical_variables"]

    if "recode_numerical_variables" in kwargs:
        warnings.warn(
            renaming_message(
                "recode_numerical_variables", "numerical_recoding_method", "10.0"
            ),
            stacklevel=2,
        )
        del kwargs["recode_numerical_variables"]

    if "recode_bivariate_variables" in kwargs:
        warnings.warn(
            renaming_message(
                "recode_bivariate_variables", "pairs_recoding_method", "10.0"
            ),
            stacklevel=2,
        )
        del kwargs["recode_bivariate_variables"]

    # Check the type of non basic keyword arguments specific to this function
    if additional_data_tables and not is_dict_like(additional_data_tables):
        raise TypeError(
            type_error_message(
                "additional_data_tables", additional_data_tables, "dict-like"
            )
        )
    if construction_rules and not is_list_like(construction_rules):
        raise TypeError(
            type_error_message("construction_rules", construction_rules, "list-like")
        )
    if specific_pairs:
        if not is_list_like(specific_pairs):
            raise TypeError(
                type_error_message("specific_pairs", specific_pairs, "list-like")
            )
        for pair_index, pair in enumerate(specific_pairs):
            if not isinstance(pair, tuple):
                raise TypeError(
                    "specific_pairs list elements must be 'tuple'. "
                    f"Found '{type(pair).__name__}' at position {pair_index}"
                )
            if len(pair) != 2:
                raise ValueError("specific_pairs elements must have length 2")

    # Resolve the database format parameters
    disable_detect_format, header_line, field_separator = _resolve_format_spec(
        detect_format, header_line, field_separator
    )

    # Handle discretization/grouping default values
    if not discretization_method:
        if target_variable:
            discretization_method = "MODL"
        else:
            discretization_method = "EqualWidth"
    if not grouping_method:
        if target_variable:
            grouping_method = "MODL"
        else:
            grouping_method = "BasicGrouping"

    # Get or create the execution dictionary file
    execution_dictionary_file_path = _get_or_create_execution_dictionary_file(
        dictionary_file_path_or_domain, trace
    )

    # Disambiguate the results directory path if necessary
    results_dir = _create_unambiguous_khiops_path(results_dir)

    # Create the specific pairs file if they are available
    if specific_pairs is None:
        specific_pairs = []

    # Create the scenario parameters
    data_tables = {dictionary_name: data_table_path}
    if additional_data_tables:
        data_tables.update(additional_data_tables)

    used_rules = {}
    if construction_rules:
        used_rules = {rule: "true" for rule in construction_rules}

    scenario_params = {
        "__train_database_files__": DatabaseParameter(
            name="TrainDatabase", data_tables=data_tables
        ),
        "__dictionary_file__": PathParameter(execution_dictionary_file_path),
        "__results_dir__": PathParameter(results_dir),
        "__construction_rules_spec__": KeyValueListParameter(
            name="ConstructionRules", value_field_name="Used", keyvalues=used_rules
        ),
        "__specific_pairs_spec__": RecordListParameter(
            name="SpecificAttributePairs",
            records_header=("FirstName", "SecondName"),
            records=specific_pairs,
        ),
        "__dictionary__": dictionary_name,
        "__header_line__": header_line,
        "__field_separator__": field_separator,
        "__disable_detect_format__": disable_detect_format,
        "__sample_percentage__": str(sample_percentage),
        "__sampling_mode__": sampling_mode,
        "__selection_variable__": selection_variable,
        "__selection_value__": str(selection_value),
        "__variable_construction_unselect_all__": "" if construction_rules else "// ",
        "__target_variable__": target_variable,
        "__max_constructed_variables__": str(max_constructed_variables),
        "__max_trees__": str(max_trees),
        "__max_pairs__": str(max_pairs),
        "__all_possible_pairs__": str(all_possible_pairs).lower(),
        "__results_prefix__": results_prefix,
        "__informative_variables_only__": str(informative_variables_only).lower(),
        "__max_variables__": str(max_variables),
        "__keep_categorical_variables__": str(
            keep_initial_categorical_variables
        ).lower(),
        "__keep_numerical_variables__": str(keep_initial_numerical_variables).lower(),
        "__categorical_recoding_method__": categorical_recoding_method,
        "__numerical_recoding_method__": numerical_recoding_method,
        "__pairs_recoding_method__": pairs_recoding_method,
        "__group_target_value__": str(group_target_value).lower(),
        "__supervised_discretization_method__": (
            discretization_method if target_variable else "MODL"
        ),
        "__unsupervised_discretization_method__": (
            "EqualWidth" if target_variable else discretization_method
        ),
        "__min_interval_frequency__": str(min_interval_frequency),
        "__max_intervals__": str(max_intervals),
        "__supervised_grouping_method__": (
            grouping_method if target_variable else "MODL"
        ),
        "__unsupervised_grouping_method__": (
            "BasicGrouping" if target_variable else grouping_method
        ),
        "__min_group_frequency__": str(min_group_frequency),
        "__max_groups__": str(max_groups),
    }

    # Set scenario parameter available only in legacy mode
    if get_runner().khiops_version < KhiopsVersion("10"):
        scenario_params["__only_pairs_with__"] = ""
        if "only_pairs_with" in kwargs:
            scenario_params["__only_pairs_with__"] = kwargs["only_pairs_with"]
            del kwargs["only_pairs_with"]

    # Create the scenario
    scenario = ConfigurableKhiopsScenario(
        get_scenario_file_path("train_recoder", get_runner().khiops_version),
        scenario_params,
    )

    # Execute Khiops and cleanup when necessary
    try:
        get_runner().run(
            "khiops",
            scenario,
            batch_mode=batch_mode,
            log_file_path=log_file_path,
            output_scenario_path=output_scenario_path,
            task_file_path=task_file_path,
            trace=trace,
            **kwargs,
        )
    finally:
        if isinstance(dictionary_file_path_or_domain, DictionaryDomain) and not trace:
            fs.create_resource(execution_dictionary_file_path).remove()

    # Return the paths of the JSON report and modelling dictionary file
    reports_file_name = results_prefix
    if get_runner().khiops_version < KhiopsVersion("10"):
        reports_file_name += "AllReports.json"
    else:
        reports_file_name += "AllReports.khj"
    reports_file_res = fs.create_resource(results_dir).create_child(reports_file_name)

    if target_variable != "":
        modeling_dictionary_file_res = fs.create_resource(results_dir).create_child(
            f"{results_prefix}Modeling.kdic"
        )
    else:
        modeling_dictionary_file_res = None

    return (
        reports_file_res.uri,
        modeling_dictionary_file_res.uri if modeling_dictionary_file_res else None,
    )


def deploy_model(
    dictionary_file_path_or_domain,
    dictionary_name,
    data_table_path,
    output_data_table_path,
    detect_format=True,
    header_line=None,
    field_separator=None,
    sample_percentage=100,
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
        its field separator. It's ignored if ``header_line`` or ``field_separator`` are
        set.
    header_line : bool, optional (default ``True`` if ``detect_format`` is ``False``)
        If ``True`` it uses the first line of the data as column names. Overrides
        ``detect_format`` if set.
    field_separator : str, optional (default "\\t" if ``detect_format`` is ``False``)
        A field separator character, overrides ``detect_format`` if set ("" counts
        as "\\t").
    sample_percentage : int, default 100
        See ``sampling_mode`` option below.
    sampling_mode : "Include sample" or "Exclude sample"
        If equal to "Include sample" deploys the model with  ``sample_percentage``
        percent of data and if equal to "Exclude sample" with ``100 -
        sample_percentage`` percent of data.
    selection_variable : str, default ""
        It deploys only the records such that the value of ``selection_variable`` is
        equal to ``selection_value``. Ignored if equal to "".
    selection_value : str, default ""
        See ``selection_variable`` option above. Ignored if equal to "".
    additional_data_tables : dict, optional
        A dictionary containing the data paths and file paths for a multi-table
        dictionary file. For more details see :doc:`/multi_table_tasks`.
    output_header_line : bool, default ``True``
        If ``True`` writes a header line with the column names in the output table.
    output_field_separator : str, default "\\t"
        The field separator character for the output table ("" counts as "\\t").
    output_additional_data_tables : dict, optional
        A dictionary containing the output data paths and file paths for a multi-table
        dictionary file. For more details see :doc:`/multi_table_tasks`.
    results_prefix : str, default ""
        Prefix of the result files.
    ... :
        Options of the `.PyKhiopsRunner.run` method from the class `.PyKhiopsRunner`.

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
    # Handle renamed/removed parameters
    if "dictionary_domain" in kwargs:
        warnings.warn(
            removal_message(
                "dictionary_domain",
                "10.0",
                replacement="dictionary_file_path_or_domain",
            ),
            stacklevel=2,
        )
        del kwargs["dictionary_domain"]

    # Check the type of non basic keyword arguments specific to this function
    if additional_data_tables and not is_dict_like(additional_data_tables):
        raise TypeError(
            type_error_message(
                "additional_data_tables", additional_data_tables, "dict-like"
            )
        )
    if output_additional_data_tables and not is_dict_like(
        output_additional_data_tables
    ):
        raise TypeError(
            type_error_message(
                "output_additional_data_tables",
                output_additional_data_tables,
                "dict-like",
            )
        )

    # Resolve the database format parameters
    disable_detect_format, header_line, field_separator = _resolve_format_spec(
        detect_format, header_line, field_separator
    )
    _, output_header_line, output_field_separator = _resolve_format_spec(
        False, output_header_line, output_field_separator
    )

    # Get or create the execution dictionary file
    execution_dictionary_file_path = _get_or_create_execution_dictionary_file(
        dictionary_file_path_or_domain, trace
    )

    # Create the scenario
    data_tables = {dictionary_name: data_table_path}
    if additional_data_tables:
        data_tables.update(additional_data_tables)
    output_data_tables = {dictionary_name: output_data_table_path}
    if output_additional_data_tables:
        output_data_tables.update(output_additional_data_tables)

    scenario_params = {
        "__source_database_files__": DatabaseParameter(
            name="SourceDatabase", data_tables=data_tables
        ),
        "__target_database_files__": DatabaseParameter(
            name="TargetDatabase", data_tables=output_data_tables
        ),
        "__dictionary_file__": PathParameter(execution_dictionary_file_path),
        "__dictionary__": dictionary_name,
        "__header_line__": header_line,
        "__field_separator__": field_separator,
        "__disable_detect_format__": disable_detect_format,
        "__sample_percentage__": str(sample_percentage),
        "__sampling_mode__": sampling_mode,
        "__selection_variable__": selection_variable,
        "__selection_value__": str(selection_value),
        "__output_header_line__": output_header_line,
        "__output_field_separator__": output_field_separator,
    }

    # Create the scenario
    scenario = ConfigurableKhiopsScenario(
        get_scenario_file_path("deploy_model", get_runner().khiops_version),
        params=scenario_params,
    )

    # Execute Khiops and cleanup when necessary
    try:
        get_runner().run(
            "khiops",
            scenario,
            batch_mode=batch_mode,
            log_file_path=log_file_path,
            output_scenario_path=output_scenario_path,
            task_file_path=task_file_path,
            trace=trace,
            **kwargs,
        )
    finally:
        if isinstance(dictionary_file_path_or_domain, DictionaryDomain) and not trace:
            fs.create_resource(execution_dictionary_file_path).remove()


def build_deployed_dictionary(
    dictionary_file_path_or_domain,
    dictionary_name,
    output_dictionary_file_path,
    batch_mode=True,
    log_file_path=None,
    output_scenario_path=None,
    task_file_path=None,
    trace=False,
    **kwargs,
):
    """Builds a dictionary file to read a deployed data table

    Parameters
    ----------
    dictionary_file_path_or_domain : str or `.DictionaryDomain`
        Path of a Khiops dictionary file or a DictionaryDomain object.
    dictionary_name : str
        Name of the dictionary to be analyzed.
    output_dictionary_file_path : str
        Path of the output dictionary file.
    ... :
        Options of the `.PyKhiopsRunner.run` method from the class `.PyKhiopsRunner`.

    Raises
    ------
    `TypeError`
        Invalid type of an argument

    Examples
    --------
    See the following functions of the ``samples.py`` documentation script:
        - `samples.build_deployed_dictionary()`
    """
    # Handle renamed/removed parameters
    if "dictionary_domain" in kwargs:
        warnings.warn(
            removal_message(
                "dictionary_domain",
                "10.0",
                replacement="dictionary_file_path_or_domain",
            ),
            stacklevel=2,
        )
        del kwargs["dictionary_domain"]

    # Get or create the execution dictionary file
    execution_dictionary_file_path = _get_or_create_execution_dictionary_file(
        dictionary_file_path_or_domain, trace
    )

    # Create the scenario parameters
    scenario_params = {
        "__dictionary_file__": PathParameter(execution_dictionary_file_path),
        "__output_dictionary_file__": PathParameter(output_dictionary_file_path),
        "__dictionary__": dictionary_name,
    }

    # Create the scenario
    scenario = ConfigurableKhiopsScenario(
        get_scenario_file_path(
            "build_deployed_dictionary", get_runner().khiops_version
        ),
        scenario_params,
    )

    # Execute Khiops and cleanup when necessary
    try:
        get_runner().run(
            "khiops",
            scenario,
            batch_mode=batch_mode,
            log_file_path=log_file_path,
            output_scenario_path=output_scenario_path,
            task_file_path=task_file_path,
            trace=trace,
            **kwargs,
        )
    finally:
        if isinstance(dictionary_file_path_or_domain, DictionaryDomain) and not trace:
            fs.create_resource(execution_dictionary_file_path).remove()


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
        its field separator. It's ignored if ``header_line`` or ``field_separator`` are
        set.
    header_line : bool, optional (default ``True`` if ``detect_format`` is ``False``)
        If ``True`` it uses the first line of the data as column names. Overrides
        ``detect_format`` if set.
    field_separator : str, optional (default "\\t" if ``detect_format`` is ``False``)
        A field separator character, overrides ``detect_format`` if set ("" counts
        as "\\t").
    output_header_line : bool, default ``True``
        If ``True`` writes a header line with the column names in the output table.
    output_field_separator : str, default "\\t"
        The field separator character for the output table ("" counts as "\\t").
    ... :
        Options of the `.PyKhiopsRunner.run` method from the class `.PyKhiopsRunner`.

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
    # Handle renamed/removed parameters
    if "dictionary_domain" in kwargs:
        warnings.warn(
            removal_message(
                "dictionary_domain",
                "10.0",
                replacement="dictionary_file_path_or_domain",
            ),
            stacklevel=2,
        )
        del kwargs["dictionary_domain"]

    # Check the type of non basic keyword arguments specific to this function
    if sort_variables and not is_list_like(sort_variables):
        raise TypeError(
            type_error_message("sort_variables", sort_variables, "list-like")
        )

    # Resolve the database format parameters
    disable_detect_format, header_line, field_separator = _resolve_format_spec(
        detect_format, header_line, field_separator
    )
    _, output_header_line, output_field_separator = _resolve_format_spec(
        False, output_header_line, output_field_separator
    )

    # Get or create the execution dictionary file
    execution_dictionary_file_path = _get_or_create_execution_dictionary_file(
        dictionary_file_path_or_domain, trace
    )

    # Create the scenario parameters
    scenario_params = {
        "__dictionary_file__": PathParameter(execution_dictionary_file_path),
        "__data_file__": PathParameter(data_table_path),
        "__output_data_file__": PathParameter(output_data_table_path),
        "__sort_variables__": RecordListParameter(
            name="SortAttributes",
            records_header="Name",
            records=sort_variables if sort_variables is not None else [],
        ),
        "__dictionary__": dictionary_name,
        "__disable_sort_by_key_variables__": "// " if sort_variables else "",
        "__header_line__": header_line,
        "__field_separator__": field_separator,
        "__disable_detect_format__": disable_detect_format,
        "__output_header_line__": output_header_line,
        "__output_field_separator__": output_field_separator,
    }

    # Create the scenario
    scenario = ConfigurableKhiopsScenario(
        get_scenario_file_path("sort_data_table", get_runner().khiops_version),
        scenario_params,
    )

    # Execute Khiops and cleanup when necessary
    try:
        get_runner().run(
            "khiops",
            scenario,
            batch_mode=batch_mode,
            log_file_path=log_file_path,
            output_scenario_path=output_scenario_path,
            task_file_path=task_file_path,
            trace=trace,
            **kwargs,
        )
    finally:
        if isinstance(dictionary_file_path_or_domain, DictionaryDomain) and not trace:
            fs.create_resource(execution_dictionary_file_path).remove()


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
        its field separator. Ignored if ``header_line`` or ``field_separator`` are set.
    header_line : bool, optional (default ``True`` if ``detect_format`` is ``False``)
        If ``True`` it uses the first line of the data as column names. Overrides
        ``detect_format`` if set.
    field_separator : str, optional (default "\\t" if ``detect_format`` is ``False``)
        A field separator character, overrides ``detect_format`` if set ("" counts
        as "\\t").
    output_header_line : bool, default ``True``
        If ``True`` writes a header line with the column names in the output table.
    output_field_separator : str, default "\\t"
        The field separator character for the output table ("" counts as "\\t").
    ... :
        Options of the `.PyKhiopsRunner.run` method from the class `.PyKhiopsRunner`.

    Raises
    ------
    `TypeError`
        Invalid type of an argument.

    Examples
    --------
    See the following function of the ``samples.py`` documentation script:
        - `samples.extract_keys_from_data_table()`
    """
    # Handle renamed/removed parameters
    if "dictionary_domain" in kwargs:
        warnings.warn(
            removal_message(
                "dictionary_domain",
                "10.0",
                replacement="dictionary_file_path_or_domain",
            ),
            stacklevel=2,
        )
        del kwargs["dictionary_domain"]

    # Resolve the database format parameters
    disable_detect_format, header_line, field_separator = _resolve_format_spec(
        detect_format, header_line, field_separator
    )
    _, output_header_line, output_field_separator = _resolve_format_spec(
        False, output_header_line, output_field_separator
    )

    # Get or create the execution dictionary file
    execution_dictionary_file_path = _get_or_create_execution_dictionary_file(
        dictionary_file_path_or_domain, trace
    )

    # Create the scenario parameters
    scenario_params = {
        "__dictionary_file__": PathParameter(execution_dictionary_file_path),
        "__data_file__": PathParameter(data_table_path),
        "__output_data_file__": PathParameter(output_data_table_path),
        "__dictionary__": dictionary_name,
        "__header_line__": header_line,
        "__field_separator__": field_separator,
        "__disable_detect_format__": disable_detect_format,
        "__output_header_line__": output_header_line,
        "__output_field_separator__": output_field_separator,
    }

    # Create the scenario
    scenario = ConfigurableKhiopsScenario(
        get_scenario_file_path(
            "extract_keys_from_data_table", get_runner().khiops_version
        ),
        scenario_params,
    )

    # Execute Khiops and cleanup when necessary
    try:
        get_runner().run(
            "khiops",
            scenario,
            batch_mode=batch_mode,
            log_file_path=log_file_path,
            output_scenario_path=output_scenario_path,
            task_file_path=task_file_path,
            trace=trace,
            **kwargs,
        )
    finally:
        if isinstance(dictionary_file_path_or_domain, DictionaryDomain) and not trace:
            fs.create_resource(execution_dictionary_file_path).remove()


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

    Parameters
    ----------
    dictionary_file_path_or_domain : str or `.DictionaryDomain`
        Path of a Khiops dictionary file or a DictionaryDomain object.
    root_dictionary_name : str
        Name for the new root dictionary
    secondary_table_variable_name : str
        Name, in the root dictionary, for the "table" variable of the secondary table.
    output_dictionary_file_path : str
        Path of the output dictionary path.
    overwrite_dictionary_file : bool, default ``False``
        If ``True`` it will overwrite input dictionary if it is a file.
    ... :
        Options of the `.PyKhiopsRunner.run` method from the class `.PyKhiopsRunner`.

    Raises
    ------
    `ValueError`
        Invalid values of an argument
    """
    _check_dictionary_file_path_or_domain(dictionary_file_path_or_domain)

    # Create the execution dictionary file if it is domain or a file with no overwrite
    domain = None
    if isinstance(dictionary_file_path_or_domain, DictionaryDomain):
        domain = dictionary_file_path_or_domain
    elif overwrite_dictionary_file:
        execution_dictionary_file_path = dictionary_file_path_or_domain
    else:
        domain = read_dictionary_file(dictionary_file_path_or_domain)

    if domain is not None:
        execution_dictionary_file_path = get_runner().create_temp_file(
            "_build_multi_table_dictionary_", ".kdic"
        )
        domain.export_khiops_dictionary_file(execution_dictionary_file_path)

    # Create the scenario parameters
    scenario_params = {
        "__dictionary_file__": PathParameter(execution_dictionary_file_path),
        "__output_dictionary_file_path__": PathParameter(output_dictionary_file_path),
        "__root_dictionary_name__": root_dictionary_name,
        "__secondary_table_variable_name__": secondary_table_variable_name,
    }

    # Create the scenario
    scenario = ConfigurableKhiopsScenario(
        get_scenario_file_path(
            "build_multi_table_dictionary", get_runner().khiops_version
        ),
        scenario_params,
    )

    # Execute Khiops and cleanup when necessary
    try:
        get_runner().run(
            "khiops",
            scenario,
            batch_mode=batch_mode,
            log_file_path=log_file_path,
            output_scenario_path=output_scenario_path,
            task_file_path=task_file_path,
            trace=trace,
        )
    finally:
        if not trace and not overwrite_dictionary_file:
            fs.create_resource(execution_dictionary_file_path).remove()


def train_coclustering(
    dictionary_file_path_or_domain,
    dictionary_name,
    data_table_path,
    coclustering_variables,
    results_dir,
    detect_format=True,
    header_line=None,
    field_separator=None,
    sample_percentage=100,
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
        its field separator. It's ignored if ``header_line`` or ``field_separator`` are
        set.
    header_line : bool, optional (default ``True`` if ``detect_format`` is ``False``)
        If ``True`` it uses the first line of the data as column names. Overrides
        ``detect_format`` if set.
    field_separator : str, optional (default "\\t" if ``detect_format`` is ``False``)
        A field separator character, overrides ``detect_format`` if set ("" counts
        as "\\t").
    sample_percentage : int, default 100
        See ``sampling_mode`` option below.
    sampling_mode : "Include sample" or "Exclude sample"
        If equal to "Include sample" it trains the coclustering with
        ``sample_percentage`` percent of data and if equal to "Exclude sample" with
        ``100 - sample_percentage`` percent of data.
    selection_variable : str, default ""
        It trains with only the records such that the value of ``selection_variable`` is
        equal to ``selection_value``. Ignored if equal to "".
    selection_value : str, default ""
        See ``selection_variable`` option above. Ignored if equal to "".
    additional_data_tables : dict, optional
        A dictionary containing the data paths and file paths for a multi-table
        dictionary file. For more details see :doc:`/multi_table_tasks`.
    frequency_variable : str, default ""
        Name of frequency variable.
    min_optimization_time : int, default 0
        Minimum optimization time in seconds.
    results_prefix : str, default ""
        Prefix of the result files.
    ... :
        Options of the `.PyKhiopsRunner.run` method from the class `.PyKhiopsRunner`.

    Returns
    -------
    str
        The path of the of the resulting coclustering file.

    Raises
    ------
    `ValueError`
        Number of coclustering variables out of the range 2-10
    `TypeError`
        Invalid type of an argument

    Examples
    --------
    See the following function of the ``samples.py`` documentation script:
        - `samples.train_coclustering()`
    """
    # Handle renamed/removed parameters
    if "dictionary_domain" in kwargs:
        warnings.warn(
            removal_message(
                "dictionary_domain",
                "10.0",
                replacement="dictionary_file_path_or_domain",
            ),
            stacklevel=2,
        )
        del kwargs["dictionary_domain"]

    # Check the type of non basic keyword arguments specific to this function
    if coclustering_variables and not is_list_like(coclustering_variables):
        raise TypeError(
            type_error_message(
                "coclustering_variables", coclustering_variables, "list-like"
            )
        )
    if additional_data_tables and not is_dict_like(additional_data_tables):
        raise TypeError(
            type_error_message(
                "additional_data_tables", additional_data_tables, "dict-like"
            )
        )

    # Raise an error if the number of coclustering variables is out of range
    if len(coclustering_variables) < 2:
        raise ValueError("coclustering_variables must have at least 2 elements")
    elif len(coclustering_variables) > 10:
        raise ValueError("coclustering_variables must have at most 10 elements")

    # Resolve the database format parameters
    disable_detect_format, header_line, field_separator = _resolve_format_spec(
        detect_format, header_line, field_separator
    )

    # Get or create the execution dictionary file
    execution_dictionary_file_path = _get_or_create_execution_dictionary_file(
        dictionary_file_path_or_domain, trace
    )

    # Disambiguate the results directory path if necessary
    results_dir = _create_unambiguous_khiops_path(results_dir)

    # Create a dictionary for table key-value section (main table plus additional ones)

    # Create the scenario parameters
    data_tables = {dictionary_name: data_table_path}
    if additional_data_tables:
        data_tables.update(additional_data_tables)

    scenario_params = {
        "__database_files__": DatabaseParameter(
            name="Database", data_tables=data_tables
        ),
        "__dictionary_file__": PathParameter(execution_dictionary_file_path),
        "__results_dir__": PathParameter(results_dir),
        "__coclustering_variables__": RecordListParameter(
            name="AnalysisSpec.CoclusteringParameters.Attributes",
            records_header="Name",
            records=coclustering_variables,
        ),
        "__dictionary__": dictionary_name,
        "__header_line__": header_line,
        "__field_separator__": field_separator,
        "__disable_detect_format__": disable_detect_format,
        "__sample_percentage__": str(sample_percentage),
        "__sampling_mode__": sampling_mode,
        "__selection_variable__": selection_variable,
        "__selection_value__": str(selection_value),
        "__frequency_variable__": frequency_variable,
        "__min_optimization_time__": str(min_optimization_time),
        "__results_prefix__": results_prefix,
    }

    # Create the scenario
    scenario = ConfigurableKhiopsScenario(
        get_scenario_file_path(
            "train_coclustering", get_runner().khiops_version, coclustering=True
        ),
        params=scenario_params,
    )

    # Execute Khiops and cleanup when necessary
    try:
        get_runner().run(
            "khiops_coclustering",
            scenario,
            batch_mode=batch_mode,
            log_file_path=log_file_path,
            output_scenario_path=output_scenario_path,
            task_file_path=task_file_path,
            trace=trace,
            **kwargs,
        )
    finally:
        if isinstance(dictionary_file_path_or_domain, DictionaryDomain) and not trace:
            fs.create_resource(execution_dictionary_file_path).remove()

    # Return the path of the coclustering file
    coclustering_file_name = results_prefix + "Coclustering.khcj"
    coclustering_file_res = fs.create_resource(results_dir).create_child(
        coclustering_file_name
    )
    return coclustering_file_res.uri


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
        Options of the `.PyKhiopsRunner.run` method from the class `.PyKhiopsRunner`.

    Raises
    ------
    `TypeError`
        Invalid type of an argument.

    Examples
    --------
    See the following function of the ``samples.py`` documentation script:
        - `samples.simplify_coclustering()`
    """
    # Renamed parameters warnings
    if "max_cell_number" in kwargs:
        warnings.warn(
            renaming_message("max_cell_number", "max_cells", "10.0"), stacklevel=2
        )
        del kwargs["max_cell_number"]

    # Check the type of non basic keyword arguments specific to this function
    if max_part_numbers and not is_dict_like(max_part_numbers):
        raise TypeError(
            type_error_message("max_part_numbers", max_part_numbers, "dict-like")
        )

    # Disambiguate the results directory path if necessary
    results_dir = _create_unambiguous_khiops_path(results_dir)

    # Create the scenario parameters
    max_part_numbers_string = {}
    if max_part_numbers is not None:
        for variable_name, max_parts in max_part_numbers.items():
            max_part_numbers_string[variable_name] = str(max_parts)

    scenario_params = {
        "__coclustering_file__": PathParameter(coclustering_file_path),
        "__simplified_coclustering_file__": PathParameter(
            simplified_coclustering_file_path
        ),
        "__results_dir__": PathParameter(results_dir),
        "__max_part_number_spec__": KeyValueListParameter(
            name="PostProcessingSpec.PostProcessedAttributes",
            value_field_name="MaxPartNumber",
            keyvalues=max_part_numbers_string,
        ),
        "__max_preserved_information__": str(max_preserved_information),
        "__max_cells__": str(max_cells),
        "__max_total_parts__": str(max_total_parts),
        "__results_prefix__": results_prefix,
    }

    # Create the scenario
    scenario = ConfigurableKhiopsScenario(
        get_scenario_file_path(
            "simplify_coclustering", get_runner().khiops_version, coclustering=True
        ),
        params=scenario_params,
    )

    # Execute Khiops Coclustering
    get_runner().run(
        "khiops_coclustering",
        scenario,
        batch_mode=batch_mode,
        log_file_path=log_file_path,
        output_scenario_path=output_scenario_path,
        task_file_path=task_file_path,
        trace=trace,
        **kwargs,
    )


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
    **kwargs,
):
    """Prepares a *individual-variable* coclustering deployment

    Parameters
    ----------
    dictionary_file_path_or_domain : str or `.DictionaryDomain`
        Path of a Khiops dictionary file or a DictionaryDomain object.
    dictionary_name : str
        Name of the dictionary to be analyzed.
    data_table_path : str
        Path of the data table file.
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
        Options of the `.PyKhiopsRunner.run` method from the class `.PyKhiopsRunner`.

    Raises
    ------
    `TypeError`
        Invalid type of an argument

    Examples
    --------
    See the following function of the ``samples.py`` documentation script:
        - `samples.deploy_model_expert()`
    """
    # Handle renamed/removed parameters
    if "max_cell_number" in kwargs:
        warnings.warn(
            renaming_message("max_cell_number", "max_cells", "10.0"), stacklevel=2
        )
        del kwargs["max_cell_number"]

    if "dictionary_domain" in kwargs:
        warnings.warn(
            removal_message(
                "dictionary_domain",
                "10.0",
                replacement="dictionary_file_path_or_domain",
            ),
            stacklevel=2,
        )
        del kwargs["dictionary_domain"]

    # Check the type of non basic keyword arguments specific to this function
    if max_part_numbers and not is_dict_like(max_part_numbers):
        raise TypeError(
            type_error_message("max_part_numbers", max_part_numbers, "dict-like")
        )

    # Get or create the execution dictionary file
    execution_dictionary_file_path = _get_or_create_execution_dictionary_file(
        dictionary_file_path_or_domain, trace
    )

    # Disambiguate the results directory path if necessary
    results_dir = _create_unambiguous_khiops_path(results_dir)

    # Create the scenario parameters
    max_part_numbers_string = {}
    if max_part_numbers is not None:
        for variable_name, max_parts in max_part_numbers.items():
            max_part_numbers_string[variable_name] = str(max_parts)

    scenario_params = {
        "__dictionary_file__": PathParameter(execution_dictionary_file_path),
        "__coclustering_file__": PathParameter(coclustering_file_path),
        "__results_dir__": PathParameter(results_dir),
        "__max_part_number_spec__": KeyValueListParameter(
            name="PostProcessingSpec.PostProcessedAttributes",
            value_field_name="MaxPartNumber",
            keyvalues=max_part_numbers_string,
        ),
        "__dictionary__": dictionary_name,
        "__max_preserved_information__": str(max_preserved_information),
        "__max_cells__": str(max_cells),
        "__table_variable__": table_variable,
        "__deployed_variable__": deployed_variable_name,
        "__build_cluster_variable__": str(build_cluster_variable).lower(),
        "__build_distance_variables__": str(build_distance_variables).lower(),
        "__build_frequency_variables__": str(build_frequency_variables).lower(),
        "__variables_prefix__": variables_prefix,
        "__results_prefix__": results_prefix,
    }

    # Create the scenario
    scenario = ConfigurableKhiopsScenario(
        get_scenario_file_path(
            "prepare_coclustering_deployment",
            get_runner().khiops_version,
            coclustering=True,
        ),
        scenario_params,
    )

    # Execute Khiops and cleanup when necessary
    try:
        get_runner().run(
            "khiops_coclustering",
            scenario,
            batch_mode=batch_mode,
            log_file_path=log_file_path,
            output_scenario_path=output_scenario_path,
            task_file_path=task_file_path,
            trace=trace,
            **kwargs,
        )
    finally:
        if isinstance(dictionary_file_path_or_domain, DictionaryDomain) and not trace:
            fs.create_resource(execution_dictionary_file_path).remove()


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
        Options of the `.PyKhiopsRunner.run` method from the class `.PyKhiopsRunner`.

    Examples
    --------
    See the following function of the ``samples.py`` documentation script:
        - `samples.extract_clusters()`
    """
    # Obtain the directory and the file name of the clusters file path
    execution_clusters_file_path = _create_unambiguous_khiops_path(clusters_file_path)
    clusters_file_name = os.path.basename(execution_clusters_file_path)
    clusters_file_res = fs.create_resource(execution_clusters_file_path)
    clusters_file_dir_res = clusters_file_res.create_parent()
    execution_cluster_file_dir = _create_unambiguous_khiops_path(
        clusters_file_dir_res.uri
    )

    # Create the scenario parameters
    scenario_params = {
        "__coclustering_file__": PathParameter(coclustering_file_path),
        "__clusters_file_name__": PathParameter(clusters_file_name),
        "__results_dir__": PathParameter(execution_cluster_file_dir),
        "__cluster_variable__": cluster_variable,
        "__max_preserved_information__": str(max_preserved_information),
        "__max_cells__": str(max_cells),
    }

    # Create the scenario
    scenario = ConfigurableKhiopsScenario(
        get_scenario_file_path(
            "extract_clusters", get_runner().khiops_version, coclustering=True
        ),
        scenario_params,
    )

    # Execute Khiops
    get_runner().run(
        "khiops_coclustering",
        scenario,
        batch_mode=batch_mode,
        log_file_path=log_file_path,
        output_scenario_path=output_scenario_path,
        task_file_path=task_file_path,
        trace=trace,
        **kwargs,
    )


def detect_data_table_format(
    data_table_path,
    dictionary_file_path_or_domain=None,
    dictionary_name=None,
    trace=False,
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

        These are exactly the parameters expected in many pyKhiops API functions.

    Examples
    --------
    See the following function of the ``samples.py`` documentation script:
        - `samples.detect_data_table_format()`
    """
    # Raise an exception if the Khiops backend is older than 10.0.1
    if get_runner().khiops_version < KhiopsVersion("10.0.1"):
        raise PyKhiopsEnvironmentError(
            "detect_format is not available for Khiops version "
            f"{get_runner().khiops_version}. Upgrade to Khiops 10.0.1 or newer."
        )

    # Without dictionary
    if dictionary_file_path_or_domain is None:
        # Create the scenario parameters
        scenario_params = {"__data_file__": PathParameter(data_table_path)}

        # Create the scenario
        scenario = ConfigurableKhiopsScenario(
            get_scenario_file_path(
                "detect_data_table_format", get_runner().khiops_version
            ),
            scenario_params,
        )
    # With dictionary
    else:
        if dictionary_name is None:
            raise ValueError(
                "dictionary_name must be specified with "
                "dictionary_file_path_or_domain"
            )

        # Get or create the execution dictionary file
        execution_dictionary_file_path = _get_or_create_execution_dictionary_file(
            dictionary_file_path_or_domain, trace
        )

        # Create the scenario parameters
        scenario_params = {
            "__data_file__": PathParameter(data_table_path),
            "__dictionary_file__": PathParameter(execution_dictionary_file_path),
            "__dictionary__": dictionary_name,
        }

        # Create the scenario
        scenario = ConfigurableKhiopsScenario(
            get_scenario_file_path(
                "detect_data_table_format_with_dictionary", get_runner().khiops_version
            ),
            scenario_params,
        )

    # Create log file to save the detect format output
    log_file_path = get_runner().create_temp_file("_detect_data_table_format_", ".log")

    # Execute Khiops
    get_runner().run(
        "khiops",
        scenario,
        batch_mode=True,
        log_file_path=log_file_path,
        trace=trace,
    )

    # Parse the log file to obtain the header_line and field_separator parameters
    # Note: If there is an error the run method will raise an exception so at this
    #       stage we have a warning in the worst case
    log_file_res = fs.create_resource(log_file_path)
    log_file_contents = io.BytesIO(log_file_res.read())
    with io.TextIOWrapper(log_file_contents, encoding="ascii") as log_file:
        log_file_lines = log_file.readlines()

    # Obtain the first line that contains the format spec
    header_line = None
    field_separator = None
    format_line_pattern = "File format detected: "
    for line in log_file_lines:
        if line.startswith(format_line_pattern):
            raw_format_spec = line.rstrip().replace("File format detected: ", "")
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
        raise PyKhiopsRuntimeError(
            "Khiops did not write the log line with the data table file format."
        )

    # Clean up the log file if necessary
    if trace:
        print(f"detect_format log file: {log_file_path}")
    else:
        fs.create_resource(log_file_path).remove()

    return header_line, field_separator


########################
# Deprecated functions #
########################


def get_khiops_info():
    """Returns the Khiops license information

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
    if get_runner().khiops_version >= KhiopsVersion("10.1"):
        return get_khiops_version(), None, None, None
    elif get_runner().khiops_version >= KhiopsVersion("10.0"):
        return _get_tool_info_khiops10(get_runner(), "khiops")
    else:
        return _get_tool_info_khiops9(get_runner(), "khiops")


def get_khiops_coclustering_info():
    """Returns the Khiops Coclustering license information

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
    if get_runner().khiops_version >= KhiopsVersion("10.1"):
        return get_khiops_version(), None, None, None
    elif get_runner().khiops_version >= KhiopsVersion("10.0"):
        return _get_tool_info_khiops10(get_runner(), "khiops_coclustering")
    else:
        return _get_tool_info_khiops9(get_runner(), "khiops_coclustering")
