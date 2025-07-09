######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Helper functions for specific and/or advanced treatments"""
import os
import platform
import subprocess

import khiops.core.internals.filesystems as fs
from khiops.core import api
from khiops.core.coclustering_results import read_coclustering_results_file
from khiops.core.dictionary import (
    Dictionary,
    DictionaryDomain,
    Variable,
    read_dictionary_file,
)
from khiops.core.internals.common import is_list_like, type_error_message
from khiops.core.internals.runner import get_runner


def _build_multi_table_dictionary_domain(
    dictionary_domain,
    root_dictionary_name,
    secondary_table_variable_name,
    update_secondary_table_name=False,
):
    """Builds a multi-table dictionary domain from a dictionary with a key
    Parameters
    ----------
    dictionary_domain : `.DictionaryDomain`
        DictionaryDomain object. Its root dictionary must have its key set.
    root_dictionary_name : str
        Name for the new root dictionary
    secondary_table_variable_name : str
        Name, in the root dictionary, for the "table" variable of the secondary table.
    update_secondary_table_name : bool, default `False`
        If ``True``, then update the secondary table name according to the
        secondary table variable name. If not set, keep original table name.

    Returns
    -------
    `.DictionaryDomain`
        The new dictionary domain

    Raises
    ------
    `TypeError`
        Invalid type of an argument
    `ValueError`
        Invalid values of an argument:
        - the dictionary domain doesn't contain at least a dictionary
        - the dictionary domain's root dictionary doesn't have a key set
    """

    # This is a special-purpose function whose goal is to assist in preparing the
    # coclustering deployment.
    # This function builds a new root dictionary and adds it to an existing dictionary
    # domain.
    # The new root dictionary only contains one field, which references a preexisting
    # dictionary from the input dictionary domain as a new (secondary) Table variable.
    # The preexisting dictionary must have a key set on it, as this is the join key
    # with the new root table.

    # Check that `dictionary_domain` is a `DictionaryDomain`
    if not isinstance(dictionary_domain, DictionaryDomain):
        raise TypeError(
            type_error_message("dictionary_domain", dictionary_domain, DictionaryDomain)
        )

    # Check input types
    if not isinstance(root_dictionary_name, str):
        raise TypeError(
            type_error_message("root_dictionary_name", root_dictionary_name, str)
        )
    if not isinstance(secondary_table_variable_name, str):
        raise TypeError(
            type_error_message(
                "secondary_table_variable_name", secondary_table_variable_name, str
            )
        )

    # Check that dictionary_domain has one dictionary (i.e. one table)
    if len(dictionary_domain.dictionaries) < 1:
        raise ValueError("'dictionary_domain' must contain at least one dictionary")

    # Get root source dictionary
    root_source_dictionary = dictionary_domain.dictionaries[0]

    # Check that root_source_dictionary has its key set:
    if not root_source_dictionary.key:
        raise ValueError("'root_source_dictionary' must have its key set")

    # Build and initialize root target dictionary from the input dictionary domain
    root_target_dictionary = Dictionary()
    root_target_dictionary.name = root_dictionary_name
    root_target_dictionary.key = root_source_dictionary.key
    root_target_dictionary.root = True

    # Copy the key variables from source dictionary to target dictionary
    for source_variable in root_source_dictionary.variables:
        if source_variable.name in root_source_dictionary.key:
            root_target_dictionary.add_variable(source_variable)

    # Build target variable for the target root dictionary
    target_variable = Variable()
    target_variable.name = secondary_table_variable_name
    target_variable.type = "Table"
    if update_secondary_table_name:
        target_variable.object_type = secondary_table_variable_name
    else:
        target_variable.object_type = root_source_dictionary.name
    root_target_dictionary.add_variable(target_variable)

    # Build secondary target dictionary, by copying root source dictionary
    secondary_target_dictionary = root_source_dictionary.copy()
    secondary_target_dictionary.root = False
    if update_secondary_table_name:
        secondary_target_dictionary.name = secondary_table_variable_name

    # Build target domain and add dictionaries to it
    target_domain = DictionaryDomain()
    target_domain.add_dictionary(root_target_dictionary)
    target_domain.add_dictionary(secondary_target_dictionary)

    return target_domain


# Disable the protected access rule because we need to call on the private API
# method for checking that a variable is a str or a DictionaryDomain
# pylint: disable=protected-access


def deploy_coclustering(
    dictionary_file_path_or_domain,
    dictionary_name,
    data_table_path,
    coclustering_file_path,
    key_variable_names,
    deployed_variable_name,
    coclustering_dictionary_file_path,
    output_data_table_path,
    detect_format=True,
    header_line=None,
    field_separator=None,
    output_header_line=True,
    output_field_separator="\t",
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
):
    r"""Deploys a coclustering on a data table

    This procedure generates the following files:
        - ``coclustering_dictionary_file_path``: A multi-table dictionary file for
          further deployments of the coclustering with deploy_model
        - ``output_data_table_path``: A data table file containing the deployed
          coclustering model

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
        .. note::

            Instance-variable coclustering is not currently supported.

    key_variable_names : list of str
        Names of the variables forming the unique keys of the individuals.
    deployed_variable_name : str
        Name of the coclustering variable to deploy.
    coclustering_dictionary_file_path : str
        Path of the coclustering dictionary file to deploy.
    output_data_table_path : str
        Path of the output data file.
    detect_format : bool, default ``True``
        If True detects automatically whether the data table file has a header and its
        field separator. It's ignored if ``header_line`` or ``field_separator`` are set.
    header_line : bool, optional (default ``True`` if ``detect_format`` is False)
        If True it uses the first line of the data as column names. Overrides
        ``detect_format`` if set.
    field_separator : str, optional (default "\\t" if ``detect_format`` is False)
        A field separator character, overrides ``detect_format`` if set ("" counts
        as "\\t").
    output_header_line : bool, default ``True``
        If True writes a header line containing the column names in the output table.
    output_field_separator : str, default "\\t"
        A field separator character (empty string counts as tab).
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
      Dictionary associating variable names to their maximum number of parts to
      preserve in the simplified coclustering. For variables not present in
      ``max_part_numbers`` there is no limit.
    build_cluster_variable : bool, default ``True``
        If True includes a cluster id variable in the deployment.
    build_distance_variables : bool, default False
        If True includes a cluster distance variable in the deployment.
    build_frequency_variables : bool, default False
        If True includes the frequency variables in the deployment.
    variables_prefix : str, default ""
        Prefix for the variables in the deployment dictionary.
    ... :
        Options of the `.KhiopsRunner.run` method from the class `.KhiopsRunner`.

    Returns
    -------
    tuple
        A 2-tuple containing:

        - The deployed data table path
        - The deployment dictionary file path.

    Raises
    ------
    `TypeError`
        Invalid type ``dictionary_file_path_or_domain`` or ``key_variable_names``
    `ValueError`
        If the type of the dictionary key variables is not equal to ``Categorical``
    `NotImplementedError`
        If the coclustering to be deployed is of the instance-variable type

    Examples
    --------
    See the following function of the ``samples.py`` documentation script:
        - `samples.deploy_coclustering()`
    """
    # Fail early for instance-variable coclustering, which is not supported
    if any(
        dimension.is_variable_part
        for dimension in read_coclustering_results_file(
            coclustering_file_path
        ).coclustering_report.dimensions
    ):
        raise NotImplementedError(
            "Instance-variable coclustering deployment is not yet implemented."
        )

    # Obtain the dictionary of the table where the coclustering variables are
    api._check_dictionary_file_path_or_domain(dictionary_file_path_or_domain)
    if isinstance(dictionary_file_path_or_domain, DictionaryDomain):
        domain = dictionary_file_path_or_domain
    else:
        domain = read_dictionary_file(dictionary_file_path_or_domain)

    # Check the type of non basic keyword arguments specific to this function
    if not is_list_like(key_variable_names):
        raise TypeError(
            type_error_message("key_variable_names", key_variable_names, "ListLike")
        )

    # Detect the format once and for all to avoid inconsistencies
    if detect_format and header_line is None and field_separator is None:
        header_line, field_separator = api.detect_data_table_format(
            data_table_path, dictionary_file_path_or_domain, dictionary_name
        )
    else:
        if header_line is None:
            header_line = True
        if field_separator is None:
            field_separator = "\t"

    # Access the dictionary in the relevant variables
    dictionary = domain.get_dictionary(dictionary_name)

    # Verify that the key variables are categorical
    for key_variable_name in key_variable_names:
        key_variable = dictionary.get_variable(key_variable_name)
        if key_variable.type != "Categorical":
            raise ValueError(
                "key variable types must be 'Categorical', "
                f"variable '{key_variable_name}' has type '{key_variable.type}'"
            )
    # Make a copy of the dictionary and set the id_variable as key
    tmp_dictionary = dictionary.copy()
    tmp_dictionary.key = key_variable_names
    tmp_domain = DictionaryDomain()
    tmp_domain.add_dictionary(tmp_dictionary)

    # Create a root dictionary containing the keys
    root_dictionary_name = "CC_" + dictionary_name
    table_variable_name = "Table_" + dictionary_name
    domain = _build_multi_table_dictionary_domain(
        tmp_domain, root_dictionary_name, table_variable_name
    )

    # Create the deployment dictionary
    api.prepare_coclustering_deployment(
        domain,
        root_dictionary_name,
        coclustering_file_path,
        table_variable_name,
        deployed_variable_name,
        coclustering_dictionary_file_path,
        max_preserved_information=max_preserved_information,
        max_cells=max_cells,
        max_total_parts=max_total_parts,
        max_part_numbers=max_part_numbers,
        build_cluster_variable=build_cluster_variable,
        build_distance_variables=build_distance_variables,
        build_frequency_variables=build_frequency_variables,
        variables_prefix=variables_prefix,
        log_file_path=log_file_path,
        output_scenario_path=output_scenario_path,
        task_file_path=task_file_path,
        trace=trace,
    )

    # Extract the keys from the tables to a temporary file
    data_table_file_name = os.path.basename(data_table_path)
    keys_table_file_path = get_runner().create_temp_file(
        prefix="Keys", suffix=data_table_file_name
    )
    api.extract_keys_from_data_table(
        domain,
        dictionary_name,
        data_table_path,
        keys_table_file_path,
        header_line=header_line,
        field_separator=field_separator,
        output_header_line=header_line,
        output_field_separator=field_separator,
        trace=trace,
    )

    additional_data_tables = {table_variable_name: data_table_path}
    api.deploy_model(
        coclustering_dictionary_file_path,
        root_dictionary_name,
        keys_table_file_path,
        output_data_table_path,
        header_line=header_line,
        field_separator=field_separator,
        output_header_line=output_header_line,
        output_field_separator=output_field_separator,
        additional_data_tables=additional_data_tables,
        trace=trace,
    )

    # Delete auxiliary file, no longer useful
    if fs.exists(keys_table_file_path):
        fs.remove(keys_table_file_path)

    return output_data_table_path, coclustering_dictionary_file_path


def deploy_predictor_for_metrics(
    dictionary_file_path_or_domain,
    dictionary_name,
    data_table_path,
    output_data_table_path,
    detect_format=True,
    header_line=None,
    field_separator=None,
    sample_percentage=70,
    sampling_mode="Include sample",
    additional_data_tables=None,
    output_header_line=True,
    output_field_separator="\t",
    trace=False,
):
    r"""Deploys the necessary data to estimate the performance metrics of a predictor

    For each instance for each instance it deploys:

    - The true value of the target variable
    - The predicted value of the target variable
    - The probabilities of each value of the target variable *(classifier only)*

    .. note::
        To obtain the data of the default Khiops test dataset use ``sample_percentage =
        70`` and ``sampling_mode = "Exclude sample"``.

    Parameters
    ----------
    dictionary_file_path_or_domain : str or `.DictionaryDomain`
        Path of a Khiops dictionary file or a DictionaryDomain object.
    dictionary_name : str
        Name of the predictor dictionary.
    data_table_path : str
        Path of the data table file.
    output_data_table_path : str
        Path of the scores output data file.
    detect_format : bool, default ``True``
        If True detects automatically whether the data table file has a header and its
        field separator. It's ignored if ``header_line`` or ``field_separator`` are set.
    header_line : bool, optional (default ``True`` if ``detect_format`` is ``False``)
        If True it uses the first line of the data as column names. Overrides
        ``detect_format`` if set.
    field_separator : str, optional (default "\\t" if ``detect_format`` is ``False``)
        A field separator character, overrides ``detect_format`` if set ("" counts
        as "\\t").
    sample_percentage : int, default 70
        See ``sampling_mode`` option below.
    sampling_mode : "Include sample" or "Exclude sample", default "Include sample"
        If equal to "Include sample" deploys the predictor on ``sample_percentage``
        percent of data and if equal to "Exclude sample" on the complementary ``100 -
        sample_percentage`` percent of data.
    additional_data_tables : dict, optional
        A dictionary containing the data paths and file paths for a multi-table
        dictionary file. For more details see :doc:`/multi_table_primer` documentation.
    output_header_line : bool, default ``True``
        If True writes a header line containing the column names in the output table.
    output_field_separator : str, default "\\t"
        A field separator character ("" counts as "\\t").
    ... :
        Options of the `.KhiopsRunner.run` method from the class `.KhiopsRunner`.
    """
    # Check the dictionary domain
    api._check_dictionary_file_path_or_domain(dictionary_file_path_or_domain)

    # Load the dictionary file into a domain if necessary
    if isinstance(dictionary_file_path_or_domain, DictionaryDomain):
        predictor_domain = dictionary_file_path_or_domain.copy()
    else:
        predictor_domain = read_dictionary_file(dictionary_file_path_or_domain)

    # Check that the specified dictionary is a predictor
    predictor_dictionary = predictor_domain.get_dictionary(dictionary_name)
    if "PredictorType" not in predictor_dictionary.meta_data:
        raise ValueError(f"Dictionary '{predictor_dictionary.name}' is not a predictor")

    # Set the type of classifier
    predictor_type = predictor_dictionary.meta_data.get_value("PredictorType")
    is_classifier = predictor_type == "Classifier"

    # Use the necessary columns
    predictor_dictionary.use_all_variables(False)
    for variable in predictor_dictionary.variables:
        if "TargetVariable" in variable.meta_data:
            variable.used = True
        elif is_classifier:
            if "Prediction" in variable.meta_data:
                variable.used = True
            for key in variable.meta_data.keys:
                if key.startswith("TargetProb"):
                    variable.used = True
        elif not is_classifier and "Mean" in variable.meta_data:
            variable.used = True

    # Deploy the scores
    api.deploy_model(
        predictor_domain,
        dictionary_name,
        data_table_path,
        output_data_table_path,
        detect_format=detect_format,
        header_line=header_line,
        field_separator=field_separator,
        sample_percentage=sample_percentage,
        sampling_mode=sampling_mode,
        additional_data_tables=additional_data_tables,
        output_header_line=output_header_line,
        output_field_separator=output_field_separator,
        trace=trace,
    )


# pylint: enable=protected-access


def visualize_report(report_path):
    """Opens a Khiops or Khiops Coclustering report with the desktop visualization app

    Before using this function, make sure you have installed the Khiops Visualization
    app and/or the Khiops Co-Visualization app. More info at
    `<https://khiops.org/setup/visualization/>`_

    Parameters
    ----------
    report_path : str
        The path of the report file to be open. It must have extension '.khj' (Khiops
        report) or '.khcj' (Khiops Coclustering report).

    Raises
    ------
    `ValueError`
        If the report file path does not have extension '.khj' or '.khcj'.
    `FileNotFoundError`
        If the report file does not exist.
    `RuntimeError`
        If the report file is executable.
    """
    # Check that the report path:
    # - has a valid file extension
    # - exists
    # - is not executable
    #   - Skip this check on Windows because generated reports are executable
    _, ext = os.path.splitext(report_path)
    if ext not in [".khj", ".khcj"]:
        raise ValueError(
            "'report_path' must have extension '.khj' or '.khcj'. "
            f"Path: {report_path}"
        )
    if not os.path.exists(report_path):
        raise FileNotFoundError(report_path)
    if platform.system() != "Windows" and os.access(report_path, os.X_OK):
        raise RuntimeError(f"Report file cannot be executable. Path: {report_path}")

    # Open it with the associated application
    try:
        if platform.system() == "Windows":
            subprocess.call(["explorer", report_path])
        elif platform.system() == "Darwin":
            subprocess.call(["open", report_path])
        else:
            subprocess.call(["xdg-open", report_path])
    # On failure we just print the error to not break the execution
    except OSError as error:
        print(f"Could not open report file: {error}. Path: {report_path}")
