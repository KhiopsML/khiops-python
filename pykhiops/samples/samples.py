######################################################################################
# Copyright (c) 2018 - 2022 Orange - All Rights Reserved                             #
# * This software is the confidential and proprietary information of Orange.         #
# * You shall not disclose such Restricted Information and shall use it only in      #
#   accordance with the terms of the license agreement you entered into with Orange  #
#   named the "Khiops - Python Library Evaluation License".                          #
# * Unauthorized copying of this file, via any medium is strictly prohibited.        #
# * See the "LICENSE.md" file for more details.                                      #
######################################################################################
"""pyKhiops samples
The functions in this script demonstrate the basic use of the pykhiops library.

This script is fully compatible with the latest Khiops version. If you are using an
older version some samples may fail.
"""
import argparse
import os
from math import sqrt
from os import path

import pykhiops
from pykhiops import core as pk

# Disable open files without encoding because samples are simple code snippets
# pylint: disable=unspecified-encoding


def get_khiops_version():
    """Shows the Khiops version"""
    print(f"Khiops version: {pk.get_khiops_version()}")


def export_dictionary_as_json():
    """Exports a dictionary file ('.kdic') in JSON format ('.kdicj')"""
    # Set the file paths
    dictionary_file_path = path.join(pk.get_samples_dir(), "Adult", "Adult.kdic")
    json_dictionary_file_path = path.join(
        "pk_samples",
        "export_dictionary_as_json",
        "Adult.kdicj",
    )

    # Export the input dictionary as JSON
    pk.export_dictionary_as_json(dictionary_file_path, json_dictionary_file_path)


def build_dictionary_from_data_table():
    """Automatically creates a dictionary file from a data table"""
    # Set the file paths
    data_table_path = path.join(pk.get_samples_dir(), "Adult", "Adult.txt")
    dictionary_name = "AutoAdult"
    dictionary_file_path = path.join(
        "pk_samples", "build_dictionary_from_data_table", "AutoAdult.kdic"
    )

    # Create the dictionary from the data table
    pk.build_dictionary_from_data_table(
        data_table_path, dictionary_name, dictionary_file_path
    )


def detect_data_table_format():
    """Detects the format of a data table with and without a dictionary file

    The user may provide a dictionary file or dictionary domain object specifying the
    table schema. The detection heuristic is more accurate with this information.
    """
    # Set the file paths
    data_table_path = path.join(pk.get_samples_dir(), "Adult", "Adult.txt")
    dictionary_file_path = path.join(pk.get_samples_dir(), "Adult", "Adult.kdic")
    results_dir = path.join("pk_samples", "detect_data_table_format")
    transformed_data_table_path = path.join(results_dir, "AdultWithAnotherFormat.txt")

    # Create the output directory
    if not path.isdir(results_dir):
        os.mkdir(results_dir)

    # Detect the format of the table
    format_spec = pk.detect_data_table_format(data_table_path)
    print("Format specification (header_line, field_separator)")
    print("Format detected on original table:", format_spec)

    # Make a deployment to change the format of the data table
    pk.deploy_model(
        dictionary_file_path,
        "Adult",
        data_table_path,
        transformed_data_table_path,
        output_header_line=False,
        output_field_separator=",",
    )

    # Detect the new format of the table without a dictionary file
    format_spec = pk.detect_data_table_format(transformed_data_table_path)
    print("Format detected on reformatted table:", format_spec)

    # Detect the new format of the table with a dictionary file
    format_spec = pk.detect_data_table_format(
        transformed_data_table_path,
        dictionary_file_path_or_domain=dictionary_file_path,
        dictionary_name="Adult",
    )
    print("Format detected (with dictionary file) on reformatted table:", format_spec)


def check_database():
    """Runs an integrity check of a database

    The results are stored in the specified log file with at most 50 error messages.
    """
    # Set the file paths
    dictionary_file_path = path.join(pk.get_samples_dir(), "Adult", "Adult.kdic")
    data_table_path = path.join(pk.get_samples_dir(), "Adult", "Adult.txt")
    log_file = path.join("pk_samples", "check_database", "check_database.log")

    # Check the database
    pk.check_database(
        dictionary_file_path,
        "Adult",
        data_table_path,
        log_file_path=log_file,
        max_messages=50,
    )


def export_dictionary_files():
    """Exports a customized dictionary to ".kdic" and to ".kdicj" (JSON)"""
    # Set the file paths
    dictionary_file_path = path.join(pk.get_samples_dir(), "Adult", "Adult.kdic")
    results_dir = path.join("pk_samples", "export_dictionary_file")
    output_dictionary_file_path = path.join(results_dir, "ModifiedAdult.kdic")
    output_dictionary_json_path = path.join(results_dir, "ModifiedAdult.kdicj")
    alt_output_dictionary_json_path = path.join(results_dir, "AltModifiedAdult.kdicj")

    # Load the dictionary domain from initial dictionary file
    # Then obtain the "Adult" dictionary within
    domain = pk.read_dictionary_file(dictionary_file_path)
    dictionary = domain.get_dictionary("Adult")

    # Set some of its variables to unused
    fnlwgt_variable = dictionary.get_variable("fnlwgt")
    fnlwgt_variable.used = False
    label_variable = dictionary.get_variable("Label")
    label_variable.used = False

    # Create output directory if necessary
    if not path.exists("pk_samples"):
        os.mkdir("pk_samples")
        os.mkdir(results_dir)
    else:
        if not path.exists(results_dir):
            os.mkdir(results_dir)

    # Export to kdic
    domain.export_khiops_dictionary_file(output_dictionary_file_path)

    # Export to kdicj either from the domain or from a kdic file
    # Requires a Khiops execution, that's why it is not a method of DictionaryDomain
    pk.export_dictionary_as_json(domain, output_dictionary_json_path)
    pk.export_dictionary_as_json(
        output_dictionary_file_path, alt_output_dictionary_json_path
    )


def train_predictor():
    """Trains a predictor with a minimal setup"""
    # Set the file paths
    dictionary_file_path = path.join(pk.get_samples_dir(), "Adult", "Adult.kdic")
    data_table_path = path.join(pk.get_samples_dir(), "Adult", "Adult.txt")
    results_dir = path.join("pk_samples", "train_predictor")

    # Train the predictor
    pk.train_predictor(
        dictionary_file_path,
        "Adult",
        data_table_path,
        "class",
        results_dir,
        max_trees=0,
    )


def train_predictor_file_paths():
    """Trains a predictor and stores the return value of the function"""
    # Set the file paths
    dictionary_file_path = path.join(pk.get_samples_dir(), "Adult", "Adult.kdic")
    data_table_path = path.join(pk.get_samples_dir(), "Adult", "Adult.txt")
    results_dir = path.join("pk_samples", "train_predictor_file_paths")

    # Train the predictor
    report_file_path, modeling_dictionary_file_path = pk.train_predictor(
        dictionary_file_path,
        "Adult",
        data_table_path,
        "class",
        results_dir,
        max_trees=0,
    )
    print("Reports file available at " + report_file_path)
    print("Modeling dictionary file available at " + modeling_dictionary_file_path)


def train_predictor_error_handling():
    """Shows how to handle errors when training a predictor

    Trains the predictor and handles the errors by printing a custom message. When the
    Khiops application fails pyKhiops will raise a PyKhiopsRuntimeError reporting the
    errors encountered by Khiops.

    If the latter information is not enough to diagnose the problem, it is possible to
    save the temporary log file by activating the "trace" flag in the call to
    `~.api.train_predictor`. The path of the log file will be printed to the standard
    output, as well as that of the dictionary and scenario files (note that the "trace"
    keyword argument is available in all functions of the `pykhiops.core.api`
    submodule).
    """
    # Set the file paths with a nonexistent dictionary file
    dictionary_file_path = "NONEXISTENT_DICTIONARY_FILE.kdic"
    data_table_path = path.join(pk.get_samples_dir(), "Adult", "Adult.txt")
    results_dir = path.join("pk_samples", "train_predictor_error_handling")
    log_file_path = path.join(results_dir, "khiops.log")
    scenario_path = path.join(results_dir, "scenario._kh")

    # Train the predictor and handle the error
    try:
        pk.train_predictor(
            dictionary_file_path,
            "Adult",
            data_table_path,
            "class",
            results_dir,
            trace=True,
            log_file_path=log_file_path,
            output_scenario_path=scenario_path,
        )
    except pk.PyKhiopsRuntimeError as error:
        print("Khiops training failed! Below the PyKhiopsRuntimeError message:")
        print(error)

    print("\nFull log contents:")
    print("------------------")
    with open(log_file_path) as log_file:
        for line in log_file:
            print(line, end="")

    print("\nExecuted scenario")
    print("-----------------")
    with open(scenario_path) as scenario_file:
        for line in scenario_file:
            print(line, end="")


def train_predictor_mt():
    """Trains a multi-table predictor in the simplest way possible

    It is a call to `~.api.train_predictor` with additional parameters to handle
    multi-table learning
    """
    # Set the file paths
    accidents_dir = path.join(pk.get_samples_dir(), "AccidentsSummary")
    dictionary_file_path = path.join(accidents_dir, "Accidents.kdic")
    accidents_table_path = path.join(accidents_dir, "Accidents.txt")
    vehicles_table_path = path.join(accidents_dir, "Vehicles.txt")
    results_dir = path.join("pk_samples", "train_predictor_mt")

    # Train the predictor. Besides the mandatory parameters, we specify:
    # - A python dictionary linking data paths to file paths for non-root tables
    # - To not construct any decision tree
    # The default number of automatic features is 100
    pk.train_predictor(
        dictionary_file_path,
        "Accident",
        accidents_table_path,
        "Gravity",
        results_dir,
        additional_data_tables={"Accident`Vehicles": vehicles_table_path},
        max_trees=0,
    )


def train_predictor_mt_with_specific_rules():
    """Trains a multi-table predictor with specific construction rules

    It is the same as `.train_predictor_mt` but with the specification of the allowed
    variable construction rules. The list of available rules is found in the field
    ``pk.all_construction_rules``
    """
    # Set the file paths
    accidents_dir = path.join(pk.get_samples_dir(), "AccidentsSummary")
    dictionary_file_path = path.join(accidents_dir, "Accidents.kdic")
    accidents_table_path = path.join(accidents_dir, "Accidents.txt")
    vehicles_table_path = path.join(accidents_dir, "Vehicles.txt")
    results_dir = path.join("pk_samples", "train_predictor_mt_with_specific_rules")

    # Train the predictor. Besides the mandatory parameters, it is specified:
    # - A python dictionary linking data paths to file paths for non-root tables
    # - The maximum number of aggregate variables to construct (1000)
    # - The construction rules allowed to automatically create aggregates
    # - To not construct any decision tree
    pk.train_predictor(
        dictionary_file_path,
        "Accident",
        accidents_table_path,
        "Gravity",
        results_dir,
        additional_data_tables={"Accident`Vehicles": vehicles_table_path},
        max_constructed_variables=1000,
        construction_rules=["TableMode", "TableSelection"],
        max_trees=0,
    )


def train_predictor_mt_snowflake():
    """Trains a multi-table predictor for a dataset with a snowflake schema"""

    # Set the file paths
    accidents_dir = path.join(pk.get_samples_dir(), "Accidents")
    dictionary_file_path = path.join(accidents_dir, "Accidents.kdic")
    accidents_table_path = path.join(accidents_dir, "Accidents.txt")
    vehicles_table_path = path.join(accidents_dir, "Vehicles.txt")
    users_table_path = path.join(accidents_dir, "Users.txt")
    places_table_path = path.join(accidents_dir, "Places.txt")
    results_dir = path.join("pk_samples", "train_predictor_mt_snowflake")

    # Train the predictor. Besides the mandatory parameters, we specify:
    # - A python dictionary linking data paths to file paths for non-root tables
    # - To not construct any decision tree
    # The default number of automatic features is 100
    pk.train_predictor(
        dictionary_file_path,
        "Accident",
        accidents_table_path,
        "Gravity",
        results_dir,
        additional_data_tables={
            "Accident`Vehicles": vehicles_table_path,
            "Accident`Vehicles`Users": users_table_path,
            "Accident`Place": places_table_path,
        },
        max_trees=0,
    )


def train_predictor_with_train_percentage():
    """Trains a predictor with a 90%-10% train-test split

    Note: The default is a 70%-30% split
    """
    # Set the file paths
    dictionary_file_path = path.join(pk.get_samples_dir(), "Adult", "Adult.kdic")
    data_table_path = path.join(pk.get_samples_dir(), "Adult", "Adult.txt")
    results_dir = path.join("pk_samples", "train_predictor_with_train_percentage")

    # Train the predictor. Besides the mandatory parameters, it is specified:
    # - A 90% sampling rate for the training dataset
    # - Set the test dataset as the complement of the training dataset (10%)
    # - No trees
    pk.train_predictor(
        dictionary_file_path,
        "Adult",
        data_table_path,
        "class",
        results_dir,
        sample_percentage=90,
        use_complement_as_test=True,
        max_trees=0,
        results_prefix="P90_",
    )


def train_predictor_with_trees():
    """Trains a predictor based on 15 trees with a 80%-20% train-test split"""
    # Set the file paths
    dictionary_file_path = path.join(pk.get_samples_dir(), "Letter", "Letter.kdic")
    data_table_path = path.join(pk.get_samples_dir(), "Letter", "Letter.txt")
    results_dir = path.join("pk_samples", "train_predictor_with_trees")

    # Train the predictor with at most 15 trees (default 10)
    pk.train_predictor(
        dictionary_file_path,
        "Letter",
        data_table_path,
        "lettr",
        results_dir,
        sample_percentage=80,
        use_complement_as_test=True,
        results_prefix="P80_",
        max_trees=15,
    )


def train_predictor_with_pairs():
    """Trains a predictor with user specified pairs of variables"""
    # Set the file paths
    dictionary_file_path = path.join(pk.get_samples_dir(), "Adult", "Adult.kdic")
    data_table_path = path.join(pk.get_samples_dir(), "Adult", "Adult.txt")
    results_dir = path.join("pk_samples", "train_predictor_with_pairs")

    # Train the predictor with at most 10 pairs as follows:
    # - Include pairs age-race and capital_gain-capital_loss
    # - Include all possible pairs having relationship as component
    pk.train_predictor(
        dictionary_file_path,
        "Adult",
        data_table_path,
        "class",
        results_dir,
        use_complement_as_test=True,
        max_trees=0,
        max_pairs=10,
        specific_pairs=[
            ("age", "race"),
            ("capital_gain", "capital_loss"),
            ("relationship", ""),
        ],
    )


def train_predictor_with_multiple_parameters():
    """Trains a predictor with various additional parameters

    Some of these parameters are specific to `~.api.train_predictor` and others generic
    to any Khiops execution.

    In this example, we specify the following parameters in the call:
     - A main target value
     - The path where to store the "Khiops scenario" script
     - The path where to store the log of the process
     - The flag to show the execution trace (generic to any `pykhiops.core.api`
       function)

    Additionally the pyKhiops runner is set such that the learning is executed with only
    1000 MB of memory.
    """
    # Set the file paths
    dictionary_file_path = path.join(pk.get_samples_dir(), "Adult", "Adult.kdic")
    data_table_path = path.join(pk.get_samples_dir(), "Adult", "Adult.txt")
    results_dir = path.join("pk_samples", "train_predictor_with_multiple_parameters")
    output_script_path = path.join(results_dir, "output_scenario._kh")
    log_path = path.join(results_dir, "log.txt")

    # Set memory limit to 1000 Mb and train with Khiops
    pk.get_runner().max_memory_mb = 1000

    # Train the predictor. Besides the mandatory parameters, we specify:
    # - The value "more" as main target value
    # - The output Khiops script file location (generic)
    # - The log file location (generic)
    # - To show the debug trace (generic)
    pk.train_predictor(
        dictionary_file_path,
        "Adult",
        data_table_path,
        "class",
        results_dir,
        main_target_value="more",
        output_scenario_path=output_script_path,
        log_file_path=log_path,
        trace=True,
    )

    # Reset memory limit to default Khiops tool value
    pk.get_runner().max_memory_mb = 0


def train_predictor_detect_format():
    """Trains a predictor without specifying the table format"""
    # Set the file paths
    dictionary_file_path = path.join(pk.get_samples_dir(), "Iris", "Iris.kdic")
    data_table_path = path.join(pk.get_samples_dir(), "Iris", "Iris.txt")
    results_dir = path.join("pk_samples", "train_predictor_detect_format")
    transformed_data_table_path = path.join(results_dir, "TransformedIris.txt")

    # Transform the database format from header_line=True and field_separator=TAB
    # to header_line=False and field_separator=","
    # See the deploy_model examples below for more details
    pk.deploy_model(
        dictionary_file_path,
        "Iris",
        data_table_path,
        transformed_data_table_path,
        output_header_line=False,
        output_field_separator=",",
    )

    # Try to learn with the old format
    try:
        pk.train_predictor(
            dictionary_file_path,
            "Iris",
            transformed_data_table_path,
            "Class",
            results_dir,
            header_line=True,
            field_separator="",
        )
    except pk.PyKhiopsRuntimeError as error:
        print(
            "This failed because of a bad data table format spec. "
            + "Below the PyKhiopsRuntimeError message"
        )
        print(error)

    # Train without specifyng the format (detect_format is True by default)
    pk.train_predictor(
        dictionary_file_path,
        "Iris",
        transformed_data_table_path,
        "Class",
        results_dir,
    )


def train_predictor_with_cross_validation():
    """Trains a predictor with a 5-fold cross-validation"""
    # Set the file paths
    dictionary_file_path = path.join(pk.get_samples_dir(), "Adult", "Adult.kdic")
    data_table_path = path.join(pk.get_samples_dir(), "Adult", "Adult.txt")
    results_dir = path.join("pk_samples", "train_predictor_with_cross_validation")
    fold_dictionary_file_path = path.join(results_dir, "AdultWithFolding.kdic")

    # Create the output directory
    if not path.isdir(results_dir):
        os.mkdir(results_dir)

    # Load the learning dictionary object
    domain = pk.read_dictionary_file(dictionary_file_path)
    dictionary = domain.get_dictionary("Adult")

    # Add a random fold index variable to the learning dictionary
    fold_number = 5
    fold_index_variable = pk.Variable()
    fold_index_variable.name = "FoldIndex"
    fold_index_variable.type = "Numerical"
    fold_index_variable.used = False
    fold_index_variable.rule = "Ceil(Product(" + str(fold_number) + ",  Random()))"
    dictionary.add_variable(fold_index_variable)

    # Add variables that indicate if the instance is in the train dataset:
    for fold_index in range(1, fold_number + 1):
        is_in_train_dataset_variable = pk.Variable()
        is_in_train_dataset_variable.name = "IsInTrainDataset" + str(fold_index)
        is_in_train_dataset_variable.type = "Numerical"
        is_in_train_dataset_variable.used = False
        is_in_train_dataset_variable.rule = "NEQ(FoldIndex, " + str(fold_index) + ")"
        dictionary.add_variable(is_in_train_dataset_variable)

    # Print dictionary with fold variables
    print("Dictionary file with fold variables")
    domain.export_khiops_dictionary_file(fold_dictionary_file_path)
    with open(fold_dictionary_file_path) as fold_dictionary_file:
        for line in fold_dictionary_file:
            print(line, end="")

    # For each fold k:
    print("Training Adult with " + str(fold_number) + " folds")
    print("\tfold\ttrain auc\ttest auc")
    train_aucs = []
    test_aucs = []
    for fold_index in range(1, fold_number + 1):
        # Train a model from the sub-dataset where IsInTrainDataset<k> is 1
        train_reports_path, modeling_dictionary_file_path = pk.train_predictor(
            domain,
            "Adult",
            data_table_path,
            "class",
            results_dir,
            sample_percentage=100,
            selection_variable="IsInTrainDataset" + str(fold_index),
            selection_value=1,
            max_trees=0,
            results_prefix="Fold" + str(fold_index),
        )

        # Evaluate the resulting model in the subsets where IsInTrainDataset is 0
        test_evaluation_report_path = pk.evaluate_predictor(
            modeling_dictionary_file_path,
            "Adult",
            data_table_path,
            results_dir,
            sample_percentage=100,
            selection_variable="IsInTrainDataset" + str(fold_index),
            selection_value=0,
            results_prefix="Fold" + str(fold_index),
        )

        # Obtain the train AUC from the train report and the test AUC from the
        # evaluation report and print them
        train_results = pk.read_analysis_results_file(train_reports_path)
        test_evaluation_results = pk.read_analysis_results_file(
            test_evaluation_report_path
        )
        train_auc = train_results.train_evaluation_report.get_snb_performance().auc
        test_auc = test_evaluation_results.evaluation_report.get_snb_performance().auc
        print("\t" + str(fold_index) + "\t" + str(train_auc) + "\t" + str(test_auc))

        # Store the train and test AUCs in arrays
        train_aucs.append(train_auc)
        test_aucs.append(test_auc)

    # Print the mean +- error aucs for both train and test
    mean_train_auc = sum(train_aucs) / fold_number
    squared_error_train_aucs = [(auc - mean_train_auc) ** 2 for auc in train_aucs]
    sd_train_auc = sqrt(sum(squared_error_train_aucs) / (fold_number - 1))

    mean_test_auc = sum(test_aucs) / fold_number
    squared_error_test_aucs = [(auc - mean_test_auc) ** 2 for auc in test_aucs]
    sd_test_auc = sqrt(sum(squared_error_test_aucs) / (fold_number - 1))

    print("final auc")
    print("train auc: " + str(mean_train_auc) + " +- " + str(sd_train_auc))
    print("test  auc: " + str(mean_test_auc) + " +- " + str(sd_test_auc))


def multiple_train_predictor():
    """Trains a sequence of models with a decreasing number of variables

    This example illustrates the use of the pykhiops classes `.DictionaryDomain` (for
    reading dictionary files) and `.AnalysisResults` (for reading training/evaluation
    results from JSON)
    """

    def display_test_results(json_result_file_path):
        """Display some of the training results"""
        results = pk.read_analysis_results_file(json_result_file_path)
        train_performance = results.train_evaluation_report.get_snb_performance()
        test_performance = results.test_evaluation_report.get_snb_performance()
        print(
            "\t"
            + str(len(results.preparation_report.variables_statistics))
            + "\t"
            + str(train_performance.auc)
            + "\t"
            + str(test_performance.auc)
        )

    # Set the file paths
    dictionary_file_path = path.join(pk.get_samples_dir(), "Adult", "Adult.kdic")
    data_table_path = path.join(pk.get_samples_dir(), "Adult", "Adult.txt")
    results_dir = path.join("pk_samples", "multiple_train_predictor")

    # Read the dictionary file to obtain an instance of class Dictionary
    dictionary_domain = pk.read_dictionary_file(dictionary_file_path)
    dictionary = dictionary_domain.get_dictionary("Adult")

    # Train a SNB model using all the variables
    print("\t#vars\ttrain auc\ttest auc")
    json_result_file_path, _ = pk.train_predictor(
        dictionary_file_path,
        "Adult",
        data_table_path,
        "class",
        results_dir,
        sample_percentage=70,
        use_complement_as_test=True,
        max_trees=0,
    )
    display_test_results(json_result_file_path)

    # Read results to obtain the variables sorted by decreasing Level
    analysis_results = pk.read_analysis_results_file(json_result_file_path)
    preparation_results = analysis_results.preparation_report

    # Train a sequence of models with a decreasing number of variables
    # We disable variables one-by-one in increasing level (predictive power) order
    variable_number = len(preparation_results.variables_statistics)
    for i in reversed(range(variable_number)):
        # Search the next variable
        variable = preparation_results.variables_statistics[i]

        # Disable this variable and save the dictionary with the Khiops format
        dictionary.get_variable(variable.name).used = False

        # Train the model with this dictionary domain object
        prefix = f"V{variable_number - 1 - i}_"
        json_result_file_path, _ = pk.train_predictor(
            dictionary_domain,
            "Adult",
            data_table_path,
            "class",
            results_dir,
            sample_percentage=70,
            use_complement_as_test=True,
            results_prefix=prefix,
            max_trees=0,
        )

        # Show a preview of the results
        display_test_results(json_result_file_path)


def evaluate_predictor():
    """Evaluates a predictor in the simplest way possible

    It calls `~.api.evaluate_predictor` with only its mandatory parameters.
    """
    # Set the file paths
    dictionary_file_path = path.join(pk.get_samples_dir(), "Adult", "Adult.kdic")
    data_table_path = path.join(pk.get_samples_dir(), "Adult", "Adult.txt")
    results_dir = path.join("pk_samples", "evaluate_predictor")
    model_dictionary_file_path = path.join(results_dir, "Modeling.kdic")

    # Train the predictor
    pk.train_predictor(
        dictionary_file_path,
        "Adult",
        data_table_path,
        "class",
        results_dir,
        max_trees=0,
    )

    # Evaluate the predictor
    report_file_path = pk.evaluate_predictor(
        model_dictionary_file_path, "Adult", data_table_path, results_dir
    )
    print("Evaluation report available at " + report_file_path)


def access_predictor_evaluation_report():
    """Shows the performance metrics of a predictor

    See `evaluate_predictor` or `train_predictor_with_train_percentage` to see examples
    on how to evaluate a model.
    """
    # Set the file paths
    dictionary_file_path = path.join(pk.get_samples_dir(), "Adult", "Adult.kdic")
    data_table_path = path.join(pk.get_samples_dir(), "Adult", "Adult.txt")
    results_dir = path.join("pk_samples", "access_predictor_evaluation_report")
    evaluation_report_path = path.join(results_dir, "AllReports.khj")

    # Train the SNB predictor and some univariate predictors
    # Note: Evaluation in test is 30% by default
    pk.train_predictor(
        dictionary_file_path,
        "Adult",
        data_table_path,
        "class",
        results_dir,
        max_trees=0,
        univariate_predictor_number=4,
    )

    # Obtain the evaluation results
    results = pk.read_analysis_results_file(evaluation_report_path)
    evaluation_report = results.test_evaluation_report
    snb_performance = evaluation_report.get_snb_performance()

    # Print univariate metrics for the SNB
    print("\nperformance metrics for " + snb_performance.name)
    for metric_name in snb_performance.get_metric_names():
        print(metric_name + ": " + str(snb_performance.get_metric(metric_name)))

    # Print the confusion matrix
    print("\nconfusion matrix:")
    confusion_matrix = snb_performance.confusion_matrix

    for target_value in confusion_matrix.values:
        print("\t" + target_value, end="")
    print("")

    for i, target_value in enumerate(confusion_matrix.values):
        observed_frequencies = confusion_matrix.matrix[i]
        print(target_value, end="")
        for frequency in observed_frequencies:
            print("\t" + str(frequency), end="")
        print("")

    # Print the head of the lift curves for the 'more' modality
    print("\nfirst five values of the lift curves for 'more'")

    snb_lift_curve = evaluation_report.get_snb_lift_curve("more")
    optimal_lift_curve = evaluation_report.get_classifier_lift_curve("Optimal", "more")
    random_lift_curve = evaluation_report.get_classifier_lift_curve("Random", "more")

    for i in range(5):
        print(
            str(snb_lift_curve.values[i])
            + "\t"
            + str(optimal_lift_curve.values[i])
            + "\t"
            + str(random_lift_curve.values[i])
        )

    # Print univariate metrics for an univariate predictor
    predictor_performance = evaluation_report.get_predictor_performance(
        "Univariate relationship"
    )
    print("\n\nperformance metrics for " + predictor_performance.name)
    for metric_name in predictor_performance.get_metric_names():
        print(metric_name + ": " + str(predictor_performance.get_metric(metric_name)))


def train_recoder():
    """Train a database recoder in the simplest way possible

    It is a call to `~.api.train_recoder` with only its mandatory parameters.
    """
    # Set the file paths
    dictionary_file_path = path.join(pk.get_samples_dir(), "Adult", "Adult.kdic")
    data_table_path = path.join(pk.get_samples_dir(), "Adult", "Adult.txt")
    results_dir = path.join("pk_samples", "train_recoder")

    # Train the recoder model
    pk.train_recoder(
        dictionary_file_path, "Adult", data_table_path, "class", results_dir
    )


def train_recoder_with_multiple_parameters():
    """Trains a recoder that transforms variable values to their respective part labels

    It also creates 10 pair features.
    """
    # Set the file paths
    dictionary_file_path = path.join(pk.get_samples_dir(), "Adult", "Adult.kdic")
    data_table_path = path.join(pk.get_samples_dir(), "Adult", "Adult.txt")
    results_dir = path.join("pk_samples", "train_recoder_with_multiple_parameters")

    # Train the recoder model
    pk.train_recoder(
        dictionary_file_path,
        "Adult",
        data_table_path,
        "class",
        results_dir,
        max_pairs=10,
        categorical_recoding_method="part label",
        numerical_recoding_method="part label",
    )


def train_recoder_mt_flatten():
    """Trains a recoder that flattens a multi-table database into a single table

    The constructed variables are all kept and no recoding is performed on their values
    """
    # Set the file paths
    accidents_dir = path.join(pk.get_samples_dir(), "AccidentsSummary")
    dictionary_file_path = path.join(accidents_dir, "Accidents.kdic")
    accidents_table_path = path.join(accidents_dir, "Accidents.txt")
    vehicles_table_path = path.join(accidents_dir, "Vehicles.txt")
    results_dir = path.join("pk_samples", "train_recoder_mt_flatten")

    # Train the recoder. Besides the mandatory parameters, it is specified:
    # - A python dictionary linking data paths to file paths for non-root tables
    # - The maximum number of aggregate variables to construct (1000)
    # - To keep all the created variables independently of their informativeness (level)
    # - To not recode the variables values
    pk.train_recoder(
        dictionary_file_path,
        "Accident",
        accidents_table_path,
        "Gravity",
        results_dir,
        additional_data_tables={"Accident`Vehicles": vehicles_table_path},
        max_constructed_variables=1000,
        informative_variables_only=False,
        categorical_recoding_method="none",
        numerical_recoding_method="none",
        keep_initial_categorical_variables=True,
        keep_initial_numerical_variables=True,
    )


def deploy_model():
    """Deploys a model in the simplest possible

    It is a call to `~.api.deploy_model` with its mandatory parameters.

    In this example, a Selective Naive Bayes (SNB) model is deployed by applying its
    associated dictionary to the input database. The model predictions are written to
    the output database.
    """
    # Set the file paths
    dictionary_file_path = path.join(pk.get_samples_dir(), "Adult", "Adult.kdic")
    data_table_path = path.join(pk.get_samples_dir(), "Adult", "Adult.txt")
    results_dir = path.join("pk_samples", "deploy_model")
    model_dictionary_file_path = path.join(results_dir, "Modeling.kdic")
    output_data_table_path = path.join(results_dir, "ScoresAdult.txt")

    # Train the predictor
    pk.train_predictor(
        dictionary_file_path,
        "Adult",
        data_table_path,
        "class",
        results_dir,
        max_trees=0,
    )

    # Deploy the model on the database
    # It will score it according to the trained predictor
    pk.deploy_model(
        model_dictionary_file_path, "SNB_Adult", data_table_path, output_data_table_path
    )


def deploy_model_mt():
    """Deploys a multi-table classifier in the simplest way possible

    It is a call to `~.api.deploy_model` with additional parameters to handle
    multi-table deployment.

    In this example, a Selective Naive Bayes (SNB) model is deployed by applying its
    associated dictionary to the input database. The model predictions are written to
    the output database.
    """
    # Set the file paths
    accidents_dir = path.join(pk.get_samples_dir(), "AccidentsSummary")
    dictionary_file_path = path.join(accidents_dir, "Accidents.kdic")
    accidents_table_path = path.join(accidents_dir, "Accidents.txt")
    vehicles_table_path = path.join(accidents_dir, "Vehicles.txt")
    results_dir = path.join("pk_samples", "deploy_model_mt")
    model_dictionary_file_path = path.join(results_dir, "Modeling.kdic")
    output_data_table_path = path.join(results_dir, "TransferredAccidents.txt")

    # Train the predictor (see train_predictor_mt for details)
    pk.train_predictor(
        dictionary_file_path,
        "Accident",
        accidents_table_path,
        "Gravity",
        results_dir,
        additional_data_tables={"Accident`Vehicles": vehicles_table_path},
        max_trees=0,
    )

    # Deploy the model on the database
    # Besides the mandatory parameters, it is specified:
    # - A python dictionary linking data paths to file paths for non-root tables
    pk.deploy_model(
        model_dictionary_file_path,
        "SNB_Accident",
        accidents_table_path,
        output_data_table_path,
        additional_data_tables={"SNB_Accident`Vehicles": vehicles_table_path},
    )


def deploy_model_mt_snowflake():
    """Deploys a classifier model on a dataset with a snowflake schema"""
    # Set the file paths
    accidents_dir = path.join(pk.get_samples_dir(), "Accidents")
    dictionary_file_path = path.join(accidents_dir, "Accidents.kdic")
    accidents_table_path = path.join(accidents_dir, "Accidents.txt")
    vehicles_table_path = path.join(accidents_dir, "Vehicles.txt")
    users_table_path = path.join(accidents_dir, "Users.txt")
    places_table_path = path.join(accidents_dir, "Places.txt")
    results_dir = path.join("pk_samples", "deploy_model_mt_snowflake")
    model_dictionary_file_path = path.join(results_dir, "Modeling.kdic")
    output_data_table_path = path.join(results_dir, "TransferredAccidents.txt")

    # Train the predictor. Besides the mandatory parameters, we specify:
    # - A python dictionary linking data paths to file paths for non-root tables
    # - To not construct any decision tree
    # The default number of automatic features is 100
    pk.train_predictor(
        dictionary_file_path,
        "Accident",
        accidents_table_path,
        "Gravity",
        results_dir,
        additional_data_tables={
            "Accident`Vehicles": vehicles_table_path,
            "Accident`Vehicles`Users": users_table_path,
            "Accident`Place": places_table_path,
        },
        max_trees=0,
    )

    # Deploy the model on the database
    # Besides the mandatory parameters, it is specified:
    # - A python dictionary linking data paths to file paths for non-root tables
    pk.deploy_model(
        model_dictionary_file_path,
        "SNB_Accident",
        accidents_table_path,
        output_data_table_path,
        additional_data_tables={
            "SNB_Accident`Vehicles": vehicles_table_path,
            "SNB_Accident`Vehicles`Users": users_table_path,
            "SNB_Accident`Place": places_table_path,
        },
    )


def deploy_model_expert():
    """Deploys a model with a specification of additional variables to be included

    In this example, a Selective Naive Bayes (SNB) model is deployed by applying its
    associated dictionary to the input database. Specifically, the output file contains:

    - The model predictions
    - The probabilities of all modalities of the target variable.

    The "expert" part of this example is the use of the pykhiops dictionary interface
    and the `.DictionaryDomain` class
    """
    # Set the file paths
    dictionary_file_path = path.join(pk.get_samples_dir(), "Adult", "Adult.kdic")
    data_table_path = path.join(pk.get_samples_dir(), "Adult", "Adult.txt")
    results_dir = path.join("pk_samples", "deploy_model_expert")
    model_dictionary_file_path = path.join(results_dir, "Modeling.kdic")
    output_data_table_path = path.join(results_dir, "ScoresAdult.txt")

    # Train the predictor
    pk.train_predictor(
        dictionary_file_path,
        "Adult",
        data_table_path,
        "class",
        results_dir,
        max_trees=0,
    )

    # Read the dictionary file to obtain an instance of class Dictionary
    model_domain = pk.read_dictionary_file(model_dictionary_file_path)
    snb_dictionary = model_domain.get_dictionary("SNB_Adult")

    # Select Label (identifier)
    snb_dictionary.get_variable("Label").used = True

    # Select the variables containing the probabilities for each class
    for variable in snb_dictionary.variables:
        # The variable must have a meta data with key that start with "target_prob"
        for key in variable.meta_data.keys:
            if key.startswith("TargetProb"):
                variable.used = True

    # Deploy the model. Besides the mandatory parameters, it is specified:
    # - A DictionaryDomain object to use instead of the mandatory dictionary file
    pk.deploy_model(model_domain, "SNB_Adult", data_table_path, output_data_table_path)


def deploy_classifier_for_metrics():
    """Constructs a small precision-recall curve for a classifier"""
    # Set the file paths
    dictionary_file_path = path.join(pk.get_samples_dir(), "Adult", "Adult.kdic")
    data_table_path = path.join(pk.get_samples_dir(), "Adult", "Adult.txt")
    results_dir = path.join("pk_samples", "deploy_classifier_for_metrics")
    output_data_table_path = path.join(results_dir, "ScoresAdult.txt")

    # Train the classifier for the target "class"
    _, modeling_dictionary_file_path = pk.train_predictor(
        dictionary_file_path,
        "Adult",
        data_table_path,
        "class",
        results_dir,
        max_trees=0,
    )

    # Obtain the scores of the SNB on the test dataset to calculate the PR curve
    pk.deploy_predictor_for_metrics(
        modeling_dictionary_file_path,
        "SNB_Adult",
        data_table_path,
        output_data_table_path,
        sampling_mode="Exclude sample",
        output_header_line=False,
    )

    # We estimate the precision/recall for the class "more" and increasing thresholds
    # Note: Normally one would do this with a package (eg. sklearn.metrics)
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    true_positives = {thres: 0 for thres in thresholds}
    false_positives = {thres: 0 for thres in thresholds}
    false_negatives = {thres: 0 for thres in thresholds}
    with open(output_data_table_path) as output_data_table:
        for line in output_data_table:
            fields = line.split("\t")
            true_target = fields[0]
            proba_more = float(fields[3])
            for thres in thresholds:
                if true_target == "more" and proba_more >= thres:
                    true_positives[thres] += 1
                elif true_target == "more" and proba_more < thres:
                    false_negatives[thres] += 1
                elif true_target == "less" and proba_more >= thres:
                    false_positives[thres] += 1

    precision = {
        thres: true_positives[thres] / (true_positives[thres] + false_positives[thres])
        for thres in thresholds
    }
    recall = {
        thres: true_positives[thres] / (true_positives[thres] + false_negatives[thres])
        for thres in thresholds
    }

    # Print the curve at the selected points
    print("Precision and Recall for class 'more'")
    print("threshold\trecall\tprecision")
    thresholds.reverse()
    for thres in thresholds:
        print(str(thres) + "\t" + str(recall[thres]) + "\t" + str(precision[thres]))


def deploy_regressor_for_metrics():
    """Estimates the R2 coefficient of a regressor"""
    # Set the file paths
    dictionary_file_path = path.join(pk.get_samples_dir(), "Adult", "Adult.kdic")
    data_table_path = path.join(pk.get_samples_dir(), "Adult", "Adult.txt")
    results_dir = path.join("pk_samples", "deploy_regressor_for_metrics")
    output_data_table_path = path.join(results_dir, "TrueAndPredictedAges.txt")

    # Train the regressor for the target "age" (with 20% train to be quick)
    _, modeling_dictionary_file_path = pk.train_predictor(
        dictionary_file_path,
        "Adult",
        data_table_path,
        "age",
        results_dir,
        sample_percentage=20,
        max_trees=0,
    )

    # Obtain the predicted regression values of the SNB on the test dataset estimate R2
    pk.deploy_predictor_for_metrics(
        modeling_dictionary_file_path,
        "SNB_Adult",
        data_table_path,
        output_data_table_path,
        sample_percentage=20,
        sampling_mode="Exclude sample",
        output_header_line=False,
    )
    # Estimate R2
    # Note: Normally one would do this with a package (eg. sklearn.metrics)
    # First pass to estimate sums of residuals and the mean
    ss_res = 0
    mean = 0
    n_instances = 0
    with open(output_data_table_path) as output_data_table:
        for line in output_data_table:
            fields = line.split("\t")
            true_target = float(fields[0])
            predicted_target = float(fields[1])
            ss_res += (true_target - predicted_target) ** 2
            mean += true_target
            n_instances += 1
        mean /= n_instances

    # Second pass to estimate the total sums of squares and finish the R2 estimation
    ss_tot = 0
    with open(output_data_table_path) as output_data_table:
        for line in output_data_table:
            fields = line.split("\t")
            true_target = float(fields[0])
            ss_tot += (true_target - mean) ** 2
    r2_score = 1 - ss_res / ss_tot

    # Print results
    print("Adult 'age' regression (30% train)")
    print(f"R2 (explained variance) = {r2_score}")


def sort_data_table():
    """Sorts a database in the simplest way possible

    It is a call to `~.api.sort_data_table` with only its mandatory parameters. This
    sorts a data table by its default key variable (specified in the table's
    dictionary).
    """
    # Set the file paths
    accidents_dir = path.join(pk.get_samples_dir(), "AccidentsSummary")
    dictionary_file_path = path.join(accidents_dir, "Accidents.kdic")
    accidents_table_path = path.join(accidents_dir, "Accidents.txt")
    output_data_table_path = path.join(
        "pk_samples",
        "sort_data_table",
        "SortedAccidents.txt",
    )

    # Sort table
    pk.sort_data_table(
        dictionary_file_path, "Accident", accidents_table_path, output_data_table_path
    )


def sort_data_table_expert():
    """Sorts a database by a field other than the default table key

    It is a call to `~.api.sort_data_table` with additional parameters to specify the
    sorting fields.
    """
    # Set the file paths
    accidents_dir = path.join(pk.get_samples_dir(), "AccidentsSummary")
    dictionary_file_path = path.join(accidents_dir, "Accidents.kdic")
    vehicles_table_path = path.join(accidents_dir, "Vehicles.txt")
    output_data_table_path = path.join(
        "pk_samples", "sort_data_table_expert", "SortedVehicles.txt"
    )

    # Sort table. Besides the mandatory parameters, it is specified:
    # - A list containing the sorting fields
    pk.sort_data_table(
        dictionary_file_path,
        "Vehicle",
        vehicles_table_path,
        output_data_table_path,
        sort_variables=["AccidentId", "VehicleId"],
    )


def extract_keys_from_data_table():
    """Extracts the keys from a database

    It is a call to `~.api.extract_keys_from_data_table` with only its mandatory
    parameters.

    Pre-requisite: the database must be sorted by its key.
    """
    # Set the file paths
    splice_dir = path.join(pk.get_samples_dir(), "SpliceJunction")
    dictionary_file_path = path.join(splice_dir, "SpliceJunction.kdic")
    data_table_path = path.join(splice_dir, "SpliceJunctionDNA.txt")
    output_data_table_path = path.join(
        "pk_samples",
        "extract_keys_from_data_table",
        "KeysSpliceJunction.txt",
    )

    # Extract keys from table "SpliceJunctionDNA" to the output table
    pk.extract_keys_from_data_table(
        dictionary_file_path,
        "SpliceJunctionDNA",
        data_table_path,
        output_data_table_path,
    )


def train_coclustering():
    """Trains a coclustering model in the simplest way possible

    It is a call to `~.api.train_coclustering` with only its mandatory parameters.
    """
    # Set the file paths
    splice_dir = path.join(pk.get_samples_dir(), "SpliceJunction")
    dictionary_file_path = path.join(splice_dir, "SpliceJunction.kdic")
    data_table_path = path.join(splice_dir, "SpliceJunctionDNA.txt")
    results_dir = path.join("pk_samples", "train_coclustering")

    # Train a coclustering model for variables "SampleId" and "Char"
    coclustering_file_path = pk.train_coclustering(
        dictionary_file_path,
        "SpliceJunctionDNA",
        data_table_path,
        ["SampleId", "Char"],
        results_dir,
    )
    print("Coclustering file available at " + coclustering_file_path)


def simplify_coclustering():
    """Simplifies a coclustering model while preserving 80% of its information"""
    # Set the file paths
    splice_dir = path.join(pk.get_samples_dir(), "SpliceJunction")
    dictionary_file_path = path.join(splice_dir, "SpliceJunction.kdic")
    data_table_path = path.join(splice_dir, "SpliceJunctionDNA.txt")
    results_dir = path.join("pk_samples", "simplify_coclustering")
    coclustering_file_path = path.join(results_dir, "Coclustering.khc")
    simplified_coclustering_file_name = "simplified_coclustering.khc"

    # Train coclustering model for variables "SampleId" and "Char"
    pk.train_coclustering(
        dictionary_file_path,
        "SpliceJunctionDNA",
        data_table_path,
        ["SampleId", "Char"],
        results_dir,
    )

    # Simplify the trained coclustering with the constraints
    # - maximum information preserved: 80%
    # - maximum total parts number: 4
    pk.simplify_coclustering(
        coclustering_file_path,
        simplified_coclustering_file_name,
        results_dir,
        max_preserved_information=80,
        max_total_parts=4,
    )


def extract_clusters():
    """Extract the clusters' id, members, frequencies and typicalities into a file"""
    # Set the file paths
    splice_dir = path.join(pk.get_samples_dir(), "SpliceJunction")
    dictionary_file_path = path.join(splice_dir, "SpliceJunction.kdic")
    data_table_path = path.join(splice_dir, "SpliceJunctionDNA.txt")
    results_dir = path.join("pk_samples", "extract_clusters")
    coclustering_file_path = path.join(results_dir, "Coclustering.khc")
    clusters_file_path = path.join(results_dir, "extracted_clusters.txt")

    # Train a coclustering model for variables "SampleId" and "Char"
    pk.train_coclustering(
        dictionary_file_path,
        "SpliceJunctionDNA",
        data_table_path,
        ["SampleId", "Char"],
        results_dir,
    )

    # Extract clusters
    pk.extract_clusters(coclustering_file_path, "Char", clusters_file_path)


def deploy_coclustering():
    """Deploys an "individual-variable" coclustering"""
    # Set the initial file paths
    splice_dir = path.join(pk.get_runner().samples_dir, "SpliceJunction")
    data_table_path = path.join(splice_dir, "SpliceJunctionDNA.txt")
    dictionary_file_path = path.join(splice_dir, "SpliceJunction.kdic")
    results_dir = path.join("pk_samples", "deploy_coclustering")
    coclustering_file_path = path.join(results_dir, "Coclustering.khc")

    # Train a coclustering model for variables "SampleId" and "Char"
    pk.train_coclustering(
        dictionary_file_path,
        "SpliceJunctionDNA",
        data_table_path,
        ["SampleId", "Char"],
        results_dir,
    )

    # Deploy "Char" clusters in the training database
    pk.deploy_coclustering(
        dictionary_file_path,
        "SpliceJunctionDNA",
        data_table_path,
        coclustering_file_path,
        ["SampleId"],
        "Char",
        results_dir,
        header_line=True,
    )


def deploy_coclustering_expert():
    """Deploys an "individual-variable" coclustering step-by-step

    The `.api.prepare_coclustering_deployment` method is called twice to prepare the
    deployment at two granularity levels. Then, the model is deployed and the respective
    deployment dictionary is built.

    This is one of the most complex workflows of the Khiops suite.
    """
    # Set the initial file paths
    splice_dir = path.join(pk.get_samples_dir(), "SpliceJunction")
    dictionary_file_path = path.join(splice_dir, "SpliceJunction.kdic")
    data_table_path = path.join(splice_dir, "SpliceJunction.txt")
    secondary_data_table_path = path.join(splice_dir, "SpliceJunctionDNA.txt")
    results_dir = path.join("pk_samples", "deploy_coclustering_expert")
    coclustering_file_path = path.join(results_dir, "Coclustering.khc")

    # Train a coclustering model for variables "SampleId" and "Char"
    print("train coclustering on SpliceJunctionDNA")
    pk.train_coclustering(
        dictionary_file_path,
        "SpliceJunctionDNA",
        secondary_data_table_path,
        ["SampleId", "Char"],
        results_dir,
    )

    print("prepare_coclustering_deployment")
    # The input dictionary is extended with new coclustering based variables
    pk.prepare_coclustering_deployment(
        dictionary_file_path,
        "SpliceJunction",
        coclustering_file_path,
        "DNA",
        "SampleId",
        results_dir,
    )
    augmented_dictionary_file_path = path.join(results_dir, "Coclustering.kdic")

    print("prepare_coclustering_deployment with at most two clusters")
    # Extend the already extended dictionary with the new variables from a simplified CC
    pk.prepare_coclustering_deployment(
        augmented_dictionary_file_path,
        "SpliceJunction",
        coclustering_file_path,
        "DNA",
        "SampleId",
        results_dir,
        results_prefix="Reaugmented",
        variables_prefix="C2_",
        max_part_numbers={"SampleId": 2},
    )

    reaugmented_dictionary_file_path = path.join(
        results_dir, "ReaugmentedCoclustering.kdic"
    )
    output_data_table_path = path.join(results_dir, "TransferredSpliceJunction.txt")

    # Deploy the coclustering with the extended dictionary
    print("deploy_model with the new coclustering based variables")
    pk.deploy_model(
        reaugmented_dictionary_file_path,
        "SpliceJunction",
        data_table_path,
        output_data_table_path,
        additional_data_tables={"SpliceJunction`DNA": secondary_data_table_path},
    )

    deployed_dictionary_file_path = path.join(
        results_dir, "Transferred_Coclustering.kdic"
    )
    print("build_deployed_dictionary to get the new dictionary")
    pk.build_deployed_dictionary(
        reaugmented_dictionary_file_path,
        "SpliceJunction",
        deployed_dictionary_file_path,
    )


def scenario_prologue():
    """Trains a simple model with a prologue written in the Khiops scenario language

    .. note::
        This is an **advanced** feature.
    """
    # Set the file paths
    dictionary_file_path = path.join(pk.get_samples_dir(), "Adult", "Adult.kdic")
    data_table_path = path.join(pk.get_samples_dir(), "Adult", "Adult.txt")
    results_dir = path.join("pk_samples", "scenario_prologue")

    # Set the maximum memory "by hand" with an scenario prologue
    pk.get_runner().scenario_prologue = """
        // Max memory 2000 mb
        AnalysisSpec.SystemParameters.MemoryLimit 2000
        """

    # Train the predictor
    pk.train_predictor(
        dictionary_file_path,
        "Adult",
        data_table_path,
        "class",
        results_dir,
        max_trees=0,
    )


def build_deployed_dictionary():
    """Builds a dictionary file to read the output table of a deployed model"""
    # Set the file paths
    dictionary_file_path = path.join(pk.get_samples_dir(), "Iris", "Iris.kdic")
    data_table_path = path.join(pk.get_samples_dir(), "Iris", "Iris.txt")
    results_dir = path.join("pk_samples", "build_deployed_dictionary")
    deployed_dictionary_file_path = path.join(results_dir, "SNB_Iris_deployed.kdic")

    # Train the predictor
    _, modeling_dictionary_file_path = pk.train_predictor(
        dictionary_file_path,
        "Iris",
        data_table_path,
        "Class",
        results_dir,
        max_trees=0,
    )

    # Build the dictionary to read the output of the predictor dictionary file
    # It will contain the columns of the table generated by deploying the model
    pk.build_deployed_dictionary(
        modeling_dictionary_file_path,
        "SNB_Iris",
        deployed_dictionary_file_path,
    )

    # Print the deployed dictionary
    with open(deployed_dictionary_file_path) as deployed_dictionary_file:
        for line in deployed_dictionary_file:
            print(line, end="")


exported_samples = [
    get_khiops_version,
    build_dictionary_from_data_table,
    detect_data_table_format,
    check_database,
    export_dictionary_files,
    train_predictor,
    train_predictor_file_paths,
    train_predictor_error_handling,
    train_predictor_mt,
    train_predictor_mt_with_specific_rules,
    train_predictor_mt_snowflake,
    train_predictor_with_train_percentage,
    train_predictor_with_trees,
    train_predictor_with_pairs,
    train_predictor_with_multiple_parameters,
    train_predictor_detect_format,
    train_predictor_with_cross_validation,
    multiple_train_predictor,
    evaluate_predictor,
    access_predictor_evaluation_report,
    train_recoder,
    train_recoder_with_multiple_parameters,
    train_recoder_mt_flatten,
    deploy_model,
    deploy_model_mt,
    deploy_model_mt_snowflake,
    deploy_model_expert,
    deploy_classifier_for_metrics,
    deploy_regressor_for_metrics,
    sort_data_table,
    sort_data_table_expert,
    extract_keys_from_data_table,
    train_coclustering,
    simplify_coclustering,
    extract_clusters,
    deploy_coclustering,
    deploy_coclustering_expert,
    scenario_prologue,
    build_deployed_dictionary,
]


def execute_samples(args):
    """Executes all non-interactive samples"""
    # Create the results directory if it does not exist
    if not path.isdir("./pk_samples"):
        os.mkdir("./pk_samples")

    # Set the user-defined samples dir if any
    if args.samples_dir is not None:
        pk.get_runner().samples_dir = args.samples_dir

    # Filter the samples according to the options
    if args.include is not None:
        execution_samples = filter_samples(
            exported_samples, args.include, args.exact_match
        )
    else:
        execution_samples = exported_samples

    # Print the execution title
    if execution_samples:
        print(f"pyKhiops {pykhiops.__version__} running on Khiops ", end="")
        print(f"{pk.get_khiops_version()}\n")
        print(f"Sample datasets location: {pk.get_samples_dir()}")
        print(f"{len(execution_samples)} sample(s) to execute\n")

        for sample in execution_samples:
            print(f">>> Executing samples.{sample.__name__}")
            sample.__call__()
            print("> Done\n")

        print("*** Samples run! ***")

    else:
        print("*** No samples to run ***")


def filter_samples(sample_list, include, exact_match):
    """Filter the samples according to the command line options"""
    filtered_samples = []
    for sample in sample_list:
        for sample_name in include:
            if (exact_match and sample_name == sample.__name__) or (
                not exact_match and sample_name in sample.__name__
            ):
                filtered_samples.append(sample)

    return filtered_samples


# Run the samples if executed as a script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python samples.py",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Examples of use of the pyKhiops library",
    )
    parser.add_argument(
        "-d",
        "--samples-dir",
        metavar="URI",
        help="Location of the Khiops 'samples' directory",
    )
    parser.add_argument(
        "-i", "--include", nargs="*", help="Executes only the tests matching this list"
    )
    parser.add_argument(
        "-e",
        "--exact-match",
        action="store_true",
        help="Matches with --include are exact",
    )
    execute_samples(parser.parse_args())
