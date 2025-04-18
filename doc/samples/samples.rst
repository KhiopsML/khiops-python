:orphan:

.. currentmodule:: samples

Samples core
============

The code snippets on this page demonstrate the basic use of the :py:mod:`khiops.core` module.

Script and Jupyter notebook
---------------------------
The samples in this page are also available as:

- :download:`Python script <../../khiops/samples/samples.py>`
- :download:`Jupyter notebook <../../khiops/samples/samples.ipynb>`

Setup
-----
First make sure you have installed the sample datasets. In a configured
conda shell (ex. *Anaconda Prompt* in Windows) execute:

.. code-block:: shell

    kh-download-datasets

If that doesn't work open a python console and execute:

.. code-block:: python

    from khiops.tools import download_datasets
    download_datasets()


Samples
-------

.. autofunction:: get_khiops_version
.. code-block:: python

    print(f"Khiops version: {kh.get_khiops_version()}")
.. autofunction:: build_dictionary_from_data_table
.. code-block:: python

    # Imports
    import os
    from khiops import core as kh

    # Set the file paths
    data_table_path = os.path.join(kh.get_samples_dir(), "Adult", "Adult.txt")
    dictionary_name = "AutoAdult"
    dictionary_file_path = os.path.join(
        "kh_samples", "build_dictionary_from_data_table", "AutoAdult.kdic"
    )

    # Create the dictionary from the data table
    kh.build_dictionary_from_data_table(
        data_table_path, dictionary_name, dictionary_file_path
    )
.. autofunction:: detect_data_table_format
.. code-block:: python

    # Imports
    import os
    from khiops import core as kh

    # Set the file paths
    data_table_path = os.path.join(kh.get_samples_dir(), "Adult", "Adult.txt")
    dictionary_file_path = os.path.join(kh.get_samples_dir(), "Adult", "Adult.kdic")
    output_dir = os.path.join("kh_samples", "detect_data_table_format")
    transformed_data_table_path = os.path.join(output_dir, "AdultWithAnotherFormat.txt")

    # Create the output directory
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # Detect the format of the table
    format_spec = kh.detect_data_table_format(data_table_path)
    print("Format specification (header_line, field_separator)")
    print("Format detected on original table:", format_spec)

    # Make a deployment to change the format of the data table
    kh.deploy_model(
        dictionary_file_path,
        "Adult",
        data_table_path,
        transformed_data_table_path,
        output_header_line=False,
        output_field_separator=",",
    )

    # Detect the new format of the table without a dictionary file
    format_spec = kh.detect_data_table_format(transformed_data_table_path)
    print("Format detected on reformatted table:", format_spec)

    # Detect the new format of the table with a dictionary file
    format_spec = kh.detect_data_table_format(
        transformed_data_table_path,
        dictionary_file_path_or_domain=dictionary_file_path,
        dictionary_name="Adult",
    )
    print("Format detected (with dictionary file) on reformatted table:", format_spec)
.. autofunction:: check_database
.. code-block:: python

    # Imports
    import os
    from khiops import core as kh

    # Set the file paths
    dictionary_file_path = os.path.join(kh.get_samples_dir(), "Adult", "Adult.kdic")
    data_table_path = os.path.join(kh.get_samples_dir(), "Adult", "Adult.txt")
    log_file = os.path.join("kh_samples", "check_database", "check_database.log")

    # Check the database
    kh.check_database(
        dictionary_file_path,
        "Adult",
        data_table_path,
        log_file_path=log_file,
        max_messages=50,
    )
.. autofunction:: export_dictionary_files
.. code-block:: python

    # Imports
    import os
    from khiops import core as kh

    # Set the file paths
    dictionary_file_path = os.path.join(kh.get_samples_dir(), "Adult", "Adult.kdic")
    output_dir = os.path.join("kh_samples", "export_dictionary_file")
    output_dictionary_file_path = os.path.join(output_dir, "ModifiedAdult.kdic")
    output_dictionary_json_path = os.path.join(output_dir, "ModifiedAdult.kdicj")
    alt_output_dictionary_json_path = os.path.join(output_dir, "AltModifiedAdult.kdicj")

    # Load the dictionary domain from initial dictionary file
    # Then obtain the "Adult" dictionary within
    domain = kh.read_dictionary_file(dictionary_file_path)
    dictionary = domain.get_dictionary("Adult")

    # Set some of its variables to unused
    fnlwgt_variable = dictionary.get_variable("fnlwgt")
    fnlwgt_variable.used = False
    label_variable = dictionary.get_variable("Label")
    label_variable.used = False

    # Create output directory if necessary
    if not os.path.exists("kh_samples"):
        os.mkdir("kh_samples")
        os.mkdir(output_dir)
    else:
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

    # Export to kdic
    domain.export_khiops_dictionary_file(output_dictionary_file_path)

    # Export to kdicj either from the domain or from a kdic file
    # Requires a Khiops execution, that's why it is not a method of DictionaryDomain
    kh.export_dictionary_as_json(domain, output_dictionary_json_path)
    kh.export_dictionary_as_json(
        output_dictionary_file_path, alt_output_dictionary_json_path
    )
.. autofunction:: train_predictor
.. code-block:: python

    # Imports
    import os
    from khiops import core as kh

    # Set the file paths
    dictionary_file_path = os.path.join(kh.get_samples_dir(), "Adult", "Adult.kdic")
    data_table_path = os.path.join(kh.get_samples_dir(), "Adult", "Adult.txt")
    analysis_report_file_path = os.path.join(
        "kh_samples", "train_predictor", "AnalysisReport.khj"
    )

    # Train the predictor
    kh.train_predictor(
        dictionary_file_path,
        "Adult",
        data_table_path,
        "class",
        analysis_report_file_path,
        max_trees=0,
    )
.. autofunction:: train_predictor_file_paths
.. code-block:: python

    # Imports
    import os
    from khiops import core as kh

    # Set the file paths
    dictionary_file_path = os.path.join(kh.get_samples_dir(), "Adult", "Adult.kdic")
    data_table_path = os.path.join(kh.get_samples_dir(), "Adult", "Adult.txt")
    report_file_path = os.path.join(
        "kh_samples", "train_predictor_file_paths", "AnalysisResults.khj"
    )

    # Train the predictor
    _, modeling_dictionary_file_path = kh.train_predictor(
        dictionary_file_path,
        "Adult",
        data_table_path,
        "class",
        report_file_path,
        max_trees=0,
    )
    print("Reports file available at " + report_file_path)
    print("Modeling dictionary file available at " + modeling_dictionary_file_path)

    # If you have Khiops Visualization installed you may open the report as follows
    # kh.visualize_report(report_file_path)
.. autofunction:: train_predictor_text
.. code-block:: python

    # Imports
    import os
    from khiops import core as kh

    # Set the file paths
    dictionary_file_path = os.path.join(
        kh.get_samples_dir(), "NegativeAirlineTweets", "NegativeAirlineTweets.kdic"
    )
    data_table_path = os.path.join(
        kh.get_samples_dir(), "NegativeAirlineTweets", "NegativeAirlineTweets.txt"
    )
    report_file_path = os.path.join(
        "kh_samples", "train_predictor_text", "AnalysisResults.khj"
    )

    # Train the predictor
    kh.train_predictor(
        dictionary_file_path,
        "FlightNegativeTweets",
        data_table_path,
        "negativereason",
        report_file_path,
        max_trees=5,
        max_text_features=1000,
        text_features="words",
    )
.. autofunction:: train_predictor_error_handling
.. code-block:: python

    # Imports
    import os
    from khiops import core as kh

    # Set the file paths with a nonexistent dictionary file
    dictionary_file_path = "NONEXISTENT_DICTIONARY_FILE.kdic"
    data_table_path = os.path.join(kh.get_samples_dir(), "Adult", "Adult.txt")
    output_dir = os.path.join("kh_samples", "train_predictor_error_handling")
    report_file_path = os.path.join(output_dir, "AnalysisResults.khj")
    log_file_path = os.path.join(output_dir, "khiops.log")
    scenario_path = os.path.join(output_dir, "scenario._kh")

    # Train the predictor and handle the error
    try:
        kh.train_predictor(
            dictionary_file_path,
            "Adult",
            data_table_path,
            "class",
            report_file_path,
            trace=True,
            log_file_path=log_file_path,
            output_scenario_path=scenario_path,
        )
    except kh.KhiopsRuntimeError as error:
        print("Khiops training failed! Below the KhiopsRuntimeError message:")
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
.. autofunction:: train_predictor_mt
.. code-block:: python

    # Imports
    import os
    from khiops import core as kh

    # Set the file paths
    accidents_dir = os.path.join(kh.get_samples_dir(), "AccidentsSummary")
    dictionary_file_path = os.path.join(accidents_dir, "Accidents.kdic")
    accidents_table_path = os.path.join(accidents_dir, "Accidents.txt")
    vehicles_table_path = os.path.join(accidents_dir, "Vehicles.txt")
    report_file_path = os.path.join(
        "kh_samples", "train_predictor_mt", "AnalysisResults.khj"
    )

    # Train the predictor. Besides the mandatory parameters, we specify:
    # - A python dictionary linking data paths to file paths for non-root tables
    # - To not construct any decision tree
    # The default number of automatic features is 100
    kh.train_predictor(
        dictionary_file_path,
        "Accident",
        accidents_table_path,
        "Gravity",
        report_file_path,
        additional_data_tables={"Vehicles": vehicles_table_path},
        max_trees=0,
    )
.. autofunction:: train_predictor_mt_with_specific_rules
.. code-block:: python

    # Imports
    import os
    from khiops import core as kh

    # Set the file paths
    accidents_dir = os.path.join(kh.get_samples_dir(), "AccidentsSummary")
    dictionary_file_path = os.path.join(accidents_dir, "Accidents.kdic")
    accidents_table_path = os.path.join(accidents_dir, "Accidents.txt")
    vehicles_table_path = os.path.join(accidents_dir, "Vehicles.txt")
    report_file_path = os.path.join(
        "kh_samples",
        "train_predictor_mt_with_specific_rules",
        "AnalysisResults.khj",
    )

    # Train the predictor. Besides the mandatory parameters, it is specified:
    # - A python dictionary linking data paths to file paths for non-root tables
    # - The maximum number of aggregate variables to construct (1000)
    # - The construction rules allowed to automatically create aggregates
    # - To not construct any decision tree
    kh.train_predictor(
        dictionary_file_path,
        "Accident",
        accidents_table_path,
        "Gravity",
        report_file_path,
        additional_data_tables={"Vehicles": vehicles_table_path},
        max_constructed_variables=1000,
        construction_rules=["TableMode", "TableSelection"],
        max_trees=0,
    )
.. autofunction:: train_predictor_mt_snowflake
.. code-block:: python

    # Imports
    import os
    from khiops import core as kh

    # Set the file paths
    accidents_dir = os.path.join(kh.get_samples_dir(), "Accidents")
    dictionary_file_path = os.path.join(accidents_dir, "Accidents.kdic")
    accidents_table_path = os.path.join(accidents_dir, "Accidents.txt")
    vehicles_table_path = os.path.join(accidents_dir, "Vehicles.txt")
    users_table_path = os.path.join(accidents_dir, "Users.txt")
    places_table_path = os.path.join(accidents_dir, "Places.txt")
    report_file_path = os.path.join(
        "kh_samples", "train_predictor_mt_snowflake", "AnalysisResults.khj"
    )

    # Train the predictor. Besides the mandatory parameters, we specify:
    # - A python dictionary linking data paths to file paths for non-root tables
    # - To not construct any decision tree
    # The default number of automatic features is 100
    kh.train_predictor(
        dictionary_file_path,
        "Accident",
        accidents_table_path,
        "Gravity",
        report_file_path,
        additional_data_tables={
            "Vehicles": vehicles_table_path,
            "Vehicles/Users": users_table_path,
            "Place": places_table_path,
        },
        max_trees=0,
    )
.. autofunction:: train_predictor_with_train_percentage
.. code-block:: python

    # Imports
    import os
    from khiops import core as kh

    # Set the file paths
    dictionary_file_path = os.path.join(kh.get_samples_dir(), "Adult", "Adult.kdic")
    data_table_path = os.path.join(kh.get_samples_dir(), "Adult", "Adult.txt")
    report_file_path = os.path.join(
        "kh_samples",
        "train_predictor_with_train_percentage",
        "P90_AnalysisResults.khj",
    )

    # Train the predictor. Besides the mandatory parameters, it is specified:
    # - A 90% sampling rate for the training dataset
    # - Set the test dataset as the complement of the training dataset (10%)
    # - No trees
    kh.train_predictor(
        dictionary_file_path,
        "Adult",
        data_table_path,
        "class",
        report_file_path,
        sample_percentage=90,
        use_complement_as_test=True,
        max_trees=0,
    )
.. autofunction:: train_predictor_with_trees
.. code-block:: python

    # Imports
    import os
    from khiops import core as kh

    # Set the file paths
    dictionary_file_path = os.path.join(kh.get_samples_dir(), "Letter", "Letter.kdic")
    data_table_path = os.path.join(kh.get_samples_dir(), "Letter", "Letter.txt")
    report_file_path = os.path.join(
        "kh_samples", "train_predictor_with_trees", "P80_AnalysisResults.khj"
    )

    # Train the predictor with at most 15 trees (default 10)
    kh.train_predictor(
        dictionary_file_path,
        "Letter",
        data_table_path,
        "lettr",
        report_file_path,
        sample_percentage=80,
        use_complement_as_test=True,
        max_trees=15,
    )
.. autofunction:: train_predictor_with_pairs
.. code-block:: python

    # Imports
    import os
    from khiops import core as kh

    # Set the file paths
    dictionary_file_path = os.path.join(kh.get_samples_dir(), "Adult", "Adult.kdic")
    data_table_path = os.path.join(kh.get_samples_dir(), "Adult", "Adult.txt")
    report_file_path = os.path.join(
        "kh_samples", "train_predictor_with_pairs", "AnalysisResults.khj"
    )

    # Train the predictor with at most 10 pairs as follows:
    # - Include pairs age-race and capital_gain-capital_loss
    # - Include all possible pairs having relationship as component
    kh.train_predictor(
        dictionary_file_path,
        "Adult",
        data_table_path,
        "class",
        report_file_path,
        use_complement_as_test=True,
        max_trees=0,
        max_pairs=10,
        specific_pairs=[
            ("age", "race"),
            ("capital_gain", "capital_loss"),
            ("relationship", ""),
        ],
    )
.. autofunction:: train_predictor_with_multiple_parameters
.. code-block:: python

    # Imports
    import os
    from khiops import core as kh

    # Set the file paths
    dictionary_file_path = os.path.join(kh.get_samples_dir(), "Adult", "Adult.kdic")
    data_table_path = os.path.join(kh.get_samples_dir(), "Adult", "Adult.txt")
    output_dir = os.path.join("kh_samples", "train_predictor_with_multiple_parameters")
    report_file_path = os.path.join(output_dir, "AnalysisResults.khj")
    output_script_path = os.path.join(output_dir, "output_scenario._kh")
    log_path = os.path.join(output_dir, "log.txt")

    # Train the predictor. Besides the mandatory parameters, we specify:
    # - The value "more" as main target value
    # - The output Khiops script file location (generic)
    # - The log file location (generic)
    # - The maximum memory used, set to 1000 MB
    # - To show the debug trace (generic)
    kh.train_predictor(
        dictionary_file_path,
        "Adult",
        data_table_path,
        "class",
        report_file_path,
        main_target_value="more",
        output_scenario_path=output_script_path,
        log_file_path=log_path,
        memory_limit_mb=1000,
        trace=True,
    )
.. autofunction:: train_predictor_detect_format
.. code-block:: python

    # Imports
    import os
    from khiops import core as kh

    # Set the file paths
    dictionary_file_path = os.path.join(kh.get_samples_dir(), "Iris", "Iris.kdic")
    data_table_path = os.path.join(kh.get_samples_dir(), "Iris", "Iris.txt")
    output_dir = os.path.join("kh_samples", "train_predictor_detect_format")
    transformed_data_table_path = os.path.join(output_dir, "TransformedIris.txt")
    report_file_path = os.path.join(output_dir, "AnalysisResults.khj")

    # Transform the database format from header_line=True and field_separator=TAB
    # to header_line=False and field_separator=","
    # See the deploy_model examples below for more details
    kh.deploy_model(
        dictionary_file_path,
        "Iris",
        data_table_path,
        transformed_data_table_path,
        output_header_line=False,
        output_field_separator=",",
    )

    # Try to learn with the old format
    try:
        kh.train_predictor(
            dictionary_file_path,
            "Iris",
            transformed_data_table_path,
            "Class",
            report_file_path,
            header_line=True,
            field_separator="",
        )
    except kh.KhiopsRuntimeError as error:
        print(
            "This failed because of a bad data table format spec. "
            + "Below the KhiopsRuntimeError message"
        )
        print(error)

    # Train without specifyng the format (detect_format is True by default)
    kh.train_predictor(
        dictionary_file_path,
        "Iris",
        transformed_data_table_path,
        "Class",
        report_file_path,
    )
.. autofunction:: train_predictor_with_cross_validation
.. code-block:: python

    # Imports
    import math
    import os
    from khiops import core as kh

    # Set the file paths
    dictionary_file_path = os.path.join(kh.get_samples_dir(), "Adult", "Adult.kdic")
    data_table_path = os.path.join(kh.get_samples_dir(), "Adult", "Adult.txt")
    output_dir = os.path.join("kh_samples", "train_predictor_with_cross_validation")
    fold_dictionary_file_path = os.path.join(output_dir, "AdultWithFolding.kdic")

    # Create the output directory
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # Load the learning dictionary object
    domain = kh.read_dictionary_file(dictionary_file_path)
    dictionary = domain.get_dictionary("Adult")

    # Add a random fold index variable to the learning dictionary
    fold_number = 5
    fold_index_variable = kh.Variable()
    fold_index_variable.name = "FoldIndex"
    fold_index_variable.type = "Numerical"
    fold_index_variable.used = False
    fold_index_variable.rule = "Ceil(Product(" + str(fold_number) + ",  Random()))"
    dictionary.add_variable(fold_index_variable)

    # Add variables that indicate if the instance is in the train dataset:
    for fold_index in range(1, fold_number + 1):
        is_in_train_dataset_variable = kh.Variable()
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
        analysis_report_file_path = os.path.join(
            output_dir, "Fold" + str(fold_index) + "AnalysisResults.khj"
        )
        # Train a model from the sub-dataset where IsInTrainDataset<k> is 1
        _, modeling_dictionary_file_path = kh.train_predictor(
            domain,
            "Adult",
            data_table_path,
            "class",
            analysis_report_file_path,
            sample_percentage=100,
            selection_variable="IsInTrainDataset" + str(fold_index),
            selection_value=1,
            max_trees=0,
        )

        evaluation_report_file_path = os.path.join(
            output_dir, "Fold" + str(fold_index) + "AdultEvaluationResults.khj"
        )
        # Evaluate the resulting model in the subsets where IsInTrainDataset is 0
        test_evaluation_report_path = kh.evaluate_predictor(
            modeling_dictionary_file_path,
            "SNB_Adult",
            data_table_path,
            evaluation_report_file_path,
            sample_percentage=100,
            selection_variable="IsInTrainDataset" + str(fold_index),
            selection_value=0,
        )

        # Obtain the train AUC from the train report and the test AUC from the
        # evaluation report and print them
        train_results = kh.read_analysis_results_file(analysis_report_file_path)
        test_evaluation_results = kh.read_analysis_results_file(test_evaluation_report_path)
        train_auc = train_results.train_evaluation_report.get_snb_performance().auc
        test_auc = test_evaluation_results.evaluation_report.get_snb_performance().auc
        print("\t" + str(fold_index) + "\t" + str(train_auc) + "\t" + str(test_auc))

        # Store the train and test AUCs in arrays
        train_aucs.append(train_auc)
        test_aucs.append(test_auc)

    # Print the mean +- error aucs for both train and test
    mean_train_auc = sum(train_aucs) / fold_number
    squared_error_train_aucs = [(auc - mean_train_auc) ** 2 for auc in train_aucs]
    sd_train_auc = math.sqrt(sum(squared_error_train_aucs) / (fold_number - 1))

    mean_test_auc = sum(test_aucs) / fold_number
    squared_error_test_aucs = [(auc - mean_test_auc) ** 2 for auc in test_aucs]
    sd_test_auc = math.sqrt(sum(squared_error_test_aucs) / (fold_number - 1))

    print("final auc")
    print("train auc: " + str(mean_train_auc) + " +- " + str(sd_train_auc))
    print("test  auc: " + str(mean_test_auc) + " +- " + str(sd_test_auc))
.. autofunction:: interpret_predictor
.. code-block:: python

    # Imports
    import os
    from khiops import core as kh

    dictionary_file_path = os.path.join(kh.get_samples_dir(), "Adult", "Adult.kdic")
    data_table_path = os.path.join(kh.get_samples_dir(), "Adult", "Adult.txt")
    output_dir = os.path.join("kh_samples", "interpret_predictor")
    analysis_report_file_path = os.path.join(output_dir, "AnalysisResults.khj")
    interpretor_file_path = os.path.join(output_dir, "InterpretationModel.kdic")

    # Build prediction model
    _, predictor_file_path = kh.train_predictor(
        dictionary_file_path,
        "Adult",
        data_table_path,
        "class",
        analysis_report_file_path,
    )

    # Build interpretation model
    kh.interpret_predictor(predictor_file_path, "SNB_Adult", interpretor_file_path)

    print(f"The interpretation model is '{interpretor_file_path}'")
.. autofunction:: multiple_train_predictor
.. code-block:: python

    # Imports
    import os
    from khiops import core as kh


    def display_test_results(json_result_file_path):
        """Display some of the training results"""
        results = kh.read_analysis_results_file(json_result_file_path)
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
    dictionary_file_path = os.path.join(kh.get_samples_dir(), "Adult", "Adult.kdic")
    data_table_path = os.path.join(kh.get_samples_dir(), "Adult", "Adult.txt")
    output_dir = os.path.join("kh_samples", "multiple_train_predictor")
    report_file_path = os.path.join(output_dir, "AnalysisResults.khj")

    # Read the dictionary file to obtain an instance of class Dictionary
    dictionary_domain = kh.read_dictionary_file(dictionary_file_path)
    dictionary = dictionary_domain.get_dictionary("Adult")

    # Train a SNB model using all the variables
    print("\t#vars\ttrain auc\ttest auc")
    kh.train_predictor(
        dictionary_file_path,
        "Adult",
        data_table_path,
        "class",
        report_file_path,
        sample_percentage=70,
        use_complement_as_test=True,
        max_trees=0,
    )
    display_test_results(report_file_path)

    # Read results to obtain the variables sorted by decreasing Level
    analysis_results = kh.read_analysis_results_file(report_file_path)
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
        report_file_path = os.path.join(
            output_dir, f"V{variable_number - 1 - i}_AnalysisResults.khj"
        )
        kh.train_predictor(
            dictionary_domain,
            "Adult",
            data_table_path,
            "class",
            report_file_path,
            sample_percentage=70,
            use_complement_as_test=True,
            max_trees=0,
        )

        # Show a preview of the results
        display_test_results(report_file_path)
.. autofunction:: evaluate_predictor
.. code-block:: python

    # Imports
    import os
    from khiops import core as kh

    # Set the file paths
    dictionary_file_path = os.path.join(kh.get_samples_dir(), "Adult", "Adult.kdic")
    data_table_path = os.path.join(kh.get_samples_dir(), "Adult", "Adult.txt")
    output_dir = os.path.join("kh_samples", "evaluate_predictor")
    analysis_report_file_path = os.path.join(output_dir, "AnalysisResults.khj")

    # Train the predictor
    _, model_dictionary_file_path = kh.train_predictor(
        dictionary_file_path,
        "Adult",
        data_table_path,
        "class",
        analysis_report_file_path,
        max_trees=0,
    )

    evaluation_report_file_path = os.path.join(output_dir, "AdultEvaluationResults.khj")

    # Evaluate the predictor
    kh.evaluate_predictor(
        model_dictionary_file_path,
        "SNB_Adult",
        data_table_path,
        evaluation_report_file_path,
    )
    print("Evaluation report available at " + evaluation_report_file_path)
.. autofunction:: access_predictor_evaluation_report
.. code-block:: python

    # Imports
    import os
    from khiops import core as kh

    # Set the file paths
    dictionary_file_path = os.path.join(kh.get_samples_dir(), "Adult", "Adult.kdic")
    data_table_path = os.path.join(kh.get_samples_dir(), "Adult", "Adult.txt")
    report_file_path = os.path.join(
        "kh_samples", "access_predictor_evaluation_report", "AdultAnalysisReport.khj"
    )

    # Train the SNB predictor and some univariate predictors
    # Note: Evaluation in test is 30% by default
    kh.train_predictor(
        dictionary_file_path,
        "Adult",
        data_table_path,
        "class",
        report_file_path,
        max_trees=0,
    )

    # Obtain the evaluation results
    results = kh.read_analysis_results_file(report_file_path)
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

    # Print metrics for an SNB predictor
    predictor_performance = evaluation_report.get_predictor_performance(
        "Selective Naive Bayes"
    )
    print("\n\nperformance metrics for " + predictor_performance.name)
    for metric_name in predictor_performance.get_metric_names():
        print(metric_name + ": " + str(predictor_performance.get_metric(metric_name)))
.. autofunction:: train_recoder
.. code-block:: python

    # Imports
    import os
    from khiops import core as kh

    # Set the file paths
    dictionary_file_path = os.path.join(kh.get_samples_dir(), "Adult", "Adult.kdic")
    data_table_path = os.path.join(kh.get_samples_dir(), "Adult", "Adult.txt")
    report_file_path = os.path.join("kh_samples", "train_recoder", "AnalysisResults.khj")

    # Train the recoder model
    kh.train_recoder(
        dictionary_file_path, "Adult", data_table_path, "class", report_file_path
    )
.. autofunction:: train_recoder_with_multiple_parameters
.. code-block:: python

    # Imports
    import os
    from khiops import core as kh

    # Set the file paths
    dictionary_file_path = os.path.join(kh.get_samples_dir(), "Adult", "Adult.kdic")
    data_table_path = os.path.join(kh.get_samples_dir(), "Adult", "Adult.txt")
    report_file_path = os.path.join(
        "kh_samples",
        "train_recoder_with_multiple_parameters",
        "AnalysisResults.khj",
    )

    # Train the recoder model
    kh.train_recoder(
        dictionary_file_path,
        "Adult",
        data_table_path,
        "class",
        report_file_path,
        max_pairs=10,
        categorical_recoding_method="part label",
        numerical_recoding_method="part label",
    )
.. autofunction:: train_recoder_mt_flatten
.. code-block:: python

    # Imports
    import os
    from khiops import core as kh

    # Set the file paths
    accidents_dir = os.path.join(kh.get_samples_dir(), "AccidentsSummary")
    dictionary_file_path = os.path.join(accidents_dir, "Accidents.kdic")
    accidents_table_path = os.path.join(accidents_dir, "Accidents.txt")
    vehicles_table_path = os.path.join(accidents_dir, "Vehicles.txt")
    report_file_path = os.path.join(
        "kh_samples", "train_recoder_mt_flatten", "AnalysisResults.khj"
    )

    # Train the recoder. Besides the mandatory parameters, it is specified:
    # - A python dictionary linking data paths to file paths for non-root tables
    # - The maximum number of aggregate variables to construct (1000)
    # - To keep all the created variables independently of their informativeness (level)
    # - To not recode the variables values
    kh.train_recoder(
        dictionary_file_path,
        "Accident",
        accidents_table_path,
        "Gravity",
        report_file_path,
        additional_data_tables={"Vehicles": vehicles_table_path},
        max_constructed_variables=1000,
        informative_variables_only=False,
        categorical_recoding_method="none",
        numerical_recoding_method="none",
        keep_initial_categorical_variables=True,
        keep_initial_numerical_variables=True,
    )
.. autofunction:: deploy_model
.. code-block:: python

    # Imports
    import os
    from khiops import core as kh

    # Set the file paths
    dictionary_file_path = os.path.join(kh.get_samples_dir(), "Adult", "Adult.kdic")
    data_table_path = os.path.join(kh.get_samples_dir(), "Adult", "Adult.txt")
    output_dir = os.path.join("kh_samples", "deploy_model")
    report_file_path = os.path.join(output_dir, "AnalysisResults.khj")
    output_data_table_path = os.path.join(output_dir, "ScoresAdult.txt")

    # Train the predictor
    _, model_dictionary_file_path = kh.train_predictor(
        dictionary_file_path,
        "Adult",
        data_table_path,
        "class",
        report_file_path,
        max_trees=0,
    )

    # Deploy the model on the database
    # It will score it according to the trained predictor
    kh.deploy_model(
        model_dictionary_file_path, "SNB_Adult", data_table_path, output_data_table_path
    )
.. autofunction:: deploy_model_text
.. code-block:: python

    # Imports
    import os
    from khiops import core as kh

    # Set the file paths
    dictionary_file_path = os.path.join(
        kh.get_samples_dir(), "NegativeAirlineTweets", "NegativeAirlineTweets.kdic"
    )
    data_table_path = os.path.join(
        kh.get_samples_dir(), "NegativeAirlineTweets", "NegativeAirlineTweets.txt"
    )
    output_dir = os.path.join("kh_samples", "deploy_model_text")
    report_file_path = os.path.join(output_dir, "AnalysisResults.khj")
    output_data_table_path = os.path.join(output_dir, "ScoresNegativeAirlineTweets.txt")

    # Train the predictor
    _, model_dictionary_file_path = kh.train_predictor(
        dictionary_file_path,
        "FlightNegativeTweets",
        data_table_path,
        "negativereason",
        report_file_path,
        max_trees=5,
        max_text_features=1000,
        text_features="words",
    )

    # Deploy the model on the database
    # It will score it according to the trained predictor
    kh.deploy_model(
        model_dictionary_file_path,
        "SNB_FlightNegativeTweets",
        data_table_path,
        output_data_table_path,
    )
.. autofunction:: deploy_model_mt
.. code-block:: python

    # Imports
    import os
    from khiops import core as kh

    # Set the file paths
    accidents_dir = os.path.join(kh.get_samples_dir(), "AccidentsSummary")
    dictionary_file_path = os.path.join(accidents_dir, "Accidents.kdic")
    accidents_table_path = os.path.join(accidents_dir, "Accidents.txt")
    vehicles_table_path = os.path.join(accidents_dir, "Vehicles.txt")
    output_dir = os.path.join("kh_samples", "deploy_model_mt")
    report_file_path = os.path.join(output_dir, "AnalysisResults.khj")
    output_data_table_path = os.path.join(output_dir, "TransferredAccidents.txt")

    # Train the predictor (see train_predictor_mt for details)
    _, model_dictionary_file_path = kh.train_predictor(
        dictionary_file_path,
        "Accident",
        accidents_table_path,
        "Gravity",
        report_file_path,
        additional_data_tables={"Vehicles": vehicles_table_path},
        max_trees=0,
    )

    # Deploy the model on the database
    # Besides the mandatory parameters, it is specified:
    # - A python dictionary linking data paths to file paths for non-root tables
    kh.deploy_model(
        model_dictionary_file_path,
        "SNB_Accident",
        accidents_table_path,
        output_data_table_path,
        additional_data_tables={"Vehicles": vehicles_table_path},
    )
.. autofunction:: deploy_model_mt_with_interpretation
.. code-block:: python

    # Imports
    import os
    from khiops import core as kh

    # Set the file paths
    accidents_dir = os.path.join(kh.get_samples_dir(), "AccidentsSummary")
    dictionary_file_path = os.path.join(accidents_dir, "Accidents.kdic")
    accidents_table_path = os.path.join(accidents_dir, "Accidents.txt")
    vehicles_table_path = os.path.join(accidents_dir, "Vehicles.txt")
    output_dir = os.path.join("kh_samples", "deploy_model_mt")
    report_file_path = os.path.join(output_dir, "AnalysisResults.khj")
    interpretor_file_path = os.path.join(output_dir, "InterpretationModel.kdic")
    output_data_table_path = os.path.join(output_dir, "InterpretedAccidents.txt")

    # Train the predictor (see train_predictor_mt for details)
    # Add max_evaluated_variables so that an interpretation model can be built
    # (see https://github.com/KhiopsML/khiops/issues/577)
    _, model_dictionary_file_path = kh.train_predictor(
        dictionary_file_path,
        "Accident",
        accidents_table_path,
        "Gravity",
        report_file_path,
        additional_data_tables={"Vehicles": vehicles_table_path},
        max_trees=0,
        max_evaluated_variables=10,
    )

    # Interpret the predictor
    kh.interpret_predictor(
        model_dictionary_file_path,
        "SNB_Accident",
        interpretor_file_path,
        reinforcement_target_value="NonLethal",
    )

    # Deploy the interpretation model on the database
    # Besides the mandatory parameters, it is specified:
    # - A python dictionary linking data paths to file paths for non-root tables
    kh.deploy_model(
        interpretor_file_path,
        "Interpretation_SNB_Accident",
        accidents_table_path,
        output_data_table_path,
        additional_data_tables={"Vehicles": vehicles_table_path},
    )
.. autofunction:: deploy_model_mt_snowflake
.. code-block:: python

    # Imports
    import os
    from khiops import core as kh

    # Set the file paths
    accidents_dir = os.path.join(kh.get_samples_dir(), "Accidents")
    dictionary_file_path = os.path.join(accidents_dir, "Accidents.kdic")
    accidents_table_path = os.path.join(accidents_dir, "Accidents.txt")
    vehicles_table_path = os.path.join(accidents_dir, "Vehicles.txt")
    users_table_path = os.path.join(accidents_dir, "Users.txt")
    places_table_path = os.path.join(accidents_dir, "Places.txt")
    output_dir = os.path.join("kh_samples", "deploy_model_mt_snowflake")
    report_file_path = os.path.join(output_dir, "AnalysisResults.khj")
    output_data_table_path = os.path.join(output_dir, "TransferredAccidents.txt")

    # Train the predictor. Besides the mandatory parameters, we specify:
    # - A python dictionary linking data paths to file paths for non-root tables
    # - To not construct any decision tree
    # The default number of automatic features is 100
    _, model_dictionary_file_path = kh.train_predictor(
        dictionary_file_path,
        "Accident",
        accidents_table_path,
        "Gravity",
        report_file_path,
        additional_data_tables={
            "Vehicles": vehicles_table_path,
            "Vehicles/Users": users_table_path,
            "Place": places_table_path,
        },
        max_trees=0,
    )

    # Deploy the model on the database
    # Besides the mandatory parameters, it is specified:
    # - A python dictionary linking data paths to file paths for non-root tables
    kh.deploy_model(
        model_dictionary_file_path,
        "SNB_Accident",
        accidents_table_path,
        output_data_table_path,
        additional_data_tables={
            "Vehicles": vehicles_table_path,
            "Vehicles/Users": users_table_path,
            "Place": places_table_path,
        },
    )
.. autofunction:: deploy_model_expert
.. code-block:: python

    # Imports
    import os
    from khiops import core as kh

    # Set the file paths
    dictionary_file_path = os.path.join(kh.get_samples_dir(), "Adult", "Adult.kdic")
    data_table_path = os.path.join(kh.get_samples_dir(), "Adult", "Adult.txt")
    output_dir = os.path.join("kh_samples", "deploy_model_expert")
    report_file_path = os.path.join(output_dir, "AnalysisResults.khj")
    output_data_table_path = os.path.join(output_dir, "ScoresAdult.txt")

    # Train the predictor
    _, model_dictionary_file_path = kh.train_predictor(
        dictionary_file_path,
        "Adult",
        data_table_path,
        "class",
        report_file_path,
        max_trees=0,
    )

    # Read the dictionary file to obtain an instance of class Dictionary
    model_domain = kh.read_dictionary_file(model_dictionary_file_path)
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
    kh.deploy_model(model_domain, "SNB_Adult", data_table_path, output_data_table_path)
.. autofunction:: deploy_classifier_for_metrics
.. code-block:: python

    # Imports
    import os
    from khiops import core as kh

    # Set the file paths
    dictionary_file_path = os.path.join(kh.get_samples_dir(), "Adult", "Adult.kdic")
    data_table_path = os.path.join(kh.get_samples_dir(), "Adult", "Adult.txt")
    output_dir = os.path.join("kh_samples", "deploy_classifier_for_metrics")
    report_file_path = os.path.join(output_dir, "AnalysisResults.khj")
    output_data_table_path = os.path.join(output_dir, "ScoresAdult.txt")

    # Train the classifier for the target "class"
    _, modeling_dictionary_file_path = kh.train_predictor(
        dictionary_file_path,
        "Adult",
        data_table_path,
        "class",
        report_file_path,
        max_trees=0,
    )
    # Obtain the scores of the SNB on the test dataset to calculate the PR curve
    kh.deploy_predictor_for_metrics(
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
.. autofunction:: deploy_regressor_for_metrics
.. code-block:: python

    # Imports
    import os
    from khiops import core as kh

    # Set the file paths
    dictionary_file_path = os.path.join(kh.get_samples_dir(), "Adult", "Adult.kdic")
    data_table_path = os.path.join(kh.get_samples_dir(), "Adult", "Adult.txt")
    output_dir = os.path.join("kh_samples", "deploy_regressor_for_metrics")
    report_file_path = os.path.join(output_dir, "AnalysisResults.khj")
    output_data_table_path = os.path.join(output_dir, "TrueAndPredictedAges.txt")

    # Train the regressor for the target "age" (with 20% train to be quick)
    _, modeling_dictionary_file_path = kh.train_predictor(
        dictionary_file_path,
        "Adult",
        data_table_path,
        "age",
        report_file_path,
        sample_percentage=20,
        max_trees=0,
    )

    # Obtain the predicted regression values of the SNB on the test dataset estimate R2
    kh.deploy_predictor_for_metrics(
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
.. autofunction:: sort_data_table
.. code-block:: python

    # Imports
    import os
    from khiops import core as kh

    # Set the file paths
    accidents_dir = os.path.join(kh.get_samples_dir(), "AccidentsSummary")
    dictionary_file_path = os.path.join(accidents_dir, "Accidents.kdic")
    accidents_table_path = os.path.join(accidents_dir, "Accidents.txt")
    output_data_table_path = os.path.join(
        "kh_samples",
        "sort_data_table",
        "SortedAccidents.txt",
    )

    # Sort table
    kh.sort_data_table(
        dictionary_file_path, "Accident", accidents_table_path, output_data_table_path
    )
.. autofunction:: sort_data_table_expert
.. code-block:: python

    # Imports
    import os
    from khiops import core as kh

    # Set the file paths
    accidents_dir = os.path.join(kh.get_samples_dir(), "AccidentsSummary")
    dictionary_file_path = os.path.join(accidents_dir, "Accidents.kdic")
    vehicles_table_path = os.path.join(accidents_dir, "Vehicles.txt")
    output_data_table_path = os.path.join(
        "kh_samples", "sort_data_table_expert", "SortedVehicles.txt"
    )

    # Sort table. Besides the mandatory parameters, it is specified:
    # - A list containing the sorting fields
    kh.sort_data_table(
        dictionary_file_path,
        "Vehicle",
        vehicles_table_path,
        output_data_table_path,
        sort_variables=["AccidentId", "VehicleId"],
    )
.. autofunction:: extract_keys_from_data_table
.. code-block:: python

    # Imports
    import os
    from khiops import core as kh

    # Set the file paths
    splice_dir = os.path.join(kh.get_samples_dir(), "SpliceJunction")
    dictionary_file_path = os.path.join(splice_dir, "SpliceJunction.kdic")
    data_table_path = os.path.join(splice_dir, "SpliceJunctionDNA.txt")
    output_data_table_path = os.path.join(
        "kh_samples",
        "extract_keys_from_data_table",
        "KeysSpliceJunction.txt",
    )

    # Extract keys from table "SpliceJunctionDNA" to the output table
    kh.extract_keys_from_data_table(
        dictionary_file_path,
        "SpliceJunctionDNA",
        data_table_path,
        output_data_table_path,
    )
.. autofunction:: train_coclustering
.. code-block:: python

    # Imports
    import os
    from khiops import core as kh

    # Set the file paths
    splice_dir = os.path.join(kh.get_samples_dir(), "SpliceJunction")
    dictionary_file_path = os.path.join(splice_dir, "SpliceJunction.kdic")
    data_table_path = os.path.join(splice_dir, "SpliceJunctionDNA.txt")
    coclustering_report_path = os.path.join(
        "kh_samples", "train_coclustering", "CoclusteringResults.khcj"
    )

    # Train a coclustering model for variables "SampleId" and "Char"
    kh.train_coclustering(
        dictionary_file_path,
        "SpliceJunctionDNA",
        data_table_path,
        ["SampleId", "Char"],
        coclustering_report_path,
    )
    print(f"Coclustering report file available at {coclustering_report_path}")

    # If you have Khiops Co-Visualization installed you may open the report as follows
    # kh.visualize_report(coclustering_report_path)
.. autofunction:: train_instance_variable_coclustering
.. code-block:: python

    # Imports
    import os
    from khiops import core as kh

    # Set the file paths
    iris_dir = os.path.join(kh.get_samples_dir(), "Iris")
    dictionary_file_path = os.path.join(iris_dir, "Iris.kdic")
    data_table_path = os.path.join(iris_dir, "Iris.txt")
    coclustering_report_path = os.path.join(
        "kh_samples",
        "train_instance_variable_coclustering",
        "CoclusteringResults.khcj",
    )

    # Train a coclustering model for variables "SampleId" and "Char"
    kh.train_instance_variable_coclustering(
        dictionary_file_path,
        "Iris",
        data_table_path,
        coclustering_report_path,
    )
    print(
        "Instance-variable coclustering report file available "
        f"at {coclustering_report_path}"
    )

    # If you have Khiops Co-Visualization installed you may open the report as follows
    # kh.visualize_report(coclustering_report_path)
.. autofunction:: simplify_coclustering
.. code-block:: python

    # Imports
    import os
    from khiops import core as kh

    # Set the file paths
    splice_dir = os.path.join(kh.get_samples_dir(), "SpliceJunction")
    dictionary_file_path = os.path.join(splice_dir, "SpliceJunction.kdic")
    data_table_path = os.path.join(splice_dir, "SpliceJunctionDNA.txt")
    output_dir = os.path.join("kh_samples", "simplify_coclustering")
    coclustering_file_path = os.path.join(output_dir, "Coclustering.khcj")
    simplified_coclustering_file_path = os.path.join(
        output_dir, "simplified_coclustering.khcj"
    )

    # Train coclustering model for variables "SampleId" and "Char"
    kh.train_coclustering(
        dictionary_file_path,
        "SpliceJunctionDNA",
        data_table_path,
        ["SampleId", "Char"],
        coclustering_file_path,
    )

    # Simplify the trained coclustering with the constraints
    # - maximum information preserved: 80%
    # - maximum total parts number: 4
    kh.simplify_coclustering(
        coclustering_file_path,
        simplified_coclustering_file_path,
        max_preserved_information=80,
        max_total_parts=4,
    )
.. autofunction:: extract_clusters
.. code-block:: python

    # Set the file paths
    splice_dir = os.path.join(kh.get_samples_dir(), "SpliceJunction")
    dictionary_file_path = os.path.join(splice_dir, "SpliceJunction.kdic")
    data_table_path = os.path.join(splice_dir, "SpliceJunctionDNA.txt")
    output_dir = os.path.join("kh_samples", "extract_clusters")
    coclustering_file_path = os.path.join(output_dir, "Coclustering.khcj")
    clusters_file_path = os.path.join(output_dir, "extracted_clusters.txt")

    # Train a coclustering model for variables "SampleId" and "Char"
    kh.train_coclustering(
        dictionary_file_path,
        "SpliceJunctionDNA",
        data_table_path,
        ["SampleId", "Char"],
        coclustering_file_path,
    )

    # Extract clusters
    kh.extract_clusters(coclustering_file_path, "Char", clusters_file_path)
.. autofunction:: deploy_coclustering
.. code-block:: python

    # Imports
    import os
    from khiops import core as kh

    # Set the initial file paths
    splice_dir = os.path.join(kh.get_samples_dir(), "SpliceJunction")
    data_table_path = os.path.join(splice_dir, "SpliceJunctionDNA.txt")
    dictionary_file_path = os.path.join(splice_dir, "SpliceJunction.kdic")
    output_dir = os.path.join("kh_samples", "deploy_coclustering")
    coclustering_file_path = os.path.join(output_dir, "Coclustering.khcj")
    coclustering_dictionary_file_path = os.path.join(output_dir, "Coclustering.kdic")
    output_data_table_path = os.path.join(output_dir, "DeployedSpliceJunctionDNA.txt")

    # Train a coclustering model for variables "SampleId" and "Char"
    kh.train_coclustering(
        dictionary_file_path,
        "SpliceJunctionDNA",
        data_table_path,
        ["SampleId", "Char"],
        coclustering_file_path,
    )

    # Deploy "Char" clusters in the training database
    kh.deploy_coclustering(
        dictionary_file_path,
        "SpliceJunctionDNA",
        data_table_path,
        coclustering_file_path,
        ["SampleId"],
        "Char",
        coclustering_dictionary_file_path,
        output_data_table_path,
        header_line=True,
    )
.. autofunction:: deploy_coclustering_expert
.. code-block:: python

    # Imports
    import os
    from khiops import core as kh

    # Set the initial file paths
    splice_dir = os.path.join(kh.get_samples_dir(), "SpliceJunction")
    dictionary_file_path = os.path.join(splice_dir, "SpliceJunction.kdic")
    data_table_path = os.path.join(splice_dir, "SpliceJunction.txt")
    secondary_data_table_path = os.path.join(splice_dir, "SpliceJunctionDNA.txt")
    output_dir = os.path.join("kh_samples", "deploy_coclustering_expert")
    coclustering_file_path = os.path.join(output_dir, "Coclustering.khcj")

    # Train a coclustering model for variables "SampleId" and "Char"
    print("train coclustering on SpliceJunctionDNA")
    kh.train_coclustering(
        dictionary_file_path,
        "SpliceJunctionDNA",
        secondary_data_table_path,
        ["SampleId", "Char"],
        coclustering_file_path,
    )

    print("prepare_coclustering_deployment")
    # The input dictionary is extended with new coclustering based variables
    augmented_dictionary_file_path = os.path.join(output_dir, "Coclustering.kdic")
    kh.prepare_coclustering_deployment(
        dictionary_file_path,
        "SpliceJunction",
        coclustering_file_path,
        "DNA",
        "SampleId",
        augmented_dictionary_file_path,
    )

    print("prepare_coclustering_deployment with at most two clusters")
    # Extend the already extended dictionary with the new variables from a simplified CC
    reaugmented_dictionary_file_path = os.path.join(
        output_dir, "ReaugmentedCoclustering.kdic"
    )
    kh.prepare_coclustering_deployment(
        augmented_dictionary_file_path,
        "SpliceJunction",
        coclustering_file_path,
        "DNA",
        "SampleId",
        reaugmented_dictionary_file_path,
        variables_prefix="C2_",
        max_part_numbers={"SampleId": 2},
    )

    output_data_table_path = os.path.join(output_dir, "TransferredSpliceJunction.txt")

    # Deploy the coclustering with the extended dictionary
    print("deploy_model with the new coclustering based variables")
    kh.deploy_model(
        reaugmented_dictionary_file_path,
        "SpliceJunction",
        data_table_path,
        output_data_table_path,
        additional_data_tables={"DNA": secondary_data_table_path},
    )

    deployed_dictionary_file_path = os.path.join(
        output_dir, "Transferred_Coclustering.kdic"
    )
    print("build_deployed_dictionary to get the new dictionary")
    kh.build_deployed_dictionary(
        reaugmented_dictionary_file_path,
        "SpliceJunction",
        deployed_dictionary_file_path,
    )
.. autofunction:: scenario_prologue
.. code-block:: python

    # Imports
    import os
    from khiops import core as kh

    # Set the file paths
    dictionary_file_path = os.path.join(kh.get_samples_dir(), "Adult", "Adult.kdic")
    data_table_path = os.path.join(kh.get_samples_dir(), "Adult", "Adult.txt")
    report_file_path = os.path.join(
        "kh_samples", "scenario_prologue", "AnalysisResults.khj"
    )

    # Set the maximum memory "by hand" with an scenario prologue
    scenario_prologue = """
        // Max memory 2000 mb
        AnalysisSpec.SystemParameters.MemoryLimit 2000
        """

    # Train the predictor
    kh.train_predictor(
        dictionary_file_path,
        "Adult",
        data_table_path,
        "class",
        report_file_path,
        max_trees=0,
        scenario_prologue=scenario_prologue,
    )
.. autofunction:: build_deployed_dictionary
.. code-block:: python

    # Imports
    import os
    from khiops import core as kh

    # Set the file paths
    dictionary_file_path = os.path.join(kh.get_samples_dir(), "Iris", "Iris.kdic")
    data_table_path = os.path.join(kh.get_samples_dir(), "Iris", "Iris.txt")
    output_dir = os.path.join("kh_samples", "build_deployed_dictionary")
    deployed_dictionary_file_path = os.path.join(output_dir, "SNB_Iris_deployed.kdic")
    report_file_path = os.path.join(output_dir, "AnalysisResults.khj")

    # Train the predictor
    _, modeling_dictionary_file_path = kh.train_predictor(
        dictionary_file_path,
        "Iris",
        data_table_path,
        "Class",
        report_file_path,
        max_trees=0,
    )

    # Build the dictionary to read the output of the predictor dictionary file
    # It will contain the columns of the table generated by deploying the model
    kh.build_deployed_dictionary(
        modeling_dictionary_file_path,
        "SNB_Iris",
        deployed_dictionary_file_path,
    )

    # Print the deployed dictionary
    with open(deployed_dictionary_file_path) as deployed_dictionary_file:
        for line in deployed_dictionary_file:
            print(line, end="")
