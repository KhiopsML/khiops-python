######################################################################################
# Copyright (c) 2023 Orange. All rights reserved.                                    #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Classes to access Khiops JSON reports

Class Overview
--------------
Below we describe with diagrams the relationships of the classes in this modules. They
are mostly compositions (has-a relations) and we omit native attributes (str, int,
float, etc).

The main class of this module is `AnalysisResults` and it is largely a
composition of sub-reports objects given by the following structure::

    AnalysisResults
    |- preparation_report            ->  PreparationReport
    |- bivariate_preparation_report  ->  BivariatePreparationReport
    |- modeling_report               ->  ModelingReport
    |- train_evaluation_report      |
    |- test_evaluation_report       |->  EvaluationReport
    |- evaluation_report            |

These sub-classes in turn use other tertiary classes to represent specific information
pieces of each report. The dependencies for the classes `PreparationReport` and
`BivariatePreparationReport` are::

    PreparationReport
    |- variable_statistics -> list of VariableStatistics

    BivariatePreparationReport
    |- variable_pair_statistics -> list of VariablePairStatistics

    VariableStatistics
    |- data_grid -> DataGrid

    VariablePairStatistics
    |- data_grid -> DataGrid

    DataGrid
    |- dimensions -> list of DataGridDimension

    DataGridDimension
    |- partition -> list of PartInterval OR
    |               list of PartValue OR
    |               list of PartValueGroup

for class `ModelingReport`::

    ModelingReport
    |- trained_predictors -> list of TrainedPredictors

    TrainedPredictor
    |- selected_variables -> list of SelectedVariable

and for class `EvaluationReport`::

    EvaluationReport
    |- predictors_performance -> list of PredictorPerformance
    |- classification_lift_curves -> list of PredictorCurve (classification only)
    |- regression_rec_curves -> list of PredictorCurve (regression only)

    PredictorPerformance
    |- confusion_matrix -> ConfusionMatrix (classification only)

To have a complete illustration of the access to the information of all classes in this
module look at their ``write_report`` methods which write TSV (tab separated values)
reports.
"""
import io

from khiops.core.exceptions import PyKhiopsJSONError
from khiops.core.internals.common import type_error_message
from khiops.core.internals.io import KhiopsJSONObject, PyKhiopsOutputWriter


class AnalysisResults(KhiopsJSONObject):
    """Main class containing the information of a Khiops JSON file

    Sub-reports not available in the JSON data are optional (set to ``None``).

    Parameters
    ----------
    json_data : dict, optional
        A dictionary representing the data of a Khiops JSON report file. If not
        specified it returns an empty instance.

        .. note::
            Prefer either the `read_khiops_json_file` method or the
            `.read_analysis_results_file` function from the core API to obtain an
            instance of this class from a Khiops JSON file.

    Attributes
    ----------
    tool : str
        Name of the Khiops tool that generated the report.
    version : str
        Version of the Khiops tool that generated the report.
    short_description : str
        Short description defined by the user.
    logs : list of tuples
        2-tuples linking each sub-task name to a list containing the warnings and errors
        found during the execution of that sub-task. Available only if there were errors
        or warnings.
    preparation_report : `PreparationReport`
        A report about the variables' discretizations and groupings.
    bivariate_preparation_report : `BivariatePreparationReport`, optional
        A report of the grid models created from pairs of variables. Available only when
        pair of variables were created in the analysis.
    modeling_report : `ModelingReport`
        A report describing the predictor models. Available only in supervised analysis.
    train_evaluation_report : `EvaluationReport`
        An evaluation report of the trained models on the *train* dataset split.
        Available only in supervised analysis.
    test_evaluation_report : `EvaluationReport`
        An evaluation report of the trained models on the *test* dataset split.
        Available only in supervised analysis and when the *test* split was not empty.
    evaluation_report : `EvaluationReport`
        An `EvaluationReport` instance for evaluations created with an explicit
        evaluation (either with the `~.api.evaluate_predictor` core API function or the
        *Evaluate Predictor* feature of the Khiops desktop app). Available only when the
        report was generated with the aforementioned features.
    """

    def __init__(self, json_data=None):
        """See class docstring"""
        # Initialize super class
        super().__init__(json_data=json_data)

        # Initialize empty report attributes
        self.short_description = ""
        self.logs = None
        self.preparation_report = None
        self.bivariate_preparation_report = None
        self.modeling_report = None
        self.train_evaluation_report = None
        self.test_evaluation_report = None
        self.evaluation_report = None

        # Initialize from json data
        if json_data is not None:
            if self.tool != "Khiops":
                raise PyKhiopsJSONError(
                    f"'tool' value in JSON data must be 'Khiops', not '{self.tool}'"
                )

            # Initialize report basic data
            if "shortDescription" in json_data:
                self.short_description = json_data.get("shortDescription")
            if "logs" in json_data:
                self.logs = []
                for log in json_data.get("logs"):
                    self.logs.append((log["taskName"], log["messages"]))

            # Initialize sub-reports
            if "preparationReport" in json_data:
                self.preparation_report = PreparationReport(
                    json_data.get("preparationReport")
                )
            if "bivariatePreparationReport" in json_data:
                self.bivariate_preparation_report = BivariatePreparationReport(
                    json_data.get("bivariatePreparationReport")
                )
            if "modelingReport" in json_data:
                self.modeling_report = ModelingReport(json_data.get("modelingReport"))
            if "trainEvaluationReport" in json_data:
                self.train_evaluation_report = EvaluationReport(
                    json_data.get("trainEvaluationReport")
                )
            if "testEvaluationReport" in json_data:
                self.test_evaluation_report = EvaluationReport(
                    json_data.get("testEvaluationReport")
                )
            if "evaluationReport" in json_data:
                self.evaluation_report = EvaluationReport(
                    json_data.get("evaluationReport")
                )

    def read_khiops_json_file(self, json_file_path):
        """Constructs an instance from a Khiops JSON file

        Parameters
        ----------
        json_file_path : str
            Path of the Khiops JSON report.

        Returns
        -------
        `AnalysisResults`
            An instance of AnalysisResults containing the information on the file.
        """
        self.load_khiops_json_file(json_file_path)

    def get_reports(self):
        """Returns all available sub-reports

        Returns
        -------
        list
            All available sub-reports.
        """
        reports = []
        if self.preparation_report is not None:
            reports.append(self.preparation_report)
        if self.bivariate_preparation_report is not None:
            reports.append(self.bivariate_preparation_report)
        if self.modeling_report is not None:
            reports.append(self.modeling_report)
        if self.train_evaluation_report is not None:
            reports.append(self.train_evaluation_report)
        if self.test_evaluation_report is not None:
            reports.append(self.test_evaluation_report)
        if self.evaluation_report is not None:
            reports.append(self.evaluation_report)
        return reports

    def write_report_file(self, report_file_path):
        """Writes a TSV report file with the object's information

        Parameters
        ----------
        report_file_path : str
            Path of the output TSV report file.
        """
        with open(report_file_path, "wb") as report_file:
            report_file_writer = self.create_output_file_writer(report_file)
            self.write_report(report_file_writer)

    def write_report(self, stream_or_writer):
        """Writes the instance's TSV report into a writer object

        Parameters
        ----------
        stream_or_writer : `io.IOBase` or `.PyKhiopsOutputWriter`
            Output stream or writer.
        """
        # Check input writer/stream type
        if isinstance(stream_or_writer, io.IOBase):
            writer = self.create_output_file_writer(stream_or_writer)
        elif isinstance(stream_or_writer, PyKhiopsOutputWriter):
            writer = stream_or_writer
        else:
            raise TypeError(
                type_error_message(
                    "stream_or_writer",
                    stream_or_writer,
                    io.IOBase,
                    PyKhiopsOutputWriter,
                )
            )

        # Write nothing if tool is not defined
        if self.tool == "":
            return

        # Write report self-data to the file
        writer.writeln(f"Tool\t{self.tool}")
        writer.writeln(f"Version\t{self.version}")
        writer.writeln(f"Short description\t{self.short_description}")
        if self.logs is not None:
            writer.writeln("Logs")
            for subtask_name, messages in self.logs:
                writer.writeln(subtask_name)
                for message in messages:
                    writer.writeln(message)

        # Write sub-reports to the file
        for report in self.get_reports():
            writer.writeln("")
            writer.writeln("")
            report.write_report(writer)


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


class PreparationReport:
    """Univariate data preparation report: discretizations and groupings

    The attributes related to the target variable and null model are available only in
    the case of a supervised learning task (classification or regression).

    Parameters
    ----------
    json_data : dict, optional
        JSON data of the ``preparationReport`` field of a
        Khiops JSON report file. If not specified it returns an empty object.

    Attributes
    ----------
    report_type : "Preparation" (only possible value)
        Report type.
    dictionary : str
        Name of the training data table dictionary.
    variable_types : list of str
        The different types of variables.
    variable_numbers : list of int
        Number of variables for each type. Synchronized with ``variable_types``.
    database : str
        Path of the main training data table file.
    sample_percentage : int
        Percentage of instances used in training.
    sampling_mode : str
        Sampling mode used to split the train and datasets.
    selection_variable : str
        Name of the variable used to select training instances.
    selection_value : str
        Value of ``selection_variable`` to select training instance.
    instance_number : int
        Number of training instances.
    learning_task : str
        Name of the associated learning task. Possible values:
            - "Classification analysis",
            - "Regression analysis"
            - "Unsupervised analysis".
    target_variable : str
        Target variable name.
    main_target_value : str
        Main value of a categorical target variable.
    target_stats_min : float
        Minimum of a numerical target variable.
    target_stats_max : float
        Maximum of a numerical target variable.
    target_stats_mean : float
        Mean of a numerical target variable.
    target_stats_std_dev : float
        Standard deviation of a numerical target variable.
    target_stats_missing_number : int
        Number of missing values for a numerical target variable.
    target_stats_mode : str
        Mode of a categorical target variable.
    target_stats_mode_frequency : int
        Mode frequency of a categorical target variable.
    target_values : list of str
        Values of a categorical target variable.
    target_value_frequencies : list of int
        Frequencies for each target value. Synchronized with ``target_values``.
    evaluated_variable_number : int
        Number of variables analyzed.
    informative_variable_number : int
        *Supervised analysis only:* Number of informative variables.
    max_constructed_variables : int
        Maximum number of constructed variable specified for the analysis.
    max_trees : int
        Maximum number of constructed trees specified for the analysis.
    max_pairs : int
        Maximum number of constructed variables pairs specified for the analysis.
    discretization : str
        Type of discretization method used.
    value_grouping : str
        Type of grouping method used.
    null_model_construction_cost : float
        Coding length of the null construction model.
    null_model_preparation_cost : float
        Coding length of the null preparation model.
    null_model_data_cost : float
        Coding length of the data given the null model.
    variables_statistics : list of `VariableStatistics`
        Variable statistics for each variable analyzed.
    """

    def __init__(self, json_data=None):
        """See class docstring"""
        # Report type field
        self.report_type = ""

        # Summary attributes
        self.dictionary = ""
        self.variable_types = []
        self.variable_numbers = []
        self.database = ""
        self.sample_percentage = 0
        self.sampling_mode = ""
        self.selection_variable = None
        self.selection_value = None
        self.instance_number = 0
        self.learning_task = ""
        self.target_variable = None
        self.main_target_value = None

        # Target descriptive stats for regression
        self.target_stats_min = None
        self.target_stats_max = None
        self.target_stats_mean = None
        self.target_stats_std_dev = None
        self.target_stats_missing_number = None

        # Target descriptive stats for classification
        self.target_stats_mode = None
        self.target_stats_mode_frequency = None
        self.target_values = None
        self.target_value_frequencies = None

        # Other summary attributes
        self.evaluated_variables_number = 0
        self.informative_variable_number = 0
        self.max_constructed_variables = None
        self.max_trees = None
        self.max_pairs = None
        self.discretization = ""
        self.value_grouping = ""

        # Cost of model in the supervised case
        self.null_model_construction_cost = None
        self.null_model_preparation_cost = None
        self.null_model_data_cost = None

        # List and dictionary (internal) of variables
        self.variables_statistics = []
        self._variables_statistics_by_name = {}

        # Return if no JSON data
        if json_data is None:
            return

        # Raise exception if the preparation report is not valid
        if "reportType" not in json_data:
            raise PyKhiopsJSONError("'reportType' key not found in preparation report")
        if "summary" not in json_data:
            raise PyKhiopsJSONError("'summary' key not found in preparation report")
        if json_data.get("reportType") != "Preparation":
            raise PyKhiopsJSONError(
                "'reportType' is not 'Preparation', "
                f"""it is: '{json_data.get("reportType")}'"""
            )

        # Initialize report type
        self.report_type = json_data.get("reportType")

        # Initialize summary attributes
        json_summary = json_data.get("summary")
        self.dictionary = json_summary.get("dictionary")
        self.variable_types = json_summary.get("variables").get("types")
        self.variable_numbers = json_summary.get("variables").get("numbers")
        self.database = json_summary.get("database")
        self.instance_number = json_summary.get("instances")
        self.learning_task = json_summary.get("learningTask")
        self.target_variable = json_summary.get("targetVariable")

        # Initialze optional sampling attributes (Khiops >= 10.0)
        if json_summary.get("samplePercentage") is not None:
            self.sample_percentage = json_summary.get("samplePercentage")
            self.sampling_mode = json_summary.get("samplingMode")
            self.selection_variable = json_summary.get("selectionVariable")
            self.selection_value = json_summary.get("selectionValue")

        # Initialize target descriptive stats for regression
        if "Regression" in self.learning_task:
            stats = json_summary.get("targetDescriptiveStats")
            self.target_stats_values = stats.get("values")
            self.target_stats_min = stats.get("min")
            self.target_stats_max = stats.get("max")
            self.target_stats_mean = stats.get("mean")
            self.target_stats_std_dev = stats.get("stdDev")
            self.target_stats_missing_number = stats.get("missingNumber")

        # Initialize target descriptive stats for classification
        if "Classification" in self.learning_task:
            self.main_target_value = json_summary.get("mainTargetValue")
            stats = json_summary.get("targetDescriptiveStats")
            self.target_stats_values = stats.get("values")
            self.target_stats_mode = stats.get("mode")
            self.target_stats_mode_frequency = stats.get("modeFrequency")
            target_values = json_summary.get("targetValues")
            self.target_values = target_values.get("values")
            self.target_value_frequencies = target_values.get("frequencies")

        # Initialize other summary attributes
        self.evaluated_variable_number = json_summary.get("evaluatedVariables")
        self.informative_variable_number = json_summary.get("informativeVariables")
        json_feature_eng = json_summary.get("featureEngineering")
        if json_feature_eng is not None:
            self.max_constructed_variables = json_feature_eng.get(
                "maxNumberOfConstructedVariables"
            )
            self.max_trees = json_feature_eng.get("maxNumberOfTrees")
            self.max_pairs = json_feature_eng.get("maxNumberOfVariablePairs")
        self.discretization = json_summary.get("discretization")
        self.value_grouping = json_summary.get("valueGrouping")

        # Cost of model (supervised case and non empty database)
        null_model = json_summary.get("nullModel")
        if null_model is not None:
            self.null_model_construction_cost = null_model.get("constructionCost")
            self.null_model_preparation_cost = null_model.get("preparationCost")
            self.null_model_data_cost = null_model.get("dataCost")

        # Initialize statistics per variable
        json_variables_statistics = json_data.get("variablesStatistics")
        if json_variables_statistics is not None:
            # Initialize main attributes of all variable
            for json_variable_stats in json_variables_statistics:
                variable_statistics = VariableStatistics(json_variable_stats)
                self.variables_statistics.append(variable_statistics)

            # Store initialized variables statistics in dictionary by name
            for stats in self.variables_statistics:
                self._variables_statistics_by_name[stats.name] = stats

            # Initialize detailed statistics attributes when available
            # These are stored in JSON as a dict indexed by variables' ranks
            json_variables_detailed_statistics = json_data.get(
                "variablesDetailedStatistics"
            )
            if json_variables_detailed_statistics is not None:
                for stats in self.variables_statistics:
                    json_detailed_data = json_variables_detailed_statistics.get(
                        stats.rank
                    )
                    stats.init_details(json_detailed_data)

    def get_variable_names(self):
        """Returns the names of the variables analyzed during the preparation

        Returns
        -------
        list of str
            The names of the variables analyzed during the preparation.
        """
        return [variable_stats.name for variable_stats in self.variables_statistics]

    def get_variable_statistics(self, variable_name):
        """Returns the statistics of the specified variable

        Parameters
        ----------
        variable_name : str
            Name of the variable.

        Returns
        -------
        `VariableStatistics`
            The statistics of the specified variable.

        Raises
        ------
        `KeyError`
            If no variable with the specified names exist.
        """
        return self._variables_statistics_by_name[variable_name]

    def write_report(self, writer):
        """Writes the instance's TSV report into a writer object

        Parameters
        ----------
        writer : `.PyKhiopsOutputWriter`
            Output writer.
        """
        # Write header
        writer.writeln("Report\tPreparation")
        writer.writeln("")

        # Write summary attributes
        writer.writeln(f"Dictionary\t{self.dictionary}")
        writer.writeln("Variables")
        for i, variable_type in enumerate(self.variable_types):
            writer.write(f"\t{variable_type}")
            writer.writeln(f"\t{self.variable_numbers[i]}")
        writer.writeln(f"\tTotal\t{sum(self.variable_numbers)}")
        writer.writeln(f"Database\t{self.database}")

        # Write optional database sampling parameters (Khiops >= 10.0)
        if self.sampling_mode != "":
            writer.writeln(f"Sample percentage\t{self.sample_percentage}")
            writer.writeln(f"Sampling mode\t{self.sampling_mode}")
            writer.writeln(f"Selection variable\t{self.selection_variable}")
            writer.writeln(f"Selection value\t{self.selection_value}")

        writer.writeln(f"Instances\t{self.instance_number}")
        writer.writeln(f"Learning task\t{self.learning_task}")

        # Write classification specific attributes
        if "Classification" in self.learning_task:
            writer.writeln(f"Target variable\t{self.target_variable}")
            if self.main_target_value is not None:
                writer.writeln(f"Main target value\t{self.main_target_value}")
            writer.writeln("Target descriptive stats")
            writer.writeln(f"\tValues\t{self.target_stats_values}")
            writer.writeln(f"\tMode\t{self.target_stats_mode}")
            writer.writeln(f"\tMode frequency\t{self.target_stats_mode_frequency}")
            writer.writeln("Target variable stats")
            for i, target_value in enumerate(self.target_values):
                writer.write(f"\t{target_value}")
                writer.writeln(f"\t{self.target_value_frequencies[i]}")

        # Write regression specific attributes
        if "Regression" in self.learning_task:
            writer.writeln(f"Target variable\t{self.target_variable}")
            writer.writeln("Target descriptive stats")
            writer.writeln(f"\tValues\t{self.target_stats_values}")
            writer.writeln(f"\tMin\t{self.target_stats_min}")
            writer.writeln(f"\tMax\t{self.target_stats_max}")
            writer.writeln(f"\tMean\t{self.target_stats_mean}")
            writer.writeln(f"\tStd dev\t{self.target_stats_std_dev}")
            writer.writeln(f"\tMissing number\t{self.target_stats_missing_number}")
        # Write variable preparation summary attributes
        if len(self.variable_types) > 0 and self.instance_number > 0:
            writer.writeln(f"Evaluated variables\t{self.evaluated_variable_number}")
            writer.writeln(f"Informative variables\t{self.informative_variable_number}")
            if self.max_constructed_variables is not None:
                writer.writeln(
                    "Max number of constructed variables\t"
                    f"{self.max_constructed_variables}"
                )
            if self.max_trees is not None:
                writer.writeln(f"Max number of trees\t{self.max_trees}")
            if self.max_pairs is not None:
                writer.writeln(f"Max number of variable pairs\t{self.max_pairs}")
            writer.writeln(f"Discretization\t{self.discretization}")
            writer.writeln(f"Value grouping\t{self.value_grouping}")

        # Write preparation cost information
        if (
            "Unsupervised" not in self.learning_task
            and self.null_model_construction_cost is not None
        ):
            writer.writeln("Null model")
            writer.writeln(f"\tConstruction cost\t{self.null_model_construction_cost}")
            writer.writeln(f"\tPreparation cost\t{self.null_model_preparation_cost}")
            writer.writeln(f"\tData cost\t{self.null_model_data_cost}")

        # Write variables' statistics
        if len(self.variables_statistics) > 0:
            # Write main report
            writer.writeln("")
            writer.writeln("Variable statistics")
            self.variables_statistics[0].write_report_header_line(writer)
            for variable_statistics in self.variables_statistics:
                variable_statistics.write_report_line(writer)

            # Write detailed report
            writer.writeln("")
            writer.writeln("Detailed variable statistics")
            for variable_statistics in self.variables_statistics:
                variable_statistics.write_report_details(writer)


class BivariatePreparationReport:
    """Bivariate data preparation report: 2D grid models

    The attributes related to the target variable and null model are available only in
    the case of a supervised learning task (only classification in the bivariate case).

    Parameters
    ----------
    json_data : dict, optional
        JSON data of the ``bivariatePreparationReport``
        field of a Khiops JSON report file. If not specified it returns an empty object.

    Attributes
    ----------
    report_type : "BivariatePreparation" (only possible value)
        Report type.
    dictionary : str
        Name of the training data table dictionary.
    variable_types : list of str
        The different types of variables.
    variable_numbers : list of int
        The number of variables for each type in ``variables_types`` (synchronized
        lists).
    database : str
        Path of the main training data table file.
    sample_percentage : int
        Percentage of instances used in training.
    sampling_mode : str
        Sampling mode used to split the train and datasets.
    selection_variable : str
        Variable used to select instances for training.
    selection_value : str
        Value of selection_variable to select instances for training.
    instance_number : int
        Number of training instances.
    learning_task : str
        Name of the associated learning task. Possible values:
            - "Classification analysis"
            - "Regression analysis"
            - "Unsupervised analysis"
    target_variable : str
        Target variable name in supervised analysis.
    main_target_value : str
        Main modality of the target variable in supervised case.
    target_stats_mode : str
        Mode of a categorical target variable.
    target_stats_mode_frequency : int
        Mode frequency of a categorical target variable.
    target_values : list of str
        Values of a categorical target variable.
    target_value_frequencies : list of int
        Frequencies for each value in ``target_values`` (synchronized lists).
    evaluated_pair_number : int
        Number of variable pairs evaluated.
    informative_pair_number : int
        Number of informative variable pairs. A pair is considered informative if its
        level is greater than the sum of its components' levels.
    variable_pair_statistics : list of `VariablePairStatistics`
        Statistics for each analyzed pair of variables.
    """

    def __init__(self, json_data=None):
        """See class docstring"""
        # Report type
        self.report_type = ""

        # Summary attributes
        self.dictionary = ""
        self.variable_types = []
        self.variable_numbers = []
        self.database = ""
        self.sample_percentage = 0
        self.sampling_mode = ""
        self.selection_variable = None
        self.selection_value = None
        self.instance_number = 0
        self.learning_task = ""
        self.target_variable = None
        self.main_target_value = None

        # Target descriptive stats for classification
        # Note: There is no bivariate preparation in the regression case
        self.target_stats_mode = None
        self.target_stats_mode_frequency = None
        self.target_values = None
        self.target_value_frequencies = None

        # Information of the pair evaluations
        self.evaluated_pair_number = None
        self.informative_pair_number = None

        # List and dictionary (internal) of variable pairs
        self.variables_pairs_statistics = []
        self._variables_pairs_statistics_by_name = {}

        # Return if no json_data
        if json_data is None:
            return

        # Raise exception if the bivariate preparation report is not valid
        if "reportType" not in json_data:
            raise PyKhiopsJSONError(
                "'reportType' key not found in bivariate preparation report"
            )
        if "summary" not in json_data:
            raise PyKhiopsJSONError(
                "'summary' key not found in bivariate preparation report"
            )
        if json_data.get("reportType") != "BivariatePreparation":
            raise PyKhiopsJSONError(
                "'reportType' is not 'BivariatePreparation', "
                f"""it is: '{json_data.get("reportType")}'"""
            )

        # Initialize report type
        self.report_type = json_data.get("reportType")

        # Initialize summary attributes
        json_summary = json_data.get("summary")
        self.dictionary = json_summary.get("dictionary")
        self.variable_types = json_summary.get("variables").get("types")
        self.variable_numbers = json_summary.get("variables").get("numbers")
        self.database = json_summary.get("database")
        if "samplingMode" in json_summary:
            self.sample_percentage = json_summary.get("samplePercentage")
            self.sampling_mode = json_summary.get("samplingMode")
            self.selection_variable = json_summary.get("selectionVariable")
            self.selection_value = json_summary.get("selectionValue")
        self.instance_number = json_summary.get("instances")
        self.learning_task = json_summary.get("learningTask")
        self.target_variable = json_summary.get("targetVariable")

        # Classification task: Initialize target descriptive stats
        if "Classification" in self.learning_task:
            self.main_target_value = json_summary.get("mainTargetValue")
            stats = json_summary.get("targetDescriptiveStats")
            self.target_stats_values = stats.get("values")
            self.target_stats_mode = stats.get("mode")
            self.target_stats_mode_frequency = stats.get("modeFrequency")
            target_values = json_summary.get("targetValues")
            self.target_values = target_values.get("values")
            self.target_value_frequencies = target_values.get("frequencies")

        # Initialize the information of the pair evaluations
        self.evaluated_pair_number = json_summary.get("evaluatedVariablePairs")
        self.informative_pair_number = json_summary.get("informativeVariablePairs")

        # Initialize main attributes for all variables
        json_variables_pair_statistics = json_data.get("variablesPairsStatistics")
        for json_pair_stats in json_variables_pair_statistics:
            variable_pair_statistics = VariablePairStatistics(json_pair_stats)
            self.variables_pairs_statistics.append(variable_pair_statistics)

        # Store variable stats in dict indexed by name pairs in both senses
        for stats in self.variables_pairs_statistics:
            name1 = stats.name1
            name2 = stats.name2
            self._variables_pairs_statistics_by_name[name1, name2] = stats
            self._variables_pairs_statistics_by_name[name2, name1] = stats

        # Initialize the variables' detailed statistics when available
        # These are stored in JSON as a dict indexed by the variables ranks
        if "variablesPairsDetailedStatistics" in json_data:
            json_detailed_statistics = json_data.get("variablesPairsDetailedStatistics")
            for stats in self.variables_pairs_statistics:
                json_detailed_data = json_detailed_statistics.get(stats.rank)
                stats.init_details(json_detailed_data)

    def get_variable_pair_names(self):
        """Returns the pairs of variable names available on this report

        Returns
        -------
        list of tuple
            The pair of variable names available on this report
        """
        return [
            (pair_stats.name1, pair_stats.name2)
            for pair_stats in self.variables_pairs_statistics
        ]

    def get_variable_pair_statistics(self, variable_name_1, variable_name_2):
        """Returns the statistics of the specified pair of variables

        .. note::
            The variable names can be given in any order.

        Parameters
        ----------
        variable_name_1 : str
            Name of the first variable.
        variable_name_2 : str
            Name of the second variable.

        Returns
        -------
        `VariablePairStatistics`
            The statistics of the specified pair of variables.

        Raises
        ------
        `KeyError`
            If no pair with the specified names exist.
        """
        return self._variables_pairs_statistics_by_name[
            (variable_name_1, variable_name_2)
        ]

    def write_report(self, writer):
        """Writes the instance's TSV report into a writer object

        Parameters
        ----------
        writer : `.PyKhiopsOutputWriter`
            Output writer.
        """
        # Write header
        writer.writeln("Report\tBivariate preparation")
        writer.writeln("")

        # Write summary attributes
        writer.writeln(f"Dictionary\t{self.dictionary}")
        writer.writeln("Variables")
        for i, variable_type in enumerate(self.variable_types):
            writer.write(f"\t{variable_type}")
            writer.writeln(f"\t{self.variable_numbers[i]}")
        writer.writeln(f"\tTotal\t{sum(self.variable_numbers)}")
        writer.writeln(f"Database\t{self.database}")
        if self.sampling_mode != "":
            writer.writeln(f"Sample percentage\t{self.sample_percentage}")
            writer.writeln(f"Sampling mode\t{self.sampling_mode}")
            writer.writeln(f"Selection variable\t{self.selection_variable}")
            writer.writeln(f"Selection value\t{self.selection_value}")
        writer.writeln(f"Instances\t{self.instance_number}")
        writer.writeln(f"Learning task\t{self.learning_task}")

        # Write specific summary attributes for a classification task
        if self.learning_task == "Classification analysis":
            writer.writeln(f"Target variable\t{self.target_variable}")
            if self.main_target_value is not None:
                writer.writeln(f"Main target value\t{self.main_target_value}")
            writer.writeln("Target descriptive stats")
            writer.writeln(f"\tValues\t{self.target_stats_values}")
            writer.writeln(f"\tMode\t{self.target_stats_mode}")
            writer.writeln(f"\tMode frequency\t{self.target_stats_mode_frequency}")
            writer.writeln("Target variable stats")
            for i, target_value in enumerate(self.target_values):
                writer.write(f"\t{target_value}")
                writer.writeln(f"\t{self.target_value_frequencies[i]}")

        if self.evaluated_pair_number is not None:
            writer.writeln(f"Evaluated variable pairs\t{self.evaluated_pair_number}")
        if self.informative_pair_number is not None:
            writer.writeln(
                f"Informative variable pairs\t{self.informative_pair_number}"
            )

        # Write variables' pair statistics main report
        writer.writeln("")
        writer.writeln("Variable pair statistics")
        self.variables_pairs_statistics[0].write_report_header_line(writer)
        for variable_pair_statistics in self.variables_pairs_statistics:
            variable_pair_statistics.write_report_line(writer)

        # Write variables' pair statistics reports' details
        writer.writeln("")
        writer.writeln("Detailed variable pair statistics")
        for variable_pair_statistics in self.variables_pairs_statistics:
            variable_pair_statistics.write_report_details(writer)


class ModelingReport:
    """Modeling report of all predictors created in a supervised analysis

    Parameters
    ----------
    json_data : dict, optional
        JSON data of the ``modelingReport`` field of
        Khiops JSON report file. If not specified it returns an empty object.

    Attributes
    ----------
    report_type : "Modeling" (only possible value)
        Report type.
    dictionary : str
        Name of the training data table dictionary.
    database : str
        Path of the main training data table file.
    sample_percentage : int
        Percentage of instances used in training.
    sampling_mode : "Include sample" or "Exclude sample"
        Sampling mode used to split the train and datasets.
    selection_variable : str
        Variable used to select instances for training.
    selection_value : str
        Value of ``selection_variable`` to select instances for training.
    learning_task : "Classification analysis" or "Regression analysis"
        Name of the associated learning task.
    target_variable : str
        Name of the target variable.
    main_target_value : str
        Main value of the target variable.
    trained_predictors : list of `TrainedPredictor`
        The predictors trained in the task.
    """

    def __init__(self, json_data=None):
        """See class docstring"""
        # Report type
        self.report_type = ""

        # Summary attributes
        self.dictionary = ""
        self.database = ""
        self.sample_percentage = 0
        self.sampling_mode = ""
        self.selection_variable = None
        self.selection_value = None
        self.learning_task = ""
        self.target_variable = None
        self.main_target_value = None

        # List and dictionary (internal) of trained predictors
        self.trained_predictors = []
        self._trained_predictors_by_name = {}

        # Return if no JSON data
        if json_data is None:
            return

        # Raise exception if the modeling report is not valid
        if "reportType" not in json_data:
            raise PyKhiopsJSONError("'reportType' key not found in modeling report")
        if "summary" not in json_data:
            raise PyKhiopsJSONError("'summary' key not found in modeling report")
        if json_data.get("reportType") != "Modeling":
            raise PyKhiopsJSONError(
                "'reportType' is not 'Modeling', "
                f"""it is: '{json_data.get("reportType")}'"""
            )

        # Initialize report type
        self.report_type = json_data.get("reportType")

        # Initialize the summary attributes
        json_summary = json_data.get("summary")
        self.dictionary = json_summary.get("dictionary")
        self.database = json_summary.get("database")
        if "samplingMode" in json_summary:
            self.sample_percentage = json_summary.get("samplePercentage")
            self.sampling_mode = json_summary.get("samplingMode")
            self.selection_variable = json_summary.get("selectionVariable")
            self.selection_value = json_summary.get("selectionValue")
        self.learning_task = json_summary.get("learningTask")
        self.target_variable = json_summary.get("targetVariable")
        self.main_target_value = json_summary.get("mainTargetValue")

        # Initialize specifications per trained predictor
        for json_trained_predictor in json_data.get("trainedPredictors"):
            # Initialize basic trained predictor data
            predictor = TrainedPredictor(json_trained_predictor)
            self.trained_predictors.append(predictor)
            self._trained_predictors_by_name[predictor.name] = predictor

            # Initialize detailed trained predictor data
            if "trainedPredictorsDetails" in json_data:
                json_predictors_details = json_data.get("trainedPredictorsDetails")
                json_detailed_data = json_predictors_details.get(predictor.rank)
                predictor.init_details(json_detailed_data)

    def get_predictor(self, predictor_name):
        """Returns the specified predictor

        Parameters
        ----------
        predictor_name : str
            Name of the predictor.

        Returns
        -------
        `TrainedPredictor`
            The predictor object for the specified name.

        Raises
        ------
        `KeyError`
            If there is no predictor with the specified name.
        """
        return self._trained_predictors_by_name[predictor_name]

    def get_snb_predictor(self):
        """Returns the Selective Naive Bayes predictor

        Returns
        -------
        `TrainedPredictor`
            The predictor object for "Selective Naive Bayes".

        Raises
        ------
        `KeyError`
            If there is no predictor named "Selective Naive Bayes".
        """
        return self.get_predictor("Selective Naive Bayes")

    def get_predictor_names(self):
        """Returns the names of the available predictor reports

        Returns
        -------
        list of str
            The names of the available predictor reports.

        """
        return list(self._trained_predictors_by_name.keys())

    def write_report(self, writer):
        """Writes the instance's TSV report into a writer object

        Parameters
        ----------
        writer : `.PyKhiopsOutputWriter`
            Output writer.
        """
        # Write header
        writer.writeln("Report\tModeling")
        writer.writeln("")

        # Write summary attributes
        writer.writeln(f"Dictionary\t{self.dictionary}")
        writer.writeln(f"Database\t{self.database}")
        if self.sampling_mode != "":
            writer.writeln(f"Sample percentage\t{self.sample_percentage}")
            writer.writeln(f"Sampling mode\t{self.sampling_mode}")
            writer.writeln(f"Selection variable\t{self.selection_variable}")
            writer.writeln(f"Selection value\t{self.selection_value}")
        writer.writeln(f"Learning task\t{self.learning_task}")
        writer.writeln(f"Target variable\t{self.target_variable}")
        if self.main_target_value is not None:
            writer.writeln(f"Main target value\t{self.main_target_value}")

        # Write trained predictors' reports summaries
        writer.writeln("")
        writer.writeln("Trained predictors")
        self.trained_predictors[0].write_report_header_line(writer)
        for predictor in self.trained_predictors:
            predictor.write_report_line(writer)

        # Write trained predictors' reports' details
        writer.writeln("")
        writer.writeln("Detailed trained predictors")
        for predictor in self.trained_predictors:
            predictor.write_report_details(writer)


class EvaluationReport:
    """Evaluation report for predictors

    Parameters
    ----------
    json_data : dict, optional
        JSON data of the fields:
            - ``trainEvaluationReport``: predictor training
            - ``testEvaluationReport``: predictor training & non-empty test split
            - ``evaluationReport``: explicit evaluation

        The first two fields are set when doing a supervised analysis: either with the
        "Train Model" feature of the Khiops app or the `~.api.train_predictor` function
        of the Khiops Python core API. The third field is set when doing an explicit
        evaluation: either with the *Evaluate Predictor* feature of the Khiops app or
        the `~.api.evaluate_predictor` function of the Khiops Python core API.

        If not specified it returns an empty object.

    Attributes
    ----------
    report_type : "Evaluation" (only possible value)
        Report type.
    evaluation_type : "Train", "Test" or ""
        Evaluation type. The value "" is set when the evaluation was explicit.
    dictionary : str
        Name of the training data table dictionary.
    database : str
        Path of the main training data table file.
    sample_percentage : int
        Percentage of instances used in training.
    sampling_mode : str
        Sampling mode used to split the train and datasets.
    selection_variable : str
        Variable used to select instances for training.
    selection_value : str
        Value of selection_variable to select instances for training.
    instance_number : int
        Number of training instances.
    learning_task : "Classification analysis" or "Regression analysis"
        Type of learning task.
    target_variable : str
        Name of the target variable.
    main_target_value : str
        Main value of the target variable.
    predictors_performance : list of `PredictorPerformance`
        Performance metrics for each predictor.
    regression_rec_curves : list of `PredictorCurve`
        REC curves for each regressor.
    classification_target_values : list of str
        Target variable values for which a classifier lift curve was evaluated.
    classification_lift_curves : list of `PredictorCurve`
        Lift curves for each target value in ``classification_target_values``. The lift
        curve for the optimal predictor is prepended to those of the target values.
    """

    def __init__(self, json_data=None):
        """See class docstring"""
        # Type attributes
        self.report_type = ""
        self.evaluation_type = ""

        # Summary attributes
        self.dictionary = ""
        self.database = ""
        self.sample_percentage = 0
        self.sampling_mode = ""
        self.selection_variable = None
        self.selection_value = None
        self.instance_number = 0
        self.learning_task = ""
        self.target_variable = ""
        self.main_target_value = None

        # List and private dictionary of PredictorPerformance objects
        # One for each predictor
        self.predictors_performance = []
        self._predictors_performance_by_name = {}

        # List with a PredictorCurve object for each regressor REC curve
        self.regression_rec_curves = None

        # Classification task: List of target values and their associates lift curves
        # For each evaluated target value, there is a list of lift curves:
        # one PredictorCurve object per classifier plus an extra one named "Optimal"
        # put at the beginning of the list.
        # Note that there is no curve for "Random", because it can be easily calculated
        self.classification_target_values = None
        self.classification_lift_curves = None

        # Return if no JSON data
        if json_data is None:
            return

        # Raise exception if the evaluation report is not valid
        if "reportType" not in json_data:
            raise PyKhiopsJSONError("'reportType' key not found in evaluation report")
        if "summary" not in json_data:
            raise PyKhiopsJSONError("'summary' key not found in evaluation report")
        if json_data.get("reportType") != "Evaluation":
            raise PyKhiopsJSONError(
                "'reportType' is not 'Evaluation' it is: "
                f"""'{json_data.get("reportType")}'"""
            )

        # Initialize type attributes
        self.report_type = json_data.get("reportType")
        self.evaluation_type = json_data.get("evaluationType")

        # Initialize summary attributes
        json_summary = json_data.get("summary")
        self.dictionary = json_summary.get("dictionary")
        self.database = json_summary.get("database")
        if "samplingMode" in json_summary:
            self.sample_percentage = json_summary.get("samplePercentage")
            self.sampling_mode = json_summary.get("samplingMode")
            self.selection_variable = json_summary.get("selectionVariable")
            self.selection_value = json_summary.get("selectionValue")
        self.instance_number = json_summary.get("instances")
        self.instance_number = json_summary.get("instances")
        self.learning_task = json_summary.get("learningTask")
        self.target_variable = json_summary.get("targetVariable")
        self.main_target_value = json_summary.get("mainTargetValue")

        # Initialize the performance attributes for each predictor
        json_predictors_performance = json_data.get("predictorsPerformance")
        for json_predictor_performance in json_predictors_performance:
            # Initialize main performance info
            performance = PredictorPerformance(json_predictor_performance)
            self.predictors_performance.append(performance)
            self._predictors_performance_by_name[performance.name] = performance

            # Initialize detailed performance info
            if "predictorsDetailedPerformance" in json_data:
                json_detailed_performance = json_data.get(
                    "predictorsDetailedPerformance"
                )
                json_detailed_data = json_detailed_performance.get(performance.rank)
                performance.init_details(json_detailed_data)

        # Collect REC curves for each regressor
        if "Regression" in self.learning_task:
            self.regression_rec_curves = []
            for json_rec_curve in json_data.get("recCurves"):
                rec_curve = PredictorCurve(json_rec_curve)
                self.regression_rec_curves.append(rec_curve)

        # Fill Lift curves for a classification task if available
        if "Classification" in self.learning_task:
            self.classification_target_values = []
            self.classification_lift_curves = []

            # Collect all lift curves per target value and per classifier
            for json_lift_curves in json_data.get("liftCurves"):
                target_value = json_lift_curves.get("targetValue")
                json_curves = json_lift_curves.get("curves")
                lift_curves = []

                # Collect lift curves for each classifier
                for json_curve in json_curves:
                    lift_curve = PredictorCurve(json_curve)
                    lift_curves.append(lift_curve)

                # Store collected target values with their lift curves
                self.classification_target_values.append(target_value)
                self.classification_lift_curves.append(lift_curves)

    def get_predictor_names(self):
        """Returns the names of the available predictors in the report

        Returns
        -------
        list of str
            The names of the available predictors.
        """
        return list(self._predictors_performance_by_name.keys())

    def get_predictor_performance(self, predictor_name):
        """Returns the performance metrics for the specified predictor

        Parameters
        ----------
        predictor_name : str
            A predictor name.

        Returns
        -------
        `PredictorPerformance`
            The performance metrics for the specified predictor.

        Raises
        ------
        `KeyError`
            If no predictor with the specified name exists.
        """
        return self._predictors_performance_by_name[predictor_name]

    def get_snb_performance(self):
        """Returns the performance metrics for the Selective Naive Bayes predictor

        Returns
        -------
        `PredictorPerformance`
            The performance metrics for the Selective Naive Bayes predictor.

        Raises
        ------
        `ValueError`
            If the Selective Naive Bayes information is not available in the report.
        """
        if "Selective Naive Bayes" not in self._predictors_performance_by_name:
            raise ValueError("Selective Naive Bayes predictor not available")
        return self.get_predictor_performance("Selective Naive Bayes")

    def get_regressor_rec_curve(self, regressor_name):
        """Returns the REC curve for the specified regressor

        Parameters
        ----------
        regressor_name : str
            Name of a regressor.

        Returns
        -------
        `PredictorCurve`
            The REC curve for the specified regressor.

        Raises
        ------
        `ValueError`
            If no regressor curves available. (
        `KeyError`
            If no regressor with the specified name exists.
        """
        if self.learning_task != "Regression analysis":
            raise ValueError("REC curves are available only for regression")
        for curve in self.regression_rec_curves:
            if curve.name == regressor_name:
                return curve
        raise KeyError(regressor_name)

    def get_snb_rec_curve(self):
        """Returns the REC curve for the Selective Naive Bayes regressor

        Returns
        -------
        `PredictorCurve`
            The REC curve for the Selective Naive Bayes regressor.

        Raises
        ------
        `ValueError`
            If the Selective Naive Bayes information is not available in the report.
        """
        if self.learning_task != "Regression analysis":
            raise ValueError("REC curves are available only for regression")
        for curve in self.regression_rec_curves:
            if curve.name == "Selective Naive Bayes":
                return curve
        raise ValueError("Selective Naive Bayes regressor information not available")

    def get_classifier_lift_curve(self, classifier_name, target_value):
        """Returns the lift curve for the specified classifier and target value

        Parameters
        ----------
        classifier_name : str
            A name of a classifier.
        target_value : str
            A specific value of the target variable.

        Returns
        -------
        `PredictorCurve`
            The lift curve for the specified classifier and target value.

        Raises
        ------
        `KeyError`
            If no classifier with the specified exists or no target value with the
            specified name exists.
        """
        if self.learning_task != "Classification analysis":
            raise ValueError("Lift curves are available only for classification")
        for i, value in enumerate(self.classification_target_values):
            if value == target_value:
                if classifier_name == "Random":
                    point_number = len(self.classification_lift_curves[i][0].values)
                    curve = PredictorCurve()
                    curve.type = "Lift"
                    curve.name = "Random"
                    curve.values = [
                        i * 100.0 / (point_number - 1) for i in range(point_number)
                    ]
                    return curve
                else:
                    for lift_curve in self.classification_lift_curves[i]:
                        if lift_curve.name == classifier_name:
                            return lift_curve
                    raise KeyError(classifier_name)
        raise KeyError(target_value)

    def get_snb_lift_curve(self, target_value):
        """Returns lift curve for the Selective Naive Bayes clf. given a target value

        Parameters
        ----------
        target_value : str
            A specific value of the target variable.

        Returns
        -------
        `PredictorCurve`
            The lift curve of the Selective Naive Bayes classifier for the specified
            target value.

        Raises
        ------
        `ValueError`
            If the Selective Naive Bayes classifier information is not available.
        `KeyError`
            If no target value with the specified name exists.
        """
        if self.learning_task != "Classification analysis":
            raise ValueError("Lift curves are available only for classification")
        for i, value in enumerate(self.classification_target_values):
            if value == target_value:
                for lift_curve in self.classification_lift_curves[i]:
                    if lift_curve.name == "Selective Naive Bayes":
                        return lift_curve
                raise ValueError(
                    "Selective Naive Bayes classifier information not available"
                )
        raise KeyError(target_value)

    def write_report(self, writer):
        """Writes the instance's TSV report into a writer object

        Parameters
        ----------
        writer : `.PyKhiopsOutputWriter`
            Output writer object.
        """
        # Write report header
        writer.write("Report\t")
        writer.writeln(f"Evaluation\t{self.evaluation_type}")
        writer.writeln("")

        # Write summary attributes
        writer.writeln(f"Dictionary\t{self.dictionary}")
        writer.writeln(f"Database\t{self.database}")
        if self.sampling_mode != "":
            writer.writeln(f"Sample percentage\t{self.sample_percentage}")
            writer.writeln(f"Sampling mode\t{self.sampling_mode}")
            writer.writeln(f"Selection variable\t{self.selection_variable}")
            writer.writeln(f"Selection value\t{self.selection_value}")
        writer.writeln(f"Instances\t{self.instance_number}")
        writer.writeln(f"Learning task\t{self.learning_task}")
        writer.writeln(f"Target variable\t{self.target_variable}")
        if self.main_target_value is not None:
            writer.writeln(f"Main target value\t{self.main_target_value}")

        # Write predictors' performance reports
        writer.writeln("")
        writer.writeln("Predictors performance")
        self.predictors_performance[0].write_report_header_line(writer)
        for predictor_performance in self.predictors_performance:
            predictor_performance.write_report_line(writer)

        # Write predictors' performance detailed reports if available
        writer.writeln("")
        writer.writeln("Predictors detailed performance")
        for predictor_performance in self.predictors_performance:
            predictor_performance.write_report_details(writer)

        # Write REC curves, one per regressor
        if (
            self.learning_task.startswith("Regression")
            and self.regression_rec_curves is not None
        ):
            # Write header line
            writer.writeln("")
            writer.writeln("REC curves")

            # Write curves' columns header
            writer.write("Size")
            for rec_curve in self.regression_rec_curves:
                writer.write(f"\t{rec_curve.name}")
            writer.writeln("")

            # Write curves' lines
            line_number = len(self.regression_rec_curves[0].values)
            value_number = line_number - 1
            for index in range(line_number):
                percentile = index * 100.0 / value_number
                writer.write(str(percentile))
                for rec_curve in self.regression_rec_curves:
                    writer.write(f"\t{rec_curve.values[index]}")
                writer.writeln("")

        # Write lift curves, one per target value and per classifier
        if (
            self.learning_task.startswith("Classification")
            and self.classification_target_values is not None
        ):
            for i, target_value in enumerate(self.classification_target_values):
                lift_curves = self.classification_lift_curves[i]

                # Write modality header line
                writer.writeln("")
                writer.writeln(f"Lift curves\t{target_value}")

                # Write curves' columns header
                writer.write("Size")
                writer.write("\tRandom")
                for lift_curve in lift_curves:
                    writer.write(f"\t{lift_curve.name}")
                writer.writeln("")

                # Write curves' lines
                line_number = len(lift_curves[0].values)
                value_number = line_number - 1
                for index in range(line_number):
                    percent = index * 100.0 / value_number
                    writer.write(str(percent))
                    writer.write(f"\t{percent}")  # "Random" baseline
                    for lift_curve in lift_curves:
                        writer.write(f"\t{lift_curve.values[index]}")
                    writer.writeln("")


class VariableStatistics:
    """Variable information and statistics

    .. note::
        The statistics in this class are for both numerical and categorical data.


    Parameters
    ----------
    json_data : dict, optional
        JSON data of an element of the list found at the
        ``variablesStatistics`` field within the ``preparationReport`` field of a Khiops
        JSON report file. If not specified it returns an empty object.


        .. note::
            The ``data_grid`` field is considered a "detail" and is not initialized in
            the constructor. Instead, it is initialized explicitly via the
            ``init_details`` method. This allows to make partial initializations for
            large reports.


    Attributes
    ----------
    rank : str
        Variable rank with respect to its level. Lower Rank = Higher Level.
    name : str
        Variable name.
    type : str
        Variable type. Valid values:
            - "Numerical"
            - "Categorical"
            - "Date"
            - "Time"
            - "Timestamp"
            - "Table"
            - "Entity"
            - "Structure"
    level : float
        Variable predictive importance.
    target_part_number : int
        - In regression: Number of the target intervals
        - In classification with target grouping: Number of target groups
    part_number : int
        Number of parts of the variable partition.
    value_number : int
        Number of distinct values of the variable.
    min : float
        Minimum value of the variable.
    max : float
        Maximum value of the variable.
    mean : float
        Mean value of the variable.
    std_dev : float
        Standard deviation of the variable.
    missing_number : int
        Number of missing values of the variable.
    mode : float
        Most common value.
    mode_frequency : int
        Frequency of the most common value.
    input_values : list of str
        Different values taken by the variable. If there are too many values only the
        more frequent will be available.
    input_value_frequencies : list of int
        The frequencies for each input value. Synchronized with ``input_values``.
    construction_cost : float
        Construction cost of the variable. More complex variables cost more.
    preparation_cost : float
        Partition model cost. More complex partitions cost more.
    data_cost : float
        Negative log-likelihood of the variable given a preparation model and a
        construction model.
    derivation_rule : str
        If the variable is not native it is Khiops dictionary function to derive it.
        Otherwise is set to ``None``.
    data_grid : `DataGrid`
        A density estimation of the partitioned variable with respect to the target.
    """

    def __init__(self, json_data=None):
        """See class docstring"""
        # Common attributes
        self.rank = ""
        self.name = ""
        self.type = ""
        self.level = None
        self.target_part_number = None
        self.part_number = None
        self.value_number = 0

        # Numerical variable statistics attributes
        self.min = None
        self.max = None
        self.mean = None
        self.std_dev = None
        self.missing_number = None

        # Categorical variable statistics attributes
        self.mode = None
        self.mode_frequency = None

        # Cost attributes
        self.construction_cost = None
        self.preparation_cost = None
        self.data_cost = None

        # Derivation rule
        self.derivation_rule = None

        # Details' attributes
        # Data grid for density estimation
        self.data_grid = None

        # Input values and their frequencies in case of categorical variables
        # The input values may not be the exhaustive list of all the values
        # For scalability reasons, the least frequent values are not always present
        self.input_values = None
        self.input_value_frequencies = None

        # Initialization from JSON data (except for the details' attributes)
        if json_data is not None:
            # Initialize common attributes
            self.rank = json_data.get("rank")
            self.name = json_data.get("name")
            self.type = json_data.get("type")
            self.level = json_data.get("level")
            self.target_part_number = json_data.get("targetParts")
            self.part_number = json_data.get("parts")
            self.value_number = json_data.get("values")

            # Initialize numerical variable attributes
            self.min = json_data.get("min")
            self.max = json_data.get("max")
            self.mean = json_data.get("mean")
            self.std_dev = json_data.get("stdDev")
            self.missing_number = json_data.get("missingNumber")

            # Initialize categorical variable attributes
            self.mode = json_data.get("mode")
            self.mode_frequency = json_data.get("modeFrequency")

            # Initialize cost attributes
            self.construction_cost = json_data.get("constructionCost")
            self.preparation_cost = json_data.get("preparationCost")
            self.data_cost = json_data.get("dataCost")

            # Initialize derivation rule
            self.derivation_rule = json_data.get("derivationRule")

    def init_details(self, json_detailed_data=None):
        """Initializes the details' attributes from a Python JSON object

        Parameters
        ----------
        json_data : dict, optional
            JSON data of an element of the list found at the
            ``variablesDetailedStatistics`` field within the ``preparationReport`` field
            of a Khiops JSON report file. If not specified it leaves the object as-is.

        """
        if json_detailed_data is not None:
            # Initialize data grid
            json_data_grid = json_detailed_data.get("dataGrid")
            if json_data_grid is not None:
                self.data_grid = DataGrid(json_data_grid)

            # Initialize input values and their frequencies
            json_input_values = json_detailed_data.get("inputValues")
            if json_input_values is not None:
                self.input_values = json_input_values.get("values")
                self.input_value_frequencies = json_input_values.get("frequencies")

    def is_detailed(self):
        """Returns True if the report contains any detailed information

        Returns
        -------
        bool
            True if the report contains any detailed information.
        """
        return self.data_grid is not None or (
            self.input_values is not None and self.input_value_frequencies is not None
        )

    def write_report_header_line(self, writer):
        """Writes the header line of a TSV report into a writer object

        The header is the same for all variable types.

        Parameters
        ----------
        writer : `.PyKhiopsOutputWriter`
            Output writer.
        """
        writer.write("Rank\t")
        writer.write("Name\t")
        writer.write("Type\t")
        writer.write("Level\t")
        writer.write("Target parts\t")
        writer.write("Parts\t")
        writer.write("Values\t")
        writer.write("Min\t")
        writer.write("Max\t")
        writer.write("Mean\t")
        writer.write("StdDev\t")
        writer.write("Missing number\t")
        writer.write("Mode\t")
        writer.write("Mode frequency\t")
        writer.write("Construction cost\t")
        writer.write("Preparation cost\t")
        writer.write("Data cost\t")
        writer.writeln("Derivation rule")

    def write_report_line(self, writer):
        """Writes a line of the TSV report into a writer object

        Parameters
        ----------
        writer : `.PyKhiopsOutputWriter`
            Output writer.
        """
        # Write common attributes
        writer.write(f"{self.rank}\t")
        writer.write(f"{self.name}\t")
        writer.write(f"{self.type}\t")

        # Write level if available
        if self.level is not None:
            writer.write(f"{self.level}\t")
        else:
            writer.write("\t")

        # Write target part number if available
        if self.target_part_number is not None:
            writer.write(f"{self.target_part_number}\t")
        else:
            writer.write("\t")

        # Write part number if available
        if self.part_number is not None:
            writer.write(f"{self.part_number}\t")
        else:
            writer.write("\t")
        writer.write(f"{self.value_number}\t")

        # Write attributes available only for numerical variables
        if self.type == "Numerical":
            writer.write(f"{self.min}\t")
            writer.write(f"{self.max}\t")
            writer.write(f"{self.mean}\t")
            writer.write(f"{self.std_dev}\t")
            writer.write(f"{self.missing_number}\t")
        else:
            writer.write("\t" * 5)

        # Write attributes available only for categorical variables
        if self.type == "Categorical":
            writer.write(f"{self.mode}\t")
            writer.write(f"{self.mode_frequency}\t")
        else:
            writer.write("\t\t")
        writer.write(f"{self.construction_cost}\t")

        # Write preparation cost only for the supervised case
        if self.preparation_cost is not None:
            writer.write(f"{self.preparation_cost}\t")
            writer.write(f"{self.data_cost}\t")
        else:
            writer.write("\t\t")

        # Write derivation rule if available
        if self.derivation_rule is not None:
            writer.write(self.derivation_rule)
        writer.writeln("")

    def write_report_details(self, writer):
        """Writes the details' attributes into a writer object

        Parameters
        ----------
        writer : `.PyKhiopsOutputWriter`
            Output writer.
        """
        if self.is_detailed():
            # Write header line
            writer.writeln("")
            writer.write("Rank\t")
            writer.write(f"{self.rank}\t")
            writer.write(f"{self.name}\t")
            writer.writeln(self.type)

            # Write data grid
            if self.data_grid is not None:
                writer.writeln("")
                self.data_grid.write_report(writer)

            # Write input values and their frequencies
            if self.input_values is not None:
                writer.writeln("")
                writer.writeln("Input values")
                for i, input_value in enumerate(self.input_values):
                    writer.write(f"\t{input_value}\t")
                    writer.writeln(str(self.input_value_frequencies[i]))


class VariablePairStatistics:
    """Variable pair information and statistics

    Parameters
    ----------
    json_data : dict, optional
        JSON data of an element of the list found at the
        ``variablesPairStatistics`` field within the ``bivariatePreparationReport``
        field of a Khiops JSON report file. If not specified it returns an empty object.

        .. note::
            The ``data_grid`` field is considered as "detail" and is not initialized in
            the constructor. Instead, it is initialized explicitly via the
            `init_details` method. This allows to make partial initializations for large
            reports.


    Attributes
    ----------
    rank : str
        Variable rank with respect to its level. Lower Rank = Higher Level.
    name1 : str
        Name of the pair's first variable.
    name2 : str
        Name of the pair's second variable.
    level : float
        Predictive importance of the pair.
    level1 : float
        Predictive importance of the first variable.
    level2 : float
        Predictive importance of the second variable.
    delta_level : float
        Difference between the pair's level and the sum of those of its components
        (``delta_level = level - level1 - level2``).
    variable_number : int
        Number of active variables in the pair:
            - 0 means that there is no information in any of the variables
            - 1 means that the pair information reduces to that of any of its components
            - 2 means that the two variables are jointly informative

    part_number1 : int
        Number of parts of the first variable partition.
    part_number2 : int
        Number of parts of the second variable partition.
    cell_number : int
        Number of cells generated of the pair grid.
    construction_cost : float
        *Advanced:* Construction cost of the variable. More complex variables cost more.
    preparation_cost : float
        *Advanced:* Partition model cost. More complex partitions cost more.
    data_cost : float
        *Advanced:* Negative log-likelihood of the variable given a preparation model
        and a construction model.
    data_grid : `DataGrid`
        A density estimation of the partitioned pair of variable with respect to the
        target.
    """

    def __init__(self, json_data=None):
        """See class docstring"""
        # Common attributes
        self.rank = ""
        self.name1 = ""
        self.name2 = ""
        self.level = 0
        self.level1 = None
        self.level2 = None
        self.delta_level = None
        self.variable_number = 0
        self.part_number1 = 0
        self.part_number2 = 0
        self.cell_number = 0

        # Cost attributes
        self.construction_cost = None
        self.preparation_cost = None
        self.data_cost = None

        # Data grid for density estimation (detail)
        self.data_grid = None

        # Initialization from JSON data (except for details' attributes)
        if json_data is not None:
            self.rank = json_data.get("rank")
            self.name1 = json_data.get("name1")
            self.name2 = json_data.get("name2")
            self.delta_level = json_data.get("deltaLevel")
            self.level = json_data.get("level")
            self.level1 = json_data.get("level1")
            self.level2 = json_data.get("level2")
            self.variable_number = json_data.get("variables")
            self.part_number1 = json_data.get("parts1")
            self.part_number2 = json_data.get("parts2")
            self.cell_number = json_data.get("cells")

            # Cost attributes
            self.construction_cost = json_data.get("constructionCost")
            self.preparation_cost = json_data.get("preparationCost")
            self.data_cost = json_data.get("dataCost")

    def init_details(self, json_detailed_data=None):
        """Initializes the details' attributes from a Python JSON object

        Parameters
        ----------
        json_data : dict, optional
            JSON data of an element of the list found at
            the ``variablesPairsDetailedStatistics`` field within the
            ``bivariatePreparationReport`` field of a Khiops JSON report file. If not
            specified it leaves the object as-is.
        """
        if json_detailed_data is not None:
            json_data_grid = json_detailed_data.get("dataGrid")
            if json_data_grid is not None:
                self.data_grid = DataGrid(json_data_grid)

    def is_detailed(self):
        """Returns True if the report contains any detailed information

        Returns
        -------
        bool
            True if the report contains any detailed information.
        """
        return self.data_grid is not None

    def write_report_header_line(self, writer):
        """Writes the header line of a TSV report into a writer object

        The header is the same for all variable types.

        Parameters
        ----------
        writer : `.PyKhiopsOutputWriter`
            Output writer.
        """
        # Write identifier column names
        writer.write("Rank\t")
        writer.write("Name 1\t")
        writer.write("Name 2\t")

        # Supervised case: Write level column names
        if self.delta_level is not None:
            writer.write("DeltaLevel\t")
            writer.write("Level\t")
            writer.write("Level 1\t")
            writer.write("Level 2\t")
        # Unsupervised case: Write unsupervised level column name
        else:
            writer.write("Level\t")

        # Write data grid summary column names
        writer.write("Variables\t")
        writer.write("Parts 1\t")
        writer.write("Parts 2\t")
        writer.write("Cells\t")

        # Write cost column names
        writer.write("Construction cost\t")
        writer.write("Preparation cost\t")
        writer.writeln("Data cost")

    def write_report_line(self, writer):
        """Writes a line of the TSV report into a writer object

        Parameters
        ----------
        writer : `.PyKhiopsOutputWriter`
            Output writer.
        """
        # Write identifier attributes
        writer.write(f"{self.rank}\t")
        writer.write(f"{self.name1}\t")
        writer.write(f"{self.name2}\t")

        # Supervised case: Write level attributes
        if self.delta_level is not None:
            writer.write(f"{self.delta_level}\t")
            writer.write(f"{self.level}\t")
            writer.write(f"{self.level1}\t")
            writer.write(f"{self.level2}\t")
        # Unsupervised case: Write unsupervised level attribute
        else:
            writer.write(f"{self.level}\t")

        # Write attributes summarizing the data grid structure
        writer.write(f"{self.variable_number}\t")
        writer.write(f"{self.part_number1}\t")
        writer.write(f"{self.part_number2}\t")
        writer.write(f"{self.cell_number}\t")

        # Write cost attributes
        writer.write(f"{self.construction_cost}\t")
        writer.write(f"{self.preparation_cost}\t")
        writer.writeln(str(self.data_cost))

    def write_report_details(self, writer):
        """Writes the details' attributes into a writer object

        Parameters
        ----------
        writer : `.PyKhiopsOutputWriter`
            Output writer.
        """
        if self.is_detailed():
            writer.writeln("")
            writer.writeln(f"Rank\t{self.rank}")
            writer.writeln("")
            self.data_grid.write_report(writer)


class DataGrid:
    """A piecewise constant probability density estimation

    A data grid represents one or many variables referred to as "dimensions" to
    differentiate them from the original data variables. Each dimension can be
    partitioned by:

    - Intervals for numerical variables
    - Values (singletons) / Value groups for categorical variables

    The Cartesian product of the unidimensional partitions provides a multivariate
    partition of cells whose frequencies allow to estimate the multivariate probability
    density.

    In the univariate case, the data grid is simply an histogram. In the case of
    multiple variables, the data grid may be supervised or not. If supervised, the
    target variable is the last one, and the data grid represents the conditional
    density estimator of the source variable with respect to the target. Otherwise, it
    represents a joint density estimator.

    In case of an unsupervised data grid, the cells are described by their index on the
    variable partitions, together with their frequencies. For a supervised data grid,
    the cells are described by their index on the input variables partitions, and a
    vector of target frequencies is associated to each cell.

    Parameters
    ----------
    json_data : dict, optional
        JSON data at a ``dataGrid`` field of an element of the list found at the
        ``variablesDetailedStatistics`` field within the ``preparationReport`` field of
        a Khiops JSON report file. If not specified it returns an empty object.

    Attributes
    ----------
    is_supervised : bool
        ``True`` if the data grid is supervised (there is a target).
    dimensions : list of `DataGridDimension`
        The dimensions of the data grid.
    frequencies : list of int
        *Unsupervised only:* Frequencies for each part.
    part_interests : list of float
        *Supervised univariate only:* Prediction interests for each part of the input
        dimension. Synchronized with ``dimensions[0].partition``.
    part_target_frequencies : list
        *Supervised univariate only:* List of frequencies per target value for each part
        of the input dimension. Synchronized with ``dimensions[0].partition``.
    cell_ids : list of str
        *Multivariate only:* Unique identifiers of the grid's cells.
    cell_part_indexes : list
        *Multivariate only:* List of dimension indexes defining each cell. Synchronized
        with ``cell_ids``.
    cell_frequencies : list of int
        *Unsupervised multivariate only:* Frequencies for each cell. Synchronized with
        ``cell_ids``.
    cell_target_frequencies : list
        *Supervised multivariate only:* List of frequencies per target value for each
        cell. Synchronized with ``cell_ids``.
    """

    def __init__(self, json_data=None):
        """See class docstring"""
        # Supervised data grid flag
        self.is_supervised = False

        # Dimensions: one per variable, with its partition
        self.dimensions = []

        # Frequency per part (univariate) or cell (multivariate)
        self.frequencies = None

        # Attributes for a multivariate data grid
        self.cell_ids = None
        self.cell_part_indexes = None
        self.cell_frequencies = None

        # Attributes for a supervised univariate data grid
        self.part_target_frequencies = None
        self.part_interests = None

        # Attributes for a multivariate supervised data grid
        self.cell_part_indexes = None
        self.cell_target_frequencies = None
        self.cell_interests = None

        # Return if no JSON data
        if json_data is not None:
            # Initialize from JSON data
            self.is_supervised = json_data.get("isSupervised")
            json_dimensions = json_data.get("dimensions")
            for json_dimension in json_dimensions:
                dimension = DataGridDimension(json_dimension)
                self.dimensions.append(dimension)

            # Initialize frequencies in case of univariate unsupervised data grid
            if not self.is_supervised and len(self.dimensions) == 1:
                self.frequencies = json_data.get("frequencies")

            # Initialize attributes for a multivariate unsupervised data grid
            elif not self.is_supervised and len(self.dimensions) > 1:
                self.cell_ids = json_data.get("cellIds")
                self.cell_part_indexes = json_data.get("cellPartIndexes")
                self.cell_frequencies = json_data.get("cellFrequencies")

            # Initialize attributes for a supervised data grid with one input variable
            elif self.is_supervised and len(self.dimensions) == 2:
                self.part_target_frequencies = json_data.get("partTargetFrequencies")
                self.part_interests = json_data.get("partInterests")

            # Initialize attributes for a supervised data grid with several input vars.
            elif self.is_supervised and len(self.dimensions) > 2:
                self.cell_ids = json_data.get("cellIds")
                self.cell_part_indexes = json_data.get("cellPartIndexes")
                self.cell_frequencies = json_data.get("cellFrequencies")
                self.cell_target_frequencies = json_data.get("cellTargetFrequencies")
                self.cell_interests = json_data.get("cellInterests")

    def write_report(self, writer):
        """Writes the instance's TSV report into a writer object

        Parameters
        ----------
        writer : `.PyKhiopsOutputWriter`
            Output writer.
        """
        # Write data grid type
        writer.write("Data grid\t")
        if self.is_supervised:
            writer.writeln("Supervised")
        else:
            writer.writeln("Unsupervised")

        # Write dimensions
        writer.writeln("Dimensions")
        for dimension in self.dimensions:
            dimension.write_report(writer)

        # Write data grid cells
        writer.writeln("Cells")

        # Univariate unsupervised data grid: Write frequencies per part
        if not self.is_supervised and len(self.dimensions) == 1:
            dimension = self.dimensions[0]
            for i, part in enumerate(dimension.partition):
                # Write header
                if i == 0:
                    writer.writeln(f"{part.part_type()}\tFrequency")
                # Write line
                writer.writeln(f"{part}\t{self.frequencies[i]}")

        # Multivariate unsupervised data grid: Write frequencies per cell
        elif not self.is_supervised and len(self.dimensions) > 1:
            for i, part_indexes in enumerate(self.cell_part_indexes):
                cell_id = self.cell_ids[i]
                frequency = self.cell_frequencies[i]

                # Write header
                if i == 0:
                    writer.write("Cell id\t")
                    for dimension in self.dimensions:
                        writer.write(f"{dimension.variable}\t")
                    writer.writeln("Frequency")

                # Write line
                writer.write(f"{cell_id}\t")
                for j, part_index in enumerate(part_indexes):
                    dimension = self.dimensions[j]
                    part = dimension.partition[part_index]
                    writer.write(f"{part}\t")
                writer.writeln(str(frequency))

            # Write an additional contingency table in the bidimensional case
            if len(self.dimensions) == 2:
                partition_x = self.dimensions[0].partition
                partition_y = self.dimensions[1].partition

                # Initialize table with the cell frequencies, zero by default
                frequency_matrix = [
                    [0 for _ in range(len(partition_x))]
                    for _ in range(len(partition_y))
                ]
                for i, part_indexes in enumerate(self.cell_part_indexes):
                    frequency = self.cell_frequencies[i]
                    frequency_matrix[part_indexes[1]][part_indexes[0]] = frequency

                # Write contingency table
                writer.writeln("Confusion matrix")
                for part_x in partition_x:
                    writer.write(f"\t{part_x}")
                writer.writeln("")
                for index_y, part_y in enumerate(partition_y):
                    writer.write(str(part_y))
                    for index_x in range(len(partition_x)):
                        writer.write(f"\t{frequency_matrix[index_y][index_x]}")
                    writer.writeln("")

        # Supervised data grid with one input variable:
        # Write frequencies for each input part and for each target part
        elif self.is_supervised and len(self.dimensions) == 2:
            input_dimension = self.dimensions[0]
            target_dimension = self.dimensions[1]
            for i, part in enumerate(input_dimension.partition):
                target_frequencies = self.part_target_frequencies[i]
                interest = self.part_interests[i]
                # Write header
                if i == 0:
                    writer.write(part.part_type())
                    for target_part in target_dimension.partition:
                        writer.write(f"\t{target_part}")
                    writer.writeln("\tInterest")
                # Write line
                writer.write(str(part))
                for frequency in target_frequencies:
                    writer.write(f"\t{frequency}")
                writer.writeln(f"\t{interest}")

        # Supervised data grid with several input variables:
        # Write frequencies per input cell part, for each target part
        elif self.is_supervised and len(self.dimensions) > 2:
            for i, input_part_indexes in enumerate(self.cell_part_indexes):
                cell_id = self.cell_ids[i]
                target_frequencies = self.cell_target_frequencies[i]
                interest = self.cell_interests[i]
                # Write header
                if i == 0:
                    writer.write("Cell id\t")
                    input_dimensions = self.dimensions[:-1]
                    for input_dimension in input_dimensions:
                        writer.write(f"{input_dimension.variable}\t")
                    target_dimension = self.dimensions[-1]
                    for target_part in target_dimension.partition:
                        writer.write(f"\t{target_part}")
                    writer.writeln("\tInterest")
                # Write line
                writer.write(f"{cell_id}\t")
                for j, input_part_index in enumerate(input_part_indexes):
                    dimension = self.dimensions[j]
                    part = dimension.partition[input_part_index]
                    writer.write(f"{part}\t")
                for frequency in target_frequencies:
                    writer.write(f"\t{frequency}")
                writer.writeln(f"\t{interest}")


class DataGridDimension:
    """A dimension (variable) of a data grid

    Parameters
    ----------
    json_data : dict, optional
        JSON data of an element at the ``dimensions`` field of a ``dataGrid`` field of
        an element of the list found at the ``variablesDetailedStatistics`` field within
        the ``preparationReport`` field of a Khiops JSON report file. If not specified
        it returns an empty object.

    Attributes
    ----------
    variable : str
        Variable name
    type : "Numerical" or "Categorical"
        Variable type.
    partition_type : "Intervals", "Values" or "Value groups"
        Partition type.
    partition : list
        The dimension parts. The list objects are of type:
            - `PartInterval`: If ``partition type`` is "Intervals"
            - `PartValue`: If ``partition_type`` is "Values"
            - `PartValueGroup`: If ``partition_type`` is "Value groups"
    """

    def __init__(self, json_data=None):
        """See class docstring"""
        self.variable = ""
        self.type = ""
        self.partition_type = ""
        self.partition = []

        # Initialize attributes from JSON data
        if json_data is not None:
            self.variable = json_data.get("variable")
            self.type = json_data.get("type")
            self.partition_type = json_data.get("partitionType")
            json_partition = json_data.get("partition")

            # Initialize Numerical interval partition
            if self.partition_type == "Intervals":
                # Initialize intervals
                for json_interval in json_partition:
                    interval = PartInterval(json_interval)
                    self.partition.append(interval)

                # Initialize open interval flags
                first_interval = self.partition[0]
                if first_interval.is_missing:
                    first_interval = self.partition[1]
                first_interval.is_left_open = True
                last_interval = self.partition[-1]
                last_interval.is_right_open = True

            # Initialize Value partition (singletons)
            if self.partition_type == "Values":
                for json_value in json_partition:
                    value = PartValue(json_value)
                    self.partition.append(value)

            # Initialize ValueGroups partition
            if self.partition_type == "Value groups":
                # Initialize value groups
                for json_value_group in json_partition:
                    value_group = PartValueGroup(json_value_group)
                    self.partition.append(value_group)

                # Initialize default group containing all values not specified
                # in any partition element and all unknown values
                default_group_index = json_data.get("defaultGroupIndex")
                default_group = self.partition[default_group_index]
                default_group.is_default_part = True

    def write_report(self, writer):
        """Writes the instance's TSV report into a writer object

        Parameters
        ----------
        writer : `.PyKhiopsOutputWriter`
            Output writer.
        """
        writer.write(f"{self.variable}\t")
        writer.write(f"{self.type}\t")
        writer.writeln(self.partition_type)
        for part in self.partition:
            writer.write("\t")
            part.write_report_line(writer)


class PartInterval:
    """Element of a numerical interval partition in a data grid

    Parameters
    ----------
    json_data : dict, optional
        JSON data of the ``partition`` field of a ``dataGrid`` field of an element of
        the list found at the ``variablesDetailedStatistics`` field within the
        ``preparationReport`` field of a Khiops JSON report file. If not specified it
        returns an empty object.

    Attributes
    ----------
    lower_bound : float
        The lower bound of the interval.
    upper_bound : float
        The upper bound of the interval.
    is_missing : bool
        True if it is the missing values part (bounds are ``None``).
    is_left_open : bool
        True if the interval has no minimum. ``lower_bound`` still contains the minimum
        value seen on data.
    is_right_open : bool
        True if the interval has no maximum. ``upper_bound`` still contains the minimum
        value seen on data.
    """

    def __init__(self, json_data=None):
        """See class docstring"""
        self.lower_bound = None
        self.upper_bound = None
        self.is_missing = False
        self.is_left_open = False
        self.is_right_open = False

        # Initialization from JSON data
        if json_data is not None:
            # Missing value if array of bounds is empty
            if len(json_data) == 0:
                self.is_missing = True
            # Actual interval if array of bounds not empty
            else:
                self.lower_bound = json_data[0]
                self.upper_bound = json_data[1]

    def __str__(self):
        """Returns a human readable string representation"""
        if self.is_missing:
            label = "Missing"
        else:
            if self.is_left_open:
                label = "]-inf"
            else:
                label = f"]{self.lower_bound}"
            if self.is_right_open:
                label += ";+inf["
            else:
                label += f";{self.upper_bound}]"
        return label

    def part_type(self):
        """Type of this part

        Returns
        -------
        str
            Only possible value: "Interval".

        """
        return "Interval"

    def write_report_line(self, writer):
        """Writes a line of the TSV report into a writer object

        Parameters
        ----------
        writer : `.PyKhiopsOutputWriter`
            Output writer.
        """
        # Write part label
        writer.write(str(self))

        # Write interval bounds
        if not self.is_missing:
            writer.write(f"\t{self.lower_bound}")
            writer.write(f"\t{self.upper_bound}")
        writer.writeln("")


class PartValue:
    """Element of a value partition (singletons) in a data grid

    Attributes
    ----------
      value : str
          A representation of the value defining the singleton.
    """

    def __init__(self, json_data=None):
        """See class docstring"""
        self.value = ""
        if json_data is not None:
            self.value = json_data

    def __str__(self):
        """Returns a readable string representation"""
        return self.value

    def part_type(self):
        """Type of the instance

        Returns
        -------
        str
            Only possible value: "Value".

        """
        return "Value"

    def write_report_line(self, writer):
        """Writes a line of the TSV report into a writer object

        Parameters
        ----------
        writer : `.PyKhiopsOutputWriter`
            Output writer.
        """
        writer.writeln(str(self))


class PartValueGroup:
    """Element of a categorical partition in a data grid

    Attributes
    ----------
    values : list of str
        The group's values.
    is_default_part : bool
        True if this part is dedicated to all unknown values.
    """

    def __init__(self, json_data=None):
        """See class docstring"""
        self.values = []
        self.is_default_part = False
        if json_data is not None:
            self.values = json_data

    def __str__(self):
        """Returns a human readable string representation"""
        label = "{"
        for i, value in enumerate(self.values):
            if i > 0:
                label += ", "
            if i == 3:
                label += "..."
                break
            label += value
        label += "}"
        return label

    def part_type(self):
        """Type of the instance

        Returns
        -------
        str
            Only possible value: "Value group".

        """
        return "Value group"

    def write_report_line(self, writer):
        """Writes a line of the TSV report into a writer object

        Parameters
        ----------
        writer : `.PyKhiopsOutputWriter`
            Output writer.
        """
        # Write part label
        writer.write(str(self))

        # Write group values
        for value in self.values:
            writer.write(f"\t{value}")
        if self.is_default_part:
            writer.write("\t * ")
        writer.writeln("")


class TrainedPredictor:
    """Trained predictor information

    Parameters
    ----------
    json_data : dict
        JSON data of an element of the list found at the ``trainedPredictors`` field
        within the ``modelingReport`` field of a Khiops JSON report file. If not
        specified it returns an empty object.

        .. note::
            The ``selected_variables`` field is considered a "detail" and is not
            initialized in the constructor. Instead, it is initialized explicitly via
            the `init_details` method. This allows to make partial initializations for
            large reports.

    Attributes
    ----------
    type : str
        Predictor type. Valid values are found in the ``predictor_types`` class
        attribute. They are:

        - "Selective Naive Bayes"
        - "MAP Naive Bayes" **Deprecated**
        - "Naive Bayes"
        - "Univariate"

    family : "Classifier" or "Regressor"
        Predictor family name. Valid values are found in the ``predictor_families``
        class variable.
    name : str
        Human readable predictor name.
    variable_number : int
        Number of variables used by the predictor.
    selected_variables : list of `SelectedVariable`
        Variables used by the predictor. Only for types "Selective Naive Bayes" and "MAP
        Naive Bayes".
    """

    predictor_types = ["Classifier", "Regressor"]
    predictor_families = [
        "Selective Naive Bayes",
        "MAP Naive Bayes",
        "Naive Bayes",
        "Univariate",
    ]

    def __init__(self, json_data=None):
        """See class docstring"""
        # Common attributes
        self.rank = ""
        self.type = ""
        self.family = ""
        self.name = ""
        self.variable_number = 0

        # Detailed attributes, depending on the predictor
        self.selected_variables = None

        # Initialization from JSON data (except for details' attributes)
        if json_data is not None:
            self.rank = json_data.get("rank")
            self.type = json_data.get("type")
            self.family = json_data.get("family")
            self.name = json_data.get("name")
            self.variable_number = json_data.get("variables")

    def init_details(self, json_detailed_data=None):
        """Initializes the details' attributes from a Python JSON object

        Parameters
        ----------
        json_data : dict, optional
            JSON data of the dictionary found at the ``trainedPredictorsDetails`` field
            within the ``modelingReport`` field of a Khiops JSON report file. If not
            specified it leaves the object as-is.
        """
        if json_detailed_data is not None:
            json_selected_variables = json_detailed_data.get("selectedVariables")
            if json_selected_variables is not None:
                # Initialize the selected variable attributes
                self.selected_variables = []
                for json_data in json_selected_variables:
                    selected_variable = SelectedVariable(json_data)
                    self.selected_variables.append(selected_variable)

    def is_detailed(self):
        """Returns True if the report contains any detailed information

        Returns
        -------
        bool
            True if the report contains any detailed information.
        """
        return self.selected_variables is not None

    def write_report_header_line(self, writer):
        """Writes the header line of a TSV report into a writer object

        The header is the same for all variable types.

        Parameters
        ----------
        writer : `.PyKhiopsOutputWriter`
            Output writer.
        """
        writer.write("Rank\t")
        writer.write("Type\t")
        writer.write("Family\t")
        writer.write("Name\t")
        writer.writeln("Variables")

    def write_report_line(self, writer):
        """Writes a line of the TSV report into a writer object

        Parameters
        ----------
        writer : `.PyKhiopsOutputWriter`
            Output writer.
        """
        writer.write(f"{self.rank}\t")
        writer.write(f"{self.type}\t")
        writer.write(f"{self.family}\t")
        writer.write(f"{self.name}\t")
        writer.writeln(str(self.variable_number))

    def write_report_details(self, writer):
        """Writes the details of the TSV report into a writer object

        Parameters
        ----------
        writer : `.PyKhiopsOutputWriter`
            Output writer.
        """
        if self.is_detailed():
            # Header line
            writer.writeln("")
            writer.write("Rank\t")
            writer.writeln(self.rank)
            # Selected variables
            if self.selected_variables is not None:
                writer.writeln("")
                writer.writeln("Selected variables")
                for i, selected_variable in enumerate(self.selected_variables):
                    if i == 0:
                        selected_variable.write_report_header_line(writer)
                    selected_variable.write_report_line(writer)


class SelectedVariable:
    """Information about a selected variable in a predictor

    Parameters
    ----------
    json_data : dict, optional
        JSON data representing an element of the ``selectedVariables`` list in the
        ``trainedPredictorsDetails`` field within the ``modelingReport`` field of a
        Khiops JSON report file. If not specified it leaves the object as-is.

    Attributes
    ----------
    name : str
        Human readable variable name.
    prepared_name : str
        Internal variable name.
    level : float
        Variable level.
    weight : float
        Variable weight in the model.
    importance : float
        A measure of overall importance of the variable in the model. It is the
        geometric mean of the level and weight.
    map : bool
         True if the variable is in the MAP model.
         **Deprecated**: Will be removed in Khiops Python 11.
    """

    def __init__(self, json_data=None):
        """See class docstring"""
        # Common attributes
        self.prepared_name = ""
        self.name = ""
        self.level = ""

        # Attributes specific to Selective Naive Bayes
        self.weight = None
        self.importance = None
        self.map = None

        # Initialization from JSON data (except for details' attributes)
        if json_data is not None:
            self.prepared_name = json_data.get("preparedName")
            self.name = json_data.get("name")
            self.level = json_data.get("level")
            self.weight = json_data.get("weight")
            if json_data.get("importance"):
                self.importance = json_data.get("importance")
            if json_data.get("map") is not None:
                self.map = json_data.get("map")

    def write_report_header_line(self, writer):
        """Writes the header line of a TSV report into a writer object

        The header is the same for all variable types.

        Parameters
        ----------
        writer : `.PyKhiopsOutputWriter`
            Output writer.
        """
        writer.write("Prepared name\t")
        writer.write("Name\t")
        writer.write("Level")
        writer.write("\tWeight")
        if self.importance is not None:
            writer.write("\tImportance")
        if self.map is not None:
            writer.write("\tMAP")
        writer.writeln("")

    def write_report_line(self, writer):
        """Writes a line of the TSV report into a writer object

        Parameters
        ----------
        writer : `.PyKhiopsOutputWriter`
            Output writer.
        """
        writer.write(f"{self.prepared_name}\t")
        writer.write(f"{self.name}\t")
        writer.write(str(self.level))
        writer.write(f"\t{self.weight}")
        if self.importance is not None:
            writer.write(f"\t{self.importance}")
        elif self.map is not None:
            writer.write("\t1")

        writer.writeln("")


class PredictorPerformance:
    """A predictor's performance evaluation

    This class describes the performance of a predictor (classifier or regressor).

    Parameters
    ----------
    json_data : dict
        JSON data of an element of the dictionary found at the ``predictorPerformances``
        field within the one of the evaluation report fields of a Khiops JSON report
        file. If not specified it returns an empty object.

        .. note::
            The ``confusion_matrix`` field is considered as "detail" and is not
            initialized in the constructor. Instead, it is initialized explicitly via
            the `init_details` method. This allows to make partial initializations for
            large reports.


    Attributes
    ----------
    rank : str
        An string index representing the order in the report.
    type : "Classifier" or "Regressor"
        Type of the predictor.
    name : str
        Human readable name.
    data_grid : `DataGrid`
        Data grid representing the distribution of the target values per part of the
        descriptive variable in the evaluated dataset.
    accuracy : float
        *Classifier only:* Accuracy.
    compression : float
        *Classifier only:* Compression rate.
    auc : float
        *Classifier only:* Area under the ROC curve.
    confusion_matrix : ConfusionMatrix
        *Classifier only:* Confusion matrix.
    rmse : float
        *Regressor only:* Root mean square error.
    mae : float
        *Regressor only:* Mean absolute error.
    nlpd : float
        *Regressor only:* Negative log predictive density.
    rank_rmse : float
        *Regressor only:* Root mean square error on the target's value rank.
    rank_mae : float
        *Regressor only:* Mean absolute error on the target's value rank.
    rank_nlpd : float
        *Regressor only:* Negative log predictive density on the target's value rank.
    """

    def __init__(self, json_data=None):
        """See class docstring"""
        # Common attributes
        self.rank = ""
        self.type = ""
        self.family = ""
        self.name = ""

        # Classifier evaluation criteria
        self.accuracy = None
        self.compression = None
        self.auc = None

        # Regressor evaluation criteria
        self.rmse = None
        self.mae = None
        self.nlpd = None
        self.rank_rmse = None
        self.rank_mae = None
        self.rank_nlpd = None

        # Detailed attributes, depending on the predictor
        self.confusion_matrix = None  # for a Classifier
        self.data_grid = None  # for a univariate predictor

        # Initialization from the input JSON object (not the detailed info)
        if json_data is not None:
            # Initialize common attributes
            self.rank = json_data.get("rank")
            self.type = json_data.get("type")
            self.family = json_data.get("family")
            self.name = json_data.get("name")

            # Initialize classifier evaluation criteria
            if self.type == "Classifier":
                self.accuracy = json_data.get("accuracy")
                self.compression = json_data.get("compression")
                self.auc = json_data.get("auc")

            # Initialize regressor evaluation criteria
            if self.type == "Regressor":
                self.rmse = json_data.get("rmse")
                self.mae = json_data.get("mae")
                self.nlpd = json_data.get("nlpd")
                self.rank_rmse = json_data.get("rankRmse")
                self.rank_mae = json_data.get("rankMae")
                self.rank_nlpd = json_data.get("rankNlpd")

    def init_details(self, json_detailed_data=None):
        """Initializes the details' attributes from a python JSON object"""
        if json_detailed_data is not None:
            # Confusion matrix
            json_confusion_matrix = json_detailed_data.get("confusionMatrix")
            if json_confusion_matrix is not None:
                self.confusion_matrix = ConfusionMatrix(json_confusion_matrix)

            # Data grid
            json_data_grid = json_detailed_data.get("dataGrid")
            if json_data_grid is not None:
                self.data_grid = DataGrid(json_data_grid)

    def is_detailed(self):
        """Returns True if the report contains any detailed information

        Returns
        -------
        bool
            True if the report contains any detailed information.
        """
        return self.confusion_matrix is not None or self.data_grid is not None

    def get_metric_names(self):
        """Returns the available univariate metrics

        Returns
        -------
        list of str
            The names of the available metrics.
        """
        if self.type == "Classifier":
            metric_names = ["accuracy", "compression", "auc"]
        elif self.type == "Regressor":
            metric_names = ["rmse", "mae", "nlpd", "rank_rmse", "rank_mae", "rank_nlpd"]
        else:
            raise ValueError(
                "PredictorPerformance type must be 'Classifier' or 'Regressor', "
                f"not '{self.type}'"
            )
        return metric_names

    def get_metric(self, metric_name):
        """Returns the value of the specified metric

        .. note:: The available metrics is available via the method `get_metric_names`.

        Parameters
        ----------
        metric_name : str
            A metric name (case insensitive).

        Returns
        -------
        float
            The value of the specified metric.
        """
        # Search the lower cased metric name in the list, report error if not found
        lowercase_metric_name = metric_name.lower()
        metric = None
        for name in self.get_metric_names():
            if lowercase_metric_name == name:
                metric = getattr(self, lowercase_metric_name)
        if metric is None:
            metric_list_msg = ",".join(self.get_metric_names())
            raise ValueError(
                f"Invalid metric: '{metric_name}'. Choose among {metric_list_msg}"
            )
        return metric

    def write_report_header_line(self, writer):
        """Writes the header line of a TSV report into a writer object

        The header is the same for all variable types.

        Parameters
        ----------
        writer : `.PyKhiopsOutputWriter`
            Output writer.
        """
        writer.write("Rank\t")
        writer.write("Type\t")
        writer.write("Family\t")
        writer.write("Name\t")
        if self.type == "Classifier":
            writer.write("Accuracy\t")
            writer.write("Compression\t")
            writer.writeln("AUC")
        if self.type == "Regressor":
            writer.write("RMSE\t")
            writer.write("MAE\t")
            writer.write("NLPD\t")
            writer.write("RankRMSE\t")
            writer.write("RankMAE\t")
            writer.writeln("RankNLPD")

    def write_report_line(self, writer):
        """Writes a line of the TSV report into a writer object

        Parameters
        ----------
        writer : `.PyKhiopsOutputWriter`
            Output writer.
        """
        writer.write(f"{self.rank}\t")
        writer.write(f"{self.type}\t")
        writer.write(f"{self.family}\t")
        writer.write(f"{self.name}\t")

        metrics = [str(self.get_metric(name)) for name in self.get_metric_names()]
        writer.writeln("\t".join(metrics))

    def write_report_details(self, writer):
        """Writes the details of the TSV report into a writer object

        Parameters
        ----------
        writer : `.PyKhiopsOutputWriter`
            Output writer.
        """
        if self.is_detailed():
            # Write header line
            writer.writeln("")
            writer.writeln(f"Rank\t{self.rank}")

            # Write confusion matrix if available
            if self.confusion_matrix is not None:
                writer.writeln("")
                self.confusion_matrix.write_report(writer)

            # Write data grid if available
            if self.data_grid is not None:
                writer.writeln("")
                self.data_grid.write_report(writer)


class ConfusionMatrix:
    """A classifier's confusion matrix

    Parameters
    ----------
    json_data : dict
        JSON data of the ``confusionMatrix`` field of an element of the dictionary found
        at the ``predictorsDetailedPerformances`` field within one of the evaluation
        report fields of a Khiops JSON report file. If not specified it returns an empty
        object.

    Attributes
    ----------
    values : list of str
        Values of the target variable.
    matrix : list
        Matrix of predicted frequencies vs target frequencies. This list is synchornized
        with ``values``. Each list element represents a row of the confusion matrix,
        that is, the target frequencies for a fixed predicted target value.
    """

    def __init__(self, json_data=None):
        """See class docstring"""
        self.values = []
        self.matrix = []

        # Initialization from the input JSON object
        if json_data is not None:
            self.values = json_data.get("values")
            self.matrix = json_data.get("matrix")

    def write_report(self, writer):
        """Writes the instance's TSV report into a writer object

        Parameters
        ----------
        writer : `.PyKhiopsOutputWriter`
            Output writer.
        """
        # Write header
        writer.writeln("Confusion matrix")

        # Write observed values
        for observed_value in self.values:
            writer.write(f"\t{observed_value}")
        writer.writeln("")

        # Write observed frequencies for each predicted value
        for i, predicted_value in enumerate(self.values):
            observed_frequencies = self.matrix[i]
            writer.write(f"${predicted_value}")
            for frequency in observed_frequencies:
                writer.write(f"\t{frequency}")
            writer.writeln("")


class PredictorCurve:
    """A lift curve for a classifier or a REC curve for a regressor

    Parameters
    ----------
    json_data : dict
        JSON data of an element of the ``liftCurves`` or ``recCurves`` field of one of
        the evaluation report fields of a Khiops JSON report file. If not specified it
        returns an empty object.

    Attributes
    ----------
    type : "Lift" (classifier) or "REC" (regressor)
        Type of predictor curve.
    name : str
        Name of evaluated predictor.
    values : list of float
        The curve's y-axis values.
    """

    def __init__(self, json_data=None):
        """See class docstring"""
        self.type = None
        self.name = None
        self.values = None

        # Initialization from the input JSON object
        if json_data is not None:
            if "classifier" in json_data:
                self.type = "Lift"
                self.name = json_data.get("classifier")
            elif "regressor" in json_data:
                self.type = "REC"
                self.name = json_data.get("regressor")
            else:
                raise ValueError(
                    "Either 'classifier' or 'regressor' must be in curve data"
                )
            self.values = json_data.get("values")
