######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
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
    |- preparation_report           |
    |- text_preparation_report      |->  PreparationReport
    |- tree_preparation_report      |
    |- bivariate_preparation_report  ->  BivariatePreparationReport
    |- modeling_report               ->  ModelingReport
    |- train_evaluation_report      |
    |- test_evaluation_report       |->  EvaluationReport
    |- evaluation_report            |

These sub-classes in turn use other tertiary classes to represent specific information
pieces of each report. The dependencies for the classes `PreparationReport` and
`BivariatePreparationReport` are::

    PreparationReport
    |- variables_statistics -> list of VariableStatistics
    |- trees                -> list of Tree (only for tree_preparation_report)

    BivariatePreparationReport
    |- variable_pair_statistics -> list of VariablePairStatistics

    VariableStatistics
    |- data_grid       -> DataGrid
    |- modl_histograms -> ModlHistograms

    VariablePairStatistics
    |- data_grid -> DataGrid

    Tree
    |- target_partition -> TargetPartition
    |- nodes -> list of TreeNode

    TargetPartition
    |- partition -> list of PartInterval

    DataGrid
    |- dimensions -> list of DataGridDimension

    ModlHistograms
    |- histograms -> list of Histogram

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
module look at their ``to_dict`` methods which write Python dictionaries in the
same format as the Khiops JSON reports.
"""
import io
import warnings

from khiops.core.exceptions import KhiopsJSONError
from khiops.core.internals.common import deprecation_message, type_error_message
from khiops.core.internals.io import (
    KhiopsJSONObject,
    KhiopsOutputWriter,
    flexible_json_load,
)


class AnalysisResults(KhiopsJSONObject):
    """Main class containing the information of a Khiops JSON file

    Sub-reports not available in the JSON data are optional (set to ``None``).

    Parameters
    ----------
    json_data : dict, optional
        A dictionary representing the data of a Khiops JSON report file. If not
        specified it returns an empty instance.

        .. note::
            See also the `.read_analysis_results_file` function to obtain an instance
            of this class from a Khiops JSON file.

    Attributes
    ----------
    tool : str
        Name of the Khiops tool that generated the report.
    version : str
        Version of the Khiops tool that generated the report.
    short_description : str
        Short description defined by the user.
    khiops_encoding : str
        Encoding of the Khiops report file.
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
        # Call the parent's initialization
        super().__init__(json_data=json_data)

        # Transform to an empty dictionary if json_data is not specified
        if json_data is None:
            json_data = {}
        # Otherwise check that the tool is the proper one (it was set by the parent)
        elif self.tool != "Khiops":
            raise KhiopsJSONError(
                f"'tool' value in JSON data must be 'Khiops', not '{self.tool}'"
            )

        # Initialize report basic data
        self.short_description = json_data.get("shortDescription")
        json_logs = json_data.get("logs", [])
        self.logs = []
        for log in json_logs:
            self.logs.append((log.get("taskName"), log.get("messages")))

        # Initialize sub-reports
        self.preparation_report = None
        if "preparationReport" in json_data:
            self.preparation_report = PreparationReport(json_data["preparationReport"])
        self.bivariate_preparation_report = None
        if "bivariatePreparationReport" in json_data:
            self.bivariate_preparation_report = BivariatePreparationReport(
                json_data["bivariatePreparationReport"]
            )
        self.text_preparation_report = None
        if "textPreparationReport" in json_data:
            self.text_preparation_report = PreparationReport(
                json_data["textPreparationReport"]
            )
        self.tree_preparation_report = None
        if "treePreparationReport" in json_data:
            self.tree_preparation_report = PreparationReport(
                json_data["treePreparationReport"]
            )
        self.modeling_report = None
        if "modelingReport" in json_data:
            self.modeling_report = ModelingReport(json_data["modelingReport"])
        self.train_evaluation_report = None
        if "trainEvaluationReport" in json_data:
            self.train_evaluation_report = EvaluationReport(
                json_data["trainEvaluationReport"]
            )
        self.test_evaluation_report = None
        if "testEvaluationReport" in json_data:
            self.test_evaluation_report = EvaluationReport(
                json_data["testEvaluationReport"]
            )
        self.evaluation_report = None
        if "evaluationReport" in json_data:
            self.evaluation_report = EvaluationReport(json_data["evaluationReport"])

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
        if self.text_preparation_report is not None:
            reports.append(self.text_preparation_report)
        if self.tree_preparation_report is not None:
            reports.append(self.tree_preparation_report)
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

    def write_report_file(self, report_file_path):  # pragma: no cover
        """Writes a TSV report file with the object's information

        .. warning::
            This method is *deprecated* since Khiops 11.0.0 and will be removed in
            Khiops 12. Use the `.to_dict` method instead.

        Parameters
        ----------
        report_file_path : str
            Path of the output TSV report file.
        """
        # Warn the user that this method is deprecated and will be removed
        warnings.warn(deprecation_message("write_report_file", "12.0.0"))

        # Write report to file
        with open(report_file_path, "wb") as report_file:
            report_file_writer = self.create_output_file_writer(report_file)
            self.write_report(report_file_writer)

    def to_dict(self):
        """Transforms this instance to a dict with the Khiops JSON file structure"""
        report = super().to_dict()
        if self.short_description is not None:
            report["shortDescription"] = self.short_description
        if self.logs:
            report["logs"] = []
            for task_name, messages in self.logs:
                report["logs"].append({"taskName": task_name, "messages": messages})
        if self.preparation_report is not None:
            report["preparationReport"] = self.preparation_report.to_dict()
        if self.text_preparation_report is not None:
            report["textPreparationReport"] = self.text_preparation_report.to_dict()
        if self.tree_preparation_report is not None:
            report["treePreparationReport"] = self.tree_preparation_report.to_dict()
        if self.bivariate_preparation_report is not None:
            report["bivariatePreparationReport"] = (
                self.bivariate_preparation_report.to_dict()
            )
        if self.modeling_report is not None:
            report["modelingReport"] = self.modeling_report.to_dict()
        if self.train_evaluation_report is not None:
            report["trainEvaluationReport"] = self.train_evaluation_report.to_dict()
        if self.test_evaluation_report is not None:
            report["testEvaluationReport"] = self.test_evaluation_report.to_dict()
        if self.evaluation_report is not None:
            report["evaluationReport"] = self.evaluation_report.to_dict()
        return report

    def write_report(self, stream_or_writer):  # pragma: no cover
        """Writes the instance's TSV report into a writer object

        .. warning::
            This method is *deprecated* since Khiops 11.0.0 and will be removed in
            Khiops 12. Use the `.to_dict` method instead.

        Parameters
        ----------
        stream_or_writer : `io.IOBase` or `.KhiopsOutputWriter`
            Output stream or writer.
        """
        # Warn the user that this method is deprecated and will be removed
        warnings.warn(deprecation_message("write_report", "12.0.0", "to_dict"))

        # Check input writer/stream type
        if isinstance(stream_or_writer, io.IOBase):
            writer = self.create_output_file_writer(stream_or_writer)
        elif isinstance(stream_or_writer, KhiopsOutputWriter):
            writer = stream_or_writer
        else:
            raise TypeError(
                type_error_message(
                    "stream_or_writer",
                    stream_or_writer,
                    io.IOBase,
                    KhiopsOutputWriter,
                )
            )

        # Write report self-data to the file
        writer.writeln(f"Tool\t{self.tool}")
        writer.writeln(f"Version\t{self.version}")
        writer.writeln(f"Short description\t{self.short_description}")
        if self.logs:
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
    return AnalysisResults(json_data=flexible_json_load(json_file_path))


class PreparationReport:
    """Univariate data preparation report: discretizations and groupings

    The attributes related to the target variable and null model are available only in
    the case of a supervised learning task (classification or regression).

    Parameters
    ----------
    json_data : dict, optional
        JSON data of the ``preparationReport`` field of a
        Khiops JSON report file. If not specified it returns an empty instance.

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
    constructed_variable_number : int
        Number of constructed variables.
    instance_number : int
        Number of training instances.
    learning_task : str
        Name of the associated learning task. Possible values:
            - "Classification analysis"
            - "Regression analysis"
            - "Unsupervised analysis"
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
        Number of missing values for a numerical or categorical target variable.
    target_stats_sparse_missing_number : int
        Number of missing values for a sparse block of numerical or categorical target
        variables.
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
    selected_variable_number : int
        Number of selected variables.
    native_variable_number : int
        Number of native variables.
    max_constructed_variables : int
        Maximum number of constructed variable specified for the analysis.
    max_text_features : int
        Maximum number of text features specified for the analysis.
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
    trees : list of `Tree`
        Tree details for each tree built.
    """

    def __init__(self, json_data=None):
        """See class docstring"""
        # Check the type of json_data
        if json_data is not None and not isinstance(json_data, dict):
            raise TypeError(type_error_message("json_data", json_data, dict))

        # Transform to an empty dictionary if json_data is not specified
        if json_data is None:
            json_data = {}
        # Otherwise raise an exception if the preparation report is not valid
        else:
            if "reportType" not in json_data:
                raise KhiopsJSONError("'reportType' key not found")
            if "summary" not in json_data:
                raise KhiopsJSONError("'summary' key not found in preparation report")
            if json_data["reportType"] != "Preparation":
                raise KhiopsJSONError(
                    "'reportType' is not 'Preparation', "
                    f"""it is: '{json_data.get("reportType")}'"""
                )

        # Initialize report type
        self.report_type = json_data.get("reportType", "Preparation")

        # Initialize summary attributes
        json_summary = json_data.get("summary", {})
        self.dictionary = json_summary.get("dictionary", "")
        self.database = json_summary.get("database", "")
        self.instance_number = json_summary.get("instances", 0)
        self.learning_task = json_summary.get("learningTask", "")
        self.target_variable = json_summary.get("targetVariable")
        json_variables = json_summary.get("variables", {})
        self.variable_types = json_variables.get("types", [])
        self.variable_numbers = json_variables.get("numbers", [])
        self.sample_percentage = json_summary.get("samplePercentage", 0)
        self.sampling_mode = json_summary.get("samplingMode", "")
        self.selection_variable = json_summary.get("selectionVariable")
        self.selection_value = json_summary.get("selectionValue")
        self.constructed_variable_number = json_summary.get("constructedVariables")
        self.native_variable_number = json_summary.get("nativeVariables")

        # Initialize target descriptive stats for supervised tasks
        json_stats = json_summary.get("targetDescriptiveStats", {})
        self.target_stats_values = json_stats.get("values")
        json_target_values = json_summary.get("targetValues", {})
        self.target_values = json_target_values.get("values")
        self.target_value_frequencies = json_target_values.get("frequencies")
        self.target_stats_missing_number = json_stats.get("missingNumber")
        self.target_stats_sparse_missing_number = json_stats.get("sparseMissingNumber")

        # Initialize regression only target stats
        self.target_stats_min = json_stats.get("min")
        self.target_stats_max = json_stats.get("max")
        self.target_stats_mean = json_stats.get("mean")
        self.target_stats_std_dev = json_stats.get("stdDev")

        # Initialize classification only target stats
        self.main_target_value = json_summary.get("mainTargetValue")
        self.target_stats_mode = json_stats.get("mode")
        self.target_stats_mode_frequency = json_stats.get("modeFrequency")

        # Initialize other summary attributes
        self.evaluated_variable_number = json_summary.get("evaluatedVariables", 0)
        self.informative_variable_number = json_summary.get("informativeVariables")
        self.selected_variable_number = json_summary.get("selectedVariables")
        json_feature_eng = json_summary.get("featureEngineering", {})
        self.max_constructed_variables = json_feature_eng.get(
            "maxNumberOfConstructedVariables"
        )
        self.max_text_features = json_feature_eng.get("maxNumberOfTextFeatures")
        self.max_trees = json_feature_eng.get("maxNumberOfTrees")
        self.max_pairs = json_feature_eng.get("maxNumberOfVariablePairs")
        self.discretization = json_summary.get("discretization")
        self.value_grouping = json_summary.get("valueGrouping")

        # Cost of model (supervised case and non empty database)
        json_null_model = json_summary.get("nullModel", {})
        self.null_model_construction_cost = json_null_model.get("constructionCost")
        self.null_model_preparation_cost = json_null_model.get("preparationCost")
        self.null_model_data_cost = json_null_model.get("dataCost")

        # Initialize statistics per variable
        json_variables_statistics = json_data.get("variablesStatistics", [])
        self.variables_statistics = []
        self._variables_statistics_by_name = {}
        for json_variable_stats in json_variables_statistics:
            variable_stats = VariableStatistics(json_variable_stats)
            self.variables_statistics.append(variable_stats)
            self._variables_statistics_by_name[variable_stats.name] = variable_stats

        # Initialize detailed statistics attributes when available
        # These are stored in JSON as a dict indexed by variables' ranks
        json_variables_detailed_statistics = json_data.get(
            "variablesDetailedStatistics", {}
        )
        for stats in self.variables_statistics:
            json_detailed_data = json_variables_detailed_statistics.get(stats.rank)
            stats.init_details(json_detailed_data)

        # Initialize tree details
        self.trees = []
        self._trees_by_name = {}
        json_tree_details = json_data.get("treeDetails", {}).values()
        for json_tree_detail in json_tree_details:
            tree = Tree(json_tree_detail)
            self.trees.append(tree)
            self._trees_by_name[tree.name] = tree

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

    def get_tree(self, tree_name):
        """Returns the tree with the specified name

        Parameters
        ----------
        tree_name : str
            Name of the tree.

        Returns
        -------
        `Tree`
            The tree which has the specified name.

        Raises
        ------
        `KeyError`
            If no tree with the specified name exists.
        """
        return self._trees_by_name[tree_name]

    def to_dict(self):
        """Transforms this instance to a dict with the Khiops JSON file structure"""
        report_summary = {
            "dictionary": self.dictionary,
            "variables": {
                "types": self.variable_types,
                "numbers": self.variable_numbers,
            },
            "database": self.database,
            "instances": self.instance_number,
            "learningTask": self.learning_task,
        }
        report = {
            "reportType": "Preparation",
            "summary": report_summary,
        }
        if self.sampling_mode != "":
            # Note: We use `update` because |= is not available for Python 3.8
            report_summary.update(
                {
                    "samplePercentage": self.sample_percentage,
                    "samplingMode": self.sampling_mode,
                    "selectionVariable": self.selection_variable,
                    "selectionValue": self.selection_value,
                }
            )

        # Write classification specific attributes
        if "Classification" in self.learning_task:
            report_summary.update(
                {
                    "targetVariable": self.target_variable,
                    "targetDescriptiveStats": {
                        "values": self.target_stats_values,
                        "mode": self.target_stats_mode,
                        "modeFrequency": self.target_stats_mode_frequency,
                    },
                }
            )
            if self.target_values is not None:
                report_summary["targetValues"] = {
                    "values": self.target_values,
                    "frequencies": self.target_value_frequencies,
                }
            if self.main_target_value is not None:
                report_summary["mainTargetValue"] = self.main_target_value

        # Write regression specific attributes
        if "Regression" in self.learning_task:
            report_summary.update(
                {
                    "targetVariable": self.target_variable,
                    "targetDescriptiveStats": {
                        "values": self.target_stats_values,
                        "min": self.target_stats_min,
                        "max": self.target_stats_max,
                        "mean": self.target_stats_mean,
                        "stdDev": self.target_stats_std_dev,
                    },
                }
            )
            if self.target_values is not None:
                report_summary["targetValues"] = {
                    "values": self.target_values,
                    "frequencies": self.target_value_frequencies,
                }
            if self.main_target_value is not None:
                report_summary["mainTargetValue"] = self.main_target_value

        # Write common classification and regression specific attributes
        if "Classification" in self.learning_task or "Regression" in self.learning_task:
            if self.target_stats_missing_number is not None:
                report_summary["targetDescriptiveStats"][
                    "missingNumber"
                ] = self.target_stats_missing_number
            if self.target_stats_sparse_missing_number is not None:
                report_summary["targetDescriptiveStats"][
                    "sparseMissingNumber"
                ] = self.target_stats_sparse_missing_number
            if self.selected_variable_number is not None:
                report_summary["selectedVariables"] = self.selected_variable_number

        # Write variable preparation summary attributes
        if len(self.variable_types) > 0 and self.instance_number > 0:
            report_summary["evaluatedVariables"] = self.evaluated_variable_number
            if not (
                self.max_constructed_variables is None
                and self.max_text_features is None
                and self.max_trees is None
                and self.max_pairs is None
            ):
                report_summary["featureEngineering"] = {
                    "maxNumberOfConstructedVariables": (
                        self.max_constructed_variables or 0
                    ),
                    "maxNumberOfTextFeatures": self.max_text_features or 0,
                    "maxNumberOfTrees": self.max_trees or 0,
                    "maxNumberOfVariablePairs": self.max_pairs or 0,
                }
            if self.informative_variable_number is not None:
                report_summary["informativeVariables"] = (
                    self.informative_variable_number
                )
            if self.discretization is not None:
                report_summary["discretization"] = self.discretization
            if self.value_grouping is not None:
                report_summary["valueGrouping"] = self.value_grouping
            if self.constructed_variable_number is not None:
                report_summary["constructedVariables"] = (
                    self.constructed_variable_number
                )
            if self.native_variable_number is not None:
                report_summary["nativeVariables"] = self.native_variable_number

        # Write preparation cost information
        if (
            "Unsupervised" not in self.learning_task
            and self.null_model_construction_cost is not None
        ):
            report_summary["nullModel"] = {
                "constructionCost": self.null_model_construction_cost,
                "preparationCost": self.null_model_preparation_cost,
                "dataCost": self.null_model_data_cost,
            }

        # Write variables' statistics
        if len(self.variables_statistics) > 0:
            report["variablesStatistics"] = [
                variable_statistics.to_dict()
                for variable_statistics in self.variables_statistics
            ]
            if any(
                variable_statistics.is_detailed()
                for variable_statistics in self.variables_statistics
            ):
                report["variablesDetailedStatistics"] = {
                    variable_statistics.rank: variable_statistics.to_dict(details=True)
                    for variable_statistics in self.variables_statistics
                    if variable_statistics.is_detailed()
                }

        # Write trees
        if len(self.trees) > 0:
            report["treeDetails"] = {
                self.get_variable_statistics(tree.name).rank: tree.to_dict()
                for tree in self.trees
            }
        return report

    def write_report(self, writer):  # pragma: no cover
        """Writes the instance's TSV report into a writer object

        .. warning::
            This method is *deprecated* since Khiops 11.0.0 and will be removed in
            Khiops 12. Use the `.to_dict` method instead.


        Parameters
        ----------
        writer : `.KhiopsOutputWriter`
            Output writer.
        """
        # Warn the user that this method is deprecated and will be removed
        warnings.warn(deprecation_message("write_report", "12.0.0", "to_dict"))

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

        # Write common attributes for classification and regression
        if self.target_stats_missing_number is not None:
            writer.writeln(f"\tMissing number\t{self.target_stats_missing_number}")
        if self.target_stats_sparse_missing_number is not None:
            writer.writeln(
                f"\tSparse missing number\t{self.target_stats_sparse_missing_number}"
            )

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
        # Write variable preparation summary attributes
        if len(self.variable_types) > 0 and self.instance_number > 0:
            writer.writeln(f"Evaluated variables\t{self.evaluated_variable_number}")
            writer.writeln(f"Informative variables\t{self.informative_variable_number}")
            if self.max_constructed_variables is not None:
                writer.writeln(
                    "Max number of constructed variables\t"
                    f"{self.max_constructed_variables}"
                )
            if self.max_text_features is not None:
                writer.writeln(
                    "Max number of text features\t" f"{self.max_text_features}"
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
        JSON data of the ``bivariatePreparationReport`` field of a Khiops JSON report
        file. If not specified it returns an empty instance.

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
    selected_pair_number : int
        Number of variable pairs selected.
    informative_pair_number : int
        Number of informative variable pairs. A pair is considered informative if its
        level is greater than the sum of its components' levels.
    variable_pair_statistics : list of `VariablePairStatistics`
        Statistics for each analyzed pair of variables.
    """

    def __init__(self, json_data=None):
        """See class docstring"""
        # Check the type of json_data
        if json_data is not None and not isinstance(json_data, dict):
            raise TypeError(type_error_message("json_data", json_data, dict))

        # Transform to an empty dictionary if json_data is not specified
        if json_data is None:
            json_data = {}
        # Otherwise raise exception if the basic fields are not properly set
        else:
            if "reportType" not in json_data:
                raise KhiopsJSONError(
                    "'reportType' key not found in bivariate preparation report"
                )
            if "summary" not in json_data:
                raise KhiopsJSONError(
                    "'summary' key not found in bivariate preparation report"
                )
            if json_data["reportType"] != "BivariatePreparation":
                raise KhiopsJSONError(
                    "'reportType' is not 'BivariatePreparation', "
                    f"""it is: '{json_data.get("reportType")}'"""
                )

        # Initialize report type
        self.report_type = json_data.get("reportType", "BivariatePreparation")

        # Initialize summary attributes
        json_summary = json_data.get("summary", {})
        self.dictionary = json_summary.get("dictionary", "")
        json_variables = json_summary.get("variables", {})
        self.variable_types = json_variables.get("types", [])
        self.variable_numbers = json_variables.get("numbers", [])
        self.database = json_summary.get("database", "")
        self.sample_percentage = json_summary.get("samplePercentage", 0)
        self.sampling_mode = json_summary.get("samplingMode", "")
        self.selection_variable = json_summary.get("selectionVariable")
        self.selection_value = json_summary.get("selectionValue")
        self.instance_number = json_summary.get("instances", 0)
        self.learning_task = json_summary.get("learningTask", "")
        self.target_variable = json_summary.get("targetVariable")

        # Classification task: Initialize target descriptive stats
        # Note: There is no bivariate preparation in the regression case
        self.main_target_value = json_summary.get("mainTargetValue")
        json_stats = json_summary.get("targetDescriptiveStats", {})
        self.target_stats_values = json_stats.get("values")
        self.target_stats_mode = json_stats.get("mode")
        self.target_stats_mode_frequency = json_stats.get("modeFrequency")
        json_target_values = json_summary.get("targetValues", {})
        self.target_values = json_target_values.get("values")
        self.target_value_frequencies = json_target_values.get("frequencies")
        self.target_stats_missing_number = json_stats.get("missingNumber")
        self.target_stats_sparse_missing_number = json_stats.get("sparseMissingNumber")

        # Initialize the information of the pair evaluations
        self.evaluated_pair_number = json_summary.get("evaluatedVariablePairs")
        self.selected_pair_number = json_summary.get("selectedVariablePairs")
        self.informative_pair_number = json_summary.get("informativeVariablePairs")

        # Initialize main attributes for all variables
        self.variables_pairs_statistics = []
        for json_pair_stats in json_data.get("variablesPairsStatistics", []):
            self.variables_pairs_statistics.append(
                VariablePairStatistics(json_pair_stats)
            )

        # Store variable stats in dict indexed by name pairs in both senses
        self._variables_pairs_statistics_by_name = {}
        for stats in self.variables_pairs_statistics:
            name1 = stats.name1
            name2 = stats.name2
            self._variables_pairs_statistics_by_name[name1, name2] = stats
            self._variables_pairs_statistics_by_name[name2, name1] = stats

        # Initialize the variables' detailed statistics when available
        # These are stored in JSON as a dict indexed by the variables ranks
        json_detailed_statistics = json_data.get("variablesPairsDetailedStatistics", {})
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

    def to_dict(self):
        """Transforms this instance to a dict with the Khiops JSON file structure"""
        report_summary = {
            "dictionary": self.dictionary,
            "variables": {
                "types": self.variable_types,
                "numbers": self.variable_numbers,
            },
            "database": self.database,
            "instances": self.instance_number,
            "learningTask": self.learning_task,
        }
        report = {
            "reportType": "BivariatePreparation",
            "summary": report_summary,
        }

        # Update report with data sampling specifications if available
        if self.sampling_mode != "":
            report_summary.update(
                {
                    "samplePercentage": self.sample_percentage,
                    "samplingMode": self.sampling_mode,
                    "selectionVariable": self.selection_variable,
                    "selectionValue": self.selection_value,
                }
            )

        # Write specific summary attributes for a classification task
        if self.learning_task == "Classification analysis":
            report_summary.update(
                {
                    "targetVariable": self.target_variable,
                    "targetDescriptiveStats": {
                        "values": self.target_stats_values,
                        "mode": self.target_stats_mode,
                        "modeFrequency": self.target_stats_mode_frequency,
                    },
                }
            )
            if self.target_values is not None:
                report_summary["targetValues"] = {
                    "values": self.target_values,
                    "frequencies": self.target_value_frequencies,
                }
            if self.target_stats_missing_number is not None:
                report_summary["targetDescriptiveStats"][
                    "missingNumber"
                ] = self.target_stats_missing_number
            if self.target_stats_sparse_missing_number is not None:
                report_summary["targetDescriptiveStats"][
                    "sparseMissingNumber"
                ] = self.target_stats_sparse_missing_number
            if self.main_target_value is not None:
                report_summary["mainTargetValue"] = self.main_target_value

        if self.evaluated_pair_number is not None:
            report_summary["evaluatedVariablePairs"] = self.evaluated_pair_number

        if self.selected_pair_number is not None:
            report_summary["selectedVariablePairs"] = self.selected_pair_number

        if self.informative_pair_number is not None:
            report_summary["informativeVariablePairs"] = self.informative_pair_number

        # Write variables pairs' statistics
        if len(self.variables_pairs_statistics) > 0:
            report["variablesPairsStatistics"] = [
                variable_pair_statistics.to_dict()
                for variable_pair_statistics in self.variables_pairs_statistics
            ]
            if any(
                variable_pair_statistics.is_detailed()
                for variable_pair_statistics in self.variables_pairs_statistics
            ):
                report["variablesPairsDetailedStatistics"] = {
                    variable_pair_statistics.rank: variable_pair_statistics.to_dict(
                        details=True
                    )
                    for variable_pair_statistics in self.variables_pairs_statistics
                    if variable_pair_statistics.is_detailed()
                }

        return report

    def write_report(self, writer):  # pragma: no cover
        """Writes the instance's TSV report into a writer object

        .. warning::
            This method is *deprecated* since Khiops 11.0.0 and will be removed in
            Khiops 12. Use the `.to_dict` method instead.


        Parameters
        ----------
        writer : `.KhiopsOutputWriter`
            Output writer.
        """
        # Warn the user that this method is deprecated and will be removed
        warnings.warn(deprecation_message("write_report", "12.0.0", "to_dict"))

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
        JSON data of the ``modelingReport`` field of Khiops JSON report file. If not
        specified it returns an empty instance.

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
        # Check the type of json_data
        if json_data is not None and not isinstance(json_data, dict):
            raise TypeError(type_error_message("json_data", json_data, dict))

        # Transform to an empty dictionary if json_data is not specified
        if json_data is None:
            json_data = {}
        # Otherwise raise an exception if the modeling report is not valid
        else:
            if "reportType" not in json_data:
                raise KhiopsJSONError("'reportType' key not found in modeling report")
            if "summary" not in json_data:
                raise KhiopsJSONError("'summary' key not found in modeling report")
            if json_data.get("reportType") != "Modeling":
                raise KhiopsJSONError(
                    "'reportType' is not 'Modeling', "
                    f"""it is: '{json_data.get("reportType")}'"""
                )

        # Initialize report type
        self.report_type = json_data.get("reportType", "Modeling")

        # Initialize the summary attributes
        json_summary = json_data.get("summary", {})
        self.dictionary = json_summary.get("dictionary", "")
        self.database = json_summary.get("database", "")
        self.sample_percentage = json_summary.get("samplePercentage", 0)
        self.sampling_mode = json_summary.get("samplingMode", "")
        self.selection_variable = json_summary.get("selectionVariable")
        self.selection_value = json_summary.get("selectionValue")
        self.learning_task = json_summary.get("learningTask", "")
        self.target_variable = json_summary.get("targetVariable")
        self.main_target_value = json_summary.get("mainTargetValue")

        # Initialize specifications per trained predictor
        self.trained_predictors = []
        self._trained_predictors_by_name = {}
        json_predictors_details = json_data.get("trainedPredictorsDetails", {})
        for json_trained_predictor in json_data.get("trainedPredictors", []):
            # Initialize basic trained predictor data
            predictor = TrainedPredictor(json_trained_predictor)
            self.trained_predictors.append(predictor)
            self._trained_predictors_by_name[predictor.name] = predictor

            # Initialize detailed trained predictor data
            if predictor.rank in json_predictors_details:
                predictor.init_details(json_predictors_details[predictor.rank])

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

    def to_dict(self):
        """Transforms this instance to a dict with the Khiops JSON file structure"""
        report_summary = {
            "dictionary": self.dictionary,
            "database": self.database,
            "learningTask": self.learning_task,
            "targetVariable": self.target_variable,
        }
        report = {
            "reportType": "Modeling",
            "summary": report_summary,
            "trainedPredictors": [
                predictor.to_dict() for predictor in self.trained_predictors
            ],
        }
        if any(predictor.is_detailed() for predictor in self.trained_predictors):
            report["trainedPredictorsDetails"] = {
                predictor.rank: predictor.to_dict(details=True)
                for predictor in self.trained_predictors
                if predictor.is_detailed()
            }
        if self.sampling_mode != "":
            report_summary.update(
                {
                    "samplePercentage": self.sample_percentage,
                    "samplingMode": self.sampling_mode,
                    "selectionVariable": self.selection_variable,
                    "selectionValue": self.selection_value,
                }
            )

        if self.main_target_value is not None:
            report_summary["mainTargetValue"] = self.main_target_value

        return report

    def write_report(self, writer):  # pragma: no cover
        """Writes the instance's TSV report into a writer object

        .. warning::
            This method is *deprecated* since Khiops 11.0.0 and will be removed in
            Khiops 12. Use the `.to_dict` method instead.


        Parameters
        ----------
        writer : `.KhiopsOutputWriter`
            Output writer.
        """
        # Warn the user that this method is deprecated and will be removed
        warnings.warn(deprecation_message("write_report", "12.0.0", "to_dict"))

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

        If not specified it returns an empty instance.

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
        # Check the type of json_data
        if json_data is not None and not isinstance(json_data, dict):
            raise TypeError(type_error_message("json_data", json_data, dict))

        # Transform to an empty dictionary if json_data is not specified
        if json_data is None:
            json_data = {}
        # Otherwise raise an exception if the evaluation report is not valid
        else:
            if "reportType" not in json_data:
                raise KhiopsJSONError("'reportType' key not found in evaluation report")
            if "summary" not in json_data:
                raise KhiopsJSONError("'summary' key not found in evaluation report")
            if "evaluationType" not in json_data:
                raise KhiopsJSONError(
                    "'evaluationType' key not found in evaluation report"
                )
            if json_data["reportType"] != "Evaluation":
                raise KhiopsJSONError(
                    "'reportType' is not 'Evaluation' it is: "
                    f"""'{json_data.get("reportType")}'"""
                )

        # Initialize type attributes
        self.report_type = json_data.get("reportType", "Evaluation")
        self.evaluation_type = json_data.get("evaluationType", "")

        # Initialize summary attributes
        json_summary = json_data.get("summary", {})
        self.dictionary = json_summary.get("dictionary", "")
        self.database = json_summary.get("database", "")
        self.sample_percentage = json_summary.get("samplePercentage", 0)
        self.sampling_mode = json_summary.get("samplingMode", "")
        self.selection_variable = json_summary.get("selectionVariable")
        self.selection_value = json_summary.get("selectionValue")
        self.instance_number = json_summary.get("instances", 0)
        self.learning_task = json_summary.get("learningTask", "")
        self.target_variable = json_summary.get("targetVariable", "")
        self.main_target_value = json_summary.get("mainTargetValue")

        # Initialize the performance attributes for each predictor
        self.predictors_performance = []
        self._predictors_performance_by_name = {}
        json_predictors_performance = json_data.get("predictorsPerformance", [])
        json_detailed_performance = json_data.get("predictorsDetailedPerformance", {})
        for json_predictor_performance in json_predictors_performance:
            # Initialize main performance info
            performance = PredictorPerformance(json_predictor_performance)
            self.predictors_performance.append(performance)
            self._predictors_performance_by_name[performance.name] = performance

            # Initialize detailed performance info if available
            if performance.rank in json_detailed_performance:
                performance.init_details(json_detailed_performance[performance.rank])

        # Collect REC curves for each regressor
        self.regression_rec_curves = None
        if self.learning_task == "Regression analysis":
            self.regression_rec_curves = [
                PredictorCurve(json_rec_curve)
                for json_rec_curve in json_data.get("recCurves", [])
            ]

        # Fill Lift curves for a classification task if available
        # Note that there is no curve for "Random", because it can be easily calculated
        self.classification_target_values = None
        self.classification_lift_curves = None
        if self.learning_task == "Classification analysis":
            self.classification_target_values = []
            self.classification_lift_curves = []

            # Collect all lift curves per target value and per classifier
            for json_lift_curves in json_data.get("liftCurves", []):
                # Collect lift curves for each classifier
                lift_curves = [
                    PredictorCurve(json_lift_curve)
                    for json_lift_curve in json_lift_curves.get("curves")
                ]

                # Store collected target values with their lift curves
                self.classification_target_values.append(
                    json_lift_curves.get("targetValue")
                )
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
                    return PredictorCurve(
                        {
                            "classifier": "Random",
                            "values": [
                                j * 100.0 / (point_number - 1)
                                for j in range(point_number)
                            ],
                        }
                    )
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

    def to_dict(self):
        """Transforms this instance to a dict with the Khiops JSON file structure"""
        report_summary = {
            "dictionary": self.dictionary,
            "database": self.database,
            "instances": self.instance_number,
            "learningTask": self.learning_task,
            "targetVariable": self.target_variable,
        }
        report = {
            "reportType": "Evaluation",
            "evaluationType": self.evaluation_type,
            "summary": report_summary,
            "predictorsPerformance": [
                predictor_performance.to_dict()
                for predictor_performance in self.predictors_performance
            ],
        }
        if any(
            predictor_performance.is_detailed()
            for predictor_performance in self.predictors_performance
        ):
            report["predictorsDetailedPerformance"] = {
                predictor_performance.rank: predictor_performance.to_dict(details=True)
                for predictor_performance in self.predictors_performance
                if predictor_performance.is_detailed()
            }

        if self.sampling_mode != "":
            report_summary.update(
                {
                    "samplePercentage": self.sample_percentage,
                    "samplingMode": self.sampling_mode,
                    "selectionVariable": self.selection_variable,
                    "selectionValue": self.selection_value,
                }
            )

        if self.main_target_value is not None:
            report_summary["mainTargetValue"] = self.main_target_value

        # Write lift curves, one per target value and per classifier
        if (
            self.learning_task.startswith("Classification")
            and self.classification_target_values is not None
        ):
            report["liftCurves"] = []
            for i, target_value in enumerate(self.classification_target_values):
                lift_curves = self.classification_lift_curves[i]
                report["liftCurves"].append(
                    {
                        "targetValue": target_value,
                        "curves": [
                            {
                                "classifier": lift_curve.name,
                                "values": lift_curve.values,
                            }
                            for lift_curve in lift_curves
                        ],
                    }
                )

        # Write REC curves, one per regressor
        if (
            self.learning_task.startswith("Regression")
            and self.regression_rec_curves is not None
        ):
            report["recCurves"] = [
                {"regressor": rec_curve.name, "values": rec_curve.values}
                for rec_curve in self.regression_rec_curves
            ]
        return report

    def write_report(self, writer):  # pragma: no cover
        """Writes the instance's TSV report into a writer object

        .. warning::
            This method is *deprecated* since Khiops 11.0.0 and will be removed in
            Khiops 12. Use the `.to_dict` method instead.


        Parameters
        ----------
        writer : `.KhiopsOutputWriter`
            Output writer object.
        """
        # Warn the user that this method is deprecated and will be removed
        warnings.warn(deprecation_message("write_report", "12.0.0", "to_dict"))

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
        JSON report file. If not specified it returns an empty instance.


        .. note::
            The ``data_grid`` field is considered a "detail" and is not initialized in
            the constructor. Instead, it is initialized explicitly via the
            ``init_details`` method. This allows to make partial initializations for
            large reports. If not specified it returns an empty instance.


    Attributes
    ----------
    rank : str
        Variable rank with respect to its level. Lower Rank = Higher Level.
    name : str
        Variable name.
    type : str
        Variable type. There are only two valid values for a prepared variable :
            - "Numerical"
            - "Categorical"
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
    sparse_missing_number : int
        Number of sparse missing values of the variable.
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
    modl_histograms : `ModlHistograms`
        MODL optimal histograms for for numerical variables. Only for unsupervised
        analysis.
    """

    def __init__(self, json_data=None):
        """See class docstring"""
        # Check the type of json_data
        if json_data is not None and not isinstance(json_data, dict):
            raise TypeError(type_error_message("json_data", json_data, dict))

        # Transform to an empty dictionary if json_data is not specified
        if json_data is None:
            json_data = {}

        # Initialize common attributes
        self.rank = json_data.get("rank", "")
        self.name = json_data.get("name", "")
        self.type = json_data.get("type", "")
        self.level = json_data.get("level")
        self.target_part_number = json_data.get("targetParts")
        self.part_number = json_data.get("parts")
        self.value_number = json_data.get("values", 0)
        self.missing_number = json_data.get("missingNumber")
        self.sparse_missing_number = json_data.get("sparseMissingNumber")

        # Initialize numerical variable attributes
        self.min = json_data.get("min")
        self.max = json_data.get("max")
        self.mean = json_data.get("mean")
        self.std_dev = json_data.get("stdDev")

        # Initialize categorical variable attributes
        self.mode = json_data.get("mode")
        self.mode_frequency = json_data.get("modeFrequency")

        # Initialize cost attributes
        self.construction_cost = json_data.get("constructionCost")
        self.preparation_cost = json_data.get("preparationCost")
        self.data_cost = json_data.get("dataCost")

        # Initialize derivation rule
        self.derivation_rule = json_data.get("derivationRule")

        # Details' attributes
        # They may or not be initialized with the init_details method

        # Data grid for density estimation
        self.data_grid = None

        # MODL optimal histograms for numerical data in unsupervised analysis
        self.modl_histograms = None

        # Input values and their frequencies in case of categorical variables
        # The input values may not be the exhaustive list of all the values
        # For scalability reasons, the least frequent values are not always present
        self.input_values = None
        self.input_value_frequencies = None

    def init_details(self, json_data=None):
        """Initializes the details' attributes from a Python JSON object

        Parameters
        ----------
        json_data : dict, optional
            JSON data of an element of the list found at the
            ``variablesDetailedStatistics`` field within the ``preparationReport`` field
            of a Khiops JSON report file. If not specified it leaves the object as-is.

        """
        # Check the type of json_data
        if json_data is not None and not isinstance(json_data, dict):
            raise TypeError(type_error_message("json_data", json_data, dict))

        # Initialize details if not empty
        if json_data is not None:
            # Initialize data grid
            json_data_grid = json_data.get("dataGrid")
            self.data_grid = DataGrid(json_data_grid)

            # Initialize input values and their frequencies
            json_input_values = json_data.get("inputValues", {})
            self.input_values = json_input_values.get("values")
            self.input_value_frequencies = json_input_values.get("frequencies")

            # Initialize MODL histograms if present in the JSON report
            json_modl_histograms = json_data.get("modlHistograms")
            if json_modl_histograms is not None:
                self.modl_histograms = ModlHistograms(json_modl_histograms)

        return self

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

    def to_dict(self, details=False):
        """Transforms this instance to a dict with the Khiops JSON file structure"""
        # Write report details if required and applicable
        if details and self.is_detailed():
            report = {}

            # Write data grid
            if self.data_grid is not None:
                report["dataGrid"] = self.data_grid.to_dict()

            # Write input values and their frequencies
            if self.input_values is not None:
                report["inputValues"] = {
                    "values": self.input_values,
                    "frequencies": self.input_value_frequencies,
                }

            # Write MODL histograms (unsupervised analysis)
            if self.modl_histograms is not None:
                report["modlHistograms"] = self.modl_histograms.to_dict()
            return report
        elif details is False:
            report = {
                "rank": self.rank,
                "name": self.name,
                "type": self.type,
                "values": self.value_number,
                "constructionCost": self.construction_cost,
            }

            # Write level if available
            if self.level is not None:
                report["level"] = self.level

            # Write target part number if available
            if self.target_part_number is not None:
                report["targetParts"] = self.target_part_number

            # Write part number if available
            if self.part_number is not None:
                report["parts"] = self.part_number

            # Write missing number if available
            if self.missing_number is not None:
                report["missingNumber"] = self.missing_number

            # Write sparse missing number if available
            if self.sparse_missing_number is not None:
                report["sparseMissingNumber"] = self.sparse_missing_number

            # Write attributes specific to Numerical / Categorical types
            if self.type == "Numerical":
                report.update(
                    {
                        "min": self.min,
                        "max": self.max,
                        "mean": self.mean,
                        "stdDev": self.std_dev,
                    }
                )
            elif self.type == "Categorical":
                report.update({"mode": self.mode, "modeFrequency": self.mode_frequency})

            # Write preparation cost only for the supervised case
            if self.preparation_cost is not None:
                report.update(
                    {
                        "preparationCost": self.preparation_cost,
                        "dataCost": self.data_cost,
                    }
                )

            # Write derivation rule if available
            if self.derivation_rule is not None:
                report["derivationRule"] = self.derivation_rule
            return report

    def write_report_header_line(self, writer):  # pragma: no cover
        """Writes the header line of a TSV report into a writer object

        The header is the same for all variable types.

        .. warning::
            This method is *deprecated* since Khiops 11.0.0 and will be removed in
            Khiops 12. Use the `.to_dict` method instead.


        Parameters
        ----------
        writer : `.KhiopsOutputWriter`
            Output writer.
        """
        # Warn the user that this method is deprecated and will be removed
        warnings.warn(
            deprecation_message("write_report_header_line", "12.0.0", "to_dict")
        )

        # Write report header
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
        writer.write("Sparse missing number\t")
        writer.write("Mode\t")
        writer.write("Mode frequency\t")
        writer.write("Construction cost\t")
        writer.write("Preparation cost\t")
        writer.write("Data cost\t")
        writer.writeln("Derivation rule")

    def write_report_line(self, writer):  # pragma: no cover
        """Writes a line of the TSV report into a writer object

        .. warning::
            This method is *deprecated* since Khiops 11.0.0 and will be removed in
            Khiops 12. Use the `.to_dict` method instead.


        Parameters
        ----------
        writer : `.KhiopsOutputWriter`
            Output writer.
        """
        # Warn the user that this method is deprecated and will be removed
        warnings.warn(deprecation_message("write_report_line", "12.0.0", "to_dict"))

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
            writer.write(f"{self.sparse_missing_number}\t")
        else:
            writer.write("\t" * 6)

        # Write attributes available only for categorical variables
        if self.type == "Categorical":
            writer.write(f"{self.missing_number}\t")
            writer.write(f"{self.sparse_missing_number}\t")
            writer.write(f"{self.mode}\t")
            writer.write(f"{self.mode_frequency}\t")
        else:
            writer.write("\t" * 2)

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

    def write_report_details(self, writer):  # pragma: no cover
        """Writes the details' attributes into a writer object

        .. warning::
            This method is *deprecated* since Khiops 11.0.0 and will be removed in
            Khiops 12. Use the `.to_dict` method instead.


        Parameters
        ----------
        writer : `.KhiopsOutputWriter`
            Output writer.
        """
        # Warn the user that this method is deprecated and will be removed
        warnings.warn(deprecation_message("write_report_details", "12.0.0", "to_dict"))

        # Write report if detailed report is available
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
        JSON data of an element of the list found at the ``variablesPairStatistics``
        field within the ``bivariatePreparationReport`` field of a Khiops JSON report
        file. If not specified it returns an empty instance.

        .. note::
            The ``data_grid`` field is considered as "detail" and is not initialized in
            the constructor. Instead, it is initialized explicitly via the
            `init_details` method. This allows to make partial initializations for large
            reports. If not specified it returns an empty instance.


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
        # Check the type of json_data
        if json_data is not None and not isinstance(json_data, dict):
            raise TypeError(type_error_message("json_data", json_data, dict))

        # Transform to an empty dictionary if json_data is not specified
        if json_data is None:
            json_data = {}

        # Initialization from JSON data (except for details' attributes)
        self.rank = json_data.get("rank", "")
        self.name1 = json_data.get("name1", "")
        self.name2 = json_data.get("name2", "")
        self.level = json_data.get("level", 0)
        self.level1 = json_data.get("level1")
        self.level2 = json_data.get("level2")
        self.delta_level = json_data.get("deltaLevel")
        self.variable_number = json_data.get("variables", 0)
        self.part_number1 = json_data.get("parts1", 0)
        self.part_number2 = json_data.get("parts2", 0)
        self.cell_number = json_data.get("cells", 0)

        # Initialize cost attributes
        self.construction_cost = json_data.get("constructionCost")
        self.preparation_cost = json_data.get("preparationCost")
        self.data_cost = json_data.get("dataCost")

        # Data grid for density estimation (detail)
        self.data_grid = None

    def init_details(self, json_data=None):
        """Initializes the details' attributes from a Python JSON object

        Parameters
        ----------
        json_data : dict, optional
            JSON data of an element of the list found at
            the ``variablesPairsDetailedStatistics`` field within the
            ``bivariatePreparationReport`` field of a Khiops JSON report file. If not
            specified it leaves the object as-is.
        """
        # Check the type of json_data
        if json_data is not None and not isinstance(json_data, dict):
            raise TypeError(type_error_message("json_data", json_data, dict))

        # Initialize the data grid field
        if json_data is not None:
            self.data_grid = DataGrid(json_data.get("dataGrid"))

        return self

    def is_detailed(self):
        """Returns True if the report contains any detailed information

        Returns
        -------
        bool
            True if the report contains any detailed information.
        """
        return self.data_grid is not None

    def to_dict(self, details=False):
        """Transforms this instance to a dict with the Khiops JSON file structure"""
        report = {}
        if details and self.is_detailed():
            # write detailed report
            report["dataGrid"] = self.data_grid.to_dict()
        elif details is False:
            report.update(
                {
                    "rank": self.rank,
                    "name1": self.name1,
                    "name2": self.name2,
                    "level": self.level,
                    "variables": self.variable_number,
                    "parts1": self.part_number1,
                    "parts2": self.part_number2,
                    "cells": self.cell_number,
                    "constructionCost": self.construction_cost,
                    "preparationCost": self.preparation_cost,
                    "dataCost": self.data_cost,
                }
            )

            # Supervised case: write level attributes
            if self.delta_level is not None:
                report.update(
                    {
                        "deltaLevel": self.delta_level,
                        "level1": self.level1,
                        "level2": self.level2,
                    }
                )

        return report

    def write_report_header_line(self, writer):  # pragma: no cover
        """Writes the header line of a TSV report into a writer object

        The header is the same for all variable types.

        .. warning::
            This method is *deprecated* since Khiops 11.0.0 and will be removed in
            Khiops 12. Use the `.to_dict` method instead.


        Parameters
        ----------
        writer : `.KhiopsOutputWriter`
            Output writer.
        """
        # Warn the user that this method is deprecated and will be removed
        warnings.warn(
            deprecation_message("write_report_header_line", "12.0.0", "to_dict")
        )

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

    def write_report_line(self, writer):  # pragma: no cover
        """Writes a line of the TSV report into a writer object

        .. warning::
            This method is *deprecated* since Khiops 11.0.0 and will be removed in
            Khiops 12. Use the `.to_dict` method instead.


        Parameters
        ----------
        writer : `.KhiopsOutputWriter`
            Output writer.
        """
        # Warn the user that this method is deprecated and will be removed
        warnings.warn(deprecation_message("write_report_line", "12.0.0", "to_dict"))

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

    def write_report_details(self, writer):  # pragma: no cover
        """Writes the details' attributes into a writer object

        .. warning::
            This method is *deprecated* since Khiops 11.0.0 and will be removed in
            Khiops 12. Use the `.to_dict` method instead.


        Parameters
        ----------
        writer : `.KhiopsOutputWriter`
            Output writer.
        """
        # Warn the user that this method is deprecated and will be removed
        warnings.warn(deprecation_message("write_report_details", "12.0.0", "to_dict"))

        # Write report if detailed report is available
        if self.is_detailed():
            writer.writeln("")
            writer.writeln(f"Rank\t{self.rank}")
            writer.writeln("")
            self.data_grid.write_report(writer)


class Tree:
    """A decision tree feature

    Parameters
    ----------
    json_data : dict, optional
        JSON data of a value associated to the rank key in the object found at the
        ``treeDetails`` field within the ``treePreparationReport`` field of a Khiops
        JSON report file. If not specified, it returns an empty instance.

    Attributes
    ----------
    name : str
        Name of the tree.
    variable_number : int
        Number of variables in the tree.
    depth : int
        Depth of the tree.
    target_partition : `TargetPartition`
        Summary of the target partition. For regression only.
    nodes: list of `TreeNode`
        Nodes of the tree.
    """

    def __init__(self, json_data=None):
        """ "See class docstring"""
        self.name = json_data.get("name")
        self.variable_number = json_data.get("variableNumber")
        self.depth = json_data.get("depth")
        self.target_partition = None
        if "targetPartition" in json_data:
            self.target_partition = TargetPartition(json_data["targetPartition"])

        # Initialize tree nodes as empty list
        self.nodes = []

        # Update node tree structure
        json_tree_nodes = json_data.get("treeNodes")
        self._update_nodes(json_tree_nodes, root=None)

    def _update_nodes(self, json_data, root=None):
        self.nodes.append(TreeNode(json_data, parent_id=root))
        for json_node in json_data.get("childNodes", []):
            self._update_nodes(json_node, root=json_data.get("nodeId"))

    def to_dict(self):
        """Transforms this instance to a dict with the Khiops JSON file structure"""
        report = {
            "name": self.name,
            "variableNumber": self.variable_number,
            "depth": self.depth,
        }
        if self.target_partition is not None:
            report["targetPartition"] = self.target_partition.to_dict()
        if self.nodes:
            report["treeNodes"] = self._nodes_to_json(
                root_node=self.nodes[0], past_nodes=[]
            )
            return report

    def _nodes_to_json(self, root_node, past_nodes):
        report = root_node.to_dict()
        past_nodes = [root_node]
        json_child_nodes = []
        for node in self.nodes:
            if node in past_nodes:
                continue
            if node.parent_id == root_node.id:
                past_nodes.append(node)
                json_child_nodes.append(
                    self._nodes_to_json(root_node=node, past_nodes=past_nodes)
                )
        if json_child_nodes:
            report["childNodes"] = json_child_nodes
        return report


class TargetPartition:
    """Target partition details (for regression trees only)

    Parameters
    ----------
    json_data : dict, optional
        JSON data of the ``targetPartition`` field of the ``treeDetails`` field of the
        ``treePreparationReport`` field of a Khiops JSON report file. If not specified
        it returns an empty instance.

    Attributes
    ----------
    variable : str
        Variable name.
    type : "Numerical" (only possible value)
        Variable type.
    partition_type : "Intervals" (only possible value)
        Partition type.
    partition : list
        The dimension parts. The list objects are of type `PartInterval`, as
        ``partition_type`` is "Intervals"
    frequencies : list of int
        Frequencies of the intervals in the target partition.
    """

    def __init__(self, json_data=None):
        """See class docstring"""
        # Check the type of json_data
        if json_data is not None and not isinstance(json_data, dict):
            raise TypeError(type_error_message("json_data", json_data, dict))

        # Transform to an empty dictionary if json_data is not specified
        if json_data is None:
            json_data = {}

        # Initialize basic attributes
        self.variable = json_data.get("variable", "")
        self.type = json_data.get("type", "")
        self.partition_type = json_data.get("partitionType", "")

        # Initialize partition
        self.partition = []
        if "partition" in json_data:
            json_partition = json_data["partition"]
            if not isinstance(json_partition, list):
                raise KhiopsJSONError("'partition' must be a list")
        else:
            json_partition = []

        # Numerical partition
        if self.partition_type == "Intervals":
            # Check the length of the partition
            if len(json_partition) < 1:
                raise KhiopsJSONError(
                    "'partition' for interval must have length at least 1"
                )

            # Initialize intervals
            self.partition = [PartInterval(json_part) for json_part in json_partition]

            # Initialize open interval flags
            first_interval = self.partition[0]
            if first_interval.is_missing:
                first_interval = self.partition[1]
            first_interval.is_left_open = True
            last_interval = self.partition[-1]
            last_interval.is_right_open = True

        else:
            raise KhiopsJSONError("'partitionType' must be 'Intervals'")

        # Set partition element frequencies
        self.frequencies = json_data.get("frequencies", [])

    def to_dict(self):
        """Transforms this instance to a dict with the Khiops JSON file structure"""
        report = {
            "variable": self.variable,
            "type": self.type,
            "partitionType": self.partition_type,
            "partition": [part.to_dict() for part in self.partition],
        }
        if self.frequencies:
            report["frequencies"] = self.frequencies
        return report


class TreeNode:
    """A decision tree node

    Parameters
    ----------
    json_data : dict, optional
        JSON data of either:

        - the ``treeNodes`` field of the ``treeDetails`` field of the
          ``treePreparationReport`` field of a Khiops JSON report file, or
        - an element of the ``childNodes`` field of the ``treeNodes`` field of the
          ``treeDetails`` field of the ``treePreparationReport`` field of a Khiops
          JSON report file.

        If not specified it returns an empty instance
    parent_id : str, optional
        Identifier of the parent ``TreeNode`` instance. Not set for "root" nodes.

    Attributes
    ----------
    id : str
        Identifier of the ``TreeNode`` instance.
    parent_id : str, optional
        Value of the ``id`` field of another ``TreeNode`` instance. Not set for "root"
        nodes.
    variable : str
        Name of the tree variable.
    type : str
        Khiops type of the tree variable.
    partition : list
        The tree variable partition.
    default_group_index : int
        The index of the default variable group.
    target_values : list of str
        Values of a categorical tree target variable.
    target_value_frequencies : list of int
        Frequencies of each tree target value. Synchronized with ``target_values``.
    """

    def __init__(self, json_data=None, parent_id=None):
        """See class docstring"""
        # Check the type of json_data
        if json_data is not None and not isinstance(json_data, dict):
            raise TypeError(type_error_message("json_data", json_data, dict))

        # Transform to an empty dictionary if json_data is not specified
        if json_data is None:
            json_data = {}

        # Set the parent node ID
        self.parent_id = parent_id

        # Set the target values if applicable
        json_target_values = json_data.get("targetValues")
        if json_target_values is not None:
            self.target_values = json_target_values.get("values")
            self.target_value_frequencies = json_target_values.get("frequencies")
        else:
            self.target_values = None
            self.target_value_frequencies = None

        # Set the remainder of the node attributes
        self.id = json_data.get("nodeId")
        self.variable = json_data.get("variable")
        self.type = json_data.get("type")
        self.partition = json_data.get("partition")
        self.default_group_index = json_data.get("defaultGroupIndex")

    def to_dict(self):
        """Transforms this instance to a dict with the Khiops JSON file structure"""

        report = {"nodeId": self.id}
        if self.variable is not None:
            report["variable"] = self.variable
        if self.type is not None:
            report["type"] = self.type
        if self.partition is not None:
            report["partition"] = self.partition
        if self.default_group_index is not None:
            report["defaultGroupIndex"] = self.default_group_index
        if self.target_values is not None:
            report["targetValues"] = {
                "values": self.target_values,
                "frequencies": self.target_value_frequencies,
            }
        return report


class ModlHistograms:
    """A histogram density estimation for numerical data

    A MODL histogram is a regularized piecewise-constant estimation of the probability
    density for numerical data. It has various refinement levels to ease exploratory
    analysis tasks.

    Parameters
    ----------
    json_data : dict, optional
        JSON data at a ``modlHistograms`` field of an element of the list found at the
        ``variablesDetailedStatistics`` field within the ``preparationReport`` field
        of a Khiops JSON report file. If not specified, it returns an empty instance.

    Attributes
    ----------
    histogram_number : int
        Number of available histograms.
    interpretable_histogram_number : int
        Number of interpretable histograms. Can be equal to either
        ``histogram_number`` or ``histogram_number - 1``.
    truncation_epsilon : float
        Truncation epsilon used by the truncation heuristic implemented in Khiops.
        Equals 0 if no truncation is detected in the input data.
    removed_singular_interval_number : int
        Number of singular intervals removed from the finest-grained histogram to
        obtain the first interpretable histogram.
    granularities : list of int
        Histogram granularities, sorted in increasing order.
        Synchronized with ``histograms``.
    interval_numbers : list of int
        Histogram interval numbers, sorted in increasing order.
        Synchronized with ``histograms``.
    peak_interval_numbers : list of int
        Histogram peak interval numbers, sorted in increasing order.
        Synchronized with ``histograms``.
    spike_interval_numbers : list of int
        Histogram spike interval numbers, sorted in increasing order.
        Synchronized with ``histograms``.
    empty_interval_numbers : list of int
        Histogram empty interval numbers, sorted in increasing order.
        Synchronized with ``histograms``.
    levels : list of float
        List of histogram levels, sorted in increasing order.
        Synchronized with ``histograms``.
    information_rates : list of float
        Histogram information rates, sorted in increasing order. Between 0 and
        100 for interpretable histograms.
        Synchronized with ``histograms``.
    histograms : list of `Histogram`
        The MODL histograms.

    """

    def __init__(self, json_data=None):
        """See class docstring"""
        # Check the type of json_data
        if json_data is not None and not isinstance(json_data, dict):
            raise TypeError(type_error_message("json_data", json_data, dict))

        # Transform to an empty dictionary if json_data is not specified
        if json_data is None:
            json_data = {}

        # Initialize histogram number
        self.histogram_number = json_data.get("histogramNumber")

        # Initialize interpretable_histogram_number
        self.interpretable_histogram_number = json_data.get(
            "interpretableHistogramNumber"
        )

        # Initialize truncation_epsilon
        self.truncation_epsilon = json_data.get("truncationEpsilon")

        # Initialize removed_singular_interval_number
        self.removed_singular_interval_number = json_data.get(
            "removedSingularIntervalNumber"
        )

        # Initialize histogram granularities
        self.granularities = json_data.get("granularities")

        # Initialize histogram interval numbers
        self.interval_numbers = json_data.get("intervalNumbers")

        # Initialize histogram peak interval numbers
        self.peak_interval_numbers = json_data.get("peakIntervalNumbers")

        # Initialize histogram spike interval numbers
        self.spike_interval_numbers = json_data.get("spikeIntervalNumbers")

        # Initialize histogram empty interval numbers
        self.empty_interval_numbers = json_data.get("emptyIntervalNumbers")

        # Initialize histogram levels
        self.levels = json_data.get("levels")

        # Initialize histogram information rates
        self.information_rates = json_data.get("informationRates")

        # Initialize histograms
        self.histograms = [
            Histogram(json_histogram)
            for json_histogram in json_data.get("histograms", [])
        ]

    def to_dict(self):
        """Transforms this instance to a dict with the Khiops JSON file structure"""
        return {
            "emptyIntervalNumbers": self.empty_interval_numbers,
            "granularities": self.granularities,
            "histogramNumber": self.histogram_number,
            "histograms": [histogram.to_dict() for histogram in self.histograms],
            "informationRates": self.information_rates,
            "interpretableHistogramNumber": self.interpretable_histogram_number,
            "intervalNumbers": self.interval_numbers,
            "levels": self.levels,
            "peakIntervalNumbers": self.peak_interval_numbers,
            "removedSingularIntervalNumber": self.removed_singular_interval_number,
            "spikeIntervalNumbers": self.spike_interval_numbers,
            "truncationEpsilon": self.truncation_epsilon,
        }


class Histogram:
    """A histogram

    Represents one of the refinement levels of a `ModlHistograms` object.

    Parameters
    ----------
    json_data : dict, optional
        JSON data of an element at the ``histograms`` field of a ``modlHistograms``
        field of an element of the list found at the ``variablesDetailedStatistics``
        field within the ``preparationReport`` field of a Khiops JSON report file.
        If not specified it returns an empty instance.

    Attributes
    ----------
    bounds : list of float
        Interval bounds.
    frequencies : list of int
        Interval frequencies.
    """

    def __init__(self, json_data=None):
        """See class docstring"""
        # Check the type of json_data
        if json_data is not None and not isinstance(json_data, dict):
            raise TypeError(type_error_message("json_data", json_data, dict))

        # Transform to an empty dictionary if json_data is not specified
        if json_data is None:
            json_data = {}

        # Initialize basic attributes
        self.bounds = json_data.get("bounds")
        self.frequencies = json_data.get("frequencies")

    def to_dict(self):
        """Transforms this instance to a dict with the Khiops JSON file structure"""
        return {"bounds": self.bounds, "frequencies": self.frequencies}


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
        a Khiops JSON report file. If not specified it returns an empty instance.

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
        # Check the type of json_data
        if json_data is not None and not isinstance(json_data, dict):
            raise TypeError(type_error_message("json_data", json_data, dict))

        # Transform to an empty dictionary if json_data is not specified
        if json_data is None:
            json_data = {}

        # Initialize supervised status
        self.is_supervised = json_data.get("isSupervised", False)

        # Initialize dimensions
        self.dimensions = [
            DataGridDimension(json_dimension)
            for json_dimension in json_data.get("dimensions", [])
        ]

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

        # Initialize attributes for a supervised data grid with several input variables
        elif self.is_supervised and len(self.dimensions) > 2:
            self.cell_ids = json_data.get("cellIds")
            self.cell_part_indexes = json_data.get("cellPartIndexes")
            self.cell_frequencies = json_data.get("cellFrequencies")
            self.cell_target_frequencies = json_data.get("cellTargetFrequencies")
            self.cell_interests = json_data.get("cellInterests")

    def to_dict(self):
        """Transforms this instance to a dict with the Khiops JSON file structure"""
        # Write data grid type and dimensions
        report = {
            "isSupervised": self.is_supervised,
            "dimensions": [dimension.to_dict() for dimension in self.dimensions],
        }

        # Write data grid cells
        # Univariate unsupervised data grid: Write frequencies per part
        if not self.is_supervised and len(self.dimensions) == 1:
            report["frequencies"] = self.frequencies

        # Multivariate unsupervised data grid: Write frequencies per cell
        elif not self.is_supervised and len(self.dimensions) > 1:
            report["cellIds"] = self.cell_ids
            report["cellPartIndexes"] = self.cell_part_indexes
            if self.cell_frequencies is not None:
                report["cellFrequencies"] = self.cell_frequencies

        # Supervised data grid with one input variable:
        # Write frequencies for each input part and for each target part
        elif self.is_supervised and len(self.dimensions) == 2:
            report["partTargetFrequencies"] = self.part_target_frequencies
            report["partInterests"] = self.part_interests

        # Supervised data grid with several input variables
        # Write frequencies per input cell part, for each target part
        elif self.is_supervised and len(self.dimensions) > 2:
            report["cellIds"] = self.cell_ids
            report["cellPartIndexes"] = self.cell_part_indexes
            if self.cell_frequencies is not None:
                report["cellFrequencies"] = self.cell_frequencies
            report["cellTargetFrequencies"] = self.cell_target_frequencies
            report["cellInterests"] = self.cell_interests

        return report

    def write_report(self, writer):  # pragma: no cover
        """Writes the instance's TSV report into a writer object

        .. warning::
            This method is *deprecated* since Khiops 11.0.0 and will be removed in
            Khiops 12. Use the `.to_dict` method instead.


        Parameters
        ----------
        writer : `.KhiopsOutputWriter`
            Output writer.
        """
        # Warn the user that this method is deprecated and will be removed
        warnings.warn(deprecation_message("write_report", "12.0.0", "to_dict"))

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
        it returns an empty instance.

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
        # Check the type of json_data
        if json_data is not None and not isinstance(json_data, dict):
            raise TypeError(type_error_message("json_data", json_data, dict))

        # Transform to an empty dictionary if json_data is not specified
        if json_data is None:
            json_data = {}

        # Initialize basic attributes
        self.variable = json_data.get("variable", "")
        self.type = json_data.get("type", "")
        self.partition_type = json_data.get("partitionType", "")

        # Initialize partition
        self.partition = []
        if "partition" in json_data:
            json_partition = json_data["partition"]
            if not isinstance(json_partition, list):
                raise KhiopsJSONError("'partition' must be a list")
        else:
            json_partition = []

        # Numerical partition
        if self.partition_type == "Intervals":
            # Check the length of the partition
            if len(json_partition) < 1:
                raise KhiopsJSONError(
                    "'partition' for interval must have length at least 1"
                )

            # Initialize intervals
            self.partition = [PartInterval(json_part) for json_part in json_partition]

            # Initialize open interval flags
            first_interval = self.partition[0]
            if first_interval.is_missing:
                first_interval = self.partition[1]
            first_interval.is_left_open = True
            last_interval = self.partition[-1]
            last_interval.is_right_open = True

        # Value partition (singletons)
        elif self.partition_type == "Values":
            self.partition = [PartValue(json_part) for json_part in json_partition]

        # ValueGroups partition
        elif self.partition_type == "Value groups":
            # Initialize value groups
            self.partition = [PartValueGroup(json_part) for json_part in json_partition]

            # Initialize default group containing all values not specified
            # in any partition element and all unknown values
            default_group_index = json_data["defaultGroupIndex"]
            self.partition[default_group_index].is_default_part = True

    def to_dict(self):
        """Transforms this instance to a dict with the Khiops JSON file structure"""
        report = {
            "variable": self.variable,
            "type": self.type,
            "partitionType": self.partition_type,
            "partition": [part.to_dict() for part in self.partition],
        }

        if self.partition_type == "Value groups":
            default_group_index = None
            for i, part in enumerate(self.partition):
                if part.is_default_part:
                    default_group_index = i
                    break
            if default_group_index is not None:
                report["defaultGroupIndex"] = default_group_index

        return report

    def write_report(self, writer):  # pragma: no cover
        """Writes the instance's TSV report into a writer object

        .. warning::
            This method is *deprecated* since Khiops 11.0.0 and will be removed in
            Khiops 12. Use the `.to_dict` method instead.


        Parameters
        ----------
        writer : `.KhiopsOutputWriter`
            Output writer.
        """
        # Warn the user that this method is deprecated and will be removed
        warnings.warn(deprecation_message("write_report", "12.0.0", "to_dict"))

        # Write report
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
    json_data : list, optional
        JSON data of the ``partition`` field of a ``dataGrid`` field of an element of
        the list found at the ``variablesDetailedStatistics`` field within the
        ``preparationReport`` field of a Khiops JSON report file. If not specified it
        returns an empty instance.

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
        # Check the type of json_data
        if json_data is not None:
            if not isinstance(json_data, list):
                raise TypeError(type_error_message("json_data", json_data, list))
            if len(json_data) not in (0, 2):
                raise ValueError("'json_data' must be a list of size two or empty.")

        # Transform to an empty list if json_data is not specified
        if json_data is None:
            json_data = []

        # Missing value if array of bounds is empty
        self.lower_bound = None
        self.upper_bound = None
        self.is_missing = False
        if not json_data:
            self.is_missing = True
        # Actual interval if array of bounds not empty
        else:
            self.lower_bound = json_data[0]
            self.upper_bound = json_data[1]

        # These fields are initialized by another class
        self.is_left_open = False
        self.is_right_open = False

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

    def to_dict(self):
        """Transforms this instance to a dict with the Khiops JSON file structure"""
        if not self.is_missing:
            return [self.lower_bound, self.upper_bound]
        return []

    def write_report_line(self, writer):  # pragma: no cover
        """Writes a line of the TSV report into a writer object

        .. warning::
            This method is *deprecated* since Khiops 11.0.0 and will be removed in
            Khiops 12. Use the `.to_dict` method instead.


        Parameters
        ----------
        writer : `.KhiopsOutputWriter`
            Output writer.
        """
        # Warn the user that this method is deprecated and will be removed
        warnings.warn(deprecation_message("write_report_line", "12.0.0", "to_dict"))

        # Write part label
        writer.write(str(self))

        # Write interval bounds
        if not self.is_missing:
            writer.write(f"\t{self.lower_bound}")
            writer.write(f"\t{self.upper_bound}")
        writer.writeln("")


class PartValue:
    """Element of a value partition (singletons) in a data grid

    Parameters
    ----------
    json_data : str, optional
        The value contained in this singleton part. If not specified it returns an empty
        object.

    Attributes
    ----------
      value : str
          A representation of the value defining the singleton.
    """

    def __init__(self, json_data=None):
        """See class docstring"""
        # Check the type of json_data
        if json_data is not None and not isinstance(json_data, str):
            raise TypeError(type_error_message("json_data", json_data, str))

        # Set the value
        if json_data is not None:
            self.value = json_data
        else:
            self.value = ""

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

    def to_dict(self):
        """Transforms this instance to a dict with the Khiops JSON file structure"""
        return self.value

    def write_report_line(self, writer):  # pragma: no cover
        """Writes a line of the TSV report into a writer object

        .. warning::
            This method is *deprecated* since Khiops 11.0.0 and will be removed in
            Khiops 12. Use the `.to_dict` method instead.


        Parameters
        ----------
        writer : `.KhiopsOutputWriter`
            Output writer.
        """
        # Warn the user that this method is deprecated and will be removed
        warnings.warn(deprecation_message("write_report_line", "12.0.0", "to_dict"))

        # Write object value
        writer.writeln(str(self))


class PartValueGroup:
    """Element of a categorical partition in a data grid

    Parameters
    ----------
    json_data : list of str, optional
        The list of values of the group. If not specified it returns an empty instance.

    Attributes
    ----------
    values : list of str
        The group's values.
    is_default_part : bool
        True if this part is dedicated to all unknown values.
    """

    def __init__(self, json_data=None):
        """See class docstring"""
        # Check the type of json_data
        if json_data is not None and not isinstance(json_data, list):
            raise TypeError(type_error_message("json_data", json_data, list))

        # Set the values
        self.values = json_data

        # Initialize default part to False: It will be set by DataGridDimension
        self.is_default_part = False

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

    def to_dict(self):
        """Transforms this instance to a dict with the Khiops JSON file structure"""
        return self.values

    def write_report_line(self, writer):  # pragma: no cover
        """Writes a line of the TSV report into a writer object

        .. warning::
            This method is *deprecated* since Khiops 11.0.0 and will be removed in
            Khiops 12. Use the `.to_dict` method instead.


        Parameters
        ----------
        writer : `.KhiopsOutputWriter`
            Output writer.
        """
        # Warn the user that this method is deprecated and will be removed
        warnings.warn(deprecation_message("write_report_line", "12.0.0", "to_dict"))

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
    json_data : dict, optional
        JSON data of an element of the list found at the ``trainedPredictors`` field
        within the ``modelingReport`` field of a Khiops JSON report file. If not
        specified it returns an empty instance.

        .. note::
            The ``selected_variables`` field is considered a "detail" and is not
            initialized in the constructor. Instead, it is initialized explicitly via
            the `init_details` method. This allows to make partial initializations for
            large reports.

    Attributes
    ----------
    family : str
        Predictor family name. Valid values are found in the ``predictor_families``
        class variable. They are:

        - "Baseline": for regression only,
        - "Selective Naive Bayes": in all other cases.

    type : "Classifier" or "Regressor"
        Predictor type. Valid values are found in the ``predictor_types`` class
        attribute.
    name : str
        Human readable predictor name.
    variable_number : int
        Number of variables used by the predictor.
    selected_variables : list of `SelectedVariable`
        Variables used by the predictor. Only for type "Selective Naive Bayes".
    """

    predictor_types = ["Classifier", "Regressor"]
    predictor_families = ["Selective Naive Bayes", "Baseline"]

    def __init__(self, json_data=None):
        """See class docstring"""
        # Check the type of json_data
        if json_data is not None and not isinstance(json_data, dict):
            raise TypeError(type_error_message("json_data", json_data, dict))

        # Transform to an empty dictionary if json_data is not specified
        if json_data is None:
            json_data = {}

        # Initialize common attributes
        self.rank = json_data.get("rank", "")
        self.type = json_data.get("type", "")
        self.family = json_data.get("family", "")
        self.name = json_data.get("name", "")
        self.variable_number = json_data.get("variables", 0)

        # Detailed attributes, depending on the predictor
        self.selected_variables = None

    def init_details(self, json_data=None):
        """Initializes the details' attributes from a Python JSON object

        Parameters
        ----------
        json_data : dict, optional
            JSON data of the dictionary found at the ``trainedPredictorsDetails`` field
            within the ``modelingReport`` field of a Khiops JSON report file. If not
            specified it leaves the object as-is.
        """
        # Check the type of json_data
        if json_data is not None and not isinstance(json_data, dict):
            raise TypeError(type_error_message("json_data", json_data, dict))

        # Transform to an empty dictionary if json_data is not specified
        if json_data is None:
            json_data = {}

        # Initialize the selected variable attributes
        if "selectedVariables" in json_data:
            self.selected_variables = [
                SelectedVariable(json_selected_variable)
                for json_selected_variable in json_data["selectedVariables"]
            ]

        return self

    def is_detailed(self):
        """Returns True if the report contains any detailed information

        Returns
        -------
        bool
            True if the report contains any detailed information.
        """
        return self.selected_variables is not None

    def to_dict(self, details=False):
        """Transforms this instance to a dict with the Khiops JSON file structure"""
        if details and self.is_detailed() and self.selected_variables is not None:
            return {
                "selectedVariables": [
                    selected_variable.to_dict()
                    for selected_variable in self.selected_variables
                ]
            }

        # details is False:
        return {
            "rank": self.rank,
            "type": self.type,
            "family": self.family,
            "name": self.name,
            "variables": self.variable_number,
        }

    def write_report_header_line(self, writer):  # pragma: no cover
        """Writes the header line of a TSV report into a writer object

        .. warning::
            This method is *deprecated* since Khiops 11.0.0 and will be removed in
            Khiops 12. Use the `.to_dict` method instead.


        The header is the same for all variable types.

        Parameters
        ----------
        writer : `.KhiopsOutputWriter`
            Output writer.
        """
        # Warn the user that this method is deprecated and will be removed
        warnings.warn(
            deprecation_message("write_report_header_line", "12.0.0", "to_dict")
        )

        # Write report header line
        writer.write("Rank\t")
        writer.write("Type\t")
        writer.write("Family\t")
        writer.write("Name\t")
        writer.writeln("Variables")

    def write_report_line(self, writer):  # pragma: no cover
        """Writes a line of the TSV report into a writer object

        .. warning::
            This method is *deprecated* since Khiops 11.0.0 and will be removed in
            Khiops 12. Use the `.to_dict` method instead.


        Parameters
        ----------
        writer : `.KhiopsOutputWriter`
            Output writer.
        """
        # Warn the user that this method is deprecated and will be removed
        warnings.warn(deprecation_message("write_report_line", "12.0.0", "to_dict"))

        # Write report line
        writer.write(f"{self.rank}\t")
        writer.write(f"{self.type}\t")
        writer.write(f"{self.family}\t")
        writer.write(f"{self.name}\t")
        writer.writeln(str(self.variable_number))

    def write_report_details(self, writer):  # pragma: no cover
        """Writes the details of the TSV report into a writer object

        .. warning::
            This method is *deprecated* since Khiops 11.0.0 and will be removed in
            Khiops 12. Use the `.to_dict` method instead.


        Parameters
        ----------
        writer : `.KhiopsOutputWriter`
            Output writer.
        """
        # Warn the user that this method is deprecated and will be removed
        warnings.warn(deprecation_message("write_report_details", "12.0.0", "to_dict"))

        # Write detailed report header if available
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
        Khiops JSON report file. If not specified it returns an empty instance.

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
    """

    def __init__(self, json_data=None):
        """See class docstring"""
        # Check the type of json_data
        if json_data is not None and not isinstance(json_data, dict):
            raise TypeError(type_error_message("json_data", json_data, dict))

        # Transform to an empty dictionary if json_data is not specified
        if json_data is None:
            json_data = {}

        # Initialize fierds
        self.prepared_name = json_data.get("preparedName", "")
        self.name = json_data.get("name", "")
        self.level = json_data.get("level", "")
        self.weight = json_data.get("weight")
        self.importance = json_data.get("importance")

    def to_dict(self):
        """Transforms this instance to a dict with the Khiops JSON file structure"""
        report = {
            "preparedName": self.prepared_name,
            "name": self.name,
            "level": self.level,
            "weight": self.weight,
        }
        if self.importance is not None:
            report["importance"] = self.importance

        return report

    def write_report_header_line(self, writer):  # pragma: no cover
        """Writes the header line of a TSV report into a writer object

        The header is the same for all variable types.

        .. warning::
            This method is *deprecated* since Khiops 11.0.0 and will be removed in
            Khiops 12. Use the `.to_dict` method instead.


        Parameters
        ----------
        writer : `.KhiopsOutputWriter`
            Output writer.
        """
        # Warn the user that this method is deprecated and will be removed
        warnings.warn(
            deprecation_message("write_report_header_line", "12.0.0", "to_dict")
        )

        # Write report header
        writer.write("Prepared name\t")
        writer.write("Name\t")
        writer.write("Level")
        writer.write("\tWeight")
        if self.importance is not None:
            writer.write("\tImportance")
        writer.writeln("")

    def write_report_line(self, writer):  # pragma: no cover
        """Writes a line of the TSV report into a writer object

        .. warning::
            This method is *deprecated* since Khiops 11.0.0 and will be removed in
            Khiops 12. Use the `.to_dict` method instead.


        Parameters
        ----------
        writer : `.KhiopsOutputWriter`
            Output writer.
        """
        # Warn the user that this method is deprecated and will be removed
        warnings.warn(deprecation_message("write_report_line", "12.0.0", "to_dict"))

        # Write report line
        writer.write(f"{self.prepared_name}\t")
        writer.write(f"{self.name}\t")
        writer.write(str(self.level))
        writer.write(f"\t{self.weight}")
        if self.importance is not None:
            writer.write(f"\t{self.importance}")

        writer.writeln("")


class PredictorPerformance:
    """A predictor's performance evaluation

    This class describes the performance of a predictor (classifier or regressor).

    Parameters
    ----------
    json_data : dict, optional
        JSON data of an element of the dictionary found at the ``predictorPerformances``
        field within the one of the evaluation report fields of a Khiops JSON report
        file. If not specified it returns an empty instance.

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
        # Check the type of json_data
        if json_data is not None and not isinstance(json_data, dict):
            raise TypeError(type_error_message("json_data", json_data, dict))

        # Transform to an empty dictionary if json_data is not specified
        if json_data is None:
            json_data = {}

        # Initialize common attributes
        self.rank = json_data.get("rank", "")
        self.type = json_data.get("type", "")
        self.family = json_data.get("family", "")
        self.name = json_data.get("name", "")

        # Initialize classifier evaluation criteria
        self.accuracy = json_data.get("accuracy")
        self.compression = json_data.get("compression")
        self.auc = json_data.get("auc")

        # Initialize regressor evaluation criteria
        self.rmse = json_data.get("rmse")
        self.mae = json_data.get("mae")
        self.nlpd = json_data.get("nlpd")
        self.rank_rmse = json_data.get("rankRmse")
        self.rank_mae = json_data.get("rankMae")
        self.rank_nlpd = json_data.get("rankNlpd")

        # Detailed attributes, depending on the predictor
        self.confusion_matrix = None  # for a Classifier
        self.data_grid = None  # for a univariate predictor

    def init_details(self, json_data=None):
        """Initializes the details' attributes from a python JSON object"""
        # Check the type of json_data
        if json_data is not None and not isinstance(json_data, dict):
            raise TypeError(type_error_message("json_data", json_data, dict))

        # Transform to an empty dictionary if json_data is not specified
        if json_data is None:
            json_data = {}

        # Initialize confusion matrix
        if "confusionMatrix" in json_data:
            self.confusion_matrix = ConfusionMatrix(json_data["confusionMatrix"])

        # Initialize data grid
        if "dataGrid" in json_data:
            self.data_grid = DataGrid(json_data["dataGrid"])

        return self

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
        metric_found = lowercase_metric_name in self.get_metric_names()
        if metric_found:
            metric = getattr(self, lowercase_metric_name)
        else:
            metric_list_msg = ",".join(self.get_metric_names())
            raise ValueError(
                f"Invalid metric: '{metric_name}'. Choose among {metric_list_msg}."
            )
        return metric

    def to_dict(self, details=False):
        """Transforms this instance to a dict with the Khiops JSON file structure"""
        if details and self.is_detailed():
            report = {}
            if self.data_grid is not None:
                report["dataGrid"] = self.data_grid.to_dict()
            if self.confusion_matrix is not None:
                report["confusionMatrix"] = self.confusion_matrix.to_dict()
        else:
            report = {
                "rank": self.rank,
                "type": self.type,
                "family": self.family,
                "name": self.name,
            }

            if self.type == "Classifier":
                report.update(
                    {
                        "accuracy": self.accuracy,
                        "compression": self.compression,
                        "auc": self.auc,
                    }
                )
            elif self.type == "Regressor":
                report.update(
                    {
                        "rmse": self.rmse,
                        "mae": self.mae,
                        "nlpd": self.nlpd,
                        "rankRmse": self.rank_rmse,
                        "rankMae": self.rank_mae,
                        "rankNlpd": self.rank_nlpd,
                    }
                )
        return report

    def write_report_header_line(self, writer):  # pragma: no cover
        """Writes the header line of a TSV report into a writer object

        The header is the same for all variable types.

        .. warning::
            This method is *deprecated* since Khiops 11.0.0 and will be removed in
            Khiops 12. Use the `.to_dict` method instead.


        Parameters
        ----------
        writer : `.KhiopsOutputWriter`
            Output writer.
        """
        # Warn the user that this method is deprecated and will be removed
        warnings.warn(
            deprecation_message("write_report_header_line", "12.0.0", "to_dict")
        )

        # Write report header
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

    def write_report_line(self, writer):  # pragma: no cover
        """Writes a line of the TSV report into a writer object

        .. warning::
            This method is *deprecated* since Khiops 11.0.0 and will be removed in
            Khiops 12. Use the `.to_dict` method instead.


        Parameters
        ----------
        writer : `.KhiopsOutputWriter`
            Output writer.
        """
        # Warn the user that this method is deprecated and will be removed
        warnings.warn(deprecation_message("write_report_line", "12.0.0", "to_dict"))

        # Write report line
        writer.write(f"{self.rank}\t")
        writer.write(f"{self.type}\t")
        writer.write(f"{self.family}\t")
        writer.write(f"{self.name}\t")

        metrics = [str(self.get_metric(name)) for name in self.get_metric_names()]
        writer.writeln("\t".join(metrics))

    def write_report_details(self, writer):  # pragma: no cover
        """Writes the details of the TSV report into a writer object

        .. warning::
            This method is *deprecated* since Khiops 11.0.0 and will be removed in
            Khiops 12. Use the `.to_dict` method instead.


        Parameters
        ----------
        writer : `.KhiopsOutputWriter`
            Output writer.
        """
        # Warn the user that this method is deprecated and will be removed
        warnings.warn(deprecation_message("write_report_details", "12.0.0", "to_dict"))

        # Write detailed report if available
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
    json_data : dict, optional
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
        # Check the type of json_data
        if json_data is not None and not isinstance(json_data, dict):
            raise TypeError(type_error_message("json_data", json_data, dict))

        # Transform to an empty dictionary if json_data is not specified
        if json_data is None:
            json_data = {}

        # Initialize fields
        self.values = json_data.get("values", [])
        self.matrix = json_data.get("matrix", [])

    def to_dict(self):
        """Transforms this instance to a dict with the Khiops JSON file structure"""

        return {"values": self.values, "matrix": self.matrix}

    def write_report(self, writer):  # pragma: no cover
        """Writes the instance's TSV report into a writer object

        .. warning::
            This method is *deprecated* since Khiops 11.0.0 and will be removed in
            Khiops 12. Use the `.to_dict` method instead.


        Parameters
        ----------
        writer : `.KhiopsOutputWriter`
            Output writer.
        """
        # Warn the user that this method is deprecated and will be removed
        warnings.warn(deprecation_message("write_report", "12.0.0", "to_dict"))

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
    json_data : dict, optional
        JSON data of an element of the ``liftCurves`` or ``recCurves`` field of one of
        the evaluation report fields of a Khiops JSON report file. If not specified it
        returns an empty instance.

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
        # Check the type of json_data
        if json_data is not None and not isinstance(json_data, dict):
            raise TypeError(type_error_message("json_data", json_data, dict))

        # Transform to an empty dictionary if json_data is not specified
        if json_data is None:
            json_data = {}
        # Otherwise check that either 'classifier' or 'regressor' keys are present
        else:
            if "classifier" not in json_data and "regressor" not in json_data:
                raise ValueError(
                    "Either 'classifier' or 'regressor' must be in curve data"
                )

        # Initialize the fields
        if "classifier" in json_data:
            self.type = "Lift"
            self.name = json_data.get("classifier")
        elif "regressor" in json_data:
            self.type = "REC"
            self.name = json_data.get("regressor")
        else:
            self.type = None
            self.name = None
        self.values = json_data.get("values")
