######################################################################################
# Copyright (c) 2023 Orange. All rights reserved.                                    #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""train_predictor task family"""
from pykhiops.core.internals import task as tm
from pykhiops.core.internals.types import (
    BoolType,
    DictType,
    FloatType,
    IntType,
    ListType,
    StringLikeType,
    TupleType,
)

# Disable long lines to have readable scenarios
# pylint: disable=line-too-long
TASKS = [
    tm.KhiopsTask(
        "train_predictor",
        "khiops",
        "10.0",
        [
            ("dictionary_file_path", StringLikeType),
            ("dictionary_name", StringLikeType),
            ("data_table_path", StringLikeType),
            ("target_variable", StringLikeType),
            ("results_dir", StringLikeType),
        ],
        [
            ("detect_format", BoolType, True),
            ("header_line", BoolType, True),
            ("field_separator", StringLikeType, ""),
            ("sample_percentage", FloatType, 100.0),
            ("sampling_mode", StringLikeType, "Include sample"),
            ("test_database_mode", StringLikeType, "Complementary"),
            ("selection_variable", StringLikeType, ""),
            ("selection_value", StringLikeType, ""),
            ("additional_data_tables", DictType(StringLikeType, StringLikeType), None),
            ("main_target_value", StringLikeType, ""),
            ("snb_predictor", BoolType, True),
            ("univariate_predictor_number", IntType, 0),
            ("max_evaluated_variables", IntType, 0),
            ("max_selected_variables", IntType, 0),
            ("max_constructed_variables", IntType, 0),
            ("construction_rules", ListType(StringLikeType), None),
            ("max_trees", IntType, 10),
            ("max_pairs", IntType, 0),
            ("all_possible_pairs", BoolType, True),
            (
                "specific_pairs",
                ListType(TupleType(StringLikeType, StringLikeType)),
                None,
            ),
            ("group_target_value", BoolType, False),
            ("discretization_method", StringLikeType, "MODL"),
            ("min_interval_frequency", IntType, 0),
            ("max_intervals", IntType, 0),
            ("grouping_method", StringLikeType, "MODL"),
            ("min_group_frequency", IntType, 0),
            ("max_groups", IntType, 0),
            ("results_prefix", StringLikeType, ""),
        ],
        [
            "dictionary_file_path",
            "data_table_path",
            "results_dir",
            "additional_data_tables",
        ],
        # pylint: disable=line-too-long
        # fmt: off
        """
        // Dictionary file and class settings
        ClassManagement.OpenFile
        ClassFileName __dictionary_file_path__
        OK
        ClassManagement.ClassName __dictionary_name__

        // Train/test database settings
        TrainDatabase.DatabaseFiles.List.Key __dictionary_name__
        TrainDatabase.DatabaseFiles.DataTableName __data_table_path__
        __DICT__
        __additional_data_tables__
        TrainDatabase.DatabaseFiles.List.Key
        TrainDatabase.DatabaseFiles.DataTableName
        __END_DICT__
        TrainDatabase.HeaderLineUsed __header_line__
        TrainDatabase.FieldSeparator __field_separator__
        __OPT__
        __detect_format__
        TrainDatabase.DatabaseFormatDetector.DetectFileFormat
        __END_OPT__
        TrainDatabase.SampleNumberPercentage __sample_percentage__
        TrainDatabase.SamplingMode __sampling_mode__
        TrainDatabase.SelectionAttribute __selection_variable__
        TrainDatabase.SelectionValue __selection_value__
        TrainDatabase.TestDatabaseSpecificationMode __test_database_mode__

        // Target variable
        AnalysisSpec.TargetAttributeName __target_variable__
        AnalysisSpec.MainTargetModality __main_target_value__

        // Predictors to train
        AnalysisSpec.PredictorsSpec.SelectiveNaiveBayesPredictor __snb_predictor__
        AnalysisSpec.PredictorsSpec.AdvancedSpec.UnivariatePredictorNumber __univariate_predictor_number__

        // Selective Naive Bayes settings
        AnalysisSpec.PredictorsSpec.AdvancedSpec.InspectSelectiveNaiveBayesParameters
        TrainParameters.MaxEvaluatedAttributeNumber __max_evaluated_variables__
        SelectionParameters.MaxSelectedAttributeNumber __max_selected_variables__
        Exit

        // Feature engineering
        AnalysisSpec.PredictorsSpec.ConstructionSpec.MaxTreeNumber __max_trees__
        AnalysisSpec.PredictorsSpec.ConstructionSpec.MaxAttributePairNumber __max_pairs__
        AnalysisSpec.PredictorsSpec.AdvancedSpec.InspectAttributePairsParameters
        AllAttributePairs __all_possible_pairs__
        __LIST__
        __specific_pairs__
        SpecificAttributePairs.InsertItemAfter
        SpecificAttributePairs.FirstName
        SpecificAttributePairs.SecondName
        __END_LIST__
        Exit
        AnalysisSpec.PredictorsSpec.ConstructionSpec.MaxConstructedAttributeNumber __max_constructed_variables__
        AnalysisSpec.PredictorsSpec.AdvancedSpec.InspectConstructionDomain
        __DICT__
        __construction_rules__
        UnselectAll
        ConstructionRules.List.Key
        ConstructionRules.Used
        __END_DICT__
        Exit

        // Data preparation (discretization & grouping) settings
        AnalysisSpec.PreprocessingSpec.TargetGrouped __group_target_value__
        AnalysisSpec.PreprocessingSpec.DiscretizerSpec.SupervisedMethodName __discretization_method__
        AnalysisSpec.PreprocessingSpec.DiscretizerSpec.UnsupervisedMethodName __discretization_method__
        AnalysisSpec.PreprocessingSpec.DiscretizerSpec.MinIntervalFrequency __min_interval_frequency__
        AnalysisSpec.PreprocessingSpec.DiscretizerSpec.MaxIntervalNumber __max_intervals__
        AnalysisSpec.PreprocessingSpec.GrouperSpec.SupervisedMethodName __grouping_method__
        AnalysisSpec.PreprocessingSpec.GrouperSpec.UnsupervisedMethodName __grouping_method__
        AnalysisSpec.PreprocessingSpec.GrouperSpec.MinGroupFrequency __min_group_frequency__
        AnalysisSpec.PreprocessingSpec.GrouperSpec.MaxGroupNumber __max_groups__

        // Output settings
        AnalysisResults.ResultFilesDirectory __results_dir__
        AnalysisResults.ResultFilesPrefix __results_prefix__

        // Build model
        ComputeStats
        """,
        # fmt: on
    ),
    tm.KhiopsTask(
        "train_predictor",
        "khiops",
        "9.0",
        [
            ("dictionary_file_path", StringLikeType),
            ("dictionary_name", StringLikeType),
            ("data_table_path", StringLikeType),
            ("target_variable", StringLikeType),
            ("results_dir", StringLikeType),
        ],
        [
            ("header_line", BoolType, True),
            ("field_separator", StringLikeType, ""),
            ("sample_percentage", FloatType, 100.0),
            ("sampling_mode", StringLikeType, "Include sample"),
            ("fill_test_database_settings", BoolType, False),
            ("selection_variable", StringLikeType, ""),
            ("selection_value", StringLikeType, ""),
            ("additional_data_tables", DictType(StringLikeType, StringLikeType), None),
            ("main_target_value", StringLikeType, ""),
            ("snb_predictor", BoolType, True),
            ("map_predictor", BoolType, False),
            ("univariate_predictor_number", IntType, 0),
            ("max_evaluated_variables", IntType, 0),
            ("max_selected_variables", IntType, 0),
            ("max_constructed_variables", IntType, 0),
            ("construction_rules", ListType(StringLikeType), None),
            ("max_trees", IntType, 10),
            ("max_pairs", IntType, 0),
            ("only_pairs_with", StringLikeType, ""),
            ("group_target_value", BoolType, False),
            ("discretization_method", StringLikeType, "MODL"),
            ("min_interval_frequency", IntType, 0),
            ("max_intervals", IntType, 0),
            ("grouping_method", StringLikeType, "MODL"),
            ("min_group_frequency", IntType, 0),
            ("max_groups", IntType, 0),
            ("results_prefix", StringLikeType, ""),
        ],
        [
            "dictionary_file_path",
            "data_table_path",
            "results_dir",
            "additional_data_tables",
        ],
        # fmt: off
        """
        // Dictionary file and class settings
        ClassManagement.OpenFile
        ClassFileName __dictionary_file_path__
        OK
        ClassManagement.ClassName __dictionary_name__

        // Train/test database settings
        TrainDatabase.DatabaseFiles.List.Key __dictionary_name__
        TrainDatabase.DatabaseFiles.DataTableName __data_table_path__
        __DICT__
        __additional_data_tables__
        TrainDatabase.DatabaseFiles.List.Key
        TrainDatabase.DatabaseFiles.DataTableName
        __END_DICT__
        TrainDatabase.HeaderLineUsed __header_line__
        TrainDatabase.FieldSeparator __field_separator__
        TrainDatabase.SampleNumberPercentage __sample_percentage__
        TrainDatabase.SamplingMode __sampling_mode__
        TrainDatabase.SelectionAttribute __selection_variable__
        TrainDatabase.SelectionValue __selection_value__
        __OPT__
        __fill_test_database_settings__
        TrainDatabase.FillTestDatabaseSettings
        __END_OPT__

        // Target variable
        AnalysisSpec.TargetAttributeName __target_variable__
        AnalysisSpec.MainTargetModality __main_target_value__

        // Predictors to train
        AnalysisSpec.ModelingSpec.SelectiveNaiveBayesPredictor __snb_predictor__
        AnalysisSpec.ModelingSpec.MAPNaiveBayesPredictor __map_predictor__
        AnalysisSpec.ModelingSpec.NaiveBayesPredictor false
        AnalysisSpec.ModelingSpec.UnivariatePredictorNumber __univariate_predictor_number__

        // Selective Naive Bayes settings
        AnalysisSpec.ModelingSpec.InspectSelectiveNaiveBayesParameters
        TrainParameters.MaxEvaluatedAttributeNumber __max_evaluated_variables__
        SelectionParameters.MaxSelectedAttributeNumber __max_selected_variables__
        Exit

        // Feature engineering
        AnalysisSpec.AttributeConstructionSpec.MaxTreeNumber __max_trees__
        AnalysisSpec.AttributeConstructionSpec.MaxAttributePairNumber __max_pairs__
        AnalysisSpec.AttributeConstructionSpec.MandatoryAttributeInPairs __only_pairs_with__
        AnalysisSpec.AttributeConstructionSpec.MaxConstructedAttributeNumber __max_constructed_variables__
        AnalysisSpec.AttributeConstructionSpec.InspectConstructionDomain
        __DICT__
        __construction_rules__
        UnselectAll
        ConstructionRules.List.Key
        ConstructionRules.Used
        __END_DICT__

        // Data preparation (discretization & grouping) settings
        AnalysisSpec.PreprocessingSpec.TargetGrouped __group_target_value__
        AnalysisSpec.PreprocessingSpec.DiscretizerSpec.SupervisedMethodName __discretization_method__
        AnalysisSpec.PreprocessingSpec.DiscretizerSpec.UnsupervisedMethodName __discretization_method__
        AnalysisSpec.PreprocessingSpec.DiscretizerSpec.MinIntervalFrequency __min_interval_frequency__
        AnalysisSpec.PreprocessingSpec.DiscretizerSpec.MaxIntervalNumber __max_intervals__
        AnalysisSpec.PreprocessingSpec.GrouperSpec.SupervisedMethodName __grouping_method__
        AnalysisSpec.PreprocessingSpec.GrouperSpec.UnsupervisedMethodName __grouping_method__
        AnalysisSpec.PreprocessingSpec.GrouperSpec.MinGroupFrequency __min_group_frequency__
        AnalysisSpec.PreprocessingSpec.GrouperSpec.MaxGroupNumber __max_groups__

        // Output settings
        AnalysisResults.ResultFilesDirectory __results_dir__
        AnalysisResults.ResultFilesPrefix __results_prefix__

        // Build model
        ComputeStats
        """,
        # fmt: on
    ),
]
