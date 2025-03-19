######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""train_predictor task family"""
from khiops.core.internals import task as tm
from khiops.core.internals.types import (
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
        "10.6.0-b.0",
        [
            ("dictionary_file_path", StringLikeType),
            ("dictionary_name", StringLikeType),
            ("data_table_path", StringLikeType),
            ("target_variable", StringLikeType),
            ("analysis_report_file_path", StringLikeType),
        ],
        [
            ("detect_format", BoolType, True),
            ("header_line", BoolType, True),
            ("field_separator", StringLikeType, ""),
            ("sample_percentage", FloatType, 70.0),
            ("sampling_mode", StringLikeType, "Include sample"),
            ("test_database_mode", StringLikeType, "Complementary"),
            ("selection_variable", StringLikeType, ""),
            ("selection_value", StringLikeType, ""),
            ("additional_data_tables", DictType(StringLikeType, StringLikeType), None),
            ("do_data_preparation_only", BoolType, False),
            ("main_target_value", StringLikeType, ""),
            ("keep_selected_variables_only", BoolType, True),
            ("max_evaluated_variables", IntType, 0),
            ("max_selected_variables", IntType, 0),
            ("max_constructed_variables", IntType, 1000),
            ("construction_rules", ListType(StringLikeType), None),
            ("max_text_features", IntType, 10000),
            ("max_trees", IntType, 10),
            ("max_pairs", IntType, 0),
            ("all_possible_pairs", BoolType, True),
            (
                "specific_pairs",
                ListType(TupleType(StringLikeType, StringLikeType)),
                None,
            ),
            ("text_features", StringLikeType, "words"),
            ("group_target_value", BoolType, False),
            ("discretization_method", StringLikeType, "MODL"),
            ("grouping_method", StringLikeType, "MODL"),
            ("max_parts", IntType, 0),
        ],
        [
            "dictionary_file_path",
            "data_table_path",
            "additional_data_tables",
            "analysis_report_file_path",
        ],
        # pylint: disable=line-too-long
        # fmt: off
        """
        // Dictionary file and class settings
        ClassManagement.OpenFile
        ClassFileName __dictionary_file_path__
        OK

        // Train/test database settings
        TrainDatabase.ClassName __dictionary_name__
        TrainDatabase.DatabaseSpec.Data.DatabaseFiles.List.Key
        TrainDatabase.DatabaseSpec.Data.DatabaseFiles.DataTableName __data_table_path__
        __DICT__
        __additional_data_tables__
        TrainDatabase.DatabaseSpec.Data.DatabaseFiles.List.Key
        TrainDatabase.DatabaseSpec.Data.DatabaseFiles.DataTableName
        __END_DICT__
        TrainDatabase.DatabaseSpec.Data.HeaderLineUsed __header_line__
        TrainDatabase.DatabaseSpec.Data.FieldSeparator __field_separator__
        __OPT__
        __detect_format__
        TrainDatabase.DatabaseSpec.Data.DatabaseFormatDetector.DetectFileFormat
        __END_OPT__
        TrainDatabase.DatabaseSpec.Sampling.SampleNumberPercentage __sample_percentage__
        TrainDatabase.DatabaseSpec.Sampling.SamplingMode __sampling_mode__
        TrainDatabase.DatabaseSpec.Selection.SelectionAttribute __selection_variable__
        TrainDatabase.DatabaseSpec.Selection.SelectionValue __selection_value__
        TrainDatabase.TestDatabaseSpecificationMode __test_database_mode__

        // Target variable
        AnalysisSpec.TargetAttributeName __target_variable__
        AnalysisSpec.MainTargetModality __main_target_value__

        // Do data preparation only
        AnalysisSpec.PredictorsSpec.AdvancedSpec.DataPreparationOnly __do_data_preparation_only__

        // Selective Naive Bayes settings
        AnalysisSpec.PredictorsSpec.AdvancedSpec.SelectiveNaiveBayesParameters.TrainParameters.MaxEvaluatedAttributeNumber __max_evaluated_variables__
        AnalysisSpec.PredictorsSpec.AdvancedSpec.SelectiveNaiveBayesParameters.SelectionParameters.MaxSelectedAttributeNumber __max_selected_variables__

        // Feature engineering
        AnalysisSpec.PredictorsSpec.ConstructionSpec.MaxTextFeatureNumber __max_text_features__
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
        AnalysisSpec.PredictorsSpec.ConstructionSpec.KeepSelectedAttributesOnly __keep_selected_variables_only__
        AnalysisSpec.PredictorsSpec.ConstructionSpec.MaxConstructedAttributeNumber __max_constructed_variables__
        AnalysisSpec.PredictorsSpec.AdvancedSpec.InspectConstructionDomain
        __DICT__
        __construction_rules__
        UnselectAll
        ConstructionRules.List.Key
        ConstructionRules.Used
        __END_DICT__
        Exit
        
        //  Text feature parameters
        AnalysisSpec.PredictorsSpec.AdvancedSpec.InspectTextFeaturesParameters 
        TextFeatures __text_features__
        Exit


        // Data preparation (discretization & grouping) settings
        AnalysisSpec.PreprocessingSpec.TargetGrouped __group_target_value__
        AnalysisSpec.PreprocessingSpec.InspectAdvancedParameters
        DiscretizerUnsupervisedMethodName __discretization_method__
        GrouperUnsupervisedMethodName __grouping_method__
        Exit
        
        // Max parts
        AnalysisSpec.PreprocessingSpec.MaxPartNumber __max_parts__

        // Output settings
        AnalysisResults.ReportFileName __analysis_report_file_path__

        // Build model
        ComputeStats
        """,
        # fmt: on
    ),
]
