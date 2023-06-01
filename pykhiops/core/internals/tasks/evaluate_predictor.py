######################################################################################
# Copyright (c) 2023 Orange. All rights reserved.                                    #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""evaluate_predictor task family"""
from pykhiops.core.internals import task as tm
from pykhiops.core.internals.types import BoolType, DictType, FloatType, StringLikeType

# Disable long lines to have readable scenarios
# pylint: disable=line-too-long
TASKS = [
    tm.KhiopsTask(
        "evaluate_predictor",
        "khiops",
        "10.0",
        [
            ("dictionary_file_path", StringLikeType),
            ("train_dictionary_name", StringLikeType),
            ("data_table_path", StringLikeType),
            ("evaluation_report_path", StringLikeType),
        ],
        [
            ("detect_format", BoolType, True),
            ("header_line", BoolType, True),
            ("field_separator", StringLikeType, ""),
            ("sample_percentage", FloatType, 100.0),
            ("sampling_mode", StringLikeType, "Include sample"),
            ("selection_variable", StringLikeType, ""),
            ("selection_value", StringLikeType, ""),
            ("additional_data_tables", DictType(StringLikeType, StringLikeType), None),
            ("main_target_value", StringLikeType, ""),
        ],
        [
            "dictionary_file_path",
            "data_table_path",
            "evaluation_report_path",
            "additional_data_tables",
        ],
        # fmt: off
        """
        // Dictionary file settings
        ClassManagement.OpenFile
        ClassFileName __dictionary_file_path__
        OK

        // Evaluate predictor settings
        LearningTools.EvaluatePredictors
        MainTargetModality __main_target_value__
        EvaluationDatabase.DatabaseFiles.List.Key __train_dictionary_name__
        EvaluationDatabase.DatabaseFiles.DataTableName __data_table_path__
        __DICT__
        __additional_data_tables__
        EvaluationDatabase.DatabaseFiles.List.Key
        EvaluationDatabase.DatabaseFiles.DataTableName
        __END_DICT__
        EvaluationDatabase.HeaderLineUsed __header_line__
        EvaluationDatabase.FieldSeparator __field_separator__
        __OPT__
        __detect_format__
        EvaluationDatabase.DatabaseFormatDetector.DetectFileFormat
        __END_OPT__
        EvaluationDatabase.SampleNumberPercentage __sample_percentage__
        EvaluationDatabase.SamplingMode __sampling_mode__
        EvaluationDatabase.SelectionAttribute __selection_variable__
        EvaluationDatabase.SelectionValue __selection_value__
        EvaluationFileName __evaluation_report_path__

        // Evaluate predictor
        EvaluatePredictors
        Exit
        """,
        # fmt: on
    ),
    tm.KhiopsTask(
        "evaluate_predictor",
        "khiops",
        "9.0",
        [
            ("dictionary_file_path", StringLikeType),
            ("train_dictionary_name", StringLikeType),
            ("data_table_path", StringLikeType),
            ("evaluation_report_path", StringLikeType),
        ],
        [
            ("header_line", BoolType, True),
            ("field_separator", StringLikeType, ""),
            ("sample_percentage", FloatType, 100.0),
            ("sampling_mode", StringLikeType, "Include sample"),
            ("selection_variable", StringLikeType, ""),
            ("selection_value", StringLikeType, ""),
            ("additional_data_tables", DictType(StringLikeType, StringLikeType), None),
            ("main_target_value", StringLikeType, ""),
        ],
        [
            "dictionary_file_path",
            "data_table_path",
            "evaluation_report_path",
            "additional_data_tables",
        ],
        # fmt: off
        """
        // Dictionary file settings
        ClassManagement.OpenFile
        ClassFileName __dictionary_file_path__
        OK

        // Evaluate predictor settings
        LearningTools.EvaluatePredictors
        MainTargetModality __main_target_value__
        EvaluationDatabase.DatabaseFiles.List.Key __train_dictionary_name__
        EvaluationDatabase.DatabaseFiles.DataTableName __data_table_path__
        __DICT__
        __additional_data_tables__
        EvaluationDatabase.DatabaseFiles.List.Key
        EvaluationDatabase.DatabaseFiles.DataTableName
        __END_DICT__
        EvaluationDatabase.HeaderLineUsed __header_line__
        EvaluationDatabase.FieldSeparator __field_separator__
        EvaluationDatabase.SampleNumberPercentage __sample_percentage__
        EvaluationDatabase.SamplingMode __sampling_mode__
        EvaluationDatabase.SelectionAttribute __selection_variable__
        EvaluationDatabase.SelectionValue __selection_value__
        EvaluationFileName __evaluation_report_path__

        // Evaluate predictor
        EvaluatePredictors
        Exit
        """,
        # fmt: on
    ),
]
