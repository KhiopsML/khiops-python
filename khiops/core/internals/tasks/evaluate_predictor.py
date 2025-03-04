######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""evaluate_predictor task family"""
from khiops.core.internals import task as tm
from khiops.core.internals.types import BoolType, DictType, FloatType, StringLikeType

# Disable long lines to have readable scenarios
# pylint: disable=line-too-long
TASKS = [
    tm.KhiopsTask(
        "evaluate_predictor",
        "khiops",
        "10.6.0-b.0",
        [
            ("dictionary_file_path", StringLikeType),
            ("train_dictionary_name", StringLikeType),
            ("data_table_path", StringLikeType),
            ("evaluation_report_file_path", StringLikeType),
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
            "evaluation_report_file_path",
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
        EvaluationDatabase.DatabaseSpec.Data.DatabaseFiles.List.Key
        EvaluationDatabase.DatabaseSpec.Data.DatabaseFiles.DataTableName __data_table_path__
        __DICT__
        __additional_data_tables__
        EvaluationDatabase.DatabaseSpec.Data.DatabaseFiles.List.Key
        EvaluationDatabase.DatabaseSpec.Data.DatabaseFiles.DataTableName
        __END_DICT__
        EvaluationDatabase.DatabaseSpec.Data.HeaderLineUsed __header_line__
        EvaluationDatabase.DatabaseSpec.Data.FieldSeparator __field_separator__
        __OPT__
        __detect_format__
        EvaluationDatabase.DatabaseSpec.Data.DatabaseFormatDetector.DetectFileFormat
        __END_OPT__
        EvaluationDatabase.DatabaseSpec.Sampling.SampleNumberPercentage __sample_percentage__
        EvaluationDatabase.DatabaseSpec.Sampling.SamplingMode __sampling_mode__
        EvaluatedPredictors.List.Key __train_dictionary_name__
        EvaluationDatabase.DatabaseSpec.Selection.SelectionAttribute __selection_variable__
        EvaluationDatabase.DatabaseSpec.Selection.SelectionValue __selection_value__
        ExportAsXls false
        EvaluationFileName __evaluation_report_file_path__

        // Evaluate predictor
        EvaluatePredictors
        Exit
        """,
        # fmt: on
    ),
]
