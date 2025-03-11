######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""check_database task family"""
from khiops.core.internals import task as tm
from khiops.core.internals.types import (
    BoolType,
    DictType,
    FloatType,
    IntType,
    StringLikeType,
)

# Disable long lines to have readable scenarios
# pylint: disable=line-too-long
TASKS = [
    tm.KhiopsTask(
        "check_database",
        "khiops",
        "10.6.0-b.0",
        [
            ("dictionary_file_path", StringLikeType),
            ("dictionary_name", StringLikeType),
            ("data_table_path", StringLikeType),
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
            ("max_messages", IntType, 20),
        ],
        ["dictionary_file_path", "data_table_path", "additional_data_tables"],
        # fmt: off
        """
        // Dictionary file and class settings
        ClassManagement.OpenFile
        ClassFileName __dictionary_file_path__
        OK

        // Train database settings
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

        // Log messages limit
        AnalysisSpec.SystemParameters.MaxErrorMessageNumberInLog __max_messages__

        // Execute check database
        LearningTools.CheckData
        """,
        # fmt: on
    ),
]
