######################################################################################
# Copyright (c) 2018 - 2023 Orange - All Rights Reserved                             #
# * This software is the confidential and proprietary information of Orange.         #
# * You shall not disclose such Restricted Information and shall use it only in      #
#   accordance with the terms of the license agreement you entered into with Orange  #
#   named the "Khiops - Python Library Evaluation License".                          #
# * Unauthorized copying of this file, via any medium is strictly prohibited.        #
# * See the "LICENSE.md" file for more details.                                      #
######################################################################################
"""check_database task family"""
from pykhiops.core.internals import task as tm
from pykhiops.core.internals.types import (
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
        "10.0",
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
        ClassManagement.ClassName __dictionary_name__

        // Train database settings
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

        // Log messages limit
        AnalysisSpec.SystemParameters.MaxErrorMessageNumberInLog __max_messages__

        // Execute check database
        LearningTools.CheckData
        """,
        # fmt: on
    ),
    tm.KhiopsTask(
        "check_database",
        "khiops",
        "9.0",
        [
            ("dictionary_file_path", StringLikeType),
            ("dictionary_name", StringLikeType),
            ("data_table_path", StringLikeType),
        ],
        [
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
        ClassManagement.ClassName __dictionary_name__

        // Train database settings
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

        // Log messages limit
        AnalysisSpec.SystemParameters.MaxErrorMessageNumberInLog __max_messages__

        // Execute check database
        LearningTools.CheckData
        """,
        # fmt: on
    ),
]
