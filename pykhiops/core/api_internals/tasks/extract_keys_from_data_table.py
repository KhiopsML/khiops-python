######################################################################################
# Copyright (c) 2018 - 2023 Orange - All Rights Reserved                             #
# * This software is the confidential and proprietary information of Orange.         #
# * You shall not disclose such Restricted Information and shall use it only in      #
#   accordance with the terms of the license agreement you entered into with Orange  #
#   named the "Khiops - Python Library Evaluation License".                          #
# * Unauthorized copying of this file, via any medium is strictly prohibited.        #
# * See the "LICENSE.md" file for more details.                                      #
######################################################################################
"""extract_keys_from_data_table task family"""
from pykhiops.core.api_internals import task as tm
from pykhiops.core.api_internals.types import BoolType, StringLikeType

# Disable long lines to have readable scenarios
# pylint: disable=line-too-long
TASKS = [
    tm.KhiopsTask(
        "extract_keys_from_data_table",
        "khiops",
        "10.0",
        [
            ("dictionary_file_path", StringLikeType),
            ("dictionary_name", StringLikeType),
            ("data_table_path", StringLikeType),
            ("output_data_table_path", StringLikeType),
        ],
        [
            ("detect_format", BoolType, True),
            ("header_line", BoolType, True),
            ("field_separator", StringLikeType, ""),
            ("output_header_line", BoolType, True),
            ("output_field_separator", StringLikeType, ""),
        ],
        [
            "dictionary_file_path",
            "data_table_path",
            "output_data_table_path",
        ],
        # fmt: off
        """
        // Dictionary file
        ClassManagement.OpenFile
        ClassFileName __dictionary_file_path__
        OK

        // Extract keys settings
        LearningTools.ExtractKeysFromDataTable
        ClassName __dictionary_name__
        SourceDataTable.DatabaseName __data_table_path__
        SourceDataTable.HeaderLineUsed __header_line__
        SourceDataTable.FieldSeparator __field_separator__
        __OPT__
        __detect_format__
        SourceDataTable.DatabaseFormatDetector.DetectFileFormat
        __END_OPT__
        TargetDataTable.DatabaseName __output_data_table_path__
        TargetDataTable.HeaderLineUsed __output_header_line__
        TargetDataTable.FieldSeparator __output_field_separator__
        ExtractKeysFromDataTable
        Exit
        """,
        # fmt: on
    ),
    tm.KhiopsTask(
        "extract_keys_from_data_table",
        "khiops",
        "9.0",
        [
            ("dictionary_file_path", StringLikeType),
            ("dictionary_name", StringLikeType),
            ("data_table_path", StringLikeType),
            ("output_data_table_path", StringLikeType),
        ],
        [
            ("header_line", BoolType, True),
            ("field_separator", StringLikeType, ""),
            ("output_header_line", BoolType, True),
            ("output_field_separator", StringLikeType, ""),
        ],
        [
            "dictionary_file_path",
            "data_table_path",
            "output_data_table_path",
        ],
        # fmt: off
        """
        // Dictionary file
        ClassManagement.OpenFile
        ClassFileName __dictionary_file_path__
        OK

        // Extract keys settings
        LearningTools.ExtractKeysFromDataTable
        ClassName __dictionary_name__
        SourceDataTable.DatabaseName __data_table_path__
        SourceDataTable.HeaderLineUsed __header_line__
        SourceDataTable.FieldSeparator __field_separator__
        TargetDataTable.DatabaseName __output_data_table_path__
        TargetDataTable.HeaderLineUsed __output_header_line__
        TargetDataTable.FieldSeparator __output_field_separator__
        ExtractKeysFromDataTable
        Exit
        """,
        # fmt: on
    ),
]
