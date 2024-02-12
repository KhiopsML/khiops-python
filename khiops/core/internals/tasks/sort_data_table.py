######################################################################################
# Copyright (c) 2023 Orange. All rights reserved.                                    #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""sort_data_table task family"""
from khiops.core.internals import task as tm
from khiops.core.internals.types import BoolType, ListType, StringLikeType

# Disable long lines to have readable scenarios
# pylint: disable=line-too-long
TASKS = [
    tm.KhiopsTask(
        "sort_data_table",
        "khiops",
        "10.0.0",
        [
            ("dictionary_file_path", StringLikeType),
            ("dictionary_name", StringLikeType),
            ("data_table_path", StringLikeType),
            ("output_data_table_path", StringLikeType),
        ],
        [
            ("sort_variables", ListType(StringLikeType), None),
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
        // Dictionary file and class settings
        ClassManagement.OpenFile
        ClassFileName __dictionary_file_path__
        OK
        ClassManagement.ClassName __dictionary_name__

        // Sort table settings
        LearningTools.SortDataTableByKey
        SortAttributes.SelectDefaultKeyAttributes
        __LIST__
        __sort_variables__
        SortAttributes.RemoveItem
        SortAttributes.RemoveItem
        SortAttributes.RemoveItem
        SortAttributes.RemoveItem
        SortAttributes.RemoveItem
        SortAttributes.RemoveItem
        SortAttributes.RemoveItem
        SortAttributes.RemoveItem
        SortAttributes.RemoveItem
        SortAttributes.RemoveItem
        SortAttributes.InsertItemAfter
        SortAttributes.Name
        __END_LIST__

        // Source table settings
        SourceDataTable.DatabaseName __data_table_path__
        SourceDataTable.HeaderLineUsed __header_line__
        SourceDataTable.FieldSeparator __field_separator__
        __OPT__
        __detect_format__
        SourceDataTable.DatabaseFormatDetector.DetectFileFormat
        __END_OPT__

        // Target table settings
        TargetDataTable.HeaderLineUsed __output_header_line__
        TargetDataTable.DatabaseName __output_data_table_path__
        TargetDataTable.FieldSeparator __output_field_separator__

        // Sort table
        SortDataTableByKey
        Exit
        """,
        # fmt: on
    ),
    tm.KhiopsTask(
        "sort_data_table",
        "khiops",
        "9.0.0",
        [
            ("dictionary_file_path", StringLikeType),
            ("dictionary_name", StringLikeType),
            ("data_table_path", StringLikeType),
            ("output_data_table_path", StringLikeType),
        ],
        [
            ("sort_variables", ListType(StringLikeType), None),
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
        // Dictionary file and class settings
        ClassManagement.OpenFile
        ClassFileName __dictionary_file_path__
        OK
        ClassManagement.ClassName __dictionary_name__

        // Sort table settings
        LearningTools.SortDataTableByKey
        SortAttributes.SelectDefaultKeyAttributes
        __LIST__
        __sort_variables__
        SortAttributes.RemoveItem
        SortAttributes.RemoveItem
        SortAttributes.RemoveItem
        SortAttributes.RemoveItem
        SortAttributes.RemoveItem
        SortAttributes.RemoveItem
        SortAttributes.RemoveItem
        SortAttributes.RemoveItem
        SortAttributes.RemoveItem
        SortAttributes.RemoveItem
        SortAttributes.InsertItemAfter
        SortAttributes.Name
        __END_LIST__

        // Source table settings
        SourceDataTable.DatabaseName __data_table_path__
        SourceDataTable.HeaderLineUsed __header_line__
        SourceDataTable.FieldSeparator __field_separator__

        // Target table settings
        TargetDataTable.HeaderLineUsed __output_header_line__
        TargetDataTable.DatabaseName __output_data_table_path__
        TargetDataTable.FieldSeparator __output_field_separator__

        // Sort table
        SortDataTableByKey
        Exit
        """,
        # fmt: on
    ),
]
