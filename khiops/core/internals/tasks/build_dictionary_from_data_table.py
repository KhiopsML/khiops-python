######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""build_dictionary_from_data_table task family"""
from khiops.core.internals import task as tm
from khiops.core.internals.types import BoolType, StringLikeType

# Disable long lines to have readable scenarios
# pylint: disable=line-too-long
TASKS = [
    tm.KhiopsTask(
        "build_dictionary_from_data_table",
        "khiops",
        "10.6.0-b.0",
        [
            ("data_table_path", StringLikeType),
            ("output_dictionary_name", StringLikeType),
            ("output_dictionary_file_path", StringLikeType),
        ],
        [
            ("detect_format", BoolType, True),
            ("header_line", BoolType, True),
            ("field_separator", StringLikeType, ""),
        ],
        ["data_table_path", "output_dictionary_file_path"],
        # fmt: off
        """
        // Dictionary building settings
        ClassManagement.ManageClasses
        BuildClassDefButton
        SourceDataTable.DatabaseSpec.Data.DatabaseName __data_table_path__
        SourceDataTable.DatabaseSpec.Data.HeaderLineUsed __header_line__
        SourceDataTable.DatabaseSpec.Data.FieldSeparator __field_separator__
        __OPT__
        __detect_format__
        SourceDataTable.DatabaseSpec.Data.DatabaseFormatDetector.DetectFileFormat
        __END_OPT__
        BuildClassDef
        ClassName __output_dictionary_name__
        OK
        Exit

        // Save dictionary
        ClassFileName __output_dictionary_file_path__
        OK
        Exit
        """,
        # fmt: on
    ),
]
