"""build_dictionary_from_data_table task family"""
from pykhiops.core.api_internals import task as tm
from pykhiops.core.api_internals.types import BoolType, StringLikeType

# Disable long lines to have readable scenarios
# pylint: disable=line-too-long
TASKS = [
    tm.KhiopsTask(
        "build_dictionary_from_data_table",
        "khiops",
        "10.0",
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
        ClassManagement.BuildClassDefButton
        SourceDataTable.DatabaseName __data_table_path__
        SourceDataTable.HeaderLineUsed __header_line__
        SourceDataTable.FieldSeparator __field_separator__
        __OPT__
        __detect_format__
        SourceDataTable.DatabaseFormatDetector.DetectFileFormat
        __END_OPT__
        BuildClassDef
        ClassName __output_dictionary_name__
        OK
        Exit

        // Save dictionary
        ClassFileName __output_dictionary_file_path__
        OK
        """,
        # fmt: on
    ),
    tm.KhiopsTask(
        "build_dictionary_from_data_table",
        "khiops",
        "9.0",
        [
            ("data_table_path", StringLikeType),
            ("output_dictionary_name", StringLikeType),
            ("output_dictionary_file_path", StringLikeType),
        ],
        [
            ("header_line", BoolType, True),
            ("field_separator", StringLikeType, ""),
        ],
        ["data_table_path", "output_dictionary_file_path"],
        # fmt: off
        """
        // Dictionary building settings
        ClassManagement.BuildClassDefButton
        SourceDataTable.DatabaseName __data_table_path__
        SourceDataTable.HeaderLineUsed __header_line__
        SourceDataTable.FieldSeparator __field_separator__
        BuildClassDef
        ClassName __output_dictionary_name__
        OK
        Exit

        // Save dictionary
        ClassFileName __output_dictionary_file_path__
        OK
        """,
        # fmt: on
    ),
]
