"""detect_data_table_format_with_dictionary task family"""
from pykhiops.core.api_internals import task as tm
from pykhiops.core.api_internals.types import StringLikeType

# Disable long lines to have readable scenarios
# pylint: disable=line-too-long
TASKS = [
    tm.KhiopsTask(
        "detect_data_table_format_with_dictionary",
        "khiops",
        "10.0.1",
        [
            ("dictionary_file_path", StringLikeType),
            ("dictionary_name", StringLikeType),
            ("data_table_path", StringLikeType),
        ],
        [],
        ["data_table_path"],
        # fmt: off
        """
        // Dictionary file and class settings
        ClassManagement.OpenFile
        ClassFileName __dictionary_file_path__
        OK
        ClassManagement.ClassName __dictionary_name__

        // Detect format the data table format on the "Extract Keys" window
        LearningTools.ExtractKeysFromDataTable
        SourceDataTable.DatabaseName __data_table_path__
        SourceDataTable.DatabaseFormatDetector.DetectFileFormat
        Exit
        """,
        # fmt: on
    ),
]
