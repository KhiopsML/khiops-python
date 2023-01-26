"""detect_data_table_format task family"""
from pykhiops.core.api_internals import task as tm
from pykhiops.core.api_internals.types import StringLikeType

# Disable long lines to have readable scenarios
# pylint: disable=line-too-long
TASKS = [
    tm.KhiopsTask(
        "detect_data_table_format",
        "khiops",
        "10.0.1",
        [
            ("data_table_path", StringLikeType),
        ],
        [],
        ["data_table_path"],
        # fmt: off
        """
        // Detect format on the "Build Dictionary" window
        ClassManagement.BuildClassDefButton
        SourceDataTable.DatabaseName __data_table_path__
        SourceDataTable.DatabaseFormatDetector.DetectFileFormat
        Exit
        """,
        # fmt: on
    ),
]
