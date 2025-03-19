######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""detect_data_table_format_with_dictionary task family"""
from khiops.core.internals import task as tm
from khiops.core.internals.types import StringLikeType

# Disable long lines to have readable scenarios
# pylint: disable=line-too-long
TASKS = [
    tm.KhiopsTask(
        "detect_data_table_format_with_dictionary",
        "khiops",
        "10.6.0-b.0",
        [
            ("dictionary_file_path", StringLikeType),
            ("dictionary_name", StringLikeType),
            ("data_table_path", StringLikeType),
        ],
        [],
        ["dictionary_file_path", "data_table_path"],
        # fmt: off
        """
        // Dictionary file and class settings
        ClassManagement.OpenFile
        ClassFileName __dictionary_file_path__
        OK

        TrainDatabase.ClassName __dictionary_name__

        // Detect format the data table format on the "Extract Keys" window
        LearningTools.ExtractKeysFromDataTable
        SourceDataTable.DatabaseSpec.Data.DatabaseName __data_table_path__
        SourceDataTable.DatabaseSpec.Data.DatabaseFormatDetector.DetectFileFormat
        Exit
        """,
        # fmt: on
    ),
]
