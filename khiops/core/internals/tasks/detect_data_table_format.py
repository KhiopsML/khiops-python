######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""detect_data_table_format task family"""
from khiops.core.internals import task as tm
from khiops.core.internals.types import StringLikeType

# Disable long lines to have readable scenarios
# pylint: disable=line-too-long
TASKS = [
    tm.KhiopsTask(
        "detect_data_table_format",
        "khiops",
        "10.6.0-b.0",
        [
            ("data_table_path", StringLikeType),
        ],
        [],
        ["data_table_path"],
        # fmt: off
        """
        // Detect format on the "Build Dictionary" window
        TrainDatabase.DatabaseSpec.Data.DatabaseFiles.List.Key
        TrainDatabase.DatabaseSpec.Data.DatabaseFiles.DataTableName __data_table_path__
        TrainDatabase.DatabaseSpec.Data.DatabaseFormatDetector.DetectFileFormat
        """,
        # fmt: on
    ),
]
