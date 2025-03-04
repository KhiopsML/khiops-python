######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""export_dictionary_as_json task family"""
from khiops.core.internals import task as tm
from khiops.core.internals.types import StringLikeType

# Disable long lines to have readable scenarios
# pylint: disable=line-too-long
TASKS = [
    tm.KhiopsTask(
        "export_dictionary_as_json",
        "khiops",
        "10.6.0-b.0",
        [
            ("dictionary_file_path", StringLikeType),
            ("json_dictionary_file_path", StringLikeType),
        ],
        [],
        ["dictionary_file_path", "json_dictionary_file_path"],
        """
        // Dictionary file settings
        ClassManagement.OpenFile
        ClassFileName __dictionary_file_path__
        OK

        // Export dictionary as JSON file
        ClassManagement.ExportAsJSON
        JSONFileName __json_dictionary_file_path__
        OK
        """,
    ),
]
