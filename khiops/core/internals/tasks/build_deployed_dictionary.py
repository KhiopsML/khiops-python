######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""build_deployed_dictionary task family"""
from khiops.core.internals import task as tm
from khiops.core.internals.types import StringLikeType

# Disable long lines to have readable scenarios
# pylint: disable=line-too-long
TASKS = [
    tm.KhiopsTask(
        "build_deployed_dictionary",
        "khiops",
        "10.0.0",
        [
            ("dictionary_file_path", StringLikeType),
            ("dictionary_name", StringLikeType),
            ("output_dictionary_file_path", StringLikeType),
        ],
        [],
        ["dictionary_file_path", "output_dictionary_file_path"],
        # fmt: off
        """
        // Dictionary file and class settings
        ClassManagement.OpenFile
        ClassFileName __dictionary_file_path__
        OK

        // Transfer settings
        LearningTools.TransferDatabase
        ClassName __dictionary_name__
        BuildTransferredClass
        ClassFileName __output_dictionary_file_path__
        OK
        Exit
        """,
        # fmt: on
    ),
]
