######################################################################################
# Copyright (c) 2018 - 2023 Orange - All Rights Reserved                             #
# * This software is the confidential and proprietary information of Orange.         #
# * You shall not disclose such Restricted Information and shall use it only in      #
#   accordance with the terms of the license agreement you entered into with Orange  #
#   named the "Khiops - Python Library Evaluation License".                          #
# * Unauthorized copying of this file, via any medium is strictly prohibited.        #
# * See the "LICENSE.md" file for more details.                                      #
######################################################################################
"""build_deployed_dictionary task family"""
from pykhiops.core.api_internals import task as tm
from pykhiops.core.api_internals.types import StringLikeType

# Disable long lines to have readable scenarios
# pylint: disable=line-too-long
TASKS = [
    tm.KhiopsTask(
        "build_deployed_dictionary",
        "khiops",
        "10.0",
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
