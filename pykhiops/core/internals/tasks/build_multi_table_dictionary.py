######################################################################################
# Copyright (c) 2018 - 2023 Orange - All Rights Reserved                             #
# * This software is the confidential and proprietary information of Orange.         #
# * You shall not disclose such Restricted Information and shall use it only in      #
#   accordance with the terms of the license agreement you entered into with Orange  #
#   named the "Khiops - Python Library Evaluation License".                          #
# * Unauthorized copying of this file, via any medium is strictly prohibited.        #
# * See the "LICENSE.md" file for more details.                                      #
######################################################################################
"""build_multi_table_dictionary task family"""
from pykhiops.core.internals import task as tm
from pykhiops.core.internals.types import StringLikeType

# Disable long lines to have readable scenarios
# pylint: disable=line-too-long
TASKS = [
    tm.KhiopsTask(
        "build_multi_table_dictionary",
        "khiops",
        "9.0",
        [
            ("dictionary_file_path", StringLikeType),
            ("root_dictionary_name", StringLikeType),
            ("secondary_table_variable_name", StringLikeType),
        ],
        [],
        ["dictionary_file_path"],
        # fmt: off
        """
        // Dictionary file and class settings
        ClassManagement.OpenFile
        ClassFileName __dictionary_file_path__
        OK

        // Multi-table dictionary creation
        // warning: Overwrites the loaded dictionary file
        LearningTools.ExtractKeysFromDataTable
        BuildMultiTableClass
        MultiTableClassName __root_dictionary_name__
        TableVariableName __secondary_table_variable_name__
        OK
        Exit
        """,
        # fmt: on
    ),
]
