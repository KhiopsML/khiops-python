######################################################################################
# Copyright (c) 2018 - 2023 Orange - All Rights Reserved                             #
# * This software is the confidential and proprietary information of Orange.         #
# * You shall not disclose such Restricted Information and shall use it only in      #
#   accordance with the terms of the license agreement you entered into with Orange  #
#   named the "Khiops - Python Library Evaluation License".                          #
# * Unauthorized copying of this file, via any medium is strictly prohibited.        #
# * See the "LICENSE.md" file for more details.                                      #
######################################################################################
"""export_dictionary_as_json task family"""
from pykhiops.core.api_internals import task as tm
from pykhiops.core.api_internals.types import StringLikeType

# Disable long lines to have readable scenarios
# pylint: disable=line-too-long
TASKS = [
    tm.KhiopsTask(
        "export_dictionary_as_json",
        "khiops",
        "9.0",
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
