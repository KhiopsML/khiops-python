######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""simplify_coclustering task family"""
from khiops.core.internals import task as tm
from khiops.core.internals.types import DictType, IntType, StringLikeType

# Disable long lines to have readable scenarios
# pylint: disable=line-too-long
TASKS = [
    tm.KhiopsTask(
        "simplify_coclustering",
        "khiops_coclustering",
        "10.6.0-b.0",
        [
            ("coclustering_file_path", StringLikeType),
            ("simplified_coclustering_file_path", StringLikeType),
        ],
        [
            ("max_preserved_information", IntType, 0),
            ("max_cells", IntType, 0),
            ("max_total_parts", IntType, 0),
            ("max_part_numbers", DictType(StringLikeType, IntType), None),
        ],
        [
            "coclustering_file_path",
            "simplified_coclustering_file_path",
        ],
        # fmt: off
        """
        // Simplify coclustering settings
        LearningTools.PostProcessCoclustering
        SelectInputCoclustering
        InputCoclusteringFileName __coclustering_file_path__
        OK
        PostProcessingSpec.MaxPreservedInformation __max_preserved_information__
        PostProcessingSpec.MaxCellNumber __max_cells__
        PostProcessingSpec.MaxTotalPartNumber __max_total_parts__
        __DICT__
        __max_part_numbers__
        PostProcessingSpec.PostProcessedAttributes.List.Key
        PostProcessingSpec.PostProcessedAttributes.MaxPartNumber
        __END_DICT__

        // Output settings
        PostProcessedCoclusteringFileName __simplified_coclustering_file_path__

        // Simplify Coclustering
        PostProcessCoclustering
        Exit
        """,
        # fmt: on
    ),
]
