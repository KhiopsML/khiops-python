######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""extract_clusters task family"""
from khiops.core.internals import task as tm
from khiops.core.internals.types import IntType, StringLikeType

# Disable long lines to have readable scenarios
# pylint: disable=line-too-long
TASKS = [
    tm.KhiopsTask(
        "extract_clusters",
        "khiops_coclustering",
        "10.6.0-b.0",
        [
            ("coclustering_file_path", StringLikeType),
            ("cluster_variable", StringLikeType),
            ("clusters_file_path", StringLikeType),
        ],
        [
            ("max_preserved_information", IntType, 0),
            ("max_cells", IntType, 0),
        ],
        [
            "coclustering_file_path",
        ],
        # fmt: off
        """
        // Extract cluster settings
        LearningTools.ExtractClusters
        SelectInputCoclustering
        InputCoclusteringFileName __coclustering_file_path__
        OK
        CoclusteringAttribute __cluster_variable__
        PostProcessingSpec.MaxPreservedInformation __max_preserved_information__
        PostProcessingSpec.MaxCellNumber __max_cells__

        // Output settings
        ClusterFileName __clusters_file_path__

        // Extract clusters
        ExtractClusters
        Exit
        ClassManagement.Quit
        OK
        """,
        # fmt: on
    ),
]
