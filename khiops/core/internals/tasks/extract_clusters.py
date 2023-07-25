######################################################################################
# Copyright (c) 2023 Orange. All rights reserved.                                    #
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
        "9.0",
        [
            ("coclustering_file_path", StringLikeType),
            ("cluster_variable", StringLikeType),
            ("results_dir", StringLikeType),
            ("clusters_file_name", StringLikeType),
        ],
        [
            ("max_preserved_information", IntType, 0),
            ("max_cells", IntType, 0),
        ],
        [
            "coclustering_file_path",
            "results_dir",
        ],
        # fmt: off
        """
        // Extract cluster settings
        LearningTools.ExtractClusters
        SelectInputCoclustering
        InputCoclusteringFileName __coclustering_file_path__
        OK
        CoclusteringAttributeSpec.CoclusteringAttribute __cluster_variable__
        PostProcessingSpec.MaxPreservedInformation __max_preserved_information__
        PostProcessingSpec.MaxCellNumber __max_cells__

        // Output settings
        AnalysisResults.ResultFilesDirectory __results_dir__
        AnalysisResults.ClusterFileName __clusters_file_name__

        // Extract clusters
        ExtractClusters
        Exit
        """,
        # fmt: on
    ),
]
