"""extract_clusters task family"""
from pykhiops.core.api_internals import task as tm
from pykhiops.core.api_internals.types import IntType, StringLikeType

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
