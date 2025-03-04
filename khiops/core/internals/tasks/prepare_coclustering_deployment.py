######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""prepare_coclustering_deployment task family"""
from khiops.core.internals import task as tm
from khiops.core.internals.types import BoolType, DictType, IntType, StringLikeType

# Disable long lines to have readable scenarios
# pylint: disable=line-too-long
TASKS = [
    tm.KhiopsTask(
        "prepare_coclustering_deployment",
        "khiops_coclustering",
        "10.6.0-b.0",
        [
            ("dictionary_file_path", StringLikeType),
            ("dictionary_name", StringLikeType),
            ("coclustering_file_path", StringLikeType),
            ("table_variable", StringLikeType),
            ("deployed_variable_name", StringLikeType),
            ("coclustering_dictionary_file_path", StringLikeType),
        ],
        [
            ("max_preserved_information", IntType, 0),
            ("max_cells", IntType, 0),
            ("max_total_parts", IntType, 0),
            ("max_part_numbers", DictType(StringLikeType, IntType), None),
            ("build_cluster_variable", BoolType, True),
            ("build_distance_variables", BoolType, False),
            ("build_frequency_variables", BoolType, False),
            ("variables_prefix", StringLikeType, ""),
        ],
        [
            "dictionary_file_path",
            "coclustering_file_path",
            "coclustering_dictionary_file_path",
        ],
        # fmt: off
        """
        // Dictionary file and class settings
        ClassManagement.OpenFile
        ClassFileName __dictionary_file_path__
        OK

        // Prepare deployment window
        LearningTools.PrepareDeployment

        // Coclustering file
        SelectInputCoclustering
        InputCoclusteringFileName __coclustering_file_path__
        OK

        // Simplification settings
        PostProcessingSpec.MaxPreservedInformation __max_preserved_information__
        PostProcessingSpec.MaxCellNumber __max_cells__
        PostProcessingSpec.MaxTotalPartNumber __max_total_parts__
        __DICT__
        __max_part_numbers__
        PostProcessingSpec.PostProcessedAttributes.List.Key
        PostProcessingSpec.PostProcessedAttributes.MaxPartNumber
        __END_DICT__

        // Deployment dictionary settings
        DeploymentSpec.InputClassName __dictionary_name__
        DeploymentSpec.InputObjectArrayAttributeName __table_variable__
        DeploymentSpec.DeployedAttributeName __deployed_variable_name__
        DeploymentSpec.BuildPredictedClusterAttribute __build_cluster_variable__
        DeploymentSpec.BuildClusterDistanceAttributes __build_distance_variables__
        DeploymentSpec.BuildFrequencyRecodingAttributes __build_frequency_variables__
        DeploymentSpec.OutputAttributesPrefix __variables_prefix__

        // Output settings
        CoclusteringDictionaryFileName __coclustering_dictionary_file_path__

        // Execute prepare deployment
        PrepareDeployment
        Exit
        """,
        # fmt: on
    ),
]
