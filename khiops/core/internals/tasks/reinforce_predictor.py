######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""reinforce_predictor task family"""
from khiops.core.internals import task as tm
from khiops.core.internals.types import ListType, StringLikeType

# Disable long lines to have readable scenarios
# pylint: disable=line-too-long
TASKS = [
    tm.KhiopsTask(
        "reinforce_predictor",
        "khiops",
        "10.7.3-a.0",
        [
            ("dictionary_file_path", StringLikeType),
            ("predictor_dictionary_name", StringLikeType),
            ("reinforced_predictor_file_path", StringLikeType),
        ],
        [
            ("reinforcement_target_value", StringLikeType, ""),
            ("reinforcement_lever_variables", ListType(StringLikeType), None),
        ],
        ["dictionary_file_path", "reinforced_predictor_file_path"],
        # pylint: disable=line-too-long
        # fmt: off
        """
        // Dictionary file and class settings
        ClassManagement.OpenFile
        ClassFileName __dictionary_file_path__
        OK

        // Reinforcement settings
        TrainDatabase.ClassName __predictor_dictionary_name__

        // Reinforce model
        LearningTools.ReinforcePredictor
        ReinforcedTargetValue __reinforcement_target_value__

        LeverAttributes.UnselectAll
        __DICT__
        __reinforcement_lever_variables__
        LeverAttributes.List.Key
        LeverAttributes.Used
        __END_DICT__

        // Build reinforced predictor
        BuildReinforcementClass

        // Output settings
        ClassFileName __reinforced_predictor_file_path__
        OK
        Exit
        """,
        # fmt: on
    ),
]
