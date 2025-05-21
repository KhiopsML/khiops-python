######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""interpret_predictor task family"""
from khiops.core.internals import task as tm
from khiops.core.internals.types import IntType, StringLikeType

# Disable long lines to have readable scenarios
# pylint: disable=line-too-long
TASKS = [
    tm.KhiopsTask(
        "interpret_predictor",
        "khiops",
        "10.6.0-b.0",
        [
            ("dictionary_file_path", StringLikeType),
            ("predictor_dictionary_name", StringLikeType),
            ("interpretor_file_path", StringLikeType),
        ],
        [
            ("max_variable_importances", IntType, 100),
            ("importance_ranking", StringLikeType, "Global"),
        ],
        ["dictionary_file_path", "interpretor_file_path"],
        # pylint: disable=line-too-long
        # fmt: off
        """
        // Dictionary file and class settings
        ClassManagement.OpenFile
        ClassFileName __dictionary_file_path__
        OK

        // interpretation settings
        TrainDatabase.ClassName __predictor_dictionary_name__

        // Interpret model
        LearningTools.InterpretPredictor

        // Number of predictor variables exploited in the interpretation model
        ContributionAttributeNumber __max_variable_importances__

        // Ranking of the Shapley value produced by the interpretation model
        ShapleyValueRanking __importance_ranking__

        // Build interpretation dictionary
        BuildInterpretationClass

        // Output settings
        ClassFileName __interpretor_file_path__
        OK
        Exit
        """,
        # fmt: on
    ),
]
