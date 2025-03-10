######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""train_coclustering task family"""
from khiops.core.internals import task as tm
from khiops.core.internals.types import (
    BoolType,
    DictType,
    FloatType,
    IntType,
    ListType,
    StringLikeType,
)

# Disable long lines to have readable scenarios
# pylint: disable=line-too-long
TASKS = [
    tm.KhiopsTask(
        "train_coclustering",
        "khiops_coclustering",
        "10.6.0-b.0",
        [
            ("dictionary_file_path", StringLikeType),
            ("dictionary_name", StringLikeType),
            ("data_table_path", StringLikeType),
            ("coclustering_variables", ListType(StringLikeType)),
            ("coclustering_report_file_path", StringLikeType),
        ],
        [
            ("detect_format", BoolType, True),
            ("header_line", BoolType, True),
            ("field_separator", StringLikeType, ""),
            ("sample_percentage", FloatType, 100.0),
            ("sampling_mode", StringLikeType, "Include sample"),
            ("selection_variable", StringLikeType, ""),
            ("selection_value", StringLikeType, ""),
            ("additional_data_tables", DictType(StringLikeType, StringLikeType), None),
            ("frequency_variable", StringLikeType, ""),
            ("min_optimization_time", IntType, 0),
        ],
        [
            "dictionary_file_path",
            "data_table_path",
            "coclustering_report_file_path",
            "additional_data_tables",
        ],
        # fmt: off
        """
        // Dictionary file and class settings
        ClassManagement.OpenFile
        ClassFileName __dictionary_file_path__
        OK

        // Train database settings
        Database.ClassName __dictionary_name__
        Database.DatabaseSpec.Data.DatabaseFiles.List.Key
        Database.DatabaseSpec.Data.DatabaseFiles.DataTableName __data_table_path__
        __DICT__
        __additional_data_tables__
        Database.DatabaseSpec.Data.DatabaseFiles.List.Key
        Database.DatabaseSpec.Data.DatabaseFiles.DataTableName
        __END_DICT__
        Database.DatabaseSpec.Data.HeaderLineUsed __header_line__
        Database.DatabaseSpec.Data.FieldSeparator __field_separator__
        __OPT__
        __detect_format__
        Database.DatabaseSpec.Data.DatabaseFormatDetector.DetectFileFormat
        __END_OPT__
        Database.DatabaseSpec.Sampling.SampleNumberPercentage __sample_percentage__
        Database.DatabaseSpec.Sampling.SamplingMode __sampling_mode__
        Database.DatabaseSpec.Selection.SelectionAttribute __selection_variable__
        Database.DatabaseSpec.Selection.SelectionValue __selection_value__

        // Coclustering variables settings
        __LIST__
        __coclustering_variables__
        AnalysisSpec.CoclusteringParameters.Attributes.InsertItemAfter
        AnalysisSpec.CoclusteringParameters.Attributes.Name
        __END_LIST__
        AnalysisSpec.CoclusteringParameters.FrequencyAttributeName __frequency_variable__

        // Minimum optimization time
        AnalysisSpec.SystemParameters.OptimizationTime __min_optimization_time__

        // Output settings
        AnalysisResults.CoclusteringFileName __coclustering_report_file_path__

        // Train
        BuildCoclustering
        """,
        # fmt: on
    ),
]
