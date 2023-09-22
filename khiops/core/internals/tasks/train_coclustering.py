######################################################################################
# Copyright (c) 2023 Orange. All rights reserved.                                    #
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
        "10.0",
        [
            ("dictionary_file_path", StringLikeType),
            ("dictionary_name", StringLikeType),
            ("data_table_path", StringLikeType),
            ("coclustering_variables", ListType(StringLikeType)),
            ("results_dir", StringLikeType),
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
            ("results_prefix", StringLikeType, ""),
        ],
        [
            "dictionary_file_path",
            "data_table_path",
            "results_dir",
            "additional_data_tables",
        ],
        # fmt: off
        """
        // Dictionary file and class settings
        ClassManagement.OpenFile
        ClassFileName __dictionary_file_path__
        OK
        ClassManagement.ClassName __dictionary_name__

        // Train database settings
        Database.DatabaseFiles.List.Key __dictionary_name__
        Database.DatabaseFiles.DataTableName __data_table_path__
        __DICT__
        __additional_data_tables__
        Database.DatabaseFiles.List.Key
        Database.DatabaseFiles.DataTableName
        __END_DICT__
        Database.HeaderLineUsed __header_line__
        Database.FieldSeparator __field_separator__
        __OPT__
        __detect_format__
        Database.DatabaseFormatDetector.DetectFileFormat
        __END_OPT__
        Database.SampleNumberPercentage __sample_percentage__
        Database.SamplingMode __sampling_mode__
        Database.SelectionAttribute __selection_variable__
        Database.SelectionValue __selection_value__

        // Coclustering variables settings
        __LIST__
        __coclustering_variables__
        AnalysisSpec.CoclusteringParameters.Attributes.InsertItemAfter
        AnalysisSpec.CoclusteringParameters.Attributes.Name
        __END_LIST__
        AnalysisSpec.CoclusteringParameters.FrequencyAttribute __frequency_variable__

        // Minimum optimization time
        AnalysisSpec.SystemParameters.OptimizationTime __min_optimization_time__

        // Output settings
        AnalysisResults.ResultFilesDirectory __results_dir__
        AnalysisResults.ResultFilesPrefix __results_prefix__

        // Train
        BuildCoclustering
        """,
        # fmt: on
    ),
    tm.KhiopsTask(
        "train_coclustering",
        "khiops_coclustering",
        "9.0",
        [
            ("dictionary_file_path", StringLikeType),
            ("dictionary_name", StringLikeType),
            ("data_table_path", StringLikeType),
            ("coclustering_variables", ListType(StringLikeType)),
            ("results_dir", StringLikeType),
        ],
        [
            ("header_line", BoolType, True),
            ("field_separator", StringLikeType, ""),
            ("sample_percentage", FloatType, 100.0),
            ("sampling_mode", StringLikeType, "Include sample"),
            ("selection_variable", StringLikeType, ""),
            ("selection_value", StringLikeType, ""),
            ("additional_data_tables", DictType(StringLikeType, StringLikeType), None),
            ("frequency_variable", StringLikeType, ""),
            ("min_optimization_time", IntType, 0),
            ("results_prefix", StringLikeType, ""),
        ],
        [
            "dictionary_file_path",
            "data_table_path",
            "results_dir",
            "additional_data_tables",
        ],
        # fmt: off
        """
        // Dictionary file and class settings
        ClassManagement.OpenFile
        ClassFileName __dictionary_file_path__
        OK
        ClassManagement.ClassName __dictionary_name__

        // Train database settings
        Database.DatabaseFiles.List.Key __dictionary_name__
        Database.DatabaseFiles.DataTableName __data_table_path__
        __DICT__
        __additional_data_tables__
        Database.DatabaseFiles.List.Key
        Database.DatabaseFiles.DataTableName
        __END_DICT__
        Database.HeaderLineUsed __header_line__
        Database.FieldSeparator __field_separator__
        Database.SampleNumberPercentage __sample_percentage__
        Database.SamplingMode __sampling_mode__
        Database.SelectionAttribute __selection_variable__
        Database.SelectionValue __selection_value__

        // Coclustering variables settings
        __LIST__
        __coclustering_variables__
        AnalysisSpec.CoclusteringParameters.Attributes.InsertItemAfter
        AnalysisSpec.CoclusteringParameters.Attributes.Name
        __END_LIST__
        AnalysisSpec.CoclusteringParameters.FrequencyAttribute __frequency_variable__

        // Minimum optimization time
        AnalysisSpec.SystemParameters.OptimizationTime __min_optimization_time__

        // Output settings
        AnalysisResults.ResultFilesDirectory __results_dir__
        AnalysisResults.ResultFilesPrefix __results_prefix__

        // Train
        BuildCoclustering
        """,
        # fmt: on
    ),
]
