######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""deploy_model task family"""
from khiops.core.internals import task as tm
from khiops.core.internals.types import BoolType, DictType, FloatType, StringLikeType

# Disable long lines to have readable scenarios
# pylint: disable=line-too-long
TASKS = [
    tm.KhiopsTask(
        "deploy_model",
        "khiops",
        "10.6.0-b.0",
        [
            ("dictionary_file_path", StringLikeType),
            ("dictionary_name", StringLikeType),
            ("data_table_path", StringLikeType),
            ("output_data_table_path", StringLikeType),
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
            ("output_header_line", BoolType, True),
            ("output_field_separator", StringLikeType, ""),
            (
                "output_additional_data_tables",
                DictType(StringLikeType, StringLikeType),
                None,
            ),
        ],
        [
            "dictionary_file_path",
            "data_table_path",
            "output_data_table_path",
            "additional_data_tables",
            "output_additional_data_tables",
        ],
        # fmt: off
        """
        // Dictionary file settings
        ClassManagement.OpenFile
        ClassFileName __dictionary_file_path__
        OK

        // Deploy settings
        LearningTools.TransferDatabase
        ClassName __dictionary_name__

        // Input database settings
        SourceDatabase.DatabaseSpec.Data.DatabaseFiles.List.Key
        SourceDatabase.DatabaseSpec.Data.DatabaseFiles.DataTableName __data_table_path__
        __DICT__
        __additional_data_tables__
        SourceDatabase.DatabaseSpec.Data.DatabaseFiles.List.Key
        SourceDatabase.DatabaseSpec.Data.DatabaseFiles.DataTableName
        __END_DICT__
        SourceDatabase.DatabaseSpec.Data.HeaderLineUsed __header_line__
        SourceDatabase.DatabaseSpec.Data.FieldSeparator __field_separator__
        __OPT__
        __detect_format__
        SourceDatabase.DatabaseSpec.Data.DatabaseFormatDetector.DetectFileFormat
        __END_OPT__
        SourceDatabase.DatabaseSpec.Sampling.SampleNumberPercentage __sample_percentage__
        SourceDatabase.DatabaseSpec.Sampling.SamplingMode  __sampling_mode__
        SourceDatabase.DatabaseSpec.Selection.SelectionAttribute __selection_variable__
        SourceDatabase.DatabaseSpec.Selection.SelectionValue __selection_value__

        // Output database settings
        TargetDatabase.DatabaseSpec.Data.DatabaseFiles.List.Key
        TargetDatabase.DatabaseSpec.Data.DatabaseFiles.DataTableName __output_data_table_path__
        __DICT__
        __output_additional_data_tables__
        TargetDatabase.DatabaseSpec.Data.DatabaseFiles.List.Key
        TargetDatabase.DatabaseSpec.Data.DatabaseFiles.DataTableName
        __END_DICT__
        TargetDatabase.DatabaseSpec.Data.HeaderLineUsed __output_header_line__
        TargetDatabase.DatabaseSpec.Data.FieldSeparator __output_field_separator__

        // Transfer
        TransferDatabase
        Exit
        """,
        # fmt: on
    ),
]
