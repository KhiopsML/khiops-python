######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Test the output message when the input data contains errors"""
import os
import shutil
import unittest
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from khiops.core.internals.common import type_error_message
from khiops.sklearn.dataset import Dataset, PandasTable


# Disable PEP8 variable names because of scikit-learn X,y conventions
# To capture invalid-names other than X,y run:
#   pylint --disable=all --enable=invalid-names test_dataset_errors.py
# pylint: disable=invalid-name
class AnotherType(object):
    """A placeholder class that is not of any basic type to test TypeError's"""

    pass


class DatasetSpecErrorsTests(unittest.TestCase):
    """Test the output message when the input data contains errors

    Each test covers an edge-case for the initialization of Dataset/DatasetTable and
    checks:

    - that either `TypeError` or `ValueError` is raised and
    - the error's message matches the expected one.
    """

    def setUp(self):
        """Set-up test-specific output directory"""
        self.output_dir = os.path.join("resources", "tmp", self._testMethodName)
        os.makedirs(self.output_dir, exist_ok=True)

    def tearDown(self):
        """Clean-up test-specific output directory"""
        shutil.rmtree(self.output_dir, ignore_errors=True)
        del self.output_dir

    ####################
    # Helper functions #
    ####################
    def create_fixture_dataset_spec(self, multitable=True, schema="snowflake"):
        """Creates a fixture dataset specification

        Parameters
        ----------

        multitable : bool, default ``True``
            Whether the dataset is multitable or not.
        schema : ["snowflake", "star"], default "snowflake"
            The type of multi-table schema.
        """
        if not multitable:
            reference_table = self.create_monotable_dataframe()
            features = reference_table.drop(["class"], axis=1)
            dataset_spec = {
                "main_table": (features, ["User_ID"]),
            }
            label = reference_table["class"]

        elif schema == "star":
            (
                reference_main_table,
                reference_secondary_table,
            ) = self.create_multitable_star_dataframes()
            features_reference_main_table = reference_main_table.drop("class", axis=1)
            dataset_spec = {
                "main_table": (features_reference_main_table, ["User_ID"]),
                "additional_data_tables": {
                    "logs": (reference_secondary_table, ["User_ID"]),
                },
            }
            label = reference_main_table["class"]
        else:
            assert schema == "snowflake"
            (
                reference_main_table,
                reference_secondary_table_1,
                reference_secondary_table_2,
                reference_tertiary_table,
                reference_quaternary_table,
            ) = self.create_multitable_snowflake_dataframes()

            features_reference_main_table = reference_main_table.drop("class", axis=1)
            dataset_spec = {
                "main_table": (features_reference_main_table, ["User_ID"]),
                "additional_data_tables": {
                    "B/D": (
                        reference_tertiary_table,
                        ["User_ID", "VAR_1", "VAR_2"],
                        False,
                    ),
                    "B": (reference_secondary_table_1, ["User_ID", "VAR_1"]),
                    "B/D/E": (
                        reference_quaternary_table,
                        ["User_ID", "VAR_1", "VAR_2", "VAR_3"],
                        False,
                    ),
                    "C": (reference_secondary_table_2, ["User_ID"], True),
                },
            }
            label = reference_main_table["class"]

        return dataset_spec, label

    def create_monotable_dataframe(self):
        data = {
            "User_ID": [
                "60B2Xk_3Fw",
                "J94geVHf_-",
                "jsPsQUdVAL",
                "tSSBwAcIvw",
                "-I-UlX4n-B",
                "4TQsd3FX7i",
                "7w824zHOgN",
                "Cm6fu01r99",
                "zbbZRgbqar",
                "WfkfYVhQFy",
            ],
            "Age": [33, 34, 60, 50, 47, 49, 39, 39, 24, 34],
            "Clothing ID": [
                767,
                1080,
                1077,
                1049,
                847,
                1080,
                858,
                858,
                1077,
                1077,
            ],
            "Date": pd.to_datetime(
                [
                    "2019-03-22",
                    "2019-03-23",
                    "2019-03-24",
                    "2019-03-25",
                    "2019-03-26",
                    "2019-03-27",
                    "2019-03-28",
                    "2019-03-29",
                    "2019-03-30",
                    "2019-03-31",
                ]
            ),
            "New": [
                True,
                False,
                True,
                False,
                False,
                True,
                True,
                True,
                False,
                False,
            ],
            "Title": [
                "Awesome",
                "Very lovely",
                "Some major design flaws",
                "My favorite buy!",
                "Flattering shirt",
                "Not for the very petite",
                "Cagrcoal shimmer fun",
                "Shimmer, surprisingly goes with lots",
                "Flattering",
                "Such a fun dress!",
            ],
            "Recommended IND": [1, 1, 0, 1, 1, 0, 1, 1, 1, 1],
            "Positive Feedback average": [0, 4.3, 0, 0.5, 6, 4, 3.6, 4, 0, 0],
            "class": [
                "Intimates",
                "Dresses",
                "Dresses",
                "Pants",
                "Blouses",
                "Dresses",
                "Knits",
                "Knits",
                "Dresses",
                "Dresses",
            ],
        }
        dataset = pd.DataFrame(data)
        return dataset

    def create_monotable_data_file(self, table_path):
        dataframe = self.create_monotable_dataframe()
        dataframe.to_csv(table_path, sep="\t", index=False)

    def create_multitable_star_dataframes(self):
        # Create the main table
        main_table_data = {
            "User_ID": [
                "60B2Xk_3Fw",
                "J94geVHf_-",
                "jsPsQUdVAL",
                "tSSBwAcIvw",
                "-I-UlX4n-B",
                "4TQsd3FX7i",
                "7w824zHOgN",
                "Cm6fu01r99",
                "zbbZRgbqar",
                "WfkfYVhQFy",
            ],
            "class": np.random.choice([0, 1], 10).astype("int64"),
        }
        main_table = pd.DataFrame(main_table_data)

        # Create the secondary table
        secondary_table_data = {
            "User_ID": np.random.choice(main_table["User_ID"], 20),
            "VAR_1": np.random.choice(["a", "b", "c", "d"], 20),
            "VAR_2": np.random.randint(low=1, high=20, size=20).astype("int64"),
            "VAR_3": np.random.choice([1, 0], 20).astype("int64"),
            "VAR_4": np.round(np.random.rand(1, 20)[0].tolist(), 2),
        }
        secondary_table = pd.DataFrame(secondary_table_data)

        return main_table, secondary_table

    def create_multitable_star_data_files(self, main_table_path, secondary_table_path):
        main_table, secondary_table = self.create_multitable_star_dataframes()
        main_table.to_csv(main_table_path, sep="\t", index=False)
        secondary_table.to_csv(secondary_table_path, sep="\t", index=False)

    def create_multitable_snowflake_dataframes(self):
        main_table_data = {
            "User_ID": [
                "60B2Xk_3Fw",
                "J94geVHf_-",
                "jsPsQUdVAL",
                "tSSBwAcIvw",
                "-I-UlX4n-B",
                "4TQsd3FX7i",
                "7w824zHOgN",
                "Cm6fu01r99",
                "zbbZRgbqar",
                "WfkfYVhQFy",
            ],
            "class": np.random.choice([0, 1], 10).astype("int64"),
        }
        main_table = pd.DataFrame(main_table_data)

        secondary_table_data_1 = {
            "User_ID": np.random.choice(main_table["User_ID"], 20),
            "VAR_1": np.random.choice(["a", "b", "c", "d"], 20),
            "VAR_2": np.random.randint(low=1, high=20, size=20).astype("int64"),
            "VAR_3": np.random.choice([1, 0], 20).astype("int64"),
            "VAR_4": np.round(np.random.rand(20).tolist(), 2),
        }
        secondary_table_1 = pd.DataFrame(secondary_table_data_1)

        secondary_table_data_2 = {
            "User_ID": np.random.choice(
                main_table["User_ID"], len(main_table), replace=False
            ),
            "VAR_1": np.random.choice(["W", "X", "Y", "Z"], len(main_table)),
            "VAR_2": np.random.randint(low=5, high=100, size=len(main_table)).astype(
                "int64"
            ),
            "VAR_3": np.random.choice([1, 0], len(main_table)).astype("int64"),
            "VAR_4": np.round(np.random.rand(len(main_table)).tolist(), 2),
        }
        secondary_table_2 = pd.DataFrame(secondary_table_data_2)

        tertiary_table_data = {
            "User_ID": np.random.choice(main_table["User_ID"], 100),
            "VAR_1": np.random.choice(["a", "b", "c", "d"], 100),
            "VAR_2": np.random.choice(["e", "f", "g", "h"], 100),
            "VAR_3": np.round(np.random.rand(100).tolist(), 2),
        }
        tertiary_table = pd.DataFrame(tertiary_table_data)

        quaternary_table_data = {
            "User_ID": np.random.choice(main_table["User_ID"], 50),
            "VAR_1": np.random.choice(["a", "b", "c", "d"], 50),
            "VAR_2": np.random.choice(["e", "f", "g", "h"], 50),
            "VAR_3": np.random.choice(["e", "f", "g", "h"], 50),
            "VAR_4": np.random.choice(["AB", "AC", "AR", "BD"], 50),
        }
        quaternary_table = pd.DataFrame(quaternary_table_data)

        return (
            main_table,
            secondary_table_1,
            secondary_table_2,
            tertiary_table,
            quaternary_table,
        )

    def assert_dataset_fails(
        self, dataset_spec, y, expected_exception_type, expected_msg
    ):
        """Asserts that a Dataset initialization fails with a given error and message"""
        with self.assertRaises(expected_exception_type) as context:
            Dataset(dataset_spec, y=y)
        self.assertEqual(str(context.exception), expected_msg)

    ####################
    # Basic X, y tests #
    ####################

    def test_x_must_be_df_or_sequence_or_mapping(self):
        """Test that `.Dataset` init raises TypeError when X has a wrong type"""
        bad_spec = AnotherType()
        y = "class"
        expected_msg = type_error_message(
            "X", bad_spec, "array-like", Mapping, Sequence
        )
        self.assert_dataset_fails(bad_spec, y, TypeError, expected_msg)

    def test_y_type_must_be_str_or_array_like_1d(self):
        """Test that `.Dataset` init raises TypeError when y has a wrong type"""
        # Test when X is a dataframe: expects array-like
        dataframe = self.create_monotable_dataframe()
        bad_y = "TargetColumn"
        expected_msg = (
            type_error_message("y", bad_y, "array-like")
            + " (X's tables are of type pandas.DataFrame)"
        )
        self.assert_dataset_fails(dataframe, bad_y, TypeError, expected_msg)

    def test_df_dataset_fails_if_target_column_is_already_in_the_features(self):
        """Test in-memory table failing when the target is already in the features"""
        spec, _ = self.create_fixture_dataset_spec(multitable=False, schema=None)
        features_table = spec["main_table"][0]
        bad_y = features_table["Recommended IND"]
        with self.assertRaises(ValueError) as context:
            Dataset(spec, bad_y)
        output_error_msg = str(context.exception)
        expected_msg_prefix = (
            "Target column name 'Recommended IND' is already present in the main table."
        )
        self.assertIn(expected_msg_prefix, output_error_msg)

    #####################################
    # Tests for dictionary dataset spec #
    #####################################

    def test_dict_spec_key_main_table_must_be_present(self):
        """Test Dataset raising ValueError if the 'tables' key is missing"""
        bad_spec, y = self.create_fixture_dataset_spec()
        del bad_spec["main_table"]
        expected_msg = "'main_table' entry missing from dataset dict spec"
        self.assert_dataset_fails(bad_spec, y, ValueError, expected_msg)

    def test_dict_spec_main_table_input_type_must_be_a_tuple(self):
        """Test Dataset raising TypeError when the main table spec is a list"""
        bad_spec, y = self.create_fixture_dataset_spec()
        bad_spec["main_table"] = list(bad_spec["main_table"])
        expected_msg = type_error_message(
            "'main_table' entry", bad_spec["main_table"], tuple
        )
        self.assert_dataset_fails(bad_spec, y, TypeError, expected_msg)

    def test_dict_spec_source_table_type_must_be_adequate(self):
        """Test Dataset raising TypeError when a table entry is not str nor DataFrame"""
        bad_spec, y = self.create_fixture_dataset_spec()
        bad_spec["additional_data_tables"]["B/D"] = (
            AnotherType(),
            bad_spec["additional_data_tables"]["B/D"][-1],
        )
        expected_msg = type_error_message(
            "Source of table at data path 'B/D'",
            bad_spec["additional_data_tables"]["B/D"][0],
            "array-like",
            "scipy.sparse.spmatrix",
            str,
        )
        self.assert_dataset_fails(bad_spec, y, TypeError, expected_msg)

    def test_dict_spec_table_key_must_sequence(self):
        """Test Dataset raising TypeError when a table's key is not a Sequence"""
        bad_spec, y = self.create_fixture_dataset_spec()
        bad_spec["additional_data_tables"]["B/D"] = (
            bad_spec["additional_data_tables"]["B/D"][0],
            AnotherType(),
        )
        expected_msg = type_error_message(
            "'B/D' table's key",
            bad_spec["additional_data_tables"]["B/D"][1],
            Sequence,
        )
        self.assert_dataset_fails(bad_spec, y, TypeError, expected_msg)

    def test_dict_spec_table_key_column_type_must_be_str(self):
        """Test Dataset raising TypeError when a table key contains a non-string"""
        bad_spec, y = self.create_fixture_dataset_spec()
        dataframe, _, _ = bad_spec["additional_data_tables"]["B/D"]
        bad_key = ["User_ID", AnotherType(), "VAR_2"]
        bad_spec["additional_data_tables"]["B/D"] = (dataframe, bad_key)
        expected_msg = type_error_message(
            "'B/D' table's key column name", bad_key[1], str
        )
        self.assert_dataset_fails(bad_spec, y, TypeError, expected_msg)

    def test_dict_spec_main_table_must_be_specified_for_multitable_datasets(self):
        """Test Dataset raising ValueError if 'main_table' is not a key in an MT spec"""
        bad_spec, y = self.create_fixture_dataset_spec()
        del bad_spec["main_table"]
        expected_msg = "'main_table' entry missing from dataset dict spec"
        self.assert_dataset_fails(bad_spec, y, ValueError, expected_msg)

    def test_dict_spec_main_table_must_be_str(self):
        """Test Dataset raising ValueError when 'main_table' is not a tuple"""
        bad_spec, y = self.create_fixture_dataset_spec()
        bad_spec["main_table"] = 1
        expected_msg = type_error_message(
            "'main_table' entry", bad_spec["main_table"], tuple
        )
        self.assert_dataset_fails(bad_spec, y, TypeError, expected_msg)

    def test_dict_spec_main_table_key_must_be_specified(self):
        """Test Dataset raise ValueError if an MT spec doesn't have a main table key"""
        bad_spec, y = self.create_fixture_dataset_spec()
        dataframe, _ = bad_spec["main_table"]
        bad_spec["main_table"] = (dataframe, None)
        expected_msg = (
            "The key of the main table is 'None': "
            "table keys must be specified in multi-table datasets"
        )
        self.assert_dataset_fails(bad_spec, y, ValueError, expected_msg)

    def test_dict_spec_table_key_must_be_non_empty_for_multitable_datasets(self):
        """Test Dataset raising ValueError if an MT spec have an empty table key"""
        bad_spec, y = self.create_fixture_dataset_spec()
        dataframe, _ = bad_spec["main_table"]
        bad_spec["main_table"] = (dataframe, [])
        expected_msg = (
            "The key of the main table is empty: "
            "table keys must be specified in multi-table datasets"
        )
        self.assert_dataset_fails(bad_spec, y, ValueError, expected_msg)

    def test_dict_spec_secondary_table_key_must_be_specified(self):
        """Test Dataset raise ValueError if an MT spec doesn't have a sec. table key"""
        bad_spec, y = self.create_fixture_dataset_spec()
        dataframe, _, _ = bad_spec["additional_data_tables"]["B/D"]
        bad_spec["additional_data_tables"]["B/D"] = (dataframe, None)
        expected_msg = (
            "Key of secondary table at path 'B/D' is 'None': "
            "table keys must be specified in multi-table datasets"
        )
        self.assert_dataset_fails(bad_spec, y, ValueError, expected_msg)

    def test_dict_spec_y_type_must_be_series_or_df_when_x_is_df_spec(self):
        """Test Dataset raising TypeError if X a is ds-spec and y isn't array-like"""
        spec, _ = self.create_fixture_dataset_spec(multitable=False, schema=None)
        bad_y = "TargetColumnName"
        expected_msg = (
            type_error_message("y", bad_y, "array-like")
            + " (X's tables are of type pandas.DataFrame)"
        )
        self.assert_dataset_fails(spec, bad_y, TypeError, expected_msg)

    def test_pandas_table_name_must_not_be_the_empty_string(self):
        """Test Dataset raising ValueError when a table name is empty"""
        spec, _ = self.create_fixture_dataset_spec(multitable=False, schema=None)
        with self.assertRaises(ValueError) as context:
            PandasTable("", spec)
        output_error_msg = str(context.exception)
        expected_msg = "'name' cannot be empty"
        self.assertEqual(output_error_msg, expected_msg)

    def test_dict_spec_key_type_must_be_str_or_list_like(self):
        """Test Dataset raising TypeError when a key is not of the proper type"""
        bad_key = AnotherType()
        expected_error_msg = type_error_message("key", bad_key, "list-like")
        dataset_spec, _ = self.create_fixture_dataset_spec(
            multitable=False, schema=None
        )
        features_table = dataset_spec["main_table"][0]
        with self.assertRaises(TypeError) as context:
            PandasTable(name="reviews", dataframe=features_table, key=bad_key)
        output_error_msg = str(context.exception)
        self.assertEqual(output_error_msg, expected_error_msg)

    def test_dict_spec_key_column_type_must_be_str_or_int(self):
        """Test Dataset raising TypeError when a key column is not of the proper type"""
        bad_key = [AnotherType()]
        expected_error_msg = (
            type_error_message("key[0]", AnotherType(), str, int)
            + " at table 'reviews'"
        )
        dataset_spec, _ = self.create_fixture_dataset_spec(
            multitable=False, schema=None
        )
        features_table = dataset_spec["main_table"][0]
        with self.assertRaises(TypeError) as context:
            PandasTable(name="reviews", dataframe=features_table, key=bad_key)
        output_error_msg = str(context.exception)
        self.assertEqual(expected_error_msg, output_error_msg)

    def test_dict_spec_additional_data_tables_must_be_dict(self):
        """Test Dataset raising TypeError when additional_data_tables is not dict"""
        bad_spec, y = self.create_fixture_dataset_spec()
        bad_spec["additional_data_tables"] = AnotherType()
        expected_msg = type_error_message(
            "'additional_data_tables' entry",
            bad_spec["additional_data_tables"],
            Mapping,
        )
        self.assert_dataset_fails(bad_spec, y, TypeError, expected_msg)

    def test_dict_spec_additional_data_tables_item_must_be_tuple(self):
        """Test Dataset raising TypeError when a secondary table spec is not a tuple"""
        bad_spec, y = self.create_fixture_dataset_spec()
        bad_spec["additional_data_tables"]["B"] = AnotherType()
        expected_msg = type_error_message(
            "'B' table entry", bad_spec["additional_data_tables"]["B"], tuple
        )
        self.assert_dataset_fails(bad_spec, y, TypeError, expected_msg)

    def test_dict_spec_additional_data_tables_item_must_be_of_size_2_or_3(self):
        """Test Dataset raising ValueError when a secondary table spec is not of
        size 2 or 3
        """
        bad_spec, y = self.create_fixture_dataset_spec()
        for size in [0, 1, 4, 5]:
            bad_spec["additional_data_tables"]["B"] = tuple(
                (f"Table{i}" for i in range(size))
            )
            expected_msg = f"'B' table entry must have size 2 or 3, not {size}"
            with self.subTest(tuple_size=size):
                self.assert_dataset_fails(bad_spec, y, ValueError, expected_msg)

    def test_dict_spec_secondary_table_data_path_must_be_str(self):
        """Test Dataset raising TypeError when a secondary table data path is
        not a str
        """
        # Test the error in the left table
        bad_spec, y = self.create_fixture_dataset_spec()
        first_relation = bad_spec["additional_data_tables"]["B"]
        bad_spec["additional_data_tables"][AnotherType()] = first_relation
        expected_msg = type_error_message("Table path", AnotherType(), str)
        self.assert_dataset_fails(bad_spec, y, TypeError, expected_msg)

    def test_dict_spec_entity_flag_must_be_bool(self):
        """Test Dataset raising TypeError when the entity flag is not boolean"""
        bad_spec, y = self.create_fixture_dataset_spec()
        original_bad_spec = bad_spec["additional_data_tables"]["B/D"]
        bad_spec["additional_data_tables"]["B/D"] = (
            original_bad_spec[0],
            original_bad_spec[1],
            AnotherType(),
        )
        expected_msg = type_error_message(
            "Table at data path B/D 1-1 flag",
            bad_spec["additional_data_tables"]["B/D"][2],
            bool,
        )
        self.assert_dataset_fails(bad_spec, y, TypeError, expected_msg)

    ############################
    # Tests for DatasetTable's #
    ############################

    def test_pandas_table_input_type_must_be_dataframe(self):
        """Test PandasTable raising TypeError if dataframe is not a pandas.DataFrame"""
        with self.assertRaises(TypeError) as context:
            PandasTable(name="reviews", dataframe=AnotherType())
        output_error_msg = str(context.exception)
        expected_msg = type_error_message("dataframe", AnotherType(), pd.DataFrame)
        self.assertEqual(output_error_msg, expected_msg)

    def test_pandas_table_input_table_must_not_be_empty(self):
        """Test PandasTable raising ValueError if the input dataframe is empty"""
        with self.assertRaises(ValueError) as context:
            PandasTable(name="reviews", dataframe=pd.DataFrame())
        output_error_msg = str(context.exception)
        expected_msg = "'dataframe' is empty"
        self.assertEqual(output_error_msg, expected_msg)

    def test_pandas_table_column_ids_must_all_be_int_or_str(self):
        """Test that in-memory dataset all columns ids must be int or str"""
        spec, _ = self.create_fixture_dataset_spec(multitable=False, schema=None)
        features_table = spec["main_table"][0]
        features_table.rename(columns={"User_ID": 1}, inplace=True)
        with self.assertRaises(TypeError) as context:
            PandasTable(name="reviews", dataframe=features_table)
        output_error_msg = str(context.exception)
        expected_msg = (
            "Dataframe column ids must be either all integers or all "
            "strings. Column id at index 0 ('1') is of type 'int'"
        )
        self.assertEqual(output_error_msg, expected_msg)
