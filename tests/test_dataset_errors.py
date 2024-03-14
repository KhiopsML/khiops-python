######################################################################################
# Copyright (c) 2024 Orange. All rights reserved.                                    #
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
from khiops.sklearn.tables import Dataset, FileTable, PandasTable


# Disable PEP8 variable names because of scikit-learn X,y conventions
# To capture invalid-names other than X,y run:
#   pylint --disable=all --enable=invalid-names estimators.py
# pylint: disable=invalid-name
class AnotherType(object):
    """A placeholder class that is not of any basic type to test TypeError's"""

    pass


class DatasetSpecErrorsTests(unittest.TestCase):
    """Test the output message when the input data contains errors

    Each test covers an egde-case for the initialization of Dataset/DatasetTable and
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
    def create_fixture_dataset_spec(
        self, data_type="df", multitable=True, schema="snowflake", output_dir=None
    ):
        """Creates a fixture dataset specification

        Parameters
        ----------

        data_type : ["df", "file"], default "df"
            The desired type of the dataset tables.
        multitable : bool, default ``True``
            Whether the dataset is multitable or not.
        schema : ["snowflake", "star"], default "snowflake"
            The type of multi-table schema.
        output_dir: str, optional
            The output directory for file based datasets. Mandatory if ``data_type ==
            "file"``.
        """
        if not multitable:
            if data_type == "df":
                reference_table = self.create_monotable_dataframe()
                features = reference_table.drop(["class"], axis=1)
                dataset_spec = {
                    "main_table": "Reviews",
                    "tables": {"Reviews": (features, "User_ID")},
                }
                label = reference_table["class"]
            else:
                assert data_type == "file"
                assert isinstance(output_dir, str)
                reference_table_path = os.path.join(output_dir, "Reviews.csv")
                self.create_monotable_data_file(reference_table_path)
                dataset_spec = {
                    "main_table": "Reviews",
                    "tables": {"Reviews": (reference_table_path, "User_ID")},
                    "format": ("\t", True),
                }
                label = "class"

        elif schema == "star":
            if data_type == "df":
                (
                    reference_main_table,
                    reference_secondary_table,
                ) = self.create_multitable_star_dataframes()
                features_reference_main_table = reference_main_table.drop(
                    "class", axis=1
                )
                dataset_spec = {
                    "main_table": "id_class",
                    "tables": {
                        "id_class": (features_reference_main_table, "User_ID"),
                        "logs": (reference_secondary_table, "User_ID"),
                    },
                }
                label = reference_main_table["class"]
            else:
                assert data_type == "file"
                assert isinstance(output_dir, str)
                reference_main_table_path = os.path.join(output_dir, "id_class.csv")
                reference_secondary_table_path = os.path.join(output_dir, "logs.csv")
                self.create_multitable_star_data_files(
                    reference_main_table_path, reference_secondary_table_path
                )
                dataset_spec = {
                    "main_table": "id_class",
                    "tables": {
                        "id_class": (reference_main_table_path, "User_ID"),
                        "logs": (reference_secondary_table_path, "User_ID"),
                    },
                    "format": ("\t", True),
                }
                label = "class"

        else:
            assert schema == "snowflake"
            if data_type == "df":
                (
                    reference_main_table,
                    reference_secondary_table_1,
                    reference_secondary_table_2,
                    reference_tertiary_table,
                    reference_quaternary_table,
                ) = self.create_multitable_snowflake_dataframes()

                features_reference_main_table = reference_main_table.drop(
                    "class", axis=1
                )
                dataset_spec = {
                    "main_table": "A",
                    "tables": {
                        "D": (
                            reference_tertiary_table,
                            ["User_ID", "VAR_1", "VAR_2"],
                        ),
                        "B": (reference_secondary_table_1, ["User_ID", "VAR_1"]),
                        "E": (
                            reference_quaternary_table,
                            ["User_ID", "VAR_1", "VAR_2", "VAR_3"],
                        ),
                        "C": (reference_secondary_table_2, "User_ID"),
                        "A": (features_reference_main_table, "User_ID"),
                    },
                    "relations": [
                        ("B", "D", False),
                        ("A", "C", True),
                        ("D", "E", False),
                        ("A", "B"),
                    ],
                }
                label = reference_main_table["class"]
            else:
                assert data_type == "file"
                assert isinstance(output_dir, str)
                reference_main_table_path = os.path.join(output_dir, "A.csv")
                reference_secondary_table_path_1 = os.path.join(output_dir, "B.csv")
                reference_secondary_table_path_2 = os.path.join(output_dir, "C.csv")
                reference_tertiary_table_path = os.path.join(output_dir, "D.csv")
                reference_quaternary_table_path = os.path.join(output_dir, "E.csv")

                self.create_multitable_snowflake_data_files(
                    reference_main_table_path,
                    reference_secondary_table_path_1,
                    reference_secondary_table_path_2,
                    reference_tertiary_table_path,
                    reference_quaternary_table_path,
                )
                dataset_spec = {
                    "main_table": "A",
                    "tables": {
                        "B": (
                            reference_secondary_table_path_1,
                            ["User_ID", "VAR_1"],
                        ),
                        "E": (
                            reference_quaternary_table_path,
                            ["User_ID", "VAR_1", "VAR_2", "VAR_3"],
                        ),
                        "C": (
                            reference_secondary_table_path_2,
                            "User_ID",
                        ),
                        "A": (reference_main_table_path, "User_ID"),
                        "D": (
                            reference_tertiary_table_path,
                            ["User_ID", "VAR_1", "VAR_2"],
                        ),
                    },
                    "relations": [
                        ("B", "D"),
                        ("A", "B", False),
                        ("D", "E", False),
                        ("A", "C", True),
                    ],
                    "format": ("\t", True),
                }
                label = "class"

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

    def create_multitable_snowflake_data_files(
        self,
        main_table_path,
        secondary_table_path_1,
        secondary_table_path_2,
        tertiary_table_path,
        quaternary_table_path,
    ):
        (
            main_table,
            secondary_table_1,
            secondary_table_2,
            tertiary_table,
            quaternary_table,
        ) = self.create_multitable_snowflake_dataframes()
        main_table.to_csv(main_table_path, sep="\t", index=False)
        secondary_table_1.to_csv(secondary_table_path_1, sep="\t", index=False)
        secondary_table_2.to_csv(secondary_table_path_2, sep="\t", index=False)
        tertiary_table.to_csv(tertiary_table_path, sep="\t", index=False)
        quaternary_table.to_csv(quaternary_table_path, sep="\t", index=False)

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

    def test_x_must_be_df_or_tuple_or_sequence_or_mapping(self):
        """Test that `.Dataset` init raises TypeError when X has a wrong type"""
        bad_spec = AnotherType()
        y = "class"
        expected_msg = type_error_message(
            "X", bad_spec, "array-like", tuple, Sequence, Mapping
        )
        self.assert_dataset_fails(bad_spec, y, TypeError, expected_msg)

    def test_y_type_must_be_str_or_array_like_1d(self):
        """Test that `.Dataset` init raises TypeError when y has a wrong type"""
        # Test when X is a tuple: expects str
        table_path = os.path.join(self.output_dir, "Reviews.csv")
        dataframe = self.create_monotable_dataframe()
        dataframe.to_csv(table_path, sep="\t", index=False)
        tuple_spec = (table_path, "\t")
        bad_y = dataframe["class"]
        expected_msg = type_error_message("y", bad_y, str)
        self.assert_dataset_fails(tuple_spec, bad_y, TypeError, expected_msg)

        # Test when X is a dataframe: expects array-like
        bad_y = AnotherType()
        expected_msg = type_error_message("y", bad_y, "array-like")
        self.assert_dataset_fails(dataframe, bad_y, TypeError, expected_msg)

    #########################
    # Tests for X dict spec #
    #########################

    def test_dict_spec_relations_must_be_list_like(self):
        """Test Dataset raising TypeError when dict spec "relations" is a dict-like"""
        bad_spec, y = self.create_fixture_dataset_spec()
        bad_spec["relations"] = AnotherType()
        expected_msg = type_error_message(
            "Relations at X['tables']['relations']",
            bad_spec["relations"],
            "list-like",
        )
        self.assert_dataset_fails(bad_spec, y, TypeError, expected_msg)

    def test_dict_spec_relations_must_be_tuple(self):
        """Test Dataset raising TypeError when a relation is not a tuple"""
        bad_spec, y = self.create_fixture_dataset_spec()
        bad_spec["relations"][0] = AnotherType()
        expected_msg = type_error_message("Relation", bad_spec["relations"][0], "tuple")
        self.assert_dataset_fails(bad_spec, y, TypeError, expected_msg)

    def test_dict_spec_relations_must_be_of_size_2_or_3(self):
        """Test Dataset raising ValueError when a relation is not of size 2 or 3"""
        bad_spec, y = self.create_fixture_dataset_spec()
        for size in [0, 1, 4, 5]:
            bad_spec["relations"][0] = tuple((f"Table{i}" for i in range(size)))
            expected_msg = f"A relation must be of size 2 or 3, not {size}"
            with self.subTest(tuple_size=size):
                self.assert_dataset_fails(bad_spec, y, ValueError, expected_msg)

    def test_dict_spec_table_relation_must_be_str(self):
        """Test Dataset raising TypeError when a relation table is not a str"""
        # Test the error in the left table
        bad_spec, y = self.create_fixture_dataset_spec()
        bad_spec["relations"][0] = (AnotherType(), "BTable")
        expected_msg = type_error_message(
            "Table of a relation", bad_spec["relations"][0][0], str
        )
        self.assert_dataset_fails(bad_spec, y, TypeError, expected_msg)

        # Test the error in the right table
        bad_spec["relations"][0] = ("ATable", AnotherType())
        expected_msg = type_error_message(
            "Table of a relation", bad_spec["relations"][0][1], str
        )
        self.assert_dataset_fails(bad_spec, y, TypeError, expected_msg)

    def test_dict_spec_entiy_flag_relation_must_be_bool(self):
        """Test Dataset raising TypeError when the entity flag is not boolean"""
        bad_spec, y = self.create_fixture_dataset_spec()
        bad_spec["relations"][0] = ("B", "D", AnotherType())
        expected_msg = type_error_message(
            "1-1 flag for relation (B, D)", bad_spec["relations"][0][2], bool
        )
        self.assert_dataset_fails(bad_spec, y, TypeError, expected_msg)

    def test_dict_spec_relation_tables_must_not_be_the_same(self):
        """Test Dataset raising TypeError when tables of a relation are the same"""
        bad_spec, y = self.create_fixture_dataset_spec()
        bad_spec["relations"][0] = ("Table", "Table")
        expected_msg = (
            "Tables in relation '(Table, Table)' are the same. "
            "They must be different."
        )
        self.assert_dataset_fails(bad_spec, y, ValueError, expected_msg)

    def test_dict_spec_relation_table_must_be_in_table_list(self):
        """Test Dataset raising ValueError when a rel. table is not in the table list"""
        bad_spec, y = self.create_fixture_dataset_spec()
        bad_spec["relations"][0] = ("NonExistentTable", "D")
        expected_msg = (
            "X['tables'] does not contain a table named 'NonExistentTable'. "
            "All tables in X['relations'] must be declared in X['tables']"
        )
        self.assert_dataset_fails(bad_spec, y, ValueError, expected_msg)

    def test_dict_spec_relation_must_appear_once(self):
        """Test Dataset raising ValueError if a relation appears more than once"""
        bad_spec, y = self.create_fixture_dataset_spec()
        bad_spec["relations"].append(("B", "D"))
        expected_msg = (
            "Relation '(B, D)' occurs '2' times. Each relation must be unique."
        )
        self.assert_dataset_fails(bad_spec, y, ValueError, expected_msg)

    def test_dict_spec_key_tables_must_be_present(self):
        """Test Dataset raising ValueError if the 'tables' key is missing"""
        bad_spec, y = self.create_fixture_dataset_spec()
        del bad_spec["tables"]
        expected_msg = "Mandatory key 'tables' missing from dict 'X'"
        self.assert_dataset_fails(bad_spec, y, ValueError, expected_msg)

    def test_dict_spec_key_tables_must_be_mapping(self):
        """Test Dataset raising TypeError if the 'tables' key is not a mapping"""
        bad_spec, y = self.create_fixture_dataset_spec()
        bad_spec["tables"] = AnotherType()
        expected_msg = type_error_message("X['tables']", bad_spec["tables"], Mapping)
        self.assert_dataset_fails(bad_spec, y, TypeError, expected_msg)

    def test_dict_spec_table_list_cannot_be_empty(self):
        """Test Dataset raising ValueError if the 'tables' key is empty"""
        bad_spec, y = self.create_fixture_dataset_spec()
        bad_spec["tables"] = {}
        expected_msg = "X['tables'] cannot be empty"
        self.assert_dataset_fails(bad_spec, y, ValueError, expected_msg)

    def test_dict_spec_table_input_type_must_be_a_tuple(self):
        """Test Dataset raising TypeError when a relation tuple is a list"""
        bad_spec, y = self.create_fixture_dataset_spec()
        bad_spec["tables"]["D"] = list(bad_spec["tables"]["D"])
        expected_msg = type_error_message(
            "Table input at X['tables']['D']", bad_spec["tables"]["D"], tuple
        )
        self.assert_dataset_fails(bad_spec, y, TypeError, expected_msg)

    def test_dict_spec_table_input_tuple_must_have_size_2(self):
        """Test Dataset raising ValueError when a table entry is a tuple of size != 2"""
        bad_spec, y = self.create_fixture_dataset_spec()
        bad_spec["tables"]["D"] = (*bad_spec["tables"]["D"], "AnotherT", "YetAnotherT")
        expected_msg = "Table input tuple at X['tables']['D'] must have size 2 not 4"
        self.assert_dataset_fails(bad_spec, y, ValueError, expected_msg)

    def test_dict_spec_source_table_type_must_be_array_like_or_str(self):
        """Test Dataset raising TypeError when a table entry is not str nor DataFrame"""
        bad_spec, y = self.create_fixture_dataset_spec()
        bad_spec["tables"]["D"] = (AnotherType(), bad_spec["tables"]["D"][-1])
        expected_msg = type_error_message(
            "Table source at X['tables']['D']",
            bad_spec["tables"]["D"][0],
            "array-like",
            str,
        )
        self.assert_dataset_fails(bad_spec, y, TypeError, expected_msg)

    def test_dict_spec_table_key_must_be_str_or_sequence(self):
        """Test Dataset raising TypeError when a table's key is not str or Sequence"""
        bad_spec, y = self.create_fixture_dataset_spec()
        bad_spec["tables"]["D"] = (bad_spec["tables"]["D"][0], AnotherType())
        expected_msg = type_error_message(
            "Table key at X['tables']['D']",
            bad_spec["tables"]["D"][1],
            str,
            Sequence,
        )
        self.assert_dataset_fails(bad_spec, y, TypeError, expected_msg)

    def test_dict_spec_table_key_column_type_must_be_str(self):
        """Test Dataset raising TypeError when a table key contains a non-string"""
        bad_spec, y = self.create_fixture_dataset_spec()
        dataframe, _ = bad_spec["tables"]["D"]
        bad_key = ["User_ID", AnotherType(), "VAR_2"]
        bad_spec["tables"]["D"] = (dataframe, bad_key)
        expected_msg = type_error_message(
            "Column name of table key at X['tables']['D']", bad_key[1], str
        )
        self.assert_dataset_fails(bad_spec, y, TypeError, expected_msg)

    def test_dict_spec_main_table_must_be_specified_for_multitable_datasets(self):
        """Test Dataset raising ValueError if 'main_table' is not a key in a MT spec"""
        bad_spec, y = self.create_fixture_dataset_spec()
        del bad_spec["main_table"]
        expected_msg = "'main_table' must be specified for multi-table datasets"
        self.assert_dataset_fails(bad_spec, y, ValueError, expected_msg)

    def test_dict_spec_main_table_must_be_str(self):
        """Test Dataset raising ValueError when 'main_table' is not a str"""
        bad_spec, y = self.create_fixture_dataset_spec()
        bad_spec["main_table"] = 1
        expected_msg = type_error_message(
            "X['main_table']", bad_spec["main_table"], str
        )
        self.assert_dataset_fails(bad_spec, y, TypeError, expected_msg)

    def test_dict_spec_main_table_not_declared_in_tables(self):
        """Test Dataset raising ValueError if the main table is not in the table list"""
        bad_spec, y = self.create_fixture_dataset_spec()
        del bad_spec["tables"][bad_spec["main_table"]]
        expected_msg = "X['main_table'] (A) must be present in X['tables']"
        self.assert_dataset_fails(bad_spec, y, ValueError, expected_msg)

    def test_dic_spec_main_table_key_must_be_specified(self):
        """Test Dataset raising ValueError if a MT spec doesn't have a main table key"""
        bad_spec, y = self.create_fixture_dataset_spec()
        dataframe, _ = bad_spec["tables"][bad_spec["main_table"]]
        bad_spec["tables"][bad_spec["main_table"]] = (dataframe, None)
        expected_msg = "key of the root table is 'None'"
        self.assert_dataset_fails(bad_spec, y, ValueError, expected_msg)

    def test_dict_spec_main_table_key_must_be_non_empty_for_multitable_datasets(self):
        """Test Dataset raising ValueError if a MT spec have an empty main table key"""
        bad_spec, y = self.create_fixture_dataset_spec()
        dataframe, _ = bad_spec["tables"][bad_spec["main_table"]]
        bad_spec["tables"][bad_spec["main_table"]] = (dataframe, [])
        expected_msg = (
            "key of the root table must be non-empty for multi-table datasets"
        )
        self.assert_dataset_fails(bad_spec, y, ValueError, expected_msg)

    def test_dict_spec_secondary_table_key_must_be_specified(self):
        """Test Dataset raising ValueError if a MT spec doesn't have a sec. table key"""
        bad_spec, y = self.create_fixture_dataset_spec()
        dataframe, _ = bad_spec["tables"]["D"]
        bad_spec["tables"]["D"] = (dataframe, None)
        expected_msg = (
            "key of the secondary table 'D' is 'None': "
            "table keys must be specified in multitable datasets"
        )
        self.assert_dataset_fails(bad_spec, y, ValueError, expected_msg)

    def test_dict_spec_format_must_be_tuple(self):
        """Test Dataset raising a TypeError if the format field is not a tuple"""
        bad_spec, y = self.create_fixture_dataset_spec()
        bad_spec["format"] = AnotherType()
        expected_msg = type_error_message("X['format']", bad_spec["format"], tuple)
        self.assert_dataset_fails(bad_spec, y, TypeError, expected_msg)

    def test_dict_spec_format_tuple_1st_element_must_be_str(self):
        """Test Dataset raising a TypeError if any of the format fields are not str"""
        bad_spec, y = self.create_fixture_dataset_spec()
        bad_spec["format"] = (AnotherType(), True)
        expected_msg = type_error_message(
            "X['format'] 1st element", bad_spec["format"][0], str
        )
        self.assert_dataset_fails(bad_spec, y, TypeError, expected_msg)

    def test_dict_spec_format_tuple_2nd_element_must_be_bool(self):
        """Test Dataset raising a TypeError if any of the format fields are not bool"""
        bad_spec, y = self.create_fixture_dataset_spec(
            output_dir=self.output_dir,
            data_type="file",
        )
        bad_spec["format"] = (",", AnotherType())
        expected_msg = type_error_message(
            "X['format'] 2nd element", bad_spec["format"][1], bool
        )
        self.assert_dataset_fails(bad_spec, y, TypeError, expected_msg)

    def test_dict_spec_format_tuple_1st_element_must_be_a_single_character(self):
        """Test Dataset raising a ValueError if the format sep. is not a single char"""
        bad_spec, y = self.create_fixture_dataset_spec(
            output_dir=self.output_dir,
            data_type="file",
        )
        bad_spec["format"] = (";;", True)
        expected_msg = "Separator must be a single character. Value: ;;"
        self.assert_dataset_fails(bad_spec, y, ValueError, expected_msg)

    def test_dict_spec_y_type_must_be_series_when_x_is_df_spec(self):
        """Test Dataset raising TypeError if X a is df-dict-spec and y isn't a Series"""
        spec, _ = self.create_fixture_dataset_spec(multitable=False, schema=None)
        bad_y = AnotherType()
        expected_msg = (
            type_error_message("y", bad_y, pd.Series)
            + " (X's tables are of type pandas.DataFrame)"
        )
        self.assert_dataset_fails(spec, bad_y, TypeError, expected_msg)

    def test_dict_spec_y_must_be_str_when_x_is_file_spec(self):
        """Test Dataset raising TypeError for a file-dict-spec and y not a str"""
        spec, _ = self.create_fixture_dataset_spec(
            output_dir=self.output_dir, data_type="file"
        )
        bad_y = AnotherType()
        expected_msg = (
            type_error_message("y", bad_y, str)
            + " (X's tables are of type str [file paths])"
        )
        self.assert_dataset_fails(spec, bad_y, TypeError, expected_msg)

    def test_dict_spec_target_column_must_be_specified_to_be_accessed(self):
        """Test Dataset raising ValueError when accessing a non specified target col"""
        # Disable pointless statement because it is necessary for the test
        # pylint: disable=pointless-statement
        spec, _ = self.create_fixture_dataset_spec(
            output_dir=self.output_dir, data_type="file", multitable=False, schema=None
        )
        dataset = Dataset(spec, None)
        with self.assertRaises(ValueError) as context:
            dataset.target_column_type
        output_error_msg = str(context.exception)
        expected_error_msg = "Target column is not set"
        self.assertEqual(output_error_msg, expected_error_msg)

    def test_dict_spec_table_name_must_be_str(self):
        """Test Dataset raising TypeError when a table name is not a str"""
        spec, y = self.create_fixture_dataset_spec(multitable=False, schema=None)
        features_table = spec["tables"]["Reviews"][0]
        with self.assertRaises(TypeError) as context:
            PandasTable(
                AnotherType(),
                features_table,
                target_column=y,
            )
        output_error_msg = str(context.exception)
        expected_msg = type_error_message("name", AnotherType(), str)
        self.assertEqual(output_error_msg, expected_msg)

    def test_dict_spec_table_name_is_empty_string(self):
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
        expected_error_msg = type_error_message("key", bad_key, str, int, "list-like")
        dataset_spec, label = self.create_fixture_dataset_spec(
            multitable=False, schema=None
        )
        features_table = dataset_spec["tables"]["Reviews"][0]
        with self.assertRaises(TypeError) as context:
            PandasTable(
                "reviews",
                features_table,
                target_column=label,
                categorical_target=True,
                key=bad_key,
            )
        output_error_msg = str(context.exception)
        self.assertEqual(output_error_msg, expected_error_msg)

    def test_dict_spec_key_column_type_must_be_str_or_int(self):
        """Test Dataset raising TypeError when a key column is not of the proper type"""
        bad_key = {"not-a-str-or-int": []}
        expected_error_msg = (
            type_error_message("key[0]", bad_key, str, int) + " at table 'reviews'"
        )
        dataset_spec, label = self.create_fixture_dataset_spec(
            multitable=False, schema=None
        )
        features_table = dataset_spec["tables"]["Reviews"][0]
        with self.assertRaises(TypeError) as context:
            PandasTable(
                "reviews",
                features_table,
                target_column=label,
                categorical_target=True,
                key=[bad_key],
            )
        output_error_msg = str(context.exception)
        self.assertEqual(output_error_msg, expected_error_msg)

    ############################
    # Tests for DatasetTable's #
    ############################

    def test_pandas_table_input_type_must_be_dataframe(self):
        """Test PandasTable raising TypeError if dataframe is not a pandas.DataFrame"""
        with self.assertRaises(TypeError) as context:
            PandasTable("reviews", AnotherType())
        output_error_msg = str(context.exception)
        expected_msg = type_error_message("dataframe", AnotherType(), pd.DataFrame)
        self.assertEqual(output_error_msg, expected_msg)

    def test_pandas_table_input_table_must_not_be_empty(self):
        """Test PandasTable raising ValueError if the input dataframe is empty"""
        with self.assertRaises(ValueError) as context:
            PandasTable(
                "reviews",
                pd.DataFrame(),
                target_column="class",
            )
        output_error_msg = str(context.exception)
        expected_msg = "'dataframe' is empty"
        self.assertEqual(output_error_msg, expected_msg)

    def test_pandas_table_target_column_must_be_series(self):
        """Test PandasTable raising TypeError if the input target col. isn't a Series"""
        dataset_spec, _ = self.create_fixture_dataset_spec(
            multitable=False, schema=None
        )
        features_table = dataset_spec["tables"]["Reviews"][0]
        with self.assertRaises(TypeError) as context:
            PandasTable(
                "reviews",
                features_table,
                target_column=AnotherType(),
            )
        output_error_msg = str(context.exception)
        expected_msg = type_error_message("target_column", AnotherType(), "array-like")
        self.assertEqual(output_error_msg, expected_msg)

    def test_pandas_table_fails_if_target_column_is_already_in_the_features(self):
        """Test in-memory table failing when the target is already in the features"""
        dataset_spec, _ = self.create_fixture_dataset_spec(
            multitable=False, schema=None
        )
        features_table = dataset_spec["tables"]["Reviews"][0]
        y = features_table["Recommended IND"]
        with self.assertRaises(ValueError) as context:
            PandasTable(
                "reviews",
                features_table,
                target_column=y,
            )
        output_error_msg = str(context.exception)
        expected_msg = (
            "Target series name 'Recommended IND' is already present in"
            " dataframe : ['User_ID', 'Age', 'Clothing ID', 'Date', 'New',"
            " 'Title', 'Recommended IND', 'Positive Feedback average']"
        )
        self.assertEqual(output_error_msg, expected_msg)

    def test_pandas_table_column_ids_must_all_be_int_or_str(self):
        """Test that in-memory dataset all columns ids must be int or str"""
        spec, y = self.create_fixture_dataset_spec(multitable=False, schema=None)
        features_table = spec["tables"]["Reviews"][0]
        features_table.rename(columns={"User_ID": 1}, inplace=True)
        with self.assertRaises(TypeError) as context:
            PandasTable("reviews", features_table, target_column=y)
        output_error_msg = str(context.exception)
        expected_msg = (
            "Dataframe column ids must be either all integers or all "
            "strings. Column id at index 0 ('1') is of type 'int'"
        )
        self.assertEqual(output_error_msg, expected_msg)

    def test_file_table_fails_with_non_existent_table_file(self):
        """Test FileTable failing when it is created with a non-existent file"""
        with self.assertRaises(ValueError) as context:
            FileTable("reviews", "Review.csv", target_column_id="class")
        output_error_msg = str(context.exception)
        expected_msg = "Non-existent data table file: Review.csv"
        self.assertEqual(output_error_msg, expected_msg)

    def test_file_table_fails_with_empty_table_file(self):
        """Test FileTable failing if it is created with an empty table"""
        table_path = os.path.join(self.output_dir, "empty_table.csv")
        table = pd.DataFrame(columns=["a", "b"])
        table.to_csv(table_path, sep="\t", index=False)
        with self.assertRaises(ValueError) as context:
            FileTable("empty_table", table_path, target_column_id="class")
        output_error_msg = str(context.exception)
        expected_msg_prefix = "Empty data table file"
        self.assertIn(expected_msg_prefix, output_error_msg)

    def test_file_table_internal_file_creation_fails_on_an_existing_path(self):
        """Test FileTable failing to create an internal file to a existing path"""
        spec, _ = self.create_fixture_dataset_spec(
            output_dir=self.output_dir, data_type="file", multitable=False, schema=None
        )
        old_file_path = spec["tables"]["Reviews"][0]
        new_file_path = old_file_path.replace("Reviews.csv", "copy_Reviews.txt")
        os.rename(old_file_path, new_file_path)
        file_table = FileTable(
            "Reviews",
            new_file_path,
            target_column_id="class",
            key="User_ID",
        )
        with self.assertRaises(ValueError) as context:
            file_table.create_table_file_for_khiops(self.output_dir, sort=False)
        output_error_msg = str(context.exception)
        expected_msg_prefix = "Cannot overwrite this table's path"
        self.assertIn(expected_msg_prefix, output_error_msg)

    ####################################################
    # Tests for X tuple and sequence spec (deprecated) #
    ####################################################

    def test_tuple_spec_must_have_length_2(self):
        """Test that `.Dataset` raises `ValueError` when the tuple is not of size 2"""
        # Test pour la tuple de taille 3
        bad_spec = ("a", "b", "\t")
        y = "class"
        self.assert_dataset_fails(
            bad_spec, y, ValueError, "'X' tuple input must have length 2 not 3"
        )

        # Test pour une tuple de taille 1
        bad_spec = ("a",)
        self.assert_dataset_fails(
            bad_spec, y, ValueError, "'X' tuple input must have length 2 not 1"
        )

    def test_tuple_spec_elements_must_be_str(self):
        """Test Dataset raising TypeError when the tuple spec has non-strings"""
        # Test for the first element
        bad_spec = (AnotherType(), "/some/path")
        y = "class"
        expected_msg = type_error_message("X[0]", bad_spec[0], str)
        self.assert_dataset_fails(bad_spec, y, TypeError, expected_msg)

        # Test for the second element
        bad_spec = ("table-name", AnotherType())
        expected_msg = type_error_message("X[1]", bad_spec[1], str)
        self.assert_dataset_fails(bad_spec, y, TypeError, expected_msg)

    def test_sequence_spec_must_be_a_non_empty(self):
        """Test that Datasets raises `ValueError` when X is an empty sequence"""
        bad_spec = []
        y = "class"
        expected_msg = "'X' must be a non-empty sequence"
        self.assert_dataset_fails(bad_spec, y, ValueError, expected_msg)

    def test_sequence_spec_must_be_str_or_df(self):
        """Test Dataset raising TypeError when it is a sequence with bad types"""
        # Test that the first element is not str or df
        bad_spec = [AnotherType(), "table_1"]
        y = "class"
        expected_msg = type_error_message("X[0]", bad_spec[0], str, pd.DataFrame)
        self.assert_dataset_fails(bad_spec, y, TypeError, expected_msg)

        # Test that the second element is not str
        bad_spec = ["table_1", AnotherType()]
        expected_msg = (
            type_error_message("X[1]", bad_spec[1], str) + " as the first table in X"
        )
        self.assert_dataset_fails(bad_spec, y, TypeError, expected_msg)
