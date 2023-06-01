######################################################################################
# Copyright (c) 2023 Orange. All rights reserved.                                    #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Test the output message when the input data contains errors"""
import os
import shutil
import unittest

import numpy as np
import pandas as pd

from pykhiops.sklearn.tables import Dataset, FileTable, PandasTable


class PyKhiopsTriggeringErrorsTests(unittest.TestCase):
    """Test the output message when the input data contains errors

    - The following tests allow to verify that:
        - The message output by pykhiops.sklearn, when data contains a `TypeError`
        or a `ValueError`, is consistent with the expected message.
    """

    def setUp(self):
        """Set-up test-specific output directory"""
        self.output_dir = os.path.join("resources", "tmp", self._testMethodName)
        os.makedirs(self.output_dir, exist_ok=True)

    def tearDown(self):
        """Clean-up test-specific output directory"""
        shutil.rmtree(self.output_dir, ignore_errors=True)
        del self.output_dir

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
            "User_ID": np.random.choice(main_table["User_ID"], 20),
            "VAR_1": np.random.choice(["W", "X", "Y", "Z"], 20),
            "VAR_2": np.random.randint(low=5, high=100, size=20).astype("int64"),
            "VAR_3": np.random.choice([1, 0], 20).astype("int64"),
            "VAR_4": np.round(np.random.rand(20).tolist(), 2),
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

    def create_fixture_dataset_spec(self, output_dir, data_type, multitable, schema):
        if not multitable:
            if data_type == "df":
                reference_table = self.create_monotable_dataframe()
                features = reference_table.drop(["class"], axis=1)
                dataset_spec = {
                    "main_table": "Reviews",
                    "tables": {"Reviews": (features, "User_ID")},
                }
                label = reference_table["class"]
            elif data_type == "file":
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
            elif data_type == "file":
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

        else:  # schema == "snowflake":
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
                        "C": (reference_secondary_table_2, ["User_ID", "VAR_1"]),
                        "A": (features_reference_main_table, "User_ID"),
                    },
                    "relations": [
                        ("B", "D"),
                        ("A", "C"),
                        ("D", "E"),
                        ("A", "B"),
                    ],
                }
                label = reference_main_table["class"]
            elif data_type == "file":
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
                            ["User_ID", "VAR_1"],
                        ),
                        "A": (reference_main_table_path, "User_ID"),
                        "D": (
                            reference_tertiary_table_path,
                            ["User_ID", "VAR_1", "VAR_2"],
                        ),
                    },
                    "relations": [
                        ("B", "D"),
                        ("A", "B"),
                        ("D", "E"),
                        ("A", "C"),
                    ],
                    "format": ("\t", True),
                }
                label = "class"

        return dataset_spec, label

    def test_error_x_type_must_be_df_or_tuple_or_sequence_or_mapping(self):
        """test output error when X is a string

        - This test verifies that the `TypeError` output by pykhiops, when X
        is a string, is the expected one.
        """
        expected_error = (
            "'X' type must be one of 'DataFrame' or 'tuple' or "
            "'Sequence' or 'Mapping', not 'str'"
        )
        dataset_spec = "path"
        with self.assertRaises(TypeError) as context:
            Dataset(dataset_spec, "label")
        output_error = str(context.exception)
        self.assertEqual(output_error, expected_error)

    def test_error_y_type_must_be_str_when_x_type_is_tuple(self):
        """test output error when X is a tuple and y is a pd.Series

        - This test verifies that the `TypeError` output by pykhiops, when X
        is a tuple (str, str) and y is pd.Series, is equal to the expected one.
        """
        expected_error = "'y' type must be 'str', not 'Series'"
        table_path = os.path.join(self.output_dir, "Reviews.csv")
        dataframe = self.create_monotable_dataframe()
        label = dataframe["class"]
        dataframe.to_csv(table_path, sep="\t", index=False)
        tuple_input = (table_path, "\t")
        with self.assertRaises(TypeError) as context:
            Dataset(tuple_input, label)
        output_error = str(context.exception)
        self.assertEqual(output_error, expected_error)

    def test_error_y_type_must_be_series_when_x_type_is_df(self):
        """test output error when X is a df and y is a str

        - This test verifies that the `TypeError` output by pykhiops, when X is
         a dataframe and y is a string, is equal to the expected one.
        """
        expected_error = (
            "'y' type must be 'Series', not 'str' (must be coherent with "
            "X of type pandas.DataFrame)"
        )
        dataframe = pd.DataFrame()
        with self.assertRaises(TypeError) as context:
            Dataset(dataframe, "class")
        output_error = str(context.exception)
        self.assertEqual(output_error, expected_error)

    def test_error_x_tuple_must_have_length_2(self):
        """test output error when X is a tuple of size 3

        - This test verifies that the `ValueError` output by pykhiops, when X is a tuple
        of size 3, is equal to the expected one.
        """
        expected_error = "'X' tuple input must have length 2 not 3"
        tuple_input = ("a", "b", "\t")
        with self.assertRaises(ValueError) as context:
            Dataset(tuple_input, "class")
        output_error = str(context.exception)
        self.assertEqual(output_error, expected_error)

    def test_error_1st_element_of_tuple_type_must_be_str(self):
        """test output error when X is a tuple whose 1st element is an integer

        - This test verifies that the `TypeError` output by pykhiops, when X is a tuple
        whose first element is an integer, is equal to the expected one.
        """
        expected_error = "'X[0]' type must be 'str', not 'int'"
        tuple_input = (1, "a")
        with self.assertRaises(TypeError) as context:
            Dataset(tuple_input, "class")
        output_error = str(context.exception)
        self.assertEqual(output_error, expected_error)

    def test_error_2nd_element_of_tuple_type_must_be_str(self):
        """test output error when X is a tuple whose 2nd element is an integer

        - This test verifies that the `TypeError` output by pykhiops, when X is a tuple
        whose second element is an integer, is equal to the expected one.
        """
        expected_error = "'X[1]' type must be 'str', not 'int'"
        tuple_input = ("a", 1)
        with self.assertRaises(TypeError) as context:
            Dataset(tuple_input, "class")
        output_error = str(context.exception)
        self.assertEqual(output_error, expected_error)

    def test_error_x_must_be_a_non_empty_sequence(self):
        """test output error when X is an empty sequence

        - This test verifies that the `ValueError` output by pykhiops, when X is an
        empty sequence, is equal to the expected one.
        """
        expected_error = "'X' must be a non-empty sequence"
        with self.assertRaises(ValueError) as context:
            Dataset([], "class")
        output_error = str(context.exception)
        self.assertEqual(output_error, expected_error)

    def test_error_1st_element_of_sequence_type_must_be_str_or_df(self):
        """test output error when X is a sequence whose 1st element is an integer

        - This test verifies that the `TypeError` output by pykhiops, when X is a
        sequence whose 1st element is an integer, is equal to the expected one.
        """
        expected_error = "'X[0]' type must be either 'str' or 'DataFrame'," " not 'int'"
        with self.assertRaises(TypeError) as context:
            Dataset([1, "table_1"], "class")
        output_error = str(context.exception)
        self.assertEqual(output_error, expected_error)

    def test_error_2nd_element_of_sequence_type_must_be_str_or_df(self):
        """test output error when X is a sequence whose 2nd element is an integer

        - This test verifies that the `TypeError` output by pykhiops, when X is a
        sequence whose 2nd element is an integer, is equal to the expected one.
        """
        expected_error = (
            "'X[1]' type must be 'str', not 'int' as" " the first table in X"
        )
        with self.assertRaises(TypeError) as context:
            Dataset(["table_1", 1], "class")
        output_error = str(context.exception)
        self.assertEqual(output_error, expected_error)

    def test_error_relations_of_a_dict_type_must_be_list_like(self):
        """test output error when relations of a dict are of type string

        - This test verifies that the `TypeError` output by pykhiops, when relations
        at X['tables']['relations'] are of type string, is equal to the expected one.
        """
        expected_error = (
            "Relations at X['tables']['relations'] type must be "
            "'list-like', not 'str'"
        )
        dataset_spec, label = self.create_fixture_dataset_spec(
            output_dir=None, data_type="df", multitable=True, schema="snowflake"
        )
        dataset_spec["relations"] = "relations"
        with self.assertRaises(TypeError) as context:
            Dataset(dataset_spec, label)
        output_error = str(context.exception)
        self.assertEqual(output_error, expected_error)

    def test_error_relation_type_must_be_tuple(self):
        """test output error when a relation is a list of strings

        - This test verifies that the `TypeError` output by pykhiops,
        when a relation is a list of strings, is equal
        to the expected one.
        """
        expected_error = "'Relation' type must be 'tuple', not 'list'"
        dataset_spec, label = self.create_fixture_dataset_spec(
            output_dir=None, data_type="df", multitable=True, schema="snowflake"
        )
        dataset_spec["relations"][0] = ["B", "D"]
        with self.assertRaises(TypeError) as context:
            Dataset(dataset_spec, label)
        output_error = str(context.exception)
        self.assertEqual(output_error, expected_error)

    def test_error_relation_must_be_of_size_2(self):
        """test output error when a relation is a tuple of size 3

        - This test verifies that the `ValueError` output by pykhiops,
        when a relation is a tuple of 3 strings, is equal
        to the expected one.
        """
        expected_error = "A relation must be of size 2 not 3"
        dataset_spec, label = self.create_fixture_dataset_spec(
            output_dir=None, data_type="df", multitable=True, schema="snowflake"
        )
        dataset_spec["relations"][0] = ("B", "D", "E")
        with self.assertRaises(ValueError) as context:
            Dataset(dataset_spec, label)
        output_error = str(context.exception)
        self.assertEqual(output_error, expected_error)

    def test_error_table_of_a_relation_type_must_be_str(self):
        """test output error when a relation contains an element of numeric type

        - This test verifies that the `TypeError` output by pykhiops,
        when the second element of a relation is an integer, is equal
        to the expected one.
        """
        expected_error = "Table of a relation type must be 'str', not 'int'"
        dataset_spec, label = self.create_fixture_dataset_spec(
            output_dir=None, data_type="df", multitable=True, schema="snowflake"
        )
        dataset_spec["relations"][0] = ("B", 1)
        with self.assertRaises(TypeError) as context:
            Dataset(dataset_spec, label)
        output_error = str(context.exception)
        self.assertEqual(output_error, expected_error)

    def test_error_tables_in_a_relation_are_the_same(self):
        """test output error when tables of a relation are the same

        - This test verifies that the `ValueError` output by pykhiops,
        when tables of a relation are the same, is equal
        to the expected one.
        """
        expected_error = (
            "Tables in relation '('B', 'B')' are the same. " "They must be different."
        )
        dataset_spec, label = self.create_fixture_dataset_spec(
            output_dir=None, data_type="df", multitable=True, schema="snowflake"
        )
        dataset_spec["relations"][0] = ("B", "B")
        with self.assertRaises(ValueError) as context:
            Dataset(dataset_spec, label)
        output_error = str(context.exception)
        self.assertEqual(output_error, expected_error)

    def test_error_parent_table_in_relation_not_declared_in_tables(self):
        """test output error when the parent table in a relation is not declared

        - This test verifies that the `ValueError` output by pykhiops,
        when the parent table in a relation is not declared in X['tables'], is equal
        to the expected one.
        """
        expected_error = (
            "X['tables'] does not contain a table named 'X'. "
            "All tables in X['relations'] must be declared in X['tables']"
        )
        dataset_spec, label = self.create_fixture_dataset_spec(
            output_dir=None, data_type="df", multitable=True, schema="snowflake"
        )
        dataset_spec["relations"][0] = ("X", "D")
        with self.assertRaises(ValueError) as context:
            Dataset(dataset_spec, label)
        output_error = str(context.exception)
        self.assertEqual(output_error, expected_error)

    def test_error_child_table_in_relation_not_declared_in_tables(self):
        """test output error when the child table in a relation is not declared

        - This test verifies that the `ValueError` output by pykhiops,
        when the child table in a relation is not declared in X['tables'], is equal
        to the expected one.
        """
        expected_error = (
            "X['tables'] does not contain a table named 'X'. "
            "All tables in X['relations'] must be declared in X['tables']"
        )
        dataset_spec, label = self.create_fixture_dataset_spec(
            output_dir=None, data_type="df", multitable=True, schema="snowflake"
        )
        dataset_spec["relations"][0] = ("X", "D")
        with self.assertRaises(ValueError) as context:
            Dataset(dataset_spec, label)
        output_error = str(context.exception)
        self.assertEqual(output_error, expected_error)

    def test_error_a_relation_occurs_multiple_times(self):
        """test output error when a relation occurs multiple times

        - This test verifies that the `ValueError` output by pykhiops,
        when a relation occurs more than once, is equal
        to the expected one.
        """
        expected_error = (
            "Relation '('B', 'D')' occurs '2' times. Each relation " "must be unique."
        )
        dataset_spec, label = self.create_fixture_dataset_spec(
            output_dir=None, data_type="df", multitable=True, schema="snowflake"
        )
        dataset_spec["relations"][1] = ("B", "D")
        with self.assertRaises(ValueError) as context:
            Dataset(dataset_spec, label)
        output_error = str(context.exception)
        self.assertEqual(output_error, expected_error)

    def test_error_mandatory_key_tables_missing_from_dict(self):
        """test output error when 'tables' is missing from dict X

        - This test verifies that the `ValueError` output by pykhiops,
        when mandatory key 'tables' is missing from dictionary 'X', is equal
        to the expected one.
        """
        expected_error = "Mandatory key 'tables' missing from dict 'X'"
        dataset_spec, label = self.create_fixture_dataset_spec(
            output_dir=None, data_type="df", multitable=True, schema="snowflake"
        )
        del dataset_spec["tables"]
        with self.assertRaises(ValueError) as context:
            Dataset(dataset_spec, label)
        output_error = str(context.exception)
        self.assertEqual(output_error, expected_error)

    def test_error_tables_type_must_be_mapping(self):
        """test output error when tables is an integer

        - This test verifies that the `TypeError` output by pykhiops,
        when X['tables']' type is an integer, is equal
        to the expected one.
        """
        expected_error = "'X['tables']' type must be 'Mapping', not 'int'"
        dataset_spec, label = self.create_fixture_dataset_spec(
            output_dir=None, data_type="df", multitable=True, schema="snowflake"
        )
        dataset_spec["tables"] = 1
        with self.assertRaises(TypeError) as context:
            Dataset(dataset_spec, label)
        output_error = str(context.exception)
        self.assertEqual(output_error, expected_error)

    def test_error_tables_cannot_be_empty(self):
        """test output error when tables is an empty dictionary

        - This test verifies that the `ValueError` output by pykhiops,
        when X['tables']' is an empty dictionary, is equal
        to the expected one.
        """
        expected_error = "X['tables'] cannot be empty"
        dataset_spec, label = self.create_fixture_dataset_spec(
            output_dir=None, data_type="df", multitable=True, schema="snowflake"
        )
        dataset_spec["tables"] = {}
        with self.assertRaises(ValueError) as context:
            Dataset(dataset_spec, label)
        output_error = str(context.exception)
        self.assertEqual(output_error, expected_error)

    def test_error_table_input_type_must_be_a_tuple(self):
        """test output error when a table is declared as a list

        - This test verifies that the `TypeError` output by pykhiops,
        when a table input at X['tables'] is a tuple, is equal
        to the expected one.
        """
        expected_error = (
            "Table input at X['tables']['D'] type must " "be 'tuple', not 'list'"
        )
        dataset_spec, label = self.create_fixture_dataset_spec(
            output_dir=None, data_type="df", multitable=True, schema="snowflake"
        )
        dataset_spec["tables"]["D"] = list(dataset_spec["tables"]["D"])
        with self.assertRaises(TypeError) as context:
            Dataset(dataset_spec, label)
        output_error = str(context.exception)
        self.assertEqual(output_error, expected_error)

    def test_error_table_input_tuple_must_have_size_2(self):
        """test output error when a table input is a tuple of size 3

        - This test verifies that the `ValueError` output by pykhiops,
        when a table input at X['tables'] is a tuple of size 3, is equal
        to the expected one.
        """
        expected_error = (
            "Table input tuple at X['tables']['D'] must have " "size 2 not 3"
        )
        dataset_spec, label = self.create_fixture_dataset_spec(
            output_dir=None, data_type="df", multitable=True, schema="snowflake"
        )
        dataset_spec["tables"]["D"] = (*dataset_spec["tables"]["D"], "")
        with self.assertRaises(ValueError) as context:
            Dataset(dataset_spec, label)
        output_error = str(context.exception)
        self.assertEqual(output_error, expected_error)

    def test_error_source_table_type_must_be_df_or_str(self):
        """test output error when the table source is an integer

        - This test verifies that the `TypeError` output by pykhiops,
        when a table at X['tables'] is a tuple whose first element
        (table source) is an integer, is equal to the expected one.
        """
        expected_error = (
            "Table source at X['tables']['D'] type must be either"
            " 'DataFrame' or 'str', not 'int'"
        )
        dataset_spec, label = self.create_fixture_dataset_spec(
            output_dir=None, data_type="df", multitable=True, schema="snowflake"
        )
        dataset_spec["tables"]["D"] = (1, dataset_spec["tables"]["D"][-1])
        with self.assertRaises(TypeError) as context:
            Dataset(dataset_spec, label)
        output_error = str(context.exception)
        self.assertEqual(output_error, expected_error)

    def test_error_key_table_type_must_be_str_or_sequence(self):
        """test output error when the table key is an integer

        - This test verifies that the `TypeError` output by pykhiops,
        when a table at X['tables'] is a tuple whose second element (key)
         is an integer, is equal to the expected one.
        """
        expected_error = (
            "Table key at X['tables']['D'] type must be either "
            "'str' or 'Sequence', not 'int'"
        )
        dataset_spec, label = self.create_fixture_dataset_spec(
            output_dir=None, data_type="df", multitable=True, schema="snowflake"
        )
        dataset_spec["tables"]["D"] = (dataset_spec["tables"]["D"][0], 1)
        with self.assertRaises(TypeError) as context:
            Dataset(dataset_spec, label)
        output_error = str(context.exception)
        self.assertEqual(output_error, expected_error)

    def test_error_column_name_of_table_key_type_must_be_str(self):
        """test output error when a key contains a column name of type integer

        - This test verifies that the `TypeError` output by pykhiops,
        when a key of a table in X['tables'] contains a column name of
        type integer, is equal to the expected one.
        """
        expected_error = (
            "Column name of table key at X['tables']['D'] type"
            " must be 'str', not 'int'"
        )
        dataset_spec, label = self.create_fixture_dataset_spec(
            output_dir=None, data_type="df", multitable=True, schema="snowflake"
        )
        dataframe, _ = dataset_spec["tables"]["D"]
        key = ["User_ID", 1, "VAR_2"]
        dataset_spec["tables"]["D"] = (dataframe, key)
        with self.assertRaises(TypeError) as context:
            Dataset(dataset_spec, label)
        output_error = str(context.exception)
        self.assertEqual(output_error, expected_error)

    def test_error_main_table_must_be_specified_for_multitable_datasets(self):
        """test output error when data is MT and main_table is missing

        - This test verifies that the `ValueError` output by pykhiops,
        when the dataset is multi-table and the mandatory key 'main_table' is missing,
        is equal to the expected one.
        """
        expected_error = "'main_table' must be specified for multi-table datasets"
        dataset_spec, label = self.create_fixture_dataset_spec(
            output_dir=None, data_type="df", multitable=True, schema="snowflake"
        )
        del dataset_spec["main_table"]
        with self.assertRaises(ValueError) as context:
            Dataset(dataset_spec, label)
        output_error = str(context.exception)
        self.assertEqual(output_error, expected_error)

    def test_error_main_table_type_must_be_str(self):
        """test output error when X['main_table'] is an integer

        - This test verifies that the `TypeError` output by pykhiops,
        when X['main_table'] type is an integer, is equal to the expected one.
        """
        expected_error = "'X['main_table']' type must be 'str', not 'int'"
        dataset_spec, label = self.create_fixture_dataset_spec(
            output_dir=None, data_type="df", multitable=True, schema="snowflake"
        )
        dataset_spec["main_table"] = 1
        with self.assertRaises(TypeError) as context:
            Dataset(dataset_spec, label)
        output_error = str(context.exception)
        self.assertEqual(output_error, expected_error)

    def test_error_main_table_not_declared_in_tables(self):
        """test output error when X['main_table'] is not present in X['tables']

        - This test verifies that the `ValueError` output by pykhiops,
        when X['main_table'] type is an integer, is equal to the expected one.
        """
        expected_error = "X['main_table'] (A) must be present in X['tables']"
        dataset_spec, label = self.create_fixture_dataset_spec(
            output_dir=None, data_type="df", multitable=True, schema="snowflake"
        )
        del dataset_spec["tables"][dataset_spec["main_table"]]
        with self.assertRaises(ValueError) as context:
            Dataset(dataset_spec, label)
        output_error = str(context.exception)
        self.assertEqual(output_error, expected_error)

    def test_error_key_of_the_root_table_must_not_be_none(self):
        """test output error when X['main_table'] is not present in X['tables']

        - This test verifies that the `ValueError` output by pykhiops,
        when X['main_table'] is not present in the keys of X['tables'], is equal
        to the expected one.
        """
        expected_error = "key of the root table is 'None'"
        dataset_spec, label = self.create_fixture_dataset_spec(
            output_dir=None, data_type="df", multitable=True, schema="snowflake"
        )
        main_table_name = dataset_spec["main_table"]
        dataframe, _ = dataset_spec["tables"][main_table_name]
        dataset_spec["tables"][main_table_name] = (dataframe, None)
        with self.assertRaises(ValueError) as context:
            Dataset(dataset_spec, label)
        output_error = str(context.exception)
        self.assertEqual(output_error, expected_error)

    def test_error_key_of_root_table_must_be_non_empty_for_multitable_datasets(self):
        """test output error when data is MT and key of the root table is empty

        - This test verifies that the `ValueError` output by pykhiops,
        when the dataset is multitable and the key of its main table is an empty
        list, is equal to the expected one.
        """
        expected_error = (
            "key of the root table must be non-empty for" " multi-table datasets"
        )
        dataset_spec, label = self.create_fixture_dataset_spec(
            output_dir=None, data_type="df", multitable=True, schema="snowflake"
        )
        main_table_name = dataset_spec["main_table"]
        dataframe, _ = dataset_spec["tables"][main_table_name]
        dataset_spec["tables"][main_table_name] = (dataframe, [])
        with self.assertRaises(ValueError) as context:
            Dataset(dataset_spec, label)
        output_error = str(context.exception)
        self.assertEqual(output_error, expected_error)

    def test_error_key_of_a_secondary_table_must_not_be_none(self):
        """test output error when data is MT and key of a secondary table is None

        - This test verifies that the `ValueError` output by pykhiops,
        when the dataset is multitable and the key of a secondary table is None,
        is equal to the expected one.
        """
        expected_error = (
            "key of the secondary table 'D' is 'None': table"
            " keys must be specified in multitable datasets"
        )
        dataset_spec, label = self.create_fixture_dataset_spec(
            output_dir=None, data_type="df", multitable=True, schema="snowflake"
        )
        dataframe, _ = dataset_spec["tables"]["D"]
        dataset_spec["tables"]["D"] = (dataframe, None)
        with self.assertRaises(ValueError) as context:
            Dataset(dataset_spec, label)
        output_error = str(context.exception)
        self.assertEqual(output_error, expected_error)

    def test_error_format_type_must_be_tuple(self):
        """test output error when format is a string

        - This test verifies that the `TypeError` output by pykhiops,
        when format is a string, is equal to the expected one.
        """
        expected_error = "'X['format']' type must be 'tuple', not 'str'"
        dataset_spec, label = self.create_fixture_dataset_spec(
            output_dir=self.output_dir, data_type="file", multitable=False, schema=None
        )
        dataset_spec["format"] = ","
        with self.assertRaises(TypeError) as context:
            Dataset(dataset_spec, label)
        output_error = str(context.exception)
        self.assertEqual(output_error, expected_error)

    def test_error_format_tuple_1st_element_must_be_str(self):
        """test output error when the 1st element in format is an integer

        - This test verifies that the `TypeError` output by pykhiops,
        when the first element in the format (tuple) is an integer,
        is equal to the expected one.
        """
        expected_error = "X['format'] first element type must be 'str'," " not 'int'"
        dataset_spec, label = self.create_fixture_dataset_spec(
            output_dir=self.output_dir, data_type="file", multitable=False, schema=None
        )
        dataset_spec["format"] = (1, True)
        with self.assertRaises(TypeError) as context:
            Dataset(dataset_spec, label)
        output_error = str(context.exception)
        self.assertEqual(output_error, expected_error)

    def test_error_format_tuple_2nd_element_must_be_str(self):
        """test output error when the 2nd element in format is an integer

        - This test verifies that the `TypeError` output by pykhiops,
        when the second element in the format (tuple) is an integer,
        is equal to the expected one.
        """
        expected_error = "'Header' type must be 'bool', " "not 'int'"
        dataset_spec, label = self.create_fixture_dataset_spec(
            output_dir=self.output_dir,
            data_type="file",
            multitable=True,
            schema="snowflake",
        )
        dataset_spec["format"] = (",", 1)
        with self.assertRaises(TypeError) as context:
            Dataset(dataset_spec, label)
        output_error = str(context.exception)
        self.assertEqual(output_error, expected_error)

    def test_error_format_tuple_1st_element_must_be_a_single_character(self):
        """test output error when the 1st element in format is a string of size 2

        - This test verifies that the `ValueError` output by pykhiops,
        when the first element in the format is a string of size 2,
        is equal to the expected one.
        """
        expected_error = "Separator must be a single character. Value: ;;"
        dataset_spec, label = self.create_fixture_dataset_spec(
            output_dir=self.output_dir,
            data_type="file",
            multitable=True,
            schema="snowflake",
        )
        dataset_spec["format"] = (";;", True)
        with self.assertRaises(ValueError) as context:
            Dataset(dataset_spec, label)
        output_error = str(context.exception)
        self.assertEqual(output_error, expected_error)

    def test_error_y_type_must_be_series_when_x_is_dict_of_dfs(self):
        """test output error when X's tables are DFs and y is a string

        - This test verifies that the `TypeError` output by pykhiops,
        when X's tables are dataframes and y is a string, is equal to the expected one.
        """
        expected_error = (
            "'y' type must be 'Series', not 'str' "
            "(X's tables are of type pandas.DataFrame)"
        )
        dataset_spec, _ = self.create_fixture_dataset_spec(
            output_dir=None, data_type="df", multitable=False, schema=None
        )
        with self.assertRaises(TypeError) as context:
            Dataset(dataset_spec, "class")
        output_error = str(context.exception)
        self.assertEqual(output_error, expected_error)

    def test_error_y_type_must_be_str_when_x_is_dict_of_strs(self):
        """test output error when X's tables are strings and y is a DF

        - This test verifies that the `TypeError` output by pykhiops,
        when X's tables are strings and y is a dataframe, is equal to
        the expected one."""
        expected_error = (
            "'y' type must be 'str', not 'DataFrame' (X's tables are "
            "of type str [file paths])"
        )
        dataset_spec, _ = self.create_fixture_dataset_spec(
            output_dir=self.output_dir,
            data_type="file",
            multitable=True,
            schema="snowflake",
        )
        label = pd.DataFrame()
        with self.assertRaises(TypeError) as context:
            Dataset(dataset_spec, label)
        output_error = str(context.exception)
        self.assertEqual(output_error, expected_error)

    def test_error_target_column_must_be_set(self):
        """test output error when target column is None

        - This test verifies that the `ValueError` output by pykhiops,
        when target column is None, is equal to the expected one.
        """
        expected_error = "Target column is not set"
        dataset_spec, _ = self.create_fixture_dataset_spec(
            output_dir=self.output_dir, data_type="file", multitable=False, schema=None
        )
        dataset = Dataset(dataset_spec, None)
        with self.assertRaises(ValueError) as context:
            dataset.target_column_type
        output_error = str(context.exception)
        self.assertEqual(output_error, expected_error)

    def test_error_table_name_must_be_str(self):
        """test output error when name of a table is an integer

        - This test verifies that the `TypeError` output by pykhiops,
        when name of a table is an integer, is equal to the expected one.
        """
        expected_error = "'name' type must be 'str', not 'int'"
        dataset_spec, label = self.create_fixture_dataset_spec(
            output_dir=None, data_type="df", multitable=False, schema=None
        )
        features_table = dataset_spec["tables"]["Reviews"][0]
        with self.assertRaises(TypeError) as context:
            PandasTable(
                1,
                features_table,
                target_column=label,
                categorical_target=True,
                key=None,
            )
        output_error = str(context.exception)
        self.assertEqual(output_error, expected_error)

    def test_error_table_name_is_empty_string(self):
        """test output error when name of a table is an empty str

        - This test verifies that the `ValueError` output by pykhiops,
        when name of a table is an empty string, is equal to the expected one.
        """
        expected_error = "'name' is the empty string"
        dataset_spec, label = self.create_fixture_dataset_spec(
            output_dir=None, data_type="df", multitable=False, schema=None
        )
        features_table = dataset_spec["tables"]["Reviews"][0]
        with self.assertRaises(ValueError) as context:
            PandasTable(
                "",
                features_table,
                target_column=label,
                categorical_target=True,
                key=None,
            )
        output_error = str(context.exception)
        self.assertEqual(output_error, expected_error)

    def test_error_key_type_must_be_str_or_list_like(self):
        """test output error when the key type is an integer

        - This test verifies that the `TypeError` output by pykhiops,
        when the key type is an integer, is equal to the expected one.
        """
        expected_error = "'key' type must be either 'str' or 'list-like', not 'int'"
        dataset_spec, label = self.create_fixture_dataset_spec(
            output_dir=None, data_type="df", multitable=False, schema=None
        )
        features_table = dataset_spec["tables"]["Reviews"][0]
        with self.assertRaises(TypeError) as context:
            PandasTable(
                "reviews",
                features_table,
                target_column=label,
                categorical_target=True,
                key=1,
            )
        output_error = str(context.exception)
        self.assertEqual(output_error, expected_error)

    def test_error_key_column_type_must_be_str(self):
        """test output error when a column of a key is an integer

        - This test verifies that the `TypeError` output by pykhiops,
        when a column of a key is of type integer, is equal to the expected one.
        """
        expected_error = "'key[0]' type must be 'str', not 'int' at table 'reviews'"
        dataset_spec, label = self.create_fixture_dataset_spec(
            output_dir=None, data_type="df", multitable=False, schema=None
        )
        features_table = dataset_spec["tables"]["Reviews"][0]
        with self.assertRaises(TypeError) as context:
            PandasTable(
                "reviews",
                features_table,
                target_column=label,
                categorical_target=True,
                key=[1],
            )
        output_error = str(context.exception)
        self.assertEqual(output_error, expected_error)

    def test_error_input_of_pandas_table_type_must_be_dataframe(self):
        """test output error when a string is passed to PandasTable

        - This test verifies that the `TypeError` output by pykhiops,
        when string is passed to PandasTable, is equal to the expected one.
        """
        expected_error = "'dataframe' type must be 'DataFrame', not 'str'"
        with self.assertRaises(TypeError) as context:
            PandasTable(
                "reviews",
                "data",
                target_column="class",
                categorical_target=True,
                key=None,
            )
        output_error = str(context.exception)
        self.assertEqual(output_error, expected_error)

    def test_error_input_of_pandas_table_is_empty(self):
        """test output error when an empty DF is passed to PandasTable

        - This test verifies that the `ValueError` output by pykhiops,
        when an empty df is passed to PandasTable, is equal to the expected one.
        """
        expected_error = "'dataframe' is empty"
        with self.assertRaises(ValueError) as context:
            PandasTable(
                "reviews",
                pd.DataFrame(),
                target_column="class",
                categorical_target=True,
                key=None,
            )
        output_error = str(context.exception)
        self.assertEqual(output_error, expected_error)

    def test_error_target_column_passed_to_pandas_table_must_be_series(self):
        """test output error when a string target column is passed to PandasTable

        - This test verifies that the `TypeError` output by pykhiops,
        when a string target column is passed to PandasTable, is equal to
        the expected one.
        """
        expected_error = "'target_column' type must be 'Series', not 'str'"
        dataset_spec, _ = self.create_fixture_dataset_spec(
            output_dir=None, data_type="df", multitable=False, schema=None
        )
        features_table = dataset_spec["tables"]["Reviews"][0]
        with self.assertRaises(TypeError) as context:
            PandasTable(
                "reviews",
                features_table,
                target_column="class",
                categorical_target=True,
                key=None,
            )
        output_error = str(context.exception)
        self.assertEqual(output_error, expected_error)

    def test_error_target_column_is_already_present_in_the_features(self):
        """test output error when the target is present in the features

        - This test verifies that the `ValueError` output by pykhiops,
        when the target is present in the features, is equal to the expected one.
        """
        expected_error = (
            "Target series name 'Recommended IND' is already present in"
            " dataframe : ['User_ID', 'Age', 'Clothing ID', 'Date', 'New',"
            " 'Title', 'Recommended IND', 'Positive Feedback average']"
        )
        dataset_spec, _ = self.create_fixture_dataset_spec(
            output_dir=None, data_type="df", multitable=False, schema=None
        )
        features_table = dataset_spec["tables"]["Reviews"][0]
        label = features_table["Recommended IND"]
        with self.assertRaises(ValueError) as context:
            PandasTable(
                "reviews",
                features_table,
                target_column=label,
                categorical_target=True,
                key=None,
            )
        output_error = str(context.exception)
        self.assertEqual(output_error, expected_error)

    def test_error_df_column_ids_must_all_be_int_or_str(self):
        """test output error when a column id in a DF is an integer

        - This test verifies that the `TypeError` output by pykhiops,
        when a column id in a dataframe is an integer, is equal to the expected one.
        """
        expected_error = (
            "Dataframe column ids must be either all integers or all "
            "strings. Column id at index 0 ('1') is of type 'int'"
        )
        dataset_spec, label = self.create_fixture_dataset_spec(
            output_dir=None, data_type="df", multitable=False, schema=None
        )
        features_table = dataset_spec["tables"]["Reviews"][0]
        features_table.rename(columns={"User_ID": 1}, inplace=True)
        with self.assertRaises(TypeError) as context:
            PandasTable(
                "reviews",
                features_table,
                target_column=label,
                categorical_target=True,
                key=None,
            )
        output_error = str(context.exception)
        self.assertEqual(output_error, expected_error)

    def test_error_non_existant_table_file_passed_to_file_table(self):
        """test output error when a non-existent table file is passed to FileTable

        - This test verifies that the `ValueError` output by pykhiops,
        when a non-existent table file is passed to FileTable, is equal to
        the expected one.
        """
        expected_error = "Non-existent data table file: Review.csv"
        table_path = "Review.csv"
        with self.assertRaises(ValueError) as context:
            FileTable(
                "reviews",
                table_path,
                target_column_id="class",
                categorical_target=True,
                key=None,
            )
        output_error = str(context.exception)
        self.assertEqual(output_error, expected_error)

    def test_error_empty_table_file_passed_to_file_table(self):
        """test output error when an empty table file is passed to FileTable

        - This test verifies that the `ValueError` output by pykhiops,
        when an empty table file is passed to FileTable, is equal to the expected one.
        """
        expected_error = "Empty data table file"
        table_path = os.path.join(self.output_dir, "empty_table.csv")
        table = pd.DataFrame(columns=["a", "b"])
        table.to_csv(table_path, sep="\t", index=False)
        with self.assertRaises(ValueError) as context:
            FileTable(
                "empty_table",
                table_path,
                target_column_id="class",
                categorical_target=True,
                key=None,
            )
        output_error = str(context.exception)
        self.assertIn(expected_error, output_error)

    def test_error_creation_of_a_table_file_for_khiops_on_an_existing_path(self):
        """test output error when the created file's path already exists

        - This test verifies that the `ValueError` output by pykhiops,
        when the path of the created file already exists,
         is equal to the expected one.
        """
        expected_error = "Cannot overwrite this table's path"
        dataset_spec, _ = self.create_fixture_dataset_spec(
            output_dir=self.output_dir, data_type="file", multitable=False, schema=None
        )
        old_file_path = dataset_spec["tables"]["Reviews"][0]
        new_file_path = old_file_path.replace("Reviews.csv", "copy_Reviews.txt")
        os.rename(old_file_path, new_file_path)
        file_table = FileTable(
            "Reviews",
            new_file_path,
            target_column_id="class",
            categorical_target=True,
            key="User_ID",
        )
        with self.assertRaises(ValueError) as context:
            file_table.create_table_file_for_khiops(self.output_dir, sort=False)
        output_error = str(context.exception)
        self.assertIn(expected_error, output_error)
