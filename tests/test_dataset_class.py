######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Test consistency of the created files with the input data"""
import os
import shutil
import unittest
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.testing import assert_equal
from pandas.testing import assert_frame_equal
from sklearn import datasets

from khiops.sklearn.dataset import Dataset


class DatasetInputOutputConsistencyTests(unittest.TestCase):
    """Test consistency of the created files with the input data

    The following tests allow to verify that:
    - The content of the .csv files (created by khiops.sklearn) is consistent with the
      content of the input data.
    - The content of the dictionaries (created by khiops.sklearn) is consistent with the
      content of the input data.
    - The input data used in the test is variable:
        - a monotable dataset: a dataframe or a file path.
        - a multitable dataset: a dictionary with tables of type dataframe or file
         path that are presented in a random order.
        - Data contained in the datasets is unsorted.
        - Data contained in the datasets is multi-typed: numeric, categorical and dates.
        - Two schemas of increasing complexity are considered: star and snowflake.
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
                ],
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
        # Set the random seed for reproducibility
        np.random.seed(31416)

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

        # Create the secondary tables
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

    def create_fixture_ds_spec(self, output_dir, data_type, multitable, schema):
        if not multitable:
            if data_type == "df":
                ref_table = self.create_monotable_dataframe()
                features = ref_table.drop(["class"], axis=1)
                ds_spec = {
                    "main_table": "Reviews",
                    "tables": {"Reviews": (features, "User_ID")},
                }
                label = ref_table["class"]
            else:
                assert data_type == "file"
                ref_table_path = os.path.join(output_dir, "Reviews.csv")
                self.create_monotable_data_file(ref_table_path)
                ds_spec = {
                    "main_table": "Reviews",
                    "tables": {"Reviews": (ref_table_path, "User_ID")},
                    "format": ("\t", True),
                }
                label = "class"
        elif schema == "star":
            if data_type == "df":
                (
                    ref_main_table,
                    ref_secondary_table,
                ) = self.create_multitable_star_dataframes()
                features_ref_main_table = ref_main_table.drop("class", axis=1)
                ds_spec = {
                    "main_table": "id_class",
                    "tables": {
                        "id_class": (features_ref_main_table, "User_ID"),
                        "logs": (ref_secondary_table, "User_ID"),
                    },
                }
                label = ref_main_table["class"]
            else:
                assert data_type == "file"
                ref_main_table_path = os.path.join(output_dir, "id_class.csv")
                ref_secondary_table_path = os.path.join(output_dir, "logs.csv")
                self.create_multitable_star_data_files(
                    ref_main_table_path, ref_secondary_table_path
                )
                ds_spec = {
                    "main_table": "id_class",
                    "tables": {
                        "id_class": (ref_main_table_path, "User_ID"),
                        "logs": (ref_secondary_table_path, "User_ID"),
                    },
                    "format": ("\t", True),
                }
                label = "class"
        else:
            assert schema == "snowflake"
            if data_type == "df":
                (
                    ref_main_table,
                    ref_secondary_table_1,
                    ref_secondary_table_2,
                    ref_tertiary_table,
                    ref_quaternary_table,
                ) = self.create_multitable_snowflake_dataframes()

                features_ref_main_table = ref_main_table.drop("class", axis=1)
                ds_spec = {
                    "main_table": "A",
                    "tables": {
                        "D": (
                            ref_tertiary_table,
                            ["User_ID", "VAR_1", "VAR_2"],
                        ),
                        "B": (ref_secondary_table_1, ["User_ID", "VAR_1"]),
                        "E": (
                            ref_quaternary_table,
                            ["User_ID", "VAR_1", "VAR_2", "VAR_3"],
                        ),
                        "C": (ref_secondary_table_2, ["User_ID"]),
                        "A": (features_ref_main_table, "User_ID"),
                    },
                    "relations": [
                        ("B", "D", False),
                        ("A", "C", True),
                        ("D", "E"),
                        ("A", "B", False),
                    ],
                }
                label = ref_main_table["class"]
            else:
                assert data_type == "file"
                ref_main_table_path = os.path.join(output_dir, "A.csv")
                ref_secondary_table_path_1 = os.path.join(output_dir, "B.csv")
                ref_secondary_table_path_2 = os.path.join(output_dir, "C.csv")
                ref_tertiary_table_path = os.path.join(output_dir, "D.csv")
                ref_quaternary_table_path = os.path.join(output_dir, "E.csv")

                self.create_multitable_snowflake_data_files(
                    ref_main_table_path,
                    ref_secondary_table_path_1,
                    ref_secondary_table_path_2,
                    ref_tertiary_table_path,
                    ref_quaternary_table_path,
                )
                ds_spec = {
                    "main_table": "A",
                    "tables": {
                        "B": (
                            ref_secondary_table_path_1,
                            ["User_ID", "VAR_1"],
                        ),
                        "E": (
                            ref_quaternary_table_path,
                            ["User_ID", "VAR_1", "VAR_2", "VAR_3"],
                        ),
                        "C": (
                            ref_secondary_table_path_2,
                            ["User_ID"],
                        ),
                        "A": (ref_main_table_path, "User_ID"),
                        "D": (
                            ref_tertiary_table_path,
                            ["User_ID", "VAR_1", "VAR_2"],
                        ),
                    },
                    "relations": [
                        ("B", "D", False),
                        ("A", "B", False),
                        ("D", "E"),
                        ("A", "C", True),
                    ],
                    "format": ("\t", True),
                }
                label = "class"

        return ds_spec, label

    def get_ref_var_types(self, multitable, data_type="df", schema=None):
        ref_var_types = {}
        if not multitable:
            ref_var_types["Reviews"] = {
                "User_ID": "Categorical",
                "Age": "Numerical",
                "Clothing ID": "Numerical",
                "Date": "Timestamp",
                "New": "Categorical",
                "Title": "Categorical",
                "Recommended IND": "Numerical",
                "Positive Feedback average": "Numerical",
                "class": "Categorical",
            }
            # Special type changes for file datasets:
            #  - "Date" field from "Timestamp" to "Date", the type Khiops detects
            #  - "Recommended IND" field from "Numerical" to "Categorical" because
            #    Khiops doesn't parse it well
            if data_type == "file":
                ref_var_types["Reviews"]["Date"] = "Date"
                ref_var_types["Reviews"]["Recommended IND"] = "Categorical"
                warnings.warn("Changed field `Recommended IND` to avoid a Khiops bug")
        elif schema == "star":
            ref_var_types["id_class"] = {
                "User_ID": "Categorical",
                "class": "Categorical",
                "logs": "Table",
            }
            ref_var_types["logs"] = {
                "User_ID": "Categorical",
                "VAR_1": "Categorical",
                "VAR_2": "Numerical",
                "VAR_3": "Numerical",
                "VAR_4": "Numerical",
            }
            # Special change for the file type:
            # - logs.VAR_3 is binary and detected as "Categorical" by Khiops
            if data_type == "file":
                ref_var_types["logs"]["VAR_3"] = "Categorical"
        else:
            assert (
                schema == "snowflake"
            ), f"'schema' should be 'snowflake' not '{schema}'"
            ref_var_types["A"] = {
                "User_ID": "Categorical",
                "class": "Categorical",
                "B": "Table",
                "C": "Entity",
            }
            ref_var_types["B"] = {
                "User_ID": "Categorical",
                "VAR_1": "Categorical",
                "VAR_2": "Numerical",
                "VAR_3": "Numerical",
                "VAR_4": "Numerical",
                "D": "Table",
            }
            ref_var_types["C"] = {
                "User_ID": "Categorical",
                "VAR_1": "Categorical",
                "VAR_2": "Numerical",
                "VAR_3": "Numerical",
                "VAR_4": "Numerical",
            }
            ref_var_types["D"] = {
                "User_ID": "Categorical",
                "VAR_1": "Categorical",
                "VAR_2": "Categorical",
                "VAR_3": "Numerical",
                "E": "Table",
            }
            ref_var_types["E"] = {
                "User_ID": "Categorical",
                "VAR_1": "Categorical",
                "VAR_2": "Categorical",
                "VAR_3": "Categorical",
                "VAR_4": "Categorical",
            }
            # Special change for the file type:
            # - B.VAR_3 is binary and detected as "Categorical" by Khiops
            # - C.VAR_3 is binary and detected as "Categorical" by Khiops
            if data_type == "file":
                ref_var_types["B"]["VAR_3"] = "Categorical"
                ref_var_types["C"]["VAR_3"] = "Categorical"

        return ref_var_types

    def test_dataset_is_correctly_built(self):
        """Test that the dataset structure is consistent with the input spec"""
        ds_spec, label = self.create_fixture_ds_spec(
            output_dir=None, data_type="df", multitable=True, schema="snowflake"
        )
        dataset = Dataset(ds_spec, label)

        self.assertEqual(dataset.main_table.name, "A")
        self.assertEqual(len(dataset.secondary_tables), 4)
        dataset_secondary_table_names = {
            secondary_table.name for secondary_table in dataset.secondary_tables
        }
        self.assertEqual(dataset_secondary_table_names, {"B", "C", "D", "E"})
        self.assertEqual(len(dataset.relations), 4)

        spec_relations = ds_spec["relations"]
        for relation, spec_relation in zip(dataset.relations, spec_relations):
            self.assertEqual(relation[:2], spec_relation[:2])
            if len(spec_relation) == 3:
                self.assertEqual(relation[2], spec_relation[2])
            else:
                self.assertFalse(relation[2])

    def test_out_file_from_dataframe_monotable(self):
        """Test consistency of the created data file with the input dataframe

        - This test verifies that the content of the input dataframe is equal
        to that of the csv file created by khiops.sklearn.
        """
        # Create a monotable dataset object from fixture data
        spec, y = self.create_fixture_ds_spec(
            output_dir=None, data_type="df", multitable=False, schema=None
        )
        dataset = Dataset(spec, y=y)

        # Create and load the intermediary Khiops file
        out_table_path, _ = dataset.create_table_files_for_khiops(self.output_dir)
        out_table = pd.read_csv(out_table_path, sep="\t")

        # Cast "Date" columns to datetime as we don't automatically recognize dates
        out_table["Date"] = out_table["Date"].astype("datetime64[ns]")
        ref_table = spec["tables"]["Reviews"][0]
        ref_table["class"] = y

        # Check that the dataframes are equal
        assert_frame_equal(
            ref_table.sort_values(by="User_ID").reset_index(drop=True),
            out_table,
        )

    def test_out_file_from_numpy_array_monotable(self):
        """Test consistency of the created data file with the input numpy array"""
        # Create a monotable dataset from a numpy array
        iris = datasets.load_iris()
        spec = {"tables": {"iris": (iris.data, None)}}
        dataset = Dataset(spec, y=iris.target, categorical_target=True)

        # Create and load the intermediary Khiops file
        out_table_path, _ = dataset.create_table_files_for_khiops(self.output_dir)
        out_table = np.loadtxt(out_table_path, delimiter="\t", skiprows=1, ndmin=2)

        # Check that the arrays are equal
        assert_equal(
            out_table,
            np.concatenate(
                (iris.data, iris.target.reshape(len(iris.target), 1)), axis=1
            ),
        )

    def _create_test_sparse_matrix_with_target(self):
        # Create sparse array that also contains missing data-only rows
        sparse_array = np.eye(N=100, k=2) + np.eye(N=100, k=5)

        # Create scipy sparse (CSR) matrix from the sparse array
        sparse_matrix = sp.csr_matrix(sparse_array)

        # Create targets: -1 for left-sided values; +1 for right-sided values,
        # 0 for missing-data-only rows
        target_array = np.array(50 * [-1] + 45 * [1] + 5 * [0])
        return sparse_matrix, target_array

    def _load_khiops_sparse_file(self, stream):
        # Skip header
        next(stream)

        # Read the sparse file
        target_vector = []
        feature_matrix = []
        for line in stream:
            features, target_value = line.split(b"\t")
            feature_row = np.zeros(100)
            for feature in features.strip().split(b" "):
                indexed_feature = feature.split(b":")

                # Skip missing feature
                if len(indexed_feature) < 2:
                    continue

                # Set feature value in row at the specified index
                feature_index, feature_value = indexed_feature
                feature_row[int(feature_index) - 1] = float(feature_value)
            feature_matrix.append(feature_row)
            target_vector.append(float(target_value))
        target_array = np.array(target_vector)
        sparse_matrix = sp.csr_matrix(feature_matrix)
        return sparse_matrix, target_array

    def test_out_file_from_sparse_matrix_monotable(self):
        """Test consistency of the created data file with the input sparse matrix"""

        # Load input sparse matrix and target array
        (
            input_sparse_matrix,
            input_target,
        ) = self._create_test_sparse_matrix_with_target()

        # Create monotable dataset from the sparse matrix
        dataset = Dataset(
            X=input_sparse_matrix, y=input_target, categorical_target=True
        )
        # Create and load the intermediary Khiops file
        out_table_path, _ = dataset.create_table_files_for_khiops(self.output_dir)
        with open(out_table_path, "rb") as out_table_stream:
            sparse_matrix, target_array = self._load_khiops_sparse_file(
                out_table_stream
            )

        # Check that the arrays are equal
        assert_equal(
            np.concatenate(
                (
                    sparse_matrix.toarray(),
                    target_array.reshape(-1, 1),
                ),
                axis=1,
            ),
            np.concatenate(
                (input_sparse_matrix.toarray(), input_target.reshape(-1, 1)), axis=1
            ),
        )

    def test_out_file_from_sparse_matrix_monotable_specification(self):
        """Test consistency of the created data file with the input sparse matrix"""

        # Load input sparse matrix and target array
        (
            input_sparse_matrix,
            input_target,
        ) = self._create_test_sparse_matrix_with_target()

        # Create monotable dataset from input mapping with the sparse matrix
        spec = {"tables": {"example_sparse_matrix": (input_sparse_matrix, None)}}
        dataset = Dataset(spec, y=input_target, categorical_target=True)

        # Create and load the intermediary Khiops file
        out_table_path, _ = dataset.create_table_files_for_khiops(self.output_dir)
        with open(out_table_path, "rb") as out_table_stream:
            sparse_matrix, target_array = self._load_khiops_sparse_file(
                out_table_stream
            )

        # Check that the arrays are equal
        assert_equal(
            np.concatenate(
                (
                    sparse_matrix.toarray(),
                    target_array.reshape(-1, 1),
                ),
                axis=1,
            ),
            np.concatenate(
                (input_sparse_matrix.toarray(), input_target.reshape(-1, 1)), axis=1
            ),
        )

    def test_out_file_from_data_file_monotable(self):
        """Test consistency of the created data file with the input data file

         - This test verifies that the content of the input data file is equal
        to that of the csv file created by khiops.sklearn.
        """
        # Create the test dataset
        ds_spec, label = self.create_fixture_ds_spec(
            output_dir=self.output_dir, data_type="file", multitable=False, schema=None
        )
        dataset = Dataset(ds_spec, label)

        out_table_path, _ = dataset.create_table_files_for_khiops(self.output_dir)
        out_table = pd.read_csv(out_table_path, sep="\t")

        ref_table_path = ds_spec["tables"]["Reviews"][0]
        ref_table = pd.read_csv(ref_table_path, sep="\t")

        # Check that the dataframes are equal
        assert_frame_equal(
            ref_table.sort_values(by="User_ID").reset_index(drop=True),
            out_table,
        )

    def test_out_files_from_dataframes_multitable_star(self):
        """Test consistency of the created data files with the input dataframes

        - This test verifies that the content of the input dataframes, defined through a
          dictionary, is equal to that of the csv files created by khiops.sklearn. The
          schema of the dataset is "star".
        """
        # Create the test dataset
        ds_spec, label = self.create_fixture_ds_spec(
            output_dir=None, data_type="df", multitable=True, schema="star"
        )
        dataset = Dataset(ds_spec, label)

        # Create the Khiops intermediary files
        (
            main_table_path,
            secondary_table_paths,
        ) = dataset.create_table_files_for_khiops(self.output_dir)

        # Load the intermediary files
        secondary_table_path = secondary_table_paths["logs"]
        out_main_table = pd.read_csv(main_table_path, sep="\t")
        out_secondary_table = pd.read_csv(secondary_table_path, sep="\t")

        ref_main_table = ds_spec["tables"]["id_class"][0]
        ref_main_table["class"] = label
        ref_secondary_table = ds_spec["tables"]["logs"][0]

        # Clean created test data
        assert_frame_equal(
            ref_main_table.sort_values(by="User_ID", ascending=True).reset_index(
                drop=True
            ),
            out_main_table,
        )
        assert_frame_equal(
            ref_secondary_table.sort_values(
                by=ref_secondary_table.columns.tolist(), ascending=True
            ).reset_index(drop=True),
            out_secondary_table.sort_values(
                by=out_secondary_table.columns.tolist(), ascending=True
            ).reset_index(drop=True),
        )

    def test_out_files_from_data_files_multitable_star(self):
        """Test consistency of the created data files with the input data files

         - This test verifies that the content of the input data files, defined
        through a dictionary, is equal to that of the csv files created by
        khiops.sklearn. The schema of the dataset is "star".
        """
        ds_spec, label = self.create_fixture_ds_spec(
            output_dir=self.output_dir, data_type="file", multitable=True, schema="star"
        )

        dataset = Dataset(ds_spec, label)
        main_table_path, dico_secondary_table = dataset.create_table_files_for_khiops(
            self.output_dir
        )
        secondary_table_path = dico_secondary_table["logs"]
        out_main_table = pd.read_csv(main_table_path, sep="\t")
        out_secondary_table = pd.read_csv(secondary_table_path, sep="\t")

        ref_table_path = ds_spec["tables"]["id_class"][0]
        ref_main_table = pd.read_csv(ref_table_path, sep="\t")
        ref_secondary_table_path = ds_spec["tables"]["logs"][0]
        ref_secondary_table = pd.read_csv(ref_secondary_table_path, sep="\t")

        # assertions
        assert_frame_equal(
            ref_main_table.sort_values(by="User_ID", ascending=True).reset_index(
                drop=True
            ),
            out_main_table,
        )

        assert_frame_equal(
            ref_secondary_table.sort_values(
                by=ref_secondary_table.columns.tolist(), ascending=True
            ).reset_index(drop=True),
            out_secondary_table.sort_values(
                by=out_secondary_table.columns.tolist(), ascending=True
            ).reset_index(drop=True),
        )

    def test_out_files_from_dataframes_multitable_snowflake(self):
        """Test consistency of the created data files with the input dataframes

         - This test verifies that the content of the input dataframes, defined
        through a dictionary, is equal to that of the csv files created by
        khiops.sklearn. The schema of the dataset is "snowflake".
        """
        ds_spec, label = self.create_fixture_ds_spec(
            output_dir=None, data_type="df", multitable=True, schema="snowflake"
        )
        dataset = Dataset(ds_spec, label)

        (
            main_table_path,
            additional_table_paths,
        ) = dataset.create_table_files_for_khiops(self.output_dir)

        out_main_table = pd.read_csv(main_table_path, sep="\t")
        ref_main_table = ds_spec["tables"]["A"][0]
        ref_main_table["class"] = label

        # assertions
        assert_frame_equal(
            ref_main_table.sort_values(by="User_ID", ascending=True).reset_index(
                drop=True
            ),
            out_main_table,
        )

        additional_table_names = list(additional_table_paths.keys())
        for name in additional_table_names:
            additional_table_path = additional_table_paths[name]
            out_additional_table = pd.read_csv(additional_table_path, sep="\t")
            ref_additional_table = ds_spec["tables"][name][0]
            assert_frame_equal(
                ref_additional_table.sort_values(
                    by=ref_additional_table.columns.tolist(), ascending=True
                ).reset_index(drop=True),
                out_additional_table.sort_values(
                    by=out_additional_table.columns.tolist(), ascending=True
                ).reset_index(drop=True),
            )

    def test_out_files_from_data_files_multitable_snowflake(self):
        """Test consistency of the created  s with the input data files

         - This test verifies that the content of the input data files, defined
        through a dictionary, is equal to that of the csv files created
        by khiops.sklearn. The schema of the dataset is "snowflake".
        """
        ds_spec, label = self.create_fixture_ds_spec(
            output_dir=self.output_dir,
            data_type="file",
            multitable=True,
            schema="snowflake",
        )

        dataset = Dataset(ds_spec, label)
        main_table_path, additional_table_paths = dataset.create_table_files_for_khiops(
            self.output_dir
        )

        out_main_table = pd.read_csv(main_table_path, sep="\t")
        ref_main_table_path = ds_spec["tables"]["A"][0]
        ref_main_table = pd.read_csv(ref_main_table_path, sep="\t")

        # assertions
        assert_frame_equal(
            ref_main_table.sort_values(by="User_ID", ascending=True).reset_index(
                drop=True
            ),
            out_main_table,
        )

        additional_table_names = list(additional_table_paths.keys())
        for name in additional_table_names:
            additional_table_path = additional_table_paths[name]
            out_additional_table = pd.read_csv(additional_table_path, sep="\t")
            ref_additional_table_path = ds_spec["tables"][name][0]
            ref_additional_table = pd.read_csv(ref_additional_table_path, sep="\t")
            assert_frame_equal(
                out_additional_table.sort_values(
                    by=out_additional_table.columns.tolist(), ascending=True
                ).reset_index(drop=True),
                ref_additional_table.sort_values(
                    by=ref_additional_table.columns.tolist(), ascending=True
                ).reset_index(drop=True),
            )

    def test_create_khiops_domain(self):
        """Test consistency of the dataset method create_khiops_domain"""
        fixtures = [
            {
                "output_dir": None,
                "data_type": "df",
                "multitable": False,
                "schema": None,
            },
            {
                "output_dir": self.output_dir,
                "data_type": "file",
                "multitable": False,
                "schema": None,
            },
            {
                "output_dir": None,
                "data_type": "df",
                "multitable": True,
                "schema": "star",
            },
            {
                "output_dir": self.output_dir,
                "data_type": "file",
                "multitable": True,
                "schema": "star",
            },
            {
                "output_dir": None,
                "data_type": "df",
                "multitable": True,
                "schema": "snowflake",
            },
            {
                "output_dir": self.output_dir,
                "data_type": "file",
                "multitable": True,
                "schema": "snowflake",
            },
        ]

        for fixture in fixtures:
            with self.subTest(**fixture):
                ds = Dataset(*self.create_fixture_ds_spec(**fixture))
                ref_var_types = self.get_ref_var_types(
                    multitable=fixture["multitable"],
                    data_type=fixture["data_type"],
                    schema=fixture["schema"],
                )
                self._test_domain_coherence(ds, ref_var_types)

    def _test_domain_coherence(self, ds, ref_var_types):
        # Create the dictionary domain associated to the fixture dataset
        out_domain = ds.create_khiops_dictionary_domain()

        # Check that the domain has the same number of tables as the dataset
        self.assertEqual(len(out_domain.dictionaries), 1 + len(ds.secondary_tables))

        # Check that the domain has the same table names as the reference
        ref_table_names = {
            table.name for table in [ds.main_table] + ds.secondary_tables
        }
        out_table_names = {dictionary.name for dictionary in out_domain.dictionaries}
        self.assertEqual(ref_table_names, out_table_names)

        # Check that the output domain has a root table iff the dataset is multitable
        self.assertEqual(
            ds.is_multitable, out_domain.get_dictionary(ds.main_table.name).root
        )

        # Check that:
        # - the table keys are the same as the dataset
        # - the domain has the same variable names as the reference
        for table in [ds.main_table] + ds.secondary_tables:
            with self.subTest(table=table.name):
                self.assertEqual(table.key, out_domain.get_dictionary(table.name).key)
                out_dictionary_var_types = {
                    var.name: var.type
                    for var in out_domain.get_dictionary(table.name).variables
                }
                self.assertEqual(ref_var_types[table.name], out_dictionary_var_types)
