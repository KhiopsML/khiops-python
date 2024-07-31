######################################################################################
# Copyright (c) 2024 Orange. All rights reserved.                                    #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Test consistency of the created files with the input data"""
import os
import shutil
import unittest

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.testing import assert_equal
from pandas.testing import assert_frame_equal
from sklearn import datasets

from khiops.sklearn.tables import Dataset


class KhiopsConsistensyOfFilesAndDictionariesWithInputDataTests(unittest.TestCase):
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
            else:
                assert data_type == "file"
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
                        "C": (reference_secondary_table_2, ["User_ID"]),
                        "A": (features_reference_main_table, "User_ID"),
                    },
                    "relations": [
                        ("B", "D", False),
                        ("A", "C", True),
                        ("D", "E"),
                        ("A", "B", False),
                    ],
                }
                label = reference_main_table["class"]
            else:
                assert data_type == "file"
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
                            ["User_ID"],
                        ),
                        "A": (reference_main_table_path, "User_ID"),
                        "D": (
                            reference_tertiary_table_path,
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

        return dataset_spec, label

    def get_reference_dictionaries(self, multitable, schema=None):
        reference_dictionaries = []
        if not multitable:
            reference_dictionary = {
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
            reference_dictionaries.extend([reference_dictionary])
        elif schema == "star":
            reference_main_dictionary = {
                "User_ID": "Categorical",
                "class": "Categorical",
                "logs": "Table",
            }
            reference_secondary_dictionary = {
                "User_ID": "Categorical",
                "VAR_1": "Categorical",
                "VAR_2": "Numerical",
                "VAR_3": "Numerical",
                "VAR_4": "Numerical",
            }
            reference_dictionaries.extend(
                [reference_main_dictionary, reference_secondary_dictionary]
            )
        else:
            assert schema == "snowflake"
            reference_main_dictionary = {
                "User_ID": "Categorical",
                "class": "Categorical",
                "B": "Table",
                "C": "Entity",
            }
            reference_secondary_dictionary_1 = {
                "User_ID": "Categorical",
                "VAR_1": "Categorical",
                "VAR_2": "Numerical",
                "VAR_3": "Numerical",
                "VAR_4": "Numerical",
                "D": "Table",
            }
            reference_secondary_dictionary_2 = {
                "User_ID": "Categorical",
                "VAR_1": "Categorical",
                "VAR_2": "Numerical",
                "VAR_3": "Numerical",
                "VAR_4": "Numerical",
            }
            reference_tertiary_dictionary = {
                "User_ID": "Categorical",
                "VAR_1": "Categorical",
                "VAR_2": "Categorical",
                "VAR_3": "Numerical",
                "E": "Table",
            }
            reference_quaternary_dictionary = {
                "User_ID": "Categorical",
                "VAR_1": "Categorical",
                "VAR_2": "Categorical",
                "VAR_3": "Categorical",
                "VAR_4": "Categorical",
            }
            reference_dictionaries.extend(
                [
                    reference_main_dictionary,
                    reference_secondary_dictionary_1,
                    reference_secondary_dictionary_2,
                    reference_tertiary_dictionary,
                    reference_quaternary_dictionary,
                ]
            )

        return reference_dictionaries

    def test_dataset_is_correctly_built(self):
        """Test that the dataset structure is consistent with the input spec"""
        dataset_spec, label = self.create_fixture_dataset_spec(
            output_dir=None, data_type="df", multitable=True, schema="snowflake"
        )
        dataset = Dataset(dataset_spec, label)

        self.assertEqual(dataset.main_table.name, "A")
        self.assertEqual(len(dataset.secondary_tables), 4)
        dataset_secondary_table_names = set(
            secondary_table.name for secondary_table in dataset.secondary_tables
        )
        self.assertEqual(dataset_secondary_table_names, {"B", "C", "D", "E"})
        self.assertEqual(len(dataset.relations), 4)

        spec_relations = dataset_spec["relations"]
        for relation, spec_relation in zip(dataset.relations, spec_relations):
            self.assertEqual(relation[:2], spec_relation[:2])
            if len(spec_relation) == 3:
                self.assertEqual(relation[2], spec_relation[2])
            else:
                self.assertFalse(relation[2])

    def test_created_file_from_dataframe_monotable(self):
        """Test consistency of the created data file with the input dataframe

        - This test verifies that the content of the input dataframe is equal
        to that of the csv file created by khiops.sklearn.
        """
        # Create a monotable dataset object from fixture data
        spec, y = self.create_fixture_dataset_spec(
            output_dir=None, data_type="df", multitable=False, schema=None
        )
        dataset = Dataset(spec, y=y)

        # Create and load the intermediary Khiops file
        created_table_path, _ = dataset.create_table_files_for_khiops(self.output_dir)
        created_table = pd.read_csv(created_table_path, sep="\t")

        # Cast "Date" columns to datetime as we don't automatically recognize dates
        created_table["Date"] = created_table["Date"].astype("datetime64[ns]")
        reference_table = spec["tables"]["Reviews"][0]
        reference_table["class"] = y

        # Check that the dataframes are equal
        assert_frame_equal(
            created_table,
            reference_table.sort_values(by="User_ID").reset_index(drop=True),
        )

    def test_created_file_from_numpy_array_monotable(self):
        """Test consistency of the created data file with the input numpy array"""
        # Create a monotable dataset from a numpy array
        iris = datasets.load_iris()
        spec = {"tables": {"iris": (iris.data, None)}}
        dataset = Dataset(spec, y=iris.target, categorical_target=True)

        # Create and load the intermediary Khiops file
        created_table_path, _ = dataset.create_table_files_for_khiops(self.output_dir)
        created_table = np.loadtxt(
            created_table_path, delimiter="\t", skiprows=1, ndmin=2
        )

        # Check that the arrays are equal
        assert_equal(
            created_table,
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
        # skip header
        next(stream)
        target_vector = []
        feature_matrix = []
        for line in stream:
            target, features = line.split(b"\t")
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
            target_vector.append(float(target))
        target_array = np.array(target_vector)
        sparse_matrix = sp.csr_matrix(feature_matrix)
        return sparse_matrix, target_array

    def test_created_file_from_sparse_matrix_monotable(self):
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
        created_table_path, _ = dataset.create_table_files_for_khiops(self.output_dir)
        with open(created_table_path, "rb") as created_table_stream:
            sparse_matrix, target_array = self._load_khiops_sparse_file(
                created_table_stream
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

    def test_created_file_from_sparse_matrix_monotable_specification(self):
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
        created_table_path, _ = dataset.create_table_files_for_khiops(self.output_dir)
        with open(created_table_path, "rb") as created_table_stream:
            sparse_matrix, target_array = self._load_khiops_sparse_file(
                created_table_stream
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

    def test_created_file_from_data_file_monotable(self):
        """Test consistency of the created data file with the input data file

         - This test verifies that the content of the input data file is equal
        to that of the csv file created by khiops.sklearn.
        """
        # Create the test dataset
        dataset_spec, label = self.create_fixture_dataset_spec(
            output_dir=self.output_dir, data_type="file", multitable=False, schema=None
        )
        dataset = Dataset(dataset_spec, label)

        created_table_path, _ = dataset.create_table_files_for_khiops(self.output_dir)
        created_table = pd.read_csv(created_table_path, sep="\t")

        reference_table_path = dataset_spec["tables"]["Reviews"][0]
        reference_table = pd.read_csv(reference_table_path, sep="\t")

        # Check that the dataframes are equal
        assert_frame_equal(
            created_table,
            reference_table.sort_values(by="User_ID").reset_index(drop=True),
        )

    def test_created_files_from_dataframes_multitable_star(self):
        """Test consistency of the created data files with the input dataframes

        - This test verifies that the content of the input dataframes, defined through a
          dictionary, is equal to that of the csv files created by khiops.sklearn. The
          schema of the dataset is "star".
        """
        # Create the test dataset
        dataset_spec, label = self.create_fixture_dataset_spec(
            output_dir=None, data_type="df", multitable=True, schema="star"
        )
        dataset = Dataset(dataset_spec, label)

        # Create the Khiops intermediary files
        (
            main_table_path,
            secondary_table_paths,
        ) = dataset.create_table_files_for_khiops(self.output_dir)

        # Load the intermediary files
        secondary_table_path = secondary_table_paths["logs"]
        created_main_table = pd.read_csv(main_table_path, sep="\t")
        created_secondary_table = pd.read_csv(secondary_table_path, sep="\t")

        reference_main_table = dataset_spec["tables"]["id_class"][0]
        reference_main_table["class"] = label
        reference_secondary_table = dataset_spec["tables"]["logs"][0]

        # Clean created test data
        assert_frame_equal(
            created_main_table,
            reference_main_table.sort_values(by="User_ID", ascending=True).reset_index(
                drop=True
            ),
        )
        assert_frame_equal(
            created_secondary_table.sort_values(
                by=created_secondary_table.columns.tolist(), ascending=True
            ).reset_index(drop=True),
            reference_secondary_table.sort_values(
                by=reference_secondary_table.columns.tolist(), ascending=True
            ).reset_index(drop=True),
        )

    def test_created_files_from_data_files_multitable_star(self):
        """Test consistency of the created data files with the input data files

         - This test verifies that the content of the input data files, defined
        through a dictionary, is equal to that of the csv files created by
        khiops.sklearn. The schema of the dataset is "star".
        """
        dataset_spec, label = self.create_fixture_dataset_spec(
            output_dir=self.output_dir, data_type="file", multitable=True, schema="star"
        )

        dataset = Dataset(dataset_spec, label)
        main_table_path, dico_secondary_table = dataset.create_table_files_for_khiops(
            self.output_dir
        )
        secondary_table_path = dico_secondary_table["logs"]
        created_main_table = pd.read_csv(main_table_path, sep="\t")
        created_secondary_table = pd.read_csv(secondary_table_path, sep="\t")

        reference_table_path = dataset_spec["tables"]["id_class"][0]
        reference_main_table = pd.read_csv(reference_table_path, sep="\t")
        reference_secondary_table_path = dataset_spec["tables"]["logs"][0]
        reference_secondary_table = pd.read_csv(
            reference_secondary_table_path, sep="\t"
        )

        # assertions
        assert_frame_equal(
            created_main_table,
            reference_main_table.sort_values(by="User_ID", ascending=True).reset_index(
                drop=True
            ),
        )

        assert_frame_equal(
            created_secondary_table.sort_values(
                by=created_secondary_table.columns.tolist(), ascending=True
            ).reset_index(drop=True),
            reference_secondary_table.sort_values(
                by=reference_secondary_table.columns.tolist(), ascending=True
            ).reset_index(drop=True),
        )

    def test_created_files_from_dataframes_multitable_snowflake(self):
        """Test consistency of the created data files with the input dataframes

         - This test verifies that the content of the input dataframes, defined
        through a dictionary, is equal to that of the csv files created by
        khiops.sklearn. The schema of the dataset is "snowflake".
        """
        dataset_spec, label = self.create_fixture_dataset_spec(
            output_dir=None, data_type="df", multitable=True, schema="snowflake"
        )
        dataset = Dataset(dataset_spec, label)

        (
            main_table_path,
            additional_table_paths,
        ) = dataset.create_table_files_for_khiops(self.output_dir)

        created_main_table = pd.read_csv(main_table_path, sep="\t")
        reference_main_table = dataset_spec["tables"]["A"][0]
        reference_main_table["class"] = label

        # assertions
        assert_frame_equal(
            created_main_table,
            reference_main_table.sort_values(by="User_ID", ascending=True).reset_index(
                drop=True
            ),
        )

        additional_table_names = list(additional_table_paths.keys())
        for name in additional_table_names:
            additional_table_path = additional_table_paths[name]
            created_additional_table = pd.read_csv(additional_table_path, sep="\t")
            reference_additional_table = dataset_spec["tables"][name][0]
            assert_frame_equal(
                created_additional_table.sort_values(
                    by=created_additional_table.columns.tolist(), ascending=True
                ).reset_index(drop=True),
                reference_additional_table.sort_values(
                    by=reference_additional_table.columns.tolist(), ascending=True
                ).reset_index(drop=True),
            )

    def test_created_files_from_data_files_multitable_snowflake(self):
        """Test consistency of the created  s with the input data files

         - This test verifies that the content of the input data files, defined
        through a dictionary, is equal to that of the csv files created
        by khiops.sklearn. The schema of the dataset is "snowflake".
        """
        dataset_spec, label = self.create_fixture_dataset_spec(
            output_dir=self.output_dir,
            data_type="file",
            multitable=True,
            schema="snowflake",
        )

        dataset = Dataset(dataset_spec, label)
        main_table_path, additional_table_paths = dataset.create_table_files_for_khiops(
            self.output_dir
        )

        created_main_table = pd.read_csv(main_table_path, sep="\t")
        reference_main_table_path = dataset_spec["tables"]["A"][0]
        reference_main_table = pd.read_csv(reference_main_table_path, sep="\t")

        # assertions
        assert_frame_equal(
            created_main_table,
            reference_main_table.sort_values(by="User_ID", ascending=True).reset_index(
                drop=True
            ),
        )

        additional_table_names = list(additional_table_paths.keys())
        for name in additional_table_names:
            additional_table_path = additional_table_paths[name]
            created_additional_table = pd.read_csv(additional_table_path, sep="\t")
            reference_additional_table_path = dataset_spec["tables"][name][0]
            reference_additional_table = pd.read_csv(
                reference_additional_table_path, sep="\t"
            )
            assert_frame_equal(
                created_additional_table.sort_values(
                    by=created_additional_table.columns.tolist(), ascending=True
                ).reset_index(drop=True),
                reference_additional_table.sort_values(
                    by=reference_additional_table.columns.tolist(), ascending=True
                ).reset_index(drop=True),
            )

    def test_created_dictionary_from_dataframe_monotable(self):
        """Test consistency of the created dictionary with the input dataframe

        - This test verifies that the dictionary file (.kdic) created by
        khiops.sklearn contains information that is consistent with the
        input monotable dataset. Data is here provided through a dataframe.
        """

        dataset_spec, label = self.create_fixture_dataset_spec(
            output_dir=None, data_type="df", multitable=False, schema=None
        )

        dataset = Dataset(dataset_spec, label)
        created_dictionary_domain = dataset.create_khiops_dictionary_domain()
        created_dictionary = created_dictionary_domain.dictionaries[0]
        created_dictionary_variable_types = {
            var.name: var.type for var in created_dictionary.variables
        }
        reference_dictionary_variable_types = self.get_reference_dictionaries(
            multitable=False
        )[0]

        # assertions
        self.assertEqual(len(created_dictionary_domain.dictionaries), 1)
        self.assertEqual(created_dictionary.name, "Reviews")
        self.assertEqual(created_dictionary.root, False)
        self.assertEqual(len(created_dictionary.key), 1)
        self.assertEqual(
            created_dictionary_variable_types, reference_dictionary_variable_types
        )

    def test_created_dictionary_from_data_file_monotable(self):
        """Test consistency of the created dictionary with the input data file

        - This test verifies that the dictionary file (.kdic) created by
        khiops.sklearn contains information that is consistent with the
        input monotable dataset. Data is here provided through a data file.
        """
        dataset_spec, label = self.create_fixture_dataset_spec(
            output_dir=self.output_dir, data_type="file", multitable=False, schema=None
        )
        dataset = Dataset(dataset_spec, label)
        created_dictionary_domain = dataset.create_khiops_dictionary_domain()
        created_dictionary = created_dictionary_domain.dictionaries[0]
        created_dictionary_variable_types = {
            var.name: var.type for var in created_dictionary.variables
        }
        reference_dictionary_variable_types = self.get_reference_dictionaries(
            multitable=False
        )[0]
        reference_dictionary_variable_types["Date"] = "Categorical"

        # assertions
        self.assertEqual(len(created_dictionary_domain.dictionaries), 1)
        self.assertEqual(created_dictionary.name, "Reviews")
        self.assertEqual(created_dictionary.root, False)
        self.assertEqual(len(created_dictionary.key), 1)
        self.assertEqual(
            created_dictionary_variable_types, reference_dictionary_variable_types
        )

    def test_created_dictionary_from_dataframes_multitable_star(self):
        """Test consistency of the created dictionaries with the input dataframes

        - This test verifies that the dictionary file (.kdic) created by
        khiops.sklearn contains information that is consistent with the
        input multitable dataset. Data is here provided through dataframes
        and its schema is "star".
        """

        dataset_spec, label = self.create_fixture_dataset_spec(
            output_dir=None, data_type="df", multitable=True, schema="star"
        )
        dataset = Dataset(dataset_spec, label)
        created_dictionary_domain = dataset.create_khiops_dictionary_domain()
        created_main_dictionary = created_dictionary_domain.dictionaries[0]
        created_secondary_dictionary = created_dictionary_domain.dictionaries[1]

        # assertions
        self.assertEqual(len(created_dictionary_domain.dictionaries), 2)
        self.assertEqual(created_main_dictionary.name, "id_class")
        self.assertEqual(created_secondary_dictionary.name, "logs")
        self.assertEqual(created_main_dictionary.root, True)
        self.assertEqual(created_secondary_dictionary.root, False)
        self.assertEqual(created_main_dictionary.key[0], "User_ID")

        created_main_dictionary_variable_types = {
            var.name: var.type for var in created_main_dictionary.variables
        }
        created_secondary_dictionary_variable_types = {
            var.name: var.type for var in created_secondary_dictionary.variables
        }
        reference_dictionaries = self.get_reference_dictionaries(
            multitable=True, schema="star"
        )
        reference_main_dictionary_variable_types = reference_dictionaries[0]
        reference_secondary_dictionary_variable_types = reference_dictionaries[1]

        # assertions
        self.assertEqual(
            created_main_dictionary_variable_types,
            reference_main_dictionary_variable_types,
        )
        self.assertEqual(
            created_secondary_dictionary_variable_types,
            reference_secondary_dictionary_variable_types,
        )

    def test_created_dictionary_from_data_files_multitable_star(self):
        """Test consistency of the created dictionaries with the input data files

        - This test verifies that the dictionary file (.kdic) created by
        khiops.sklearn contains information that is consistent with the
        input multitable dataset. Data is here provided through data files
        and its schema is "star".
        """
        dataset_spec, label = self.create_fixture_dataset_spec(
            output_dir=self.output_dir, data_type="file", multitable=True, schema="star"
        )

        dataset = Dataset(dataset_spec, label)
        created_dictionary_domain = dataset.create_khiops_dictionary_domain()
        created_main_dictionary = created_dictionary_domain.dictionaries[0]
        created_secondary_dictionary = created_dictionary_domain.dictionaries[1]

        # assertions
        self.assertEqual(len(created_dictionary_domain.dictionaries), 2)
        self.assertEqual(created_main_dictionary.name, "id_class")
        self.assertEqual(created_secondary_dictionary.name, "logs")
        self.assertEqual(created_main_dictionary.root, True)
        self.assertEqual(created_secondary_dictionary.root, False)
        self.assertEqual(created_main_dictionary.key[0], "User_ID")

        created_main_dictionary_variable_types = {
            var.name: var.type for var in created_main_dictionary.variables
        }
        created_secondary_dictionary_variable_types = {
            var.name: var.type for var in created_secondary_dictionary.variables
        }
        reference_dictionaries = self.get_reference_dictionaries(
            multitable=True, schema="star"
        )
        reference_main_dictionary_variable_types = reference_dictionaries[0]
        reference_secondary_dictionary_variable_types = reference_dictionaries[1]

        # assertions
        self.assertEqual(
            created_main_dictionary_variable_types,
            reference_main_dictionary_variable_types,
        )
        self.assertEqual(
            created_secondary_dictionary_variable_types,
            reference_secondary_dictionary_variable_types,
        )

    def test_created_dictionary_from_dataframes_multitable_snowflake(self):
        """Test consistency of the created dictionaries with the input dataframes

        - This test verifies that the dictionary file (.kdic) created by
        khiops.sklearn contains information that is consistent with the
        input multitable dataset. Data is here provided through dataframes
        and its schema is "snowflake".
        """
        dataset_spec, label = self.create_fixture_dataset_spec(
            output_dir=None, data_type="df", multitable=True, schema="snowflake"
        )
        dataset = Dataset(dataset_spec, label)
        created_dictionary_domain = dataset.create_khiops_dictionary_domain()
        table_names = dataset_spec["tables"].keys()

        # assertions
        self.assertEqual(len(created_dictionary_domain.dictionaries), 5)
        for name in table_names:
            created_dictionary = created_dictionary_domain.get_dictionary(name)
            self.assertEqual(created_dictionary.name, name)
            if name == "A":
                self.assertEqual(created_dictionary.root, True)
                self.assertEqual(
                    created_dictionary.key[0], dataset_spec["tables"][name][1]
                )
            else:
                self.assertEqual(created_dictionary.root, False)
                self.assertEqual(
                    created_dictionary.key, dataset_spec["tables"][name][1]
                )

        created_main_dictionary_variable_types = {
            var.name: var.type
            for var in created_dictionary_domain.get_dictionary("A").variables
        }
        created_secondary_dictionary_variable_types_1 = {
            var.name: var.type
            for var in created_dictionary_domain.get_dictionary("B").variables
        }
        created_secondary_dictionary_variable_types_2 = {
            var.name: var.type
            for var in created_dictionary_domain.get_dictionary("C").variables
        }
        created_tertiary_dictionary_variable_types = {
            var.name: var.type
            for var in created_dictionary_domain.get_dictionary("D").variables
        }
        created_quaternary_dictionary_variable_types = {
            var.name: var.type
            for var in created_dictionary_domain.get_dictionary("E").variables
        }
        reference_dictionaries = self.get_reference_dictionaries(
            multitable=True, schema="snowflake"
        )
        reference_main_dictionary_variable_types = reference_dictionaries[0]
        reference_secondary_dictionary_variable_types_1 = reference_dictionaries[1]
        reference_secondary_dictionary_variable_types_2 = reference_dictionaries[2]
        reference_tertiary_dictionary_variable_types = reference_dictionaries[3]
        reference_quaternary_dictionary_variable_types = reference_dictionaries[4]

        # assertions
        self.assertEqual(
            created_main_dictionary_variable_types,
            reference_main_dictionary_variable_types,
        )
        self.assertEqual(
            created_secondary_dictionary_variable_types_1,
            reference_secondary_dictionary_variable_types_1,
        )
        self.assertEqual(
            created_secondary_dictionary_variable_types_2,
            reference_secondary_dictionary_variable_types_2,
        )
        self.assertEqual(
            created_tertiary_dictionary_variable_types,
            reference_tertiary_dictionary_variable_types,
        )
        self.assertEqual(
            created_quaternary_dictionary_variable_types,
            reference_quaternary_dictionary_variable_types,
        )

    def test_created_dictionary_from_data_files_multitable_snowflake(self):
        """Test consistency of the created dictionaries with the input data files

        - This test verifies that the dictionary file created by khiops.sklearn
        contains information that is consistent with the input multitable dataset.
        Data is here provided through data files and its schema is "snowflake".
        """
        dataset_spec, label = self.create_fixture_dataset_spec(
            output_dir=self.output_dir,
            data_type="file",
            multitable=True,
            schema="snowflake",
        )
        dataset = Dataset(dataset_spec, label)
        created_dictionary_domain = dataset.create_khiops_dictionary_domain()
        table_names = dataset_spec["tables"].keys()

        # assertions
        self.assertEqual(len(created_dictionary_domain.dictionaries), 5)
        for name in table_names:
            created_dictionary = created_dictionary_domain.get_dictionary(name)
            self.assertEqual(created_dictionary.name, name)

            if name == "A":
                self.assertEqual(created_dictionary.root, True)
                self.assertEqual(
                    created_dictionary.key[0], dataset_spec["tables"][name][1]
                )

            else:
                self.assertEqual(created_dictionary.root, False)
                self.assertEqual(
                    created_dictionary.key, dataset_spec["tables"][name][1]
                )

        created_main_dictionary_variable_types = {
            var.name: var.type
            for var in created_dictionary_domain.get_dictionary("A").variables
        }

        created_secondary_dictionary_variable_types_1 = {
            var.name: var.type
            for var in created_dictionary_domain.get_dictionary("B").variables
        }

        created_secondary_dictionary_variable_types_2 = {
            var.name: var.type
            for var in created_dictionary_domain.get_dictionary("C").variables
        }

        created_tertiary_dictionary_variable_types = {
            var.name: var.type
            for var in created_dictionary_domain.get_dictionary("D").variables
        }

        created_quaternary_dictionary_variable_types = {
            var.name: var.type
            for var in created_dictionary_domain.get_dictionary("E").variables
        }

        reference_dictionaries = self.get_reference_dictionaries(
            multitable=True, schema="snowflake"
        )
        reference_main_dictionary_variable_types = reference_dictionaries[0]
        reference_secondary_dictionary_variable_types_1 = reference_dictionaries[1]
        reference_secondary_dictionary_variable_types_2 = reference_dictionaries[2]
        reference_tertiary_dictionary_variable_types = reference_dictionaries[3]
        reference_quaternary_dictionary_variable_types = reference_dictionaries[4]

        # assertions
        self.assertEqual(
            created_main_dictionary_variable_types,
            reference_main_dictionary_variable_types,
        )
        self.assertEqual(
            created_secondary_dictionary_variable_types_1,
            reference_secondary_dictionary_variable_types_1,
        )
        self.assertEqual(
            created_secondary_dictionary_variable_types_2,
            reference_secondary_dictionary_variable_types_2,
        )
        self.assertEqual(
            created_tertiary_dictionary_variable_types,
            reference_tertiary_dictionary_variable_types,
        )
        self.assertEqual(
            created_quaternary_dictionary_variable_types,
            reference_quaternary_dictionary_variable_types,
        )
