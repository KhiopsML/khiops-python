###############################################################################
# Copyright (c) 2022 Orange - All Rights Reserved
# * This software is the confidential and proprietary information of Orange.
# * You shall not disclose such Restricted Information and shall use it only in
#   accordance with the terms of the license agreement you entered into with
#   Orange named the "Khiops - Python Library Evaluation License".
# * Unauthorized copying of this file, via any medium is strictly prohibited.
# * See the "LICENSE.md" file for more details.
###############################################################################
"""Test consistency of the created files with the input data"""
import os
import shutil
import unittest

import numpy as np
import pandas as pd

import pykhiops.core.filesystems as fs
from pykhiops.sklearn.tables import Dataset


class PyKhiopsConsistensyOfFilesAndDictionariesWithInputDataTests(unittest.TestCase):
    """Test consistency of the created files with the input data

    - The following tests allow to verify that:
        - The content of the .csv files (created by sklearn) is consistent with
        the content of the input data.
        - The content of the dictionaries (created by sklearn) is consistent with
        the content of the input data.
    - The input data used in the test is variable:
        - a monotable dataset (dataframe and file path).
        - a multitable dataset (a dictionnary with tables of type dataframe or
        file path).
        - Data contained in the datasets is unsorted.
        - Data contained in the datasets is multi-typed: numeric, categorical and
        dates.
    """

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

    def create_monotable_datafile(self, table_path):
        dataframe = self.create_monotable_dataframe()
        dataframe.to_csv(table_path, sep="\t", index=False)

    def create_multitable_dataframes(self):

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

        secondary_table_data = {
            "User_ID": np.random.choice(main_table["User_ID"], 20),
            "VAR_1": np.random.choice(["a", "b", "c", "d"], 20),
            "VAR_2": np.random.randint(low=1, high=20, size=20).astype("int64"),
            "VAR_3": np.random.choice([1, 0], 20).astype("int64"),
            "VAR_4": np.round(np.random.rand(1, 20)[0].tolist(), 2),
        }
        secondary_table = pd.DataFrame(secondary_table_data)

        return main_table, secondary_table

    def create_multitable_datafiles(self, main_table_path, secondary_table_path):
        main_table, secondary_table = self.create_multitable_dataframes()
        main_table.to_csv(main_table_path, sep="\t", index=False)
        secondary_table.to_csv(secondary_table_path, sep="\t", index=False)

    def test_created_file_from_dataframe_monotable(self):
        """Test consistency of the created datafile with the input dataframe"""
        output_dir = os.path.join(
            "resources", "tmp", "test_created_file_from_dataframe_monotable"
        )
        output_dir_res = fs.create_resource(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        reference_table = self.create_monotable_dataframe()
        features = reference_table.drop(["class"], axis=1)
        label = reference_table["class"]
        dataset = Dataset(features, label)
        created_table_path, _ = dataset.create_table_files_for_khiops(output_dir_res)
        created_table = pd.read_csv(created_table_path, sep="\t")
        # cast the type of column "Date" to datetime as pykhiops does not automatically
        # recognize dates
        created_table["Date"] = created_table["Date"].astype("datetime64")
        self.assertTrue(created_table.equals(reference_table))

        shutil.rmtree(output_dir, ignore_errors=True)

    def test_created_file_from_data_file_monotable(self):
        """Test consistency of the created datafile with the input datafile"""
        output_dir = os.path.join(
            "resources", "tmp", "test_created_file_from_data_file_monotable"
        )
        output_dir_res = fs.create_resource(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        reference_table_path = os.path.join(output_dir, "Reviews.csv")
        self.create_monotable_datafile(reference_table_path)
        reference_table = pd.read_csv(reference_table_path, sep="\t")

        dataset_spec = {
            "main_table": "Reviews",
            "tables": {"Reviews": (reference_table_path, "User_ID")},
            "format": ("\t", True),
        }

        dataset = Dataset(dataset_spec, "class")
        created_table_path, _ = dataset.create_table_files_for_khiops(output_dir_res)
        created_table = pd.read_csv(created_table_path, sep="\t")

        self.assertTrue(
            created_table.equals(
                reference_table.sort_values(by="User_ID").reset_index(drop=True)
            )
        )

        shutil.rmtree(output_dir, ignore_errors=True)

    def test_created_files_from_dataframes_multitable(self):
        """Test consistency of the created datafiles with the input dataframes"""
        output_dir = os.path.join(
            "resources",
            "tmp",
            "test_created_files_from_dataframes_multitable",
        )
        output_dir_res = fs.create_resource(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        (
            reference_main_table,
            reference_secondary_table,
        ) = self.create_multitable_dataframes()
        label_reference_main_table = reference_main_table["class"]
        features_reference_main_table = reference_main_table.drop("class", axis=1)

        dataset_spec = {
            "main_table": "id_class",
            "tables": {
                "id_class": (features_reference_main_table, "User_ID"),
                "logs": (reference_secondary_table, "User_ID"),
            },
        }
        dataset = Dataset(dataset_spec, label_reference_main_table)

        (
            main_table_path,
            secondary_table_paths,
        ) = dataset.create_table_files_for_khiops(output_dir_res)
        secondary_table_path = secondary_table_paths["logs"]
        created_main_table = pd.read_csv(main_table_path, sep="\t")
        created_secondary_table = pd.read_csv(secondary_table_path, sep="\t")

        self.assertTrue(
            created_main_table.equals(
                reference_main_table.sort_values(
                    by="User_ID", ascending=True
                ).reset_index(drop=True)
            )
        )

        self.assertTrue(
            created_secondary_table.sort_values(
                by=created_secondary_table.columns.tolist(), ascending=True
            )
            .reset_index(drop=True)
            .equals(
                reference_secondary_table.sort_values(
                    by=reference_secondary_table.columns.tolist(), ascending=True
                ).reset_index(drop=True)
            )
        )

        shutil.rmtree(output_dir, ignore_errors=True)

    def test_created_files_from_data_files_multitable(self):
        """Test consistency of the created datafiles with the input datafiles"""
        output_dir = os.path.join(
            "resources",
            "tmp",
            "test_created_files_from_data_files_multitable",
        )
        output_dir_res = fs.create_resource(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        reference_main_table_path = os.path.join(output_dir, "id_class.csv")
        reference_secondary_table_path = os.path.join(output_dir, "logs.csv")
        self.create_multitable_datafiles(
            reference_main_table_path, reference_secondary_table_path
        )
        reference_main_table = pd.read_csv(reference_main_table_path, sep="\t")
        reference_secondary_table = pd.read_csv(
            reference_secondary_table_path, sep="\t"
        )

        dataset_spec = {
            "main_table": "id_class",
            "tables": {
                "id_class": (reference_main_table_path, "User_ID"),
                "logs": (reference_secondary_table_path, "User_ID"),
            },
            "format": ("\t", True),
        }

        dataset = Dataset(dataset_spec, "class")
        main_table_path, dico_secondary_table = dataset.create_table_files_for_khiops(
            output_dir_res
        )
        secondary_table_path = dico_secondary_table["logs"]
        created_main_table = pd.read_csv(main_table_path, sep="\t")
        created_secondary_table = pd.read_csv(secondary_table_path, sep="\t")

        self.assertTrue(
            created_main_table.equals(
                reference_main_table.sort_values(
                    by="User_ID", ascending=True
                ).reset_index(drop=True)
            )
        )

        self.assertTrue(
            created_secondary_table.sort_values(
                by=created_secondary_table.columns.tolist(), ascending=True
            )
            .reset_index(drop=True)
            .equals(
                reference_secondary_table.sort_values(
                    by=reference_secondary_table.columns.tolist(), ascending=True
                ).reset_index(drop=True)
            )
        )

        shutil.rmtree(output_dir, ignore_errors=True)

    def test_created_dictionary_from_dataframe_monotable(self):
        """Test consistency of the created dictionary with the input dataframe"""
        table = self.create_monotable_dataframe()
        features = table.drop(["class"], axis=1)
        label = table["class"]
        dataset_spec = {
            "main_table": "Reviews",
            "tables": {"Reviews": (features, ["User_ID"])},
            "format": ("\t", True),
        }

        dataset = Dataset(dataset_spec, label)
        created_dictionary_domain = dataset.create_khiops_dictionary_domain()
        self.assertEqual(len(created_dictionary_domain.dictionaries), 1)
        created_dictionary = created_dictionary_domain.dictionaries[0]
        self.assertEqual(created_dictionary.name, "Reviews")
        self.assertEqual(created_dictionary.root, False)
        self.assertEqual(len(created_dictionary.key), 1)

        created_dictionary_variable_types = {
            var.name: var.type for var in created_dictionary.variables
        }

        reference_dictionary_variable_types = {
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

        self.assertEqual(
            created_dictionary_variable_types, reference_dictionary_variable_types
        )

    def test_created_dictionary_from_data_file_monotable(self):
        """Test consistency of the created dictionary with the input datafile"""
        output_dir = os.path.join(
            "resources", "tmp", "test_created_file_from_data_file_monotable"
        )
        os.makedirs(output_dir, exist_ok=True)

        table_path = os.path.join(output_dir, "Reviews.csv")
        self.create_monotable_datafile(table_path)

        dataset_spec = {
            "main_table": "Reviews",
            "tables": {"Reviews": (table_path, ["User_ID"])},
            "format": ("\t", True),
        }

        dataset = Dataset(dataset_spec, "class")
        created_dictionary_domain = dataset.create_khiops_dictionary_domain()
        self.assertEqual(len(created_dictionary_domain.dictionaries), 1)
        created_dictionary = created_dictionary_domain.dictionaries[0]
        self.assertEqual(created_dictionary.name, "Reviews")
        self.assertEqual(created_dictionary.root, False)
        self.assertEqual(len(created_dictionary.key), 1)

        created_dictionary_variable_types = {
            var.name: var.type for var in created_dictionary.variables
        }

        reference_dictionary_variable_types = {
            "User_ID": "Categorical",
            "Age": "Numerical",
            "Clothing ID": "Numerical",
            "Date": "Categorical",
            "New": "Categorical",
            "Title": "Categorical",
            "Recommended IND": "Numerical",
            "Positive Feedback average": "Numerical",
            "class": "Categorical",
        }

        self.assertEqual(
            created_dictionary_variable_types, reference_dictionary_variable_types
        )
        shutil.rmtree(output_dir, ignore_errors=True)

    def test_created_dictionary_from_dataframe_multitable(self):
        """Test consistency of the created dictionaries with the input dataframes"""
        main_table, secondary_table = self.create_multitable_dataframes()
        label_main_table = main_table["class"]
        features_main_table = main_table.drop("class", axis=1)

        dataset_spec = {
            "main_table": "id_class",
            "tables": {
                "id_class": (features_main_table, "User_ID"),
                "logs": (secondary_table, "User_ID"),
            },
        }

        dataset = Dataset(dataset_spec, label_main_table)
        created_dictionary_domain = dataset.create_khiops_dictionary_domain()
        self.assertEqual(len(created_dictionary_domain.dictionaries), 2)
        created_main_dictionary = created_dictionary_domain.dictionaries[0]
        created_secondary_dictionary = created_dictionary_domain.dictionaries[1]
        self.assertEqual(created_main_dictionary.name, "id_class")
        self.assertEqual(created_secondary_dictionary.name, "logs")
        self.assertEqual(created_main_dictionary.root, True)
        self.assertEqual(created_secondary_dictionary.root, False)
        self.assertEqual(created_main_dictionary.key[0], "User_ID")

        created_main_dictionary_variable_types = {
            var.name: var.type for var in created_main_dictionary.variables
        }

        reference_main_dictionary_variable_types = {
            "User_ID": "Categorical",
            "class": "Categorical",
            "logs": "Table",
        }

        created_secondary_dictionary_variable_types = {
            var.name: var.type for var in created_secondary_dictionary.variables
        }

        reference_secondary_dictionary_variable_types = {
            "User_ID": "Categorical",
            "VAR_1": "Categorical",
            "VAR_2": "Numerical",
            "VAR_3": "Numerical",
            "VAR_4": "Numerical",
        }

        self.assertEqual(
            created_main_dictionary_variable_types,
            reference_main_dictionary_variable_types,
        )

        self.assertEqual(
            created_secondary_dictionary_variable_types,
            reference_secondary_dictionary_variable_types,
        )

    def test_created_dictionary_from_data_files_multitable(self):
        """Test consistency of the created dictionaries with the input datafiles"""
        output_dir = os.path.join(
            "resources",
            "tmp",
            "test_created_files_from_datafiles_multitable",
        )
        os.makedirs(output_dir, exist_ok=True)

        main_table_path = os.path.join(output_dir, "id_class.csv")
        secondary_table_path = os.path.join(output_dir, "logs.csv")
        self.create_multitable_datafiles(main_table_path, secondary_table_path)

        dataset_spec = {
            "main_table": "id_class",
            "tables": {
                "id_class": (main_table_path, "User_ID"),
                "logs": (secondary_table_path, "User_ID"),
            },
            "format": ("\t", True),
        }

        dataset = Dataset(dataset_spec, "class")
        created_dictionary_domain = dataset.create_khiops_dictionary_domain()

        self.assertEqual(len(created_dictionary_domain.dictionaries), 2)
        created_main_dictionary = created_dictionary_domain.dictionaries[0]
        created_secondary_dictionary = created_dictionary_domain.dictionaries[1]
        self.assertEqual(created_main_dictionary.name, "id_class")
        self.assertEqual(created_secondary_dictionary.name, "logs")
        self.assertEqual(created_main_dictionary.root, True)
        self.assertEqual(created_secondary_dictionary.root, False)
        self.assertEqual(created_main_dictionary.key[0], "User_ID")

        created_main_dictionary_variable_types = {
            var.name: var.type for var in created_main_dictionary.variables
        }

        reference_main_dictionary_variable_types = {
            "User_ID": "Categorical",
            "class": "Categorical",
            "logs": "Table",
        }

        created_secondary_dictionary_variable_types = {
            var.name: var.type for var in created_secondary_dictionary.variables
        }

        reference_secondary_dictionary_variable_types = {
            "User_ID": "Categorical",
            "VAR_1": "Categorical",
            "VAR_2": "Numerical",
            "VAR_3": "Numerical",
            "VAR_4": "Numerical",
        }
        self.assertEqual(
            created_main_dictionary_variable_types,
            reference_main_dictionary_variable_types,
        )

        self.assertEqual(
            created_secondary_dictionary_variable_types,
            reference_secondary_dictionary_variable_types,
        )

        shutil.rmtree(output_dir, ignore_errors=True)
