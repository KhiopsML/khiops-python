######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Tests for checking the output types of predictors"""
import io
import os
import pathlib
import platform
import stat
import unittest
import warnings

import pandas as pd

from khiops.core.dictionary import DictionaryDomain
from khiops.core.helpers import build_multi_table_dictionary_domain, visualize_report
from khiops.sklearn import train_test_split_dataset


class KhiopsHelperFunctions(unittest.TestCase):
    """Tests for checking the behaviour of the helper functions"""

    @staticmethod
    def _build_monotable_domain_specification():
        return {
            "tool": "Khiops Dictionary",
            "version": "10.0",
            "khiops_encoding": "ascii",
            "dictionaries": [
                {
                    "name": "SpliceJunctionDNA",
                    "key": ["SampleId"],
                    "root": False,
                    "variables": [
                        {"name": "SampleId", "type": "Categorical"},
                        {"name": "Pos", "type": "Numerical"},
                        {"name": "Char", "type": "Categorical"},
                    ],
                }
            ],
        }

    def test_build_multi_table_dictionary_domain_deprecation(self):
        """Test that `core.helpers.build_multi_table_dictionary_domain` raises
        deprecation warning
        """
        # Build monotable_domain, with one dictionary, holding three variables
        monotable_domain_specification = (
            KhiopsHelperFunctions._build_monotable_domain_specification()
        )
        monotable_domain = DictionaryDomain(monotable_domain_specification)

        # Build multi-table dictionary domain from the montable dictionary domain
        with warnings.catch_warnings(record=True) as warning_list:
            build_multi_table_dictionary_domain(
                monotable_domain,
                "A_Prefix_SpliceJunctionDNA",
                "A_Name_SpliceJunctionDNA",
            )

        self.assertEqual(len(warning_list), 1)
        warning = warning_list[0]
        self.assertTrue(issubclass(warning.category, UserWarning))
        warning_message = warning.message
        self.assertEqual(len(warning_message.args), 1)
        message = warning_message.args[0]
        self.assertTrue(
            "'build_multi_table_dictionary_domain'" in message
            and "deprecated" in message
        )

    def test_build_multi_table_dictionary_domain(self):
        """Test that the multi-table dictionary domain built is as expected"""
        # Build monotable_domain, with one dictionary, holding three variables
        monotable_domain_specification = (
            KhiopsHelperFunctions._build_monotable_domain_specification()
        )
        monotable_domain = DictionaryDomain(monotable_domain_specification)

        # Build reference multi-table domain, with two dictionaries, one root
        ref_multi_table_domain_specification = {
            "tool": "Khiops Dictionary",
            "version": "10.0",
            "khiops_encoding": "ascii",
            "dictionaries": [
                {
                    "name": "A_Prefix_SpliceJunctionDNA",
                    "key": ["SampleId"],
                    "root": True,
                    "variables": [
                        {"name": "SampleId", "type": "Categorical"},
                        {
                            "name": "A_Name_SpliceJunctionDNA",
                            "type": "Table",
                            "object_type": "Table(SpliceJunctionDNA)",
                        },
                    ],
                },
                {
                    "name": "SpliceJunctionDNA",
                    "key": ["SampleId"],
                    "root": False,
                    "variables": [
                        {"name": "SampleId", "type": "Categorical"},
                        {"name": "Pos", "type": "Numerical"},
                        {"name": "Char", "type": "Categorical"},
                    ],
                },
            ],
        }
        ref_multi_table_domain = DictionaryDomain(ref_multi_table_domain_specification)

        # Build multi-table dictionary domain from the montable dictionary domain
        multi_table_domain = build_multi_table_dictionary_domain(
            monotable_domain, "A_Prefix_SpliceJunctionDNA", "A_Name_SpliceJunctionDNA"
        )

        # Check that multi-table domain contents are identical
        self.assertEqual(
            len(multi_table_domain.dictionaries),
            len(ref_multi_table_domain.dictionaries),
        )
        for test_dict, ref_dict in zip(
            multi_table_domain.dictionaries, ref_multi_table_domain.dictionaries
        ):
            self.assertEqual(test_dict.name, ref_dict.name)
            self.assertEqual(test_dict.root, ref_dict.root)
            self.assertEqual(len(test_dict.key), len(ref_dict.key))
            for test_key, ref_key in zip(test_dict.key, ref_dict.key):
                self.assertEqual(test_key, ref_key)
            self.assertEqual(len(test_dict.variables), len(ref_dict.variables))
            for test_var, ref_var in zip(test_dict.variables, ref_dict.variables):
                self.assertEqual(test_var.name, ref_var.name)
                self.assertEqual(test_var.type, ref_var.type)

    def test_train_test_split_dataset_dataframe(self):
        """Tests that the train_test_split_dataset function works for df datasets"""
        # Create the fixture dataset
        clients_df = pd.read_csv(io.StringIO(CLIENTS_CSV))
        calls_df = pd.read_csv(io.StringIO(CALLS_CSV))
        connections_df = pd.read_csv(io.StringIO(CONNECTIONS_CSV))
        ds_spec = {
            "main_table": "clients",
            "tables": {
                "clients": (clients_df.drop("class", axis=1), ["id"]),
                "calls": (calls_df, ["id", "call_id"]),
                "connections": (connections_df, ["id", "call_id"]),
            },
            "relations": [("clients", "calls", False), ("calls", "connections", False)],
        }
        y = clients_df["class"]

        # Execute the train/test split function
        ds_spec_train, ds_spec_test, y_train, y_test = train_test_split_dataset(
            ds_spec, y, test_size=0.5, random_state=31614
        )

        # Check that the target are the same as the reference
        ref_y_train = pd.read_csv(io.StringIO(TRAIN_DF_TARGET_CSV))["class"]
        ref_y_test = pd.read_csv(io.StringIO(TEST_DF_TARGET_CSV))["class"]
        self._assert_series_equal(ref_y_train, y_train.reset_index()["class"])
        self._assert_series_equal(ref_y_test, y_test.reset_index()["class"])

        # Check that the dataset spec structure is the same
        self._assert_dataset_keeps_structure(ds_spec_train, ds_spec)
        self._assert_dataset_keeps_structure(ds_spec_test, ds_spec)

        # Check that the table contents match those of the references
        split_ds_specs = {
            "train": ds_spec_train,
            "test": ds_spec_test,
        }
        ref_table_dfs = {
            "train": {
                "clients": pd.read_csv(io.StringIO(TRAIN_DF_CLIENTS_CSV)),
                "calls": pd.read_csv(io.StringIO(TRAIN_DF_CALLS_CSV)),
                "connections": pd.read_csv(io.StringIO(TRAIN_DF_CONNECTIONS_CSV)),
            },
            "test": {
                "clients": pd.read_csv(io.StringIO(TEST_DF_CLIENTS_CSV)),
                "calls": pd.read_csv(io.StringIO(TEST_DF_CALLS_CSV)),
                "connections": pd.read_csv(io.StringIO(TEST_DF_CONNECTIONS_CSV)),
            },
        }
        for split, ref_tables in ref_table_dfs.items():
            for table_name in ds_spec["tables"]:
                with self.subTest(split=split, table_name=table_name):
                    self._assert_frame_equal(
                        split_ds_specs[split]["tables"][table_name][0].reset_index(
                            drop=True
                        ),
                        ref_tables[table_name].reset_index(drop=True),
                    )

    def _assert_dataset_keeps_structure(self, ds_spec, ref_ds_spec):
        """Asserts that the input dataset has the same structure as the reference

        It does not check the contents of the tables.
        """
        # Check that the spec dictionary is the same excluding the tables
        self.assertIn("main_table", ref_ds_spec)
        self.assertIn("tables", ref_ds_spec)
        self.assertIn("relations", ref_ds_spec)
        self.assertEqual(ds_spec["main_table"], ref_ds_spec["main_table"])
        self.assertEqual(ds_spec["relations"], ref_ds_spec["relations"])
        self.assertEqual(ds_spec["tables"].keys(), ref_ds_spec["tables"].keys())
        if "format" in ref_ds_spec:
            self.assertIn("format", ds_spec)
            self.assertEqual(ds_spec["format"], ref_ds_spec["format"])

        # Check that the table keys are equal
        for table_name, table_spec in ds_spec["tables"].items():
            self.assertEqual(table_spec[1], ref_ds_spec["tables"][table_name][1])

    def _assert_frame_equal(self, ref_df, out_df):
        """Wrapper for the assert_frame_equal pandas function

        In case of failure of assert_frame_equal we capture the AssertionError thrown by
        it and make a unittest call to fail. This reports the error found by
        assert_frame_equal while avoiding a double thrown exception.
        """
        failure_error = None
        try:
            pd.testing.assert_frame_equal(ref_df, out_df)
        except AssertionError as error:
            failure_error = error
        if failure_error is not None:
            self.fail(failure_error)

    def _assert_series_equal(self, ref_series, out_series):
        """Wrapper for the assert_frame_equal pandas function

        In case of failure of assert_frame_equal we capture the AssertionError thrown by
        it and make a unittest call to fail. This reports the error found by
        assert_frame_equal while avoiding a double thrown exception.
        """
        failure_error = None
        try:
            pd.testing.assert_series_equal(ref_series, out_series)
        except AssertionError as error:
            failure_error = error
        if failure_error is not None:
            self.fail(failure_error)

    def test_visualize_report_fails_on_improper_file_extensions(self):
        """Tests that visualize_report fails on files without extension .khj or .khcj"""
        tmp_report_path = "report.json"
        with self.assertRaises(ValueError) as ctx:
            visualize_report(tmp_report_path)
        self.assertIn("must have extension '.khj' or '.khcj'", str(ctx.exception))
        self.assertIn(tmp_report_path, str(ctx.exception))

    def test_visualize_report_fails_on_inexistent_file(self):
        """Test that visualize_report fails on inexistent file"""
        tmp_report_path = "INEXISTENT_REPORT.khj"
        with self.assertRaises(FileNotFoundError) as ctx:
            visualize_report(tmp_report_path)
        self.assertIn(tmp_report_path, str(ctx.exception))

    @unittest.skipIf(platform.system() == "Windows", "Test non-applicable in Windows")
    def test_visualize_report_fails_on_file_with_executable_permissions(self):
        """Test that visualize_report fails on files with executable permissions"""
        # Create a temporary report file with executable permissions
        tmp_report_path = pathlib.Path("./TEMPORARY_REPORT_ABCDEFG_123.khj")
        tmp_report_path.touch()
        os.chmod(tmp_report_path, os.stat(tmp_report_path).st_mode | stat.S_IEXEC)

        # Check that the exception is raised with the proper message
        with self.assertRaises(RuntimeError) as ctx:
            visualize_report(tmp_report_path)
        self.assertIn("Report file cannot be executable", str(ctx.exception))
        self.assertIn(str(tmp_report_path), str(ctx.exception))

        # Remove the temporary file
        os.remove(tmp_report_path)


# pylint: disable=line-too-long
# fmt: off

# Test data

CLIENTS_CSV = """
id,name,phone,email,address,numberrange,time,date,class
1,Hakeem Wilkinson,1-352-535-7028,at.pete@outlook.org,247-2921 Elit. Rd.,2,3:02 PM,"May 1, 2024",1
10,Axel Holman,1-340-743-8860,est@google.com,Ap #737-7185 Donec St.,9,1:17 PM,"Jan 8, 2025",0
13,Armando Cleveland,(520) 285-3188,amet.consectetuer@icloud.edu,Ap #167-1519 Tempus Avenue,8,1:50 PM,"Jul 24, 2024",0
4,Edward Miles,(959) 886-5744,in.nec@outlook.edu,2184 Gravida Road,6,10:02 PM,"Mar 30, 2025",1
7,Aurora Valentine,1-838-806-6257,etiam.gravida.molestie@yahoo.com,Ap #923-3118 Ante Ave,8,4:02 AM,"Dec 12, 2023",1
""".lstrip()

CALLS_CSV = """
id,call_id,duration
1,1,38
1,20,29
10,2,7
13,25,329
13,3,1
13,30,8
4,14,48
4,2,543
7,4,339
""".lstrip()

CONNECTIONS_CSV = """
id,call_id,connection_ip
1,1,277.1.56.30
1,1,147.43.67.35
1,1,164.27.26.50
1,20,199.44.70.12
1,20,169.51.97.96
10,2,170.05.79.41
10,2,118.45.57.51
13,25,193.23.02.67
13,25,146.74.18.88
13,25,118.41.87.47
13,25,161.51.79.60
13,3,115.45.02.58
13,30,12.115.90.93
4,14,16.56.66.16
4,14,19.30.36.57
4,14,15.16.40.67
4,2,10.189.71.73
4,2,10.6.76.93
7,4,16.66.64.13
7,4,15.13.69.18
""".lstrip()

UNSORTED_CLIENTS_CSV = """
id,name,phone,email,address,numberrange,time,date,class
13,Armando Cleveland,(520) 285-3188,amet.consectetuer@icloud.edu,Ap #167-1519 Tempus Avenue,8,1:50 PM,"Jul 24, 2024",0
10,Axel Holman,1-340-743-8860,est@google.com,Ap #737-7185 Donec St.,9,1:17 PM,"Jan 8, 2025",0
1,Hakeem Wilkinson,1-352-535-7028,at.pete@outlook.org,247-2921 Elit. Rd.,2,3:02 PM,"May 1, 2024",1
7,Aurora Valentine,1-838-806-6257,etiam.gravida.molestie@yahoo.com,Ap #923-3118 Ante Ave,8,4:02 AM,"Dec 12, 2023",1
4,Edward Miles,(959) 886-5744,in.nec@outlook.edu,2184 Gravida Road,6,10:02 PM,"Mar 30, 2025",1
""".lstrip()

UNSORTED_CALLS_CSV = """
id,call_id,duration
1,1,38
10,2,7
13,25,329
4,2,543
13,30,8
13,3,1
4,14,48
1,20,29
7,4,339
""".lstrip()

UNSORTED_CONNECTIONS_CSV = """
id,call_id,connection_ip
13,25,193.23.02.67
1,1,277.1.56.30
4,14,16.56.66.16
13,25,146.74.18.88
13,25,118.41.87.47
1,1,147.43.67.35
4,14,19.30.36.57
1,20,199.44.70.12
10,2,170.05.79.41
1,20,169.51.97.96
10,2,118.45.57.51
13,25,161.51.79.60
13,3,115.45.02.58
4,14,15.16.40.67
1,1,164.27.26.50
7,4,16.66.64.13
13,30,12.115.90.93
7,4,15.13.69.18
4,2,10.189.71.73
4,2,10.6.76.93
""".lstrip()

TRAIN_DF_CLIENTS_CSV = """
id,name,phone,email,address,numberrange,time,date
7,Aurora Valentine,1-838-806-6257,etiam.gravida.molestie@yahoo.com,Ap #923-3118 Ante Ave,8,4:02 AM,"Dec 12, 2023"
13,Armando Cleveland,(520) 285-3188,amet.consectetuer@icloud.edu,Ap #167-1519 Tempus Avenue,8,1:50 PM,"Jul 24, 2024"
""".lstrip()

TRAIN_DF_CALLS_CSV = """
id,call_id,duration
7,4,339
13,25,329
13,3,1
13,30,8
""".lstrip()

TRAIN_DF_TARGET_CSV = """
class
1
0
""".lstrip()

TRAIN_DF_CONNECTIONS_CSV = """
id,call_id,connection_ip
7,4,16.66.64.13
7,4,15.13.69.18
13,25,193.23.02.67
13,25,146.74.18.88
13,25,118.41.87.47
13,25,161.51.79.60
13,3,115.45.02.58
13,30,12.115.90.93
""".lstrip()


TEST_DF_CLIENTS_CSV = """
id,name,phone,email,address,numberrange,time,date
4,Edward Miles,(959) 886-5744,in.nec@outlook.edu,2184 Gravida Road,6,10:02 PM,"Mar 30, 2025"
10,Axel Holman,1-340-743-8860,est@google.com,Ap #737-7185 Donec St.,9,1:17 PM,"Jan 8, 2025"
1,Hakeem Wilkinson,1-352-535-7028,at.pete@outlook.org,247-2921 Elit. Rd.,2,3:02 PM,"May 1, 2024"
""".lstrip()

TEST_DF_TARGET_CSV = """
class
1
0
1
""".lstrip()


TEST_DF_CALLS_CSV = """
id,call_id,duration
4,14,48
4,2,543
10,2,7
1,1,38
1,20,29
""".lstrip()

TEST_DF_CONNECTIONS_CSV = """
id,call_id,connection_ip
4,14,16.56.66.16
4,14,19.30.36.57
4,14,15.16.40.67
4,2,10.189.71.73
4,2,10.6.76.93
10,2,170.05.79.41
10,2,118.45.57.51
1,1,277.1.56.30
1,1,147.43.67.35
1,1,164.27.26.50
1,20,199.44.70.12
1,20,169.51.97.96
""".lstrip()

TRAIN_FILE_CLIENTS_CSV = """
id,name,phone,email,address,numberrange,time,date,class
10,Axel Holman,1-340-743-8860,est@google.com,Ap #737-7185 Donec St.,9,1:17 PM,"Jan 8, 2025",0
13,Armando Cleveland,(520) 285-3188,amet.consectetuer@icloud.edu,Ap #167-1519 Tempus Avenue,8,1:50 PM,"Jul 24, 2024",0
4,Edward Miles,(959) 886-5744,in.nec@outlook.edu,2184 Gravida Road,6,10:02 PM,"Mar 30, 2025",1
""".lstrip()

TRAIN_FILE_CALLS_CSV = """
id,call_id,duration
10,2,7
13,25,329
13,3,1
13,30,8
4,14,48
4,2,543
""".lstrip()

TRAIN_FILE_CONNECTIONS_CSV = """
id,call_id,connection_ip
10,2,170.05.79.41
10,2,118.45.57.51
13,25,193.23.02.67
13,25,146.74.18.88
13,25,118.41.87.47
13,25,161.51.79.60
13,3,115.45.02.58
13,30,12.115.90.93
4,14,16.56.66.16
4,14,19.30.36.57
4,14,15.16.40.67
4,2,10.189.71.73
4,2,10.6.76.93
""".lstrip()


TEST_FILE_CLIENTS_CSV = """
id,name,phone,email,address,numberrange,time,date,class
1,Hakeem Wilkinson,1-352-535-7028,at.pete@outlook.org,247-2921 Elit. Rd.,2,3:02 PM,"May 1, 2024",1
7,Aurora Valentine,1-838-806-6257,etiam.gravida.molestie@yahoo.com,Ap #923-3118 Ante Ave,8,4:02 AM,"Dec 12, 2023",1
""".lstrip()

TEST_FILE_CALLS_CSV = """
id,call_id,duration
1,1,38
1,20,29
7,4,339
""".lstrip()

TEST_FILE_CONNECTIONS_CSV = """
id,call_id,connection_ip
1,1,277.1.56.30
1,1,147.43.67.35
1,1,164.27.26.50
1,20,199.44.70.12
1,20,169.51.97.96
7,4,16.66.64.13
7,4,15.13.69.18
""".lstrip()
