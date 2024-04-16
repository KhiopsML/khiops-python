######################################################################################
# Copyright (c) 2024 Orange. All rights reserved.                                    #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Tests for checking the output types of predictors"""
import contextlib
import io
import tempfile
import unittest

import pandas as pd

from khiops.core.dictionary import DictionaryDomain
from khiops.core.helpers import build_multi_table_dictionary_domain
from khiops.utils.helpers import sort_dataset


class KhiopsHelperFunctions(unittest.TestCase):
    """Tests for checking the behaviour of the helper functions"""

    def test_build_multi_table_dictionary_domain(self):
        """Test that the multi-table dictionary domain built is as expected"""
        # Build monotable_domain, with one dictionary, holding three variables
        monotable_domain_specification = {
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

    def test_sort_dataset_dataframe(self):
        """Tests that the sort_dataset function works for dataframe datasets"""
        # Create the fixture dataset
        clients_df = pd.read_csv(io.StringIO(UNSORTED_TEST_CLIENTS_CSV))
        calls_df = pd.read_csv(io.StringIO(UNSORTED_TEST_CALLS_CSV))
        ds_spec = {
            "main_table": "clients",
            "tables": {
                "clients": (clients_df, ["id"]),
                "calls": (calls_df, ["id", "call_id"]),
            },
            "relations": [("clients", "calls", False)],
        }

        # Call the sort_dataset function
        sorted_ds_spec = sort_dataset(ds_spec)
        ref_sorted_table_dfs = {
            "clients": pd.read_csv(io.StringIO(TEST_CLIENTS_CSV)),
            "calls": pd.read_csv(io.StringIO(TEST_CALLS_CSV)),
        }

        # Check that the structure of the sorted dataset
        self._assert_sorted_dataset_keeps_structure(ds_spec, sorted_ds_spec)

        # Check that the table specs are the equivalent and the tables are sorted
        for table_name in ds_spec["tables"]:
            # Check that the dataframes are equal (ignoring the index)
            self._assert_frame_equal(
                ref_sorted_table_dfs[table_name].reset_index(drop=True),
                sorted_ds_spec["tables"][table_name][0].reset_index(drop=True),
            )

    def test_sort_dataset_file(self):
        """Tests that the sort_dataset function works for file datasets"""
        # Create a execution context with temporary files and directories
        with contextlib.ExitStack() as exit_stack:
            # Create temporary files and a temporary directory
            clients_csv_file = exit_stack.enter_context(tempfile.NamedTemporaryFile())
            calls_csv_file = exit_stack.enter_context(tempfile.NamedTemporaryFile())
            tmp_dir = exit_stack.enter_context(tempfile.TemporaryDirectory())

            # Create the fixture dataset
            clients_csv_file.write(bytes(UNSORTED_TEST_CLIENTS_CSV, encoding="utf8"))
            calls_csv_file.write(bytes(UNSORTED_TEST_CALLS_CSV, encoding="utf8"))
            clients_csv_file.flush()
            calls_csv_file.flush()
            ds_spec = {
                "main_table": "clients",
                "tables": {
                    "clients": (clients_csv_file.name, ["id"]),
                    "calls": (calls_csv_file.name, ["id", "call_id"]),
                },
                "relations": [("clients", "calls", False)],
                "format": (",", True),
            }

            # Call the sort_dataset function
            sorted_ds_spec = sort_dataset(ds_spec, output_dir=tmp_dir)

            # Check that the structure of the sorted dataset
            self._assert_sorted_dataset_keeps_structure(ds_spec, sorted_ds_spec)

            # Check that the table specs are the equivalent and the tables are sorted
            ref_sorted_tables = {"clients": TEST_CLIENTS_CSV, "calls": TEST_CALLS_CSV}
            for table_name, _ in ds_spec["tables"].items():
                # Read the contents of the sorted table to a list of strings
                sorted_table_spec = sorted_ds_spec["tables"][table_name]
                sorted_table_file = exit_stack.enter_context(
                    open(sorted_table_spec[0], encoding="ascii")
                )
                sorted_table = sorted_table_file.readlines()

                # Transform the reference table string to a list of strings
                ref_sorted_table = ref_sorted_tables[table_name].splitlines(
                    keepends=True
                )

                # Check that the sorted table is equal to the reference
                self.assertEqual(ref_sorted_table, sorted_table)

    def _assert_sorted_dataset_keeps_structure(self, ds_spec, sorted_ds_spec):
        """Asserts that the sorted dataset keeps the structure of the input dataset

        It does not check the contents of the tables.
        """
        # Check that the spec dictionary is the same excluding the tables
        self.assertIn("main_table", sorted_ds_spec)
        self.assertIn("tables", sorted_ds_spec)
        self.assertIn("relations", sorted_ds_spec)
        self.assertEqual(ds_spec["main_table"], sorted_ds_spec["main_table"])
        self.assertEqual(ds_spec["relations"], sorted_ds_spec["relations"])
        self.assertEqual(ds_spec["tables"].keys(), sorted_ds_spec["tables"].keys())

        # Check that the table keys are equal
        for table_name, table_spec in ds_spec["tables"].items():
            self.assertEqual(table_spec[1], sorted_ds_spec["tables"][table_name][1])

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


# pylint: disable=line-too-long
# fmt: off
TEST_CLIENTS_CSV = """
id,name,phone,email,address,numberrange,time,date
1,Hakeem Wilkinson,1-352-535-7028,at.pede@outlook.org,247-2921 Elit. Rd.,2,3:02 PM,"May 1, 2024"
10,Axel Holman,1-340-743-8860,est@google.com,Ap #737-7185 Donec St.,9,1:17 PM,"Jan 8, 2025"
13,Armando Cleveland,(520) 285-3188,amet.consectetuer@icloud.edu,Ap #167-1519 Tempus Avenue,8,1:50 PM,"Jul 24, 2024"
4,Edward Miles,(959) 886-5744,in.nec@outlook.edu,2184 Gravida Road,6,10:02 PM,"Mar 30, 2025"
7,Aurora Valentine,1-838-806-6257,etiam.gravida.molestie@yahoo.com,Ap #923-3118 Ante Ave,8,4:02 AM,"Dec 12, 2023"
""".lstrip()

TEST_CALLS_CSV = """
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

UNSORTED_TEST_CLIENTS_CSV = """
id,name,phone,email,address,numberrange,time,date
1,Hakeem Wilkinson,1-352-535-7028,at.pede@outlook.org,247-2921 Elit. Rd.,2,3:02 PM,"May 1, 2024"
13,Armando Cleveland,(520) 285-3188,amet.consectetuer@icloud.edu,Ap #167-1519 Tempus Avenue,8,1:50 PM,"Jul 24, 2024"
7,Aurora Valentine,1-838-806-6257,etiam.gravida.molestie@yahoo.com,Ap #923-3118 Ante Ave,8,4:02 AM,"Dec 12, 2023"
4,Edward Miles,(959) 886-5744,in.nec@outlook.edu,2184 Gravida Road,6,10:02 PM,"Mar 30, 2025"
10,Axel Holman,1-340-743-8860,est@google.com,Ap #737-7185 Donec St.,9,1:17 PM,"Jan 8, 2025"
""".lstrip()

UNSORTED_TEST_CALLS_CSV = """
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
