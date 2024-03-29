######################################################################################
# Copyright (c) 2024 Orange. All rights reserved.                                    #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Tests for checking the output types of predictors"""
import unittest

from khiops.core.dictionary import DictionaryDomain
from khiops.core.helpers import build_multi_table_dictionary_domain


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
