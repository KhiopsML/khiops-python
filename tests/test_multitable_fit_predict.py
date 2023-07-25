######################################################################################
# Copyright (c) 2023 Orange. All rights reserved.                                    #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Tests for executing fit multiple times on multi-table data"""

import os
import shutil
import unittest

import khiops.core as kh
from khiops.sklearn.estimators import KhiopsClassifier
from tests.test_helper import KhiopsTestHelper


class KhiopsMultitableFitTests(unittest.TestCase, KhiopsTestHelper):
    """Test if Khiops estimator can be fitted on multi-table data"""

    def setUp(self):
        if "UNITTEST_ONLY_SHORT_TESTS" in os.environ:
            if os.environ["UNITTEST_ONLY_SHORT_TESTS"].lower() == "true":
                self.skipTest("Skipping long test")

    def test_estimator_multiple_create_and_fit_does_not_raise_exception(self):
        """Test if estimator can be fitted from paths several times"""
        # Set upt the file based dataset
        dataset_name = "SpliceJunction"
        samples_dir = kh.get_runner().samples_dir
        dataset = {
            "main_table": "SpliceJunction",
            "tables": {
                "SpliceJunction": (
                    os.path.join(samples_dir, dataset_name, "SpliceJunction.txt"),
                    "SampleId",
                ),
                "SpliceJunctionDNA": (
                    os.path.join(samples_dir, dataset_name, "SpliceJunctionDNA.txt"),
                    "SampleId",
                ),
            },
            "format": ("\t", True),
        }

        # Train classifier
        output_dir = os.path.join("resources", "tmp", "test_multitable_fit_predict")
        try:
            for _ in range(2):
                KhiopsTestHelper.fit_helper(
                    KhiopsClassifier,
                    data=(dataset, "Class"),
                    pickled=False,
                    output_dir=output_dir,
                )
        # Remove data files created during the test
        finally:
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
