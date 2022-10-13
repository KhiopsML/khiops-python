###############################################################################
# Copyright (c) 2022 Orange - All Rights Reserved
# * This software is the confidential and proprietary information of Orange.
# * You shall not disclose such Restricted Information and shall use it only in
#   accordance with the terms of the license agreement you entered into with
#   Orange named the "Khiops - Python Library Evaluation License".
# * Unauthorized copying of this file, via any medium is strictly prohibited.
# * See the "LICENSE.md" file for more details.
###############################################################################
"""Tests for executing fit multiple times on multi-table data"""

import os
import unittest

import pykhiops.core as pk
import pykhiops.core.filesystems as fs
from pykhiops.sklearn.estimators import KhiopsClassifier
from tests.test_helper import PyKhiopsTestHelper


class PyKhiopsMultitableFitTests(unittest.TestCase, PyKhiopsTestHelper):
    """Test if Khiops estimator can be fitted on multi-table data"""

    def setUp(self):
        if "UNITTEST_ONLY_SHORT_TESTS" in os.environ:
            if os.environ["UNITTEST_ONLY_SHORT_TESTS"].lower() == "true":
                self.skipTest("Skipping long test")

    def test_estimator_multiple_create_and_fit_does_not_raise_exception(self):
        """Test if estimator can be fitted from paths several times"""
        # Set upt the file based dataset
        dataset_name = "SpliceJunction"
        samples_dir = pk.get_runner().samples_dir
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
        output_dir = os.path.join(
            os.curdir, "resources", "tmp", "test_multitable_fit_predict"
        )
        try:
            for _ in range(2):
                PyKhiopsTestHelper.fit_helper(
                    KhiopsClassifier,
                    data=(dataset, "Class"),
                    pickled=False,
                    output_dir=output_dir,
                )

        # Remove data files created during the test
        finally:
            output_dir_res = fs.create_resource(output_dir)
            if output_dir_res.exists():
                for file_name in output_dir_res.list_dir():
                    output_dir_res.create_child(file_name).remove()
                output_dir_res.remove()
