######################################################################################
# Copyright (c) 2023 Orange. All rights reserved.                                    #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Tests for executing fit multiple times on multi-table data"""

import os
import shutil
import tempfile
import unittest

import khiops.core as kh
from khiops.core.internals.runner import KhiopsLocalRunner
from khiops.sklearn.estimators import KhiopsClassifier
from tests.test_helper import KhiopsTestHelper


class KhiopsCustomRunnerEnvironmentTests(unittest.TestCase):
    """Test that runners in custom environments work"""

    def test_runner_with_custom_khiops_binary_directory(self):
        """Test that local runner works with custom Khiops binary directory"""
        # Get default runner
        default_runner = kh.get_runner()

        # Create a fresh local runner and initialize its default Khiops binary dir
        runner = KhiopsLocalRunner()
        runner._initialize_khiops_bin_dir()

        # Get runner's default Khiops binary directory
        default_bin_dir = runner.khiops_bin_dir

        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmp_khiops_bin_dir:
            # Copy Khiops binaries into the temporary directory
            for binary_file in os.listdir(default_bin_dir):
                if binary_file.startswith("MODL"):
                    shutil.copy(
                        os.path.join(default_bin_dir, binary_file),
                        os.path.join(tmp_khiops_bin_dir, binary_file),
                    )

            # Change runner's Khiops binary directory to the temporary directory
            runner.khiops_bin_dir = tmp_khiops_bin_dir

            # Set current runner to the fresh runner
            kh.set_runner(runner)

            # Test the core API works
            # Call check_database (could be any other method)
            with self.assertRaises(kh.KhiopsRuntimeError) as cm:
                kh.check_database("a.kdic", "dict_name", "data.txt")
            # Test that MODL executable can be found and launched
            self.assertIn("khiops ended with return code 2", str(cm.exception))

        # Set current runner to the default runner
        kh.set_runner(default_runner)


class KhiopsMultitableFitTests(unittest.TestCase, KhiopsTestHelper):
    """Test if Khiops estimator can be fitted on multi-table data"""

    def setUp(self):
        KhiopsTestHelper.skip_long_test(self)

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
