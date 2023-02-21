######################################################################################
# Copyright (c) 2018 - 2023 Orange - All Rights Reserved                             #
# * This software is the confidential and proprietary information of Orange.         #
# * You shall not disclose such Restricted Information and shall use it only in      #
#   accordance with the terms of the license agreement you entered into with Orange  #
#   named the "Khiops - Python Library Evaluation License".                          #
# * Unauthorized copying of this file, via any medium is strictly prohibited.        #
# * See the "LICENSE.md" file for more details.                                      #
######################################################################################
"""Test run all samples"""
import os.path
import unittest

import pykhiops.core as pk
from pykhiops.samples import samples, samples_sklearn


class PyKhiopsSamplesTests(unittest.TestCase):
    """Test if all samples run without problems"""

    def setUp(self):
        if "UNITTEST_ONLY_SHORT_TESTS" in os.environ:
            if os.environ["UNITTEST_ONLY_SHORT_TESTS"].lower() == "true":
                self.skipTest("Skipping long test")

    def test_samples(self):
        """Test if all samples run without problems"""
        # Obtain the runner version and set the minimal requirements for some samples
        min_version = {
            samples.detect_data_table_format: pk.KhiopsVersion("10.0.1"),
            samples.deploy_coclustering: pk.KhiopsVersion("10.0.1"),
        }

        # Run the samples
        for sample in samples.exported_samples:
            with self.subTest(sample=sample.__name__):
                print(f"\n>>> Testing sample.{sample.__name__}")
                if sample not in min_version:
                    sample.__call__()
                elif pk.get_runner().khiops_version >= min_version[sample]:
                    sample.__call__()
                else:
                    print(
                        f"Ignored sample {sample.__name__}, "
                        f"minimum version required: {min_version[sample]}"
                    )
                print("> Done")

    def test_samples_sklearn(self):
        """Test if all sklearn samples run without problems"""
        # Run the samples
        for sample in samples_sklearn.exported_samples:
            with self.subTest(sample=sample.__name__):
                print(f"\n>>> Testing sample.{sample.__name__}")
                sample.__call__()
                print("> Done")
