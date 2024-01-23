######################################################################################
# Copyright (c) 2023 Orange. All rights reserved.                                    #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Test run all samples"""
import unittest

import khiops.core as kh
from khiops.samples import samples, samples_sklearn
from tests.test_helper import KhiopsTestHelper


class KhiopsSamplesTests(unittest.TestCase):
    """Test if all samples run without problems"""

    def setUp(self):
        KhiopsTestHelper.skip_long_test(self)

    def test_samples(self):
        """Test if all samples run without problems"""
        # Obtain the runner version and set the minimal requirements for some samples
        min_version = {
            samples.detect_data_table_format: kh.KhiopsVersion("10.0.1"),
            samples.deploy_coclustering: kh.KhiopsVersion("10.0.1"),
        }

        # Run the samples
        for sample in samples.exported_samples:
            with self.subTest(sample=sample.__name__):
                print(f"\n>>> Testing sample.{sample.__name__}")
                if sample not in min_version:
                    sample.__call__()
                elif kh.get_runner().khiops_version >= min_version[sample]:
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
