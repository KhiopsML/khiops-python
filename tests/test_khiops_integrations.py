######################################################################################
# Copyright (c) 2024 Orange. All rights reserved.                                    #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Tests for executing fit multiple times on multi-table data"""

import os
import platform
import shutil
import subprocess
import tempfile
import unittest

import khiops.core as kh
from khiops.core.exceptions import KhiopsEnvironmentError
from khiops.core.internals.runner import KhiopsLocalRunner
from khiops.extras.docker import KhiopsDockerRunner
from khiops.sklearn.estimators import KhiopsClassifier
from tests.test_helper import KhiopsTestHelper

# Eliminate protected-access check from these tests
# pylint: disable=protected-access


class KhiopsRunnerEnvironmentTests(unittest.TestCase):
    """Test that runners in different environments work"""

    @unittest.skipIf(
        platform.system() != "Linux", "Skipping test for non-Linux platform"
    )
    def test_runner_has_mpiexec_on_linux(self):
        """Test that local runner has executable mpiexec on Linux if MPI is installed"""
        # Check package is installed on supported platform:
        # Check /etc/os-release for Linux version
        linux_distribution = None
        mpich_found = None
        with open(
            os.path.join(os.sep, "etc", "os-release"), encoding="ascii"
        ) as os_release_info:
            for entry in os_release_info:
                if entry.startswith("NAME"):
                    linux_distribution = entry.split("=")[-1].strip('"\n').lower()
                    break

        # Check if MPICH is installed on the Debian Linux OS
        if linux_distribution == "ubuntu":
            with subprocess.Popen(
                ["dpkg", "-l", "mpich"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                universal_newlines=True,
            ) as mpich_query:
                stdout, _ = mpich_query.communicate()
                if mpich_query.returncode != 0:
                    mpich_found = False
                for line in stdout.splitlines():
                    if all(field in line for field in ("ii", "mpich")):
                        # MPICH installed
                        mpich_found = True
                        break
                else:
                    mpich_found = False

        # Check if MPICH is installed on the CentOS / Rocky Linux OS
        elif linux_distribution == "rocky linux":
            with subprocess.Popen(
                ["yum", "list", "installed", "mpich"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                universal_newlines=True,
            ) as mpich_query:
                stdout, _ = mpich_query.communicate()
                if mpich_query.returncode != 0:
                    mpich_found = False
                for line in stdout.splitlines():
                    if line.startswith("mpich"):
                        # MPICH installed
                        mpich_found = True
                        break
                else:
                    mpich_found = False
        else:
            self.skipTest("Skipping test: platform not Ubuntu or Rocky Linux")
        if mpich_found:
            runner = kh.get_runner()
            if not runner.mpi_command_args:
                self.fail("MPI support found, but MPI command args not set")
            mpiexec_path = runner.mpi_command_args[0]
            self.assertTrue(os.path.exists(mpiexec_path))
            self.assertTrue(os.path.isfile(mpiexec_path))
            self.assertTrue(os.access(mpiexec_path, os.X_OK))
        else:
            self.skipTest("Skipping test: MPI support is not installed")

    def test_runner_with_conda_based_environment(self):
        """Test that local runner works in non-Conda, Conda-based environments"""

        # Emulate Conda-based environment:
        # - unset `CONDA_PREFIX` if set
        # - create new KhiopsLocalRunner and initialize its Khiops binary
        #   directory
        # - check that the Khiops binary directory contains the MODL* binaries
        #   and `mpiexec` (which should be its default location)

        # Unset `CONDA_PREFIX` if existing
        if "CONDA_PREFIX" in os.environ:
            del os.environ["CONDA_PREFIX"]

        # Create a fresh local runner and initialize its default Khiops binary dir
        runner = KhiopsLocalRunner()
        runner._initialize_khiops_bin_dir()

        # Get runner's default Khiops binary directory
        default_bin_dir = runner.khiops_bin_dir

        # Check that MODL* are indeed in the runner's Khiops binary directory
        self.assertTrue(
            all(
                binary_file in os.listdir(default_bin_dir)
                for binary_file in ("MODL", "MODL_Coclustering")
            )
        )

        # Check that mpiexec is set correctly in the runner:
        mpi_command_args = runner.mpi_command_args
        self.assertTrue(len(mpi_command_args) > 0)
        mpiexec_path = runner.mpi_command_args[0]
        self.assertTrue(os.path.exists(mpiexec_path))
        self.assertTrue(os.path.isfile(mpiexec_path))
        self.assertTrue(os.access(mpiexec_path, os.X_OK))

    def test_runner_with_custom_khiops_binary_directory(self):
        """Test that local runner works with custom Khiops binary directory"""
        # Get default runner
        default_runner = kh.get_runner()

        # Test in a try block to restore the runner if there are unexpected errors
        try:
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
                # Note: The return code is not specified to support older khiops
                # versions that returned 2 instead of 0 in this case.
                self.assertIn(
                    "khiops execution had errors (return code ", str(cm.exception)
                )

        # Always set back to the default runner
        finally:
            kh.set_runner(default_runner)


class KhiopsMultitableFitTests(unittest.TestCase):
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


class DockerKhiopsEdgeCases(unittest.TestCase):
    """Test for KhiopsDocker runner edge cases"""

    def test_shared_dir_edge_cases(self):
        """Test that the existence check for shared_dir is done only for local paths"""
        with self.assertRaises(KhiopsEnvironmentError) as ctx:
            KhiopsDockerRunner("localhost://", "NONEXISTENT/DIRECTORY")
        self.assertRegex(str(ctx.exception), "^'shared_dir' does not exist.")
