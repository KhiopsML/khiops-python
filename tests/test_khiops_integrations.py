######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Tests for executing fit multiple times on multi-table data"""

import os
import platform
import shutil
import stat
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
        openmpi_found = None
        with open(
            os.path.join(os.sep, "etc", "os-release"), encoding="ascii"
        ) as os_release_info:
            for entry in os_release_info:
                if entry.startswith("NAME"):
                    linux_distribution = entry.split("=")[-1].strip('"\n').lower()
                    break

        # Check if OpenMPI is installed on the Debian Linux OS
        if linux_distribution == "ubuntu":
            with subprocess.Popen(
                ["dpkg", "-l", "openmpi-bin"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                universal_newlines=True,
            ) as openmpi_query:
                stdout, _ = openmpi_query.communicate()
                if openmpi_query.returncode != 0:
                    openmpi_found = False
                for line in stdout.splitlines():
                    # openmpi installed
                    if all(field in line for field in ("ii", "openmpi")):
                        openmpi_found = True
                        break
                else:
                    openmpi_found = False

        # Check if openmpi is installed on the CentOS / Rocky Linux OS
        elif linux_distribution == "rocky linux":
            with subprocess.Popen(
                ["yum", "list", "installed", "openmpi"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                universal_newlines=True,
            ) as openmpi_query:
                stdout, _ = openmpi_query.communicate()
                if openmpi_query.returncode != 0:
                    openmpi_found = False
                for line in stdout.splitlines():
                    # openmpi installed
                    if line.startswith("openmpi"):
                        openmpi_found = True
                        break
                else:
                    openmpi_found = False
        else:
            self.skipTest("Skipping test: platform not Ubuntu or Rocky Linux")
        if openmpi_found:
            runner = kh.get_runner()
            if not runner.mpi_command_args:
                self.fail("MPI support found, but MPI command args not set")
            mpiexec_path = runner.mpi_command_args[0]
            self.assertTrue(os.path.exists(mpiexec_path))
            self.assertTrue(os.path.isfile(mpiexec_path))
            self.assertTrue(os.access(mpiexec_path, os.X_OK))
        else:
            self.skipTest("Skipping test: MPI support is not installed")

    def test_environment_error_on_bogus_khiops_env_script(self):
        """Test that error is raised on reading erroneous khiops_env script"""

        with tempfile.TemporaryDirectory() as temp_dir:

            # Create temporary khiops_env binary dir
            # Note: The "bin" subdir is needed for Windows.
            temp_khiops_env_dir = os.path.join(temp_dir, "bin")
            os.makedirs(temp_khiops_env_dir)

            # Create temporary khiops_env path
            temp_khiops_env_file_path = os.path.join(temp_khiops_env_dir, "khiops_env")

            # On Windows, set KHIOPS_HOME to the temp dir
            original_khiops_home_env_var = os.environ.get("KHIOPS_HOME")
            if platform.system() == "Windows":
                os.environ["KHIOPS_HOME"] = temp_dir
                temp_khiops_env_file_path += ".cmd"

            # Replace the khiops_env with a script that fails showing an error message
            with open(
                temp_khiops_env_file_path, "w", encoding="utf-8"
            ) as temp_khiops_env_file:
                test_error_message = "Test Khiops environment error"
                if platform.system() == "Windows":
                    temp_khiops_env_file.write("@echo off\r\n")
                    temp_khiops_env_file.write(f"echo {test_error_message}>&2\r\n")
                    temp_khiops_env_file.write("exit /b 1")
                else:
                    temp_khiops_env_file.write("#!/bin/bash\n")
                    temp_khiops_env_file.write(f'>&2 echo "{test_error_message}"\n')
                    temp_khiops_env_file.write("exit 1")

            # Make the temporary khiops_env file executable
            os.chmod(
                temp_khiops_env_file_path,
                os.stat(temp_khiops_env_file_path).st_mode | stat.S_IEXEC,
            )

            # Store initial PATH
            original_path_env_var = os.environ["PATH"]

            # Prepend path to temporary khiops_env to PATH
            os.environ["PATH"] = (
                temp_khiops_env_dir + os.pathsep + original_path_env_var
            )

            # Create new KhiopsLocalRunner and capture the exception
            with self.assertRaises(KhiopsEnvironmentError) as context:
                _ = KhiopsLocalRunner()

            # Restore initial PATH
            os.environ["PATH"] = original_path_env_var

            # On Windows, restore initial KHIOPS_HOME
            if (
                platform.system() == "Windows"
                and original_khiops_home_env_var is not None
            ):
                os.environ["KHIOPS_HOME"] = original_khiops_home_env_var

        # Check that the script error message matches the expected one
        expected_msg = (
            "Error initializing the environment for Khiops from the "
            f"{temp_khiops_env_file_path} script. "
            f"Contents of stderr:\n{test_error_message}\n"
        )
        output_msg = str(context.exception)
        self.assertEqual(output_msg, expected_msg)

    def test_runner_environment_initialization(self):
        """Test that local runner initializes/ed its environment properly

        .. note::
            To test a real initialization this test should be executed alone.
        """
        # Obtain the current runner
        runner = kh.get_runner()

        # Check that MODL* files as set in the runner exist and are executable
        self.assertTrue(os.path.isfile(runner.khiops_path))
        self.assertTrue(os.access(runner.khiops_path, os.X_OK))
        self.assertTrue(os.path.isfile(runner.khiops_coclustering_path))
        self.assertTrue(os.access(runner.khiops_coclustering_path, os.X_OK))

        # Check that mpiexec is set correctly in the runner:
        if runner.mpi_command_args:
            mpiexec_path = runner.mpi_command_args[0]
            self.assertTrue(os.path.exists(mpiexec_path))
            self.assertTrue(os.path.isfile(mpiexec_path))
            self.assertTrue(os.access(mpiexec_path, os.X_OK))

        # Check that runner creation sets `KHIOPS_API_MODE` to `true`
        # Store original KHIOPS_API_MODE if any, then delete it from the
        # environment if present
        original_khiops_api_mode = os.environ.get("KHIOPS_API_MODE")
        if original_khiops_api_mode is not None:
            del os.environ["KHIOPS_API_MODE"]

        # Create fresh runner
        _ = KhiopsLocalRunner()

        # Get KHIOPS_API_MODE as set after runner initialization
        env_khiops_api_mode = os.environ.get("KHIOPS_API_MODE")

        # Restore original KHIOPS_API_MODE, if any
        if original_khiops_api_mode is not None:
            os.environ["KHIOPS_API_MODE"] = original_khiops_api_mode

        self.assertEqual(env_khiops_api_mode, "true")


class KhiopsMultitableFitTests(unittest.TestCase):
    """Test if Khiops estimator can be fitted on multi-table data"""

    def setUp(self):
        KhiopsTestHelper.skip_expensive_test(self)

    def test_estimator_multiple_create_and_fit_does_not_raise_exception(self):
        """Test if estimator can be fitted from dataframes several times"""
        # Set-up a dataframe-based dataset
        (
            root_table_data,
            secondary_table_data,
        ) = KhiopsTestHelper.get_two_table_data(
            "SpliceJunction", "SpliceJunction", "SpliceJunctionDNA"
        )
        (splice_junction_df, y), _ = KhiopsTestHelper.prepare_data(
            root_table_data, "Class"
        )
        (splice_junction_dna_df, _), _ = KhiopsTestHelper.prepare_data(
            secondary_table_data, "SampleId", primary_table=splice_junction_df
        )
        dataset = {
            "main_table": "SpliceJunction",
            "tables": {
                "SpliceJunction": (
                    splice_junction_df,
                    "SampleId",
                ),
                "SpliceJunctionDNA": (
                    splice_junction_dna_df,
                    "SampleId",
                ),
            },
        }

        # Train classifier
        output_dir = os.path.join("resources", "tmp", "test_multitable_fit_predict")
        khiops_classifier = KhiopsClassifier(output_dir=output_dir)
        try:
            for _ in range(2):
                khiops_classifier.fit(X=dataset, y=y)
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
