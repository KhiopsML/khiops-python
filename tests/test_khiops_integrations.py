######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Various integration tests"""

import os
import platform
import shutil
import stat
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import khiops.core as kh
import khiops.core.internals.filesystems as fs
from khiops import tools
from khiops.core.exceptions import KhiopsEnvironmentError
from khiops.core.internals.runner import (
    KhiopsLocalRunner,
    _build_khiops_process_environment,
)
from khiops.extras.docker import KhiopsDockerRunner
from khiops.sklearn.estimators import KhiopsClassifier
from tests.test_helper import KhiopsTestHelper

# Eliminate protected-access check from these tests
# pylint: disable=protected-access


class KhiopsRunnerEnvironmentTests(unittest.TestCase):
    """Test that runners in different environments work"""

    @unittest.skipIf(
        os.environ.get("SKIP_EXPENSIVE_TESTS", "false").lower() == "true",
        "Skipping expensive test",
    )
    def test_samples_are_downloaded_according_to_the_runner_setting(self):
        """Test that samples are downloaded to the runner samples directory"""

        # Save initial state
        initial_runner = kh.get_runner()
        initial_khiops_samples_dir = os.environ.get("KHIOPS_SAMPLES_DIR")

        # Test that default samples download location is consistent with the
        # KhiopsLocalRunner samples directory
        with tempfile.TemporaryDirectory() as tmp_samples_dir:

            # Set environment variable to the temporary samples dir
            os.environ["KHIOPS_SAMPLES_DIR"] = tmp_samples_dir

            # Create test runner to update samples dir to tmp_samples_dir,
            # according to the newly-set environment variable
            test_runner = KhiopsLocalRunner()

            # Set current runner to the test runner
            kh.set_runner(test_runner)

            # Check that samples are not in tmp_samples_dir
            with self.assertRaises(AssertionError):
                self.assert_samples_dir_integrity(tmp_samples_dir)

            # Download samples into existing, but empty, tmp_samples_dir
            # Check that samples have been downloaded to tmp_samples_dir
            tools.download_datasets(force_overwrite=True)
            self.assert_samples_dir_integrity(tmp_samples_dir)

        # Remove KHIOPS_SAMPLES_DIR
        # Create test runner to update samples dir to the default runner samples
        # dir, following the deletion of the KHIOPS_SAMPLES_DIR environment
        # variable
        # Set current runner to the test runner
        del os.environ["KHIOPS_SAMPLES_DIR"]
        test_runner = KhiopsLocalRunner()
        kh.set_runner(test_runner)

        # Get the default runner samples dir
        default_runner_samples_dir = kh.get_samples_dir()

        # Move existing default runner samples dir contents to temporary directory
        if os.path.isdir(default_runner_samples_dir):
            tmp_initial_samples_dir = tempfile.mkdtemp()
            shutil.copytree(
                default_runner_samples_dir,
                tmp_initial_samples_dir,
                dirs_exist_ok=True,
            )
            shutil.rmtree(default_runner_samples_dir)
        else:
            tmp_initial_samples_dir = None

        # Check that the samples are not present in the default runner
        # samples dir
        with self.assertRaises(AssertionError):
            self.assert_samples_dir_integrity(default_runner_samples_dir)

        # Download datasets to the default runner samples dir (which
        # should be created on this occasion)
        # Default samples dir does not exist anymore
        # Check that the default samples dir is populated
        tools.download_datasets()
        self.assert_samples_dir_integrity(default_runner_samples_dir)

        # Clean-up default samples dir
        shutil.rmtree(default_runner_samples_dir)

        # Restore initial state:
        # - initial samples dir contents if previously present
        # - initial KHIOPS_SAMPLES_DIR if set
        # - initial runner
        if tmp_initial_samples_dir is not None and os.path.isdir(
            tmp_initial_samples_dir
        ):
            shutil.copytree(
                tmp_initial_samples_dir,
                default_runner_samples_dir,
                dirs_exist_ok=True,
            )

            # Remove temporary directory
            shutil.rmtree(tmp_initial_samples_dir)
        if initial_khiops_samples_dir is not None:
            os.environ["KHIOPS_SAMPLES_DIR"] = initial_khiops_samples_dir
        kh.set_runner(initial_runner)

    def assert_samples_dir_integrity(self, samples_dir):
        """Checks that the samples dir has the expected structure"""
        expected_dataset_names = [
            "Accidents",
            "AccidentsSummary",
            "Adult",
            "Customer",
            "CustomerExtended",
            "Iris",
            "Letter",
            "Mushroom",
            "NegativeAirlineTweets",
            "SpliceJunction",
            "WineReviews",
        ]
        self.assertTrue(os.path.isdir(samples_dir))
        for ds_name in expected_dataset_names:
            self.assertIn(ds_name, os.listdir(samples_dir))

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

    @unittest.skipIf(
        platform.system() != "Linux", "Skipping test for non-Linux platform"
    )
    def test_runner_environment_for_openmpi5(self):
        """Test if KHIOPS_MPI_HOME is actually exported
        and HOME is corrected for OpenMPI 5+"""

        # Trigger the environment initialization
        _ = kh.get_runner().khiops_version

        khiops_env = _build_khiops_process_environment()

        # Check `KHIOPS_MPI_HOME` is correctly exported for OpenMPI 5+
        self.assertIsNotNone(os.environ.get("KHIOPS_MPI_HOME"))

        # Check HOME is corrected in the new process environment
        self.assertEqual(
            os.path.pathsep.join(
                [khiops_env.get("KHIOPS_MPI_HOME", ""), os.environ.get("HOME", "")]
            ),
            khiops_env.get("HOME"),
        )

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

    def test_khiops_and_khiops_coclustering_are_run_with_mpi(self):
        """Test that MODL and MODL_Coclustering are run with MPI"""

        # Get current runner
        runner = kh.get_runner()

        # Get path to the Iris dataset
        iris_data_dir = fs.get_child_path(runner.samples_dir, "Iris")

        # Create the subprocess.Popen mock
        mock_popen = MagicMock()
        mock_popen.return_value.__enter__.return_value.communicate.return_value = (
            b"",
            b"",
        )
        mock_popen.return_value.__enter__.return_value.returncode = 0

        # Run Khiops through an API function, using the mocked Popen, to capture
        # its arguments
        with patch("subprocess.Popen", mock_popen):
            kh.check_database(
                fs.get_child_path(iris_data_dir, "Iris.kdic"),
                "Iris",
                fs.get_child_path(iris_data_dir, "Iris.txt"),
            )

        # Check that the mocked Popen call arguments list starts with the MPI
        # arguments, followed by the Khiops command
        expected_command_args = runner.mpi_command_args + [runner.khiops_path]
        self.assertTrue(len(mock_popen.call_args.args) > 0)
        self.assertTrue(len(mock_popen.call_args.args[0]) > len(expected_command_args))
        self.assertEqual(
            mock_popen.call_args.args[0][: len(expected_command_args)],
            expected_command_args,
        )

        # Run Khiops Coclustering through an API function, using the mocked Popen
        # to capture its arguments
        # Nest context managers for Python 3.8 compatibility
        with patch("subprocess.Popen", mock_popen):
            with tempfile.TemporaryDirectory() as temp_dir:
                kh.train_coclustering(
                    fs.get_child_path(iris_data_dir, "Iris.kdic"),
                    "Iris",
                    fs.get_child_path(iris_data_dir, "Iris.txt"),
                    ["SepalLength", "PetalLength"],
                    fs.get_child_path(temp_dir, "IrisCoclusteringResults.khcj"),
                )

        # Check that the mocked Popen call arguments list starts with the MPI
        # arguments, followed by the Khiops Coclustering command
        expected_command_args = runner.mpi_command_args + [
            runner.khiops_coclustering_path
        ]
        self.assertTrue(len(mock_popen.call_args.args) > 0)
        self.assertTrue(len(mock_popen.call_args.args[0]) > len(expected_command_args))
        self.assertEqual(
            mock_popen.call_args.args[0][: len(expected_command_args)],
            expected_command_args,
        )


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
            "main_table": (splice_junction_df, ["SampleId"]),
            "additional_data_tables": {
                "SpliceJunctionDNA": (
                    splice_junction_dna_df,
                    ["SampleId"],
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
