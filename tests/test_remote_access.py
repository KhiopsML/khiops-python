######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Integration tests with remote filesystems and Khiops runners"""
import os
import shutil
import signal
import ssl
import subprocess
import time
import unittest
import uuid
from contextlib import suppress
from urllib.request import Request, urlopen

import khiops.core as kh
import khiops.core.internals.filesystems as fs
from khiops.core.internals.runner import KhiopsLocalRunner
from khiops.extras.docker import KhiopsDockerRunner
from tests.test_helper import KhiopsTestHelper


def s3_config_exists():
    # Note:
    # Instead of config files,
    # the S3 configuration may also be set with alternative environment variables
    # - `AWS_ACCESS_KEY_ID`
    # - `AWS_SECRET_ACCESS_KEY`
    # - `AWS_ENDPOINT_URL`
    # - `S3_BUCKET_NAME`
    # However storing the credentials in config files is more secure,
    # because these can be protected by access policies.
    return (
        "AWS_SHARED_CREDENTIALS_FILE" in os.environ
        and "AWS_CONFIG_FILE" in os.environ
        and "S3_BUCKET_NAME" in os.environ
    )


def gcs_config_exists():
    return (
        "GOOGLE_APPLICATION_CREDENTIALS" in os.environ
        and "GCS_BUCKET_NAME" in os.environ
    )


def docker_runner_config_exists():
    return (
        "KHIOPS_DOCKER_RUNNER_URL" in os.environ
        and "KHIOPS_DOCKER_RUNNER_SHARED_DIR" in os.environ
    )


class KhiopsRemoteAccessTestsContainer:
    """Container class to allow unittest.TestCase inheritance"""

    class KhiopsRemoteAccessTests(unittest.TestCase, KhiopsTestHelper):
        """Generic class to test remote filesystems and Khiops runners"""

        @classmethod
        def init_remote_bucket(cls, bucket_name=None, proto=None):
            # create the remote root_temp_dir
            remote_resource = fs.create_resource(
                f"{proto}://{bucket_name}/khiops-cicd/tmp"
            )
            remote_resource.make_dir()

            # copy to /samples each file
            for file in (
                "Iris/Iris.txt",
                "Iris/Iris.kdic",
                "SpliceJunction/SpliceJunction.txt",
                "SpliceJunction/SpliceJunctionDNA.txt",
                "SpliceJunction/SpliceJunction.kdic",
            ):
                fs.copy_from_local(
                    f"{proto}://{bucket_name}/khiops-cicd/samples/{file}",
                    os.path.join(kh.get_samples_dir(), file),
                )
                # symmetric call to ensure the upload was OK
                fs.copy_to_local(
                    f"{proto}://{bucket_name}/khiops-cicd/samples/{file}", "/tmp/dummy"
                )

        def results_dir_root(self):
            """To be overridden by descendants if needed

            The default is the current directory
            """
            return os.curdir

        def config_exists(self):
            """To be overridden by descendants"""
            return False

        def remote_access_test_case(self):
            """To be overridden by descendants"""
            return ""

        def should_skip_in_a_conda_env(self):
            """To be overridden by descendants"""
            return True

        def print_test_title(self):
            print(f"\n   Remote System: {self.remote_access_test_case()}")

        def skip_if_no_config(self):
            if not self.config_exists():
                self.skipTest(
                    f"Remote test case {self.remote_access_test_case()} "
                    "has no configuration available"
                )

        @staticmethod
        def is_in_a_conda_env():
            """Detects whether this is run from a Conda environment

            The way to find it out is to check if khiops-core is installed
            in the same environment as the current running Conda one
            """

            if not isinstance(kh.get_runner(), KhiopsLocalRunner):
                return False

            # Get path to the Khiops executable (temporarily disable pylint warning)
            # pylint: disable=protected-access
            khiops_path = kh.get_runner()._khiops_path
            # pylint: enable=protected-access

            # If $(dirname khiops_path) is identical to $CONDA_PREFIX/bin,
            # then return True
            conda_prefix = os.environ.get("CONDA_PREFIX")
            return conda_prefix is not None and os.path.join(
                conda_prefix, "bin"
            ) == os.path.dirname(khiops_path)

        def setUp(self):
            # Skip if only short and cheap tests are run
            KhiopsTestHelper.skip_expensive_test(self)

            self.skip_if_no_config()
            if self.is_in_a_conda_env() and self.should_skip_in_a_conda_env():
                self.skipTest(
                    f"Remote test case {self.remote_access_test_case()} "
                    "in a conda environment is currently skipped"
                )
            self.print_test_title()

        def tearDown(self):
            # Cleanup the output dir (the files within and the folder)
            if hasattr(self, "folder_name_to_clean_in_teardown"):
                for filename in fs.list_dir(self.folder_name_to_clean_in_teardown):
                    fs.remove(
                        fs.get_child_path(
                            self.folder_name_to_clean_in_teardown, filename
                        )
                    )
                fs.remove(self.folder_name_to_clean_in_teardown)

        def test_train_predictor_with_remote_access(self):
            """Test train_predictor with remote resources"""
            iris_data_dir = fs.get_child_path(kh.get_runner().samples_dir, "Iris")
            # ask for folder cleaning during tearDown
            self.folder_name_to_clean_in_teardown = output_dir = fs.get_child_path(
                self.results_dir_root(),
                f"test_{self.remote_access_test_case()}_remote_files_{uuid.uuid4()}",
            )

            # Attempt to make local directory if not existing
            if not fs.exists(output_dir) and fs.is_local_resource(output_dir):
                fs.make_dir(output_dir)

            # Set output report file path
            report_file_path = fs.get_child_path(output_dir, "IrisAnalysisResults.khj")

            # When using `kh`, the log file will be by default
            # in the runner `root_temp_dir` folder that can be remote
            _, model_file_path = kh.train_predictor(
                fs.get_child_path(iris_data_dir, "Iris.kdic"),
                dictionary_name="Iris",
                data_table_path=fs.get_child_path(iris_data_dir, "Iris.txt"),
                target_variable="Class",
                analysis_report_file_path=report_file_path,
                temp_dir=self._khiops_temp_dir,
                trace=True,
            )

            # Check the existence of the training files
            self.assertTrue(fs.exists(report_file_path))
            self.assertTrue(fs.exists(model_file_path))

        def test_train_predictor_fail_and_log_with_remote_access(self):
            """Test train_predictor failure and access to a remote log"""
            log_file_path = fs.get_child_path(
                self._khiops_temp_dir, f"khiops_log_{uuid.uuid4()}.log"
            )

            # no cleaning required as an exception would be raised
            # without any result produced
            self.folder_name_to_clean_in_teardown = output_dir = fs.get_child_path(
                self.results_dir_root(),
                f"test_{self.remote_access_test_case()}_remote_files_{uuid.uuid4()}",
            )

            # Attempt to make local directory if not existing
            if not fs.exists(output_dir) and fs.is_local_resource(output_dir):
                fs.make_dir(output_dir)

            # Set paths
            report_file_path = fs.get_child_path(output_dir, "IrisAnalysisResults.khj")
            iris_data_dir = fs.get_child_path(kh.get_runner().samples_dir, "Iris")

            # Run the test
            with self.assertRaises(kh.KhiopsRuntimeError):
                kh.train_predictor(
                    fs.get_child_path(iris_data_dir, "NONEXISTENT.kdic"),
                    dictionary_name="Iris",
                    data_table_path=fs.get_child_path(iris_data_dir, "Iris.txt"),
                    target_variable="Class",
                    analysis_report_file_path=report_file_path,
                    log_file_path=log_file_path,
                )
            # Check and remove log file
            self.assertTrue(fs.exists(log_file_path), f"Path: {log_file_path}")
            fs.remove(log_file_path)


class KhiopsS3RemoteFileTests(KhiopsRemoteAccessTestsContainer.KhiopsRemoteAccessTests):
    """Integration tests with Amazon S3 filesystems"""

    @classmethod
    def setUpClass(cls):
        """Sets up remote directories in runner"""
        if s3_config_exists():
            runner = kh.get_runner()
            bucket_name = os.environ["S3_BUCKET_NAME"]

            cls.init_remote_bucket(bucket_name=bucket_name, proto="s3")

            runner.samples_dir = f"s3://{bucket_name}/khiops-cicd/samples"
            resources_directory = KhiopsTestHelper.get_resources_dir()

            # WARNING : khiops temp files cannot be remote
            cls._khiops_temp_dir = f"{resources_directory}/tmp/khiops-cicd"

            # root_temp_dir
            # (where the log file is saved by default when using `kh`)
            # can be remote
            runner.root_temp_dir = f"s3://{bucket_name}/khiops-cicd/tmp"

    @classmethod
    def tearDownClass(cls):
        """Sets back the runner defaults"""
        if s3_config_exists():
            kh.get_runner().__init__()

    def should_skip_in_a_conda_env(self):
        # The S3 driver is now released for conda too.
        # No need to skip the tests any longer in a conda environment
        return False

    def config_exists(self):
        return s3_config_exists()

    def remote_access_test_case(self):
        return "S3"


class KhiopsGCSRemoteFileTests(
    KhiopsRemoteAccessTestsContainer.KhiopsRemoteAccessTests
):
    """Integration tests with Google Cloud Storage filesystems"""

    @classmethod
    def setUpClass(cls):
        """Sets up remote directories in runner"""
        if gcs_config_exists():
            runner = kh.get_runner()
            bucket_name = os.environ["GCS_BUCKET_NAME"]

            cls.init_remote_bucket(bucket_name=bucket_name, proto="gs")

            runner.samples_dir = f"gs://{bucket_name}/khiops-cicd/samples"
            resources_directory = KhiopsTestHelper.get_resources_dir()

            # WARNING : khiops temp files cannot be remote
            cls._khiops_temp_dir = f"{resources_directory}/tmp/khiops-cicd"

            # root_temp_dir
            # (where the log file is saved by default when using `kh`)
            # can be remote
            runner.root_temp_dir = f"gs://{bucket_name}/khiops-cicd/tmp"

    @classmethod
    def tearDownClass(cls):
        """Sets back the runner defaults"""
        if gcs_config_exists():
            kh.get_runner().__init__()

    def should_skip_in_a_conda_env(self):
        # The GCS driver is now released for conda too.
        # No need to skip the tests any longer in a conda environment
        return False

    def config_exists(self):
        return gcs_config_exists()

    def remote_access_test_case(self):
        return "GCS"


class KhiopsDockerRunnerTests(KhiopsRemoteAccessTestsContainer.KhiopsRemoteAccessTests):
    """Integration tests with the Docker runner service"""

    # pylint: disable=protected-access,consider-using-with
    @classmethod
    def setUpClass(cls):
        """Sets up docker runner service for this test case
        This method executes the following steps:
            - If environment variable ``KHIOPS_RUNNER_SERVICE_PATH`` is set then it
              launches the service and makes sure it is operational before executing the
              test case.  Otherwise it skips the test case.
            - Then it copies ``samples`` to a shared directory accessible to both the
              local Khiops runner service and the process using Khiops Python.
            - Finally it creates the `.KhiopsDockerRunner` client for the
              Khiops service and set it as current runner.
        """
        # Save the initial Khiops Python runner
        cls.initial_runner = kh.get_runner()

        if "KHIOPS_RUNNER_SERVICE_PATH" in os.environ:
            # Start runner process
            khiops_runner_process = subprocess.Popen(
                os.environ["KHIOPS_RUNNER_SERVICE_PATH"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # Wait a bit and check that it functions normally
            time.sleep(10)
            ctx = ssl._create_unverified_context(check_hostname=False)
            ctx.verify_mode = ssl.CERT_NONE
            with urlopen(
                Request(os.environ["KHIOPS_DOCKER_RUNNER_URL"]), context=ctx
            ) as response:
                if response.status != 200:
                    raise unittest.SkipTest("No running Khiops server")

            # Save the service's process id
            cls.khiops_runner_pid = khiops_runner_process.pid

        if docker_runner_config_exists():
            shared_dir = os.environ["KHIOPS_DOCKER_RUNNER_SHARED_DIR"]

            # Copy the samples directory
            source_samples_dir = cls.initial_runner.samples_dir
            target_samples_dir = os.path.join(shared_dir, "samples")
            if not fs.exists(target_samples_dir):
                fs.make_dir(target_samples_dir)
            for dir_name, _, dataset_file_names in os.walk(source_samples_dir):
                target_dataset_dir = os.path.join(
                    target_samples_dir, os.path.split(dir_name)[1]
                )
                if not fs.exists(target_dataset_dir):
                    fs.make_dir(target_dataset_dir)
                for file_name in dataset_file_names:
                    fs.copy_from_local(
                        target_dataset_dir, os.path.join(dir_name, file_name)
                    )

            # Create khiops service runner
            docker_runner = KhiopsDockerRunner(
                url=os.environ["KHIOPS_DOCKER_RUNNER_URL"],
                shared_dir=shared_dir,
                insecure=True,
            )
            cls._khiops_temp_dir = os.path.join(shared_dir, "tmp")
            docker_runner.samples_dir = target_samples_dir

            # Set current runner to the created Khiops service runner
            kh.set_runner(docker_runner)

    # pylint: enable=protected-access,consider-using-with

    @classmethod
    def tearDownClass(cls):
        """Sets back the initial runner and terminates local runner"""
        # If local path to Khiops service is available terminate it
        if "KHIOPS_RUNNER_SERVICE_PATH" in os.environ:
            with suppress(ProcessLookupError):
                os.kill(cls.khiops_runner_pid, signal.SIGTERM)

        # Cleanup: remove directories created in `setUpClass`
        if docker_runner_config_exists():

            # If Khiops samples and temp dirs exist, remove them
            with suppress(FileNotFoundError):
                shutil.rmtree(kh.get_runner().samples_dir)
                shutil.rmtree(cls._khiops_temp_dir)

        # Reset the Khiops Python runner to the initial one
        kh.set_runner(cls.initial_runner)

    def config_exists(self):
        return docker_runner_config_exists()

    def should_skip_in_a_conda_env(self):
        # Tests using a docker runner should never be skipped
        # even in a conda environment
        return False

    def remote_access_test_case(self):
        return "KhiopsDockerRunner"

    def results_dir_root(self):
        return self._khiops_temp_dir
