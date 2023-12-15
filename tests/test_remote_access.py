######################################################################################
# Copyright (c) 2023 Orange. All rights reserved.                                    #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Integration tests with remote filesystems and Khiops runners"""
import io
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

import pandas as pd

import khiops.core as kh
import khiops.core.internals.filesystems as fs
from khiops.extras.docker import KhiopsDockerRunner
from khiops.sklearn import KhiopsClassifier, KhiopsCoclustering
from tests.test_helper import KhiopsTestHelper


def s3_config_exists():
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

        def results_dir_root(self):
            """To be overridden by descendants if needed

            The default is the current directory
            """
            return os.curdir

        def config_exists(self):
            """To be overriden by descendants"""
            return False

        def remote_access_test_case(self):
            """To be overriden by descendants"""
            return ""

        def print_test_title(self):
            print(f"\n   Remote System: {self.remote_access_test_case()}")

        def skip_if_no_config(self):
            if not self.config_exists():
                self.skipTest(
                    f"Remote test case {self.remote_access_test_case()} "
                    "has no configuration available"
                )

        def setUp(self):
            self.skip_if_no_config()
            self.print_test_title()

        def test_train_predictor_with_remote_access(self):
            """Test train_predictor with remote resources"""
            iris_data_dir = fs.get_child_path(kh.get_runner().samples_dir, "Iris")
            output_dir = fs.get_child_path(
                self.results_dir_root(),
                f"test_{self.remote_access_test_case()}_remote_files",
            )
            kh.train_predictor(
                fs.get_child_path(iris_data_dir, "Iris.kdic"),
                dictionary_name="Iris",
                data_table_path=fs.get_child_path(iris_data_dir, "Iris.txt"),
                target_variable="Class",
                results_dir=output_dir,
                trace=True,
            )

            # Check the existents of the trining files
            self.assertTrue(fs.exists(fs.get_child_path(output_dir, "AllReports.khj")))
            self.assertTrue(fs.exists(fs.get_child_path(output_dir, "Modeling.kdic")))

            # Cleanup
            for filename in fs.list_dir(output_dir):
                fs.remove(fs.get_child_path(output_dir, filename))

        def test_khiops_classifier_with_remote_access(self):
            """Test the training of a khiops_classifier with remote resources"""
            # Setup paths
            output_dir = (
                kh.get_runner().khiops_temp_dir
                + f"/KhiopsClassifier_output_dir_{uuid.uuid4()}/"
            )
            iris_data_dir = fs.get_child_path(kh.get_runner().samples_dir, "Iris")
            iris_data_file_path = fs.get_child_path(iris_data_dir, "Iris.txt")
            iris_dataset = {
                "tables": {"Iris": (iris_data_file_path, None)},
                "format": ("\t", True),
            }

            # Test if the 'fit' output files were created
            classifier = KhiopsClassifier(output_dir=output_dir)
            classifier.fit(iris_dataset, "Class")
            self.assertTrue(fs.exists(fs.get_child_path(output_dir, "AllReports.khj")))
            self.assertTrue(fs.exists(fs.get_child_path(output_dir, "Modeling.kdic")))

            # Test if the 'predict' output file was created
            with io.BytesIO(fs.read(iris_data_file_path)) as iris_data_file:
                iris_df = pd.read_csv(iris_data_file, sep="\t")
                iris_df.pop("Class")
            classifier.predict(iris_df)
            self.assertTrue(fs.exists(fs.get_child_path(output_dir, "transformed.txt")))

            # Cleanup
            for filename in fs.list_dir(output_dir):
                fs.remove(fs.get_child_path(output_dir, filename))

        def test_khiops_coclustering_with_remote_access(self):
            """Test the training of a khiops_coclustering with remote resources"""
            # Skip if only short tests are run
            KhiopsTestHelper.skip_long_test(self)

            # Setup paths
            output_dir = (
                kh.get_runner().khiops_temp_dir
                + f"/KhiopsCoclustering_output_dir_{uuid.uuid4()}/"
            )
            splice_data_dir = fs.get_child_path(
                kh.get_runner().samples_dir, "SpliceJunction"
            )
            splice_data_file_path = fs.get_child_path(
                splice_data_dir, "SpliceJunctionDNA.txt"
            )

            # Read the splice junction secondary datatable
            with io.BytesIO(fs.read(splice_data_file_path)) as splice_data_file:
                splice_df = pd.read_csv(splice_data_file, sep="\t")

            # Fit the coclustering
            khcc = KhiopsCoclustering(output_dir=output_dir)
            khcc.fit(splice_df, id_column="SampleId")

            # Test if the 'fit' files were created
            self.assertTrue(
                fs.exists(fs.get_child_path(output_dir, "Coclustering.kdic"))
            )
            self.assertTrue(
                fs.exists(fs.get_child_path(output_dir, "Coclustering.khcj"))
            )

        def test_train_predictor_fail_and_log_with_remote_access(self):
            """Test train_predictor failure and access to a remote log"""
            log_file_path = fs.get_child_path(
                kh.get_runner().khiops_temp_dir, f"khiops_log_{uuid.uuid4()}.log"
            )
            iris_data_dir = fs.get_child_path(kh.get_runner().samples_dir, "Iris")
            with self.assertRaises(kh.KhiopsRuntimeError):
                kh.train_predictor(
                    fs.get_child_path(iris_data_dir, "NONEXISTENT.kdic"),
                    dictionary_name="Iris",
                    data_table_path=fs.get_child_path(iris_data_dir, "Iris.txt"),
                    target_variable="Class",
                    results_dir=fs.get_child_path(
                        self.results_dir_root(),
                        f"test_{self.remote_access_test_case()}_remote_files",
                    ),
                    log_file_path=log_file_path,
                )
            # Check and remove log file
            self.assertTrue(fs.exists(log_file_path))
            fs.remove(log_file_path)


class KhiopsS3RemoteFileTests(KhiopsRemoteAccessTestsContainer.KhiopsRemoteAccessTests):
    """Integration tests with Amazon S3 filesystems"""

    @classmethod
    def setUpClass(cls):
        """Sets up remote directories in runner"""
        if s3_config_exists():
            runner = kh.get_runner()
            bucket_name = os.environ["S3_BUCKET_NAME"]
            runner.samples_dir = f"s3://{bucket_name}/project/khiops-cicd/samples"
            runner.khiops_temp_dir = f"s3://{bucket_name}/project/khiops-cicd/tmp"
            runner.root_temp_dir = f"s3://{bucket_name}/project/khiops-cicd/tmp"

    @classmethod
    def tearDownClass(cls):
        """Sets back the runner defaults"""
        if s3_config_exists():
            kh.get_runner().__init__()

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
            runner.samples_dir = f"gs://{bucket_name}/khiops-cicd/samples"
            runner.khiops_temp_dir = f"gs://{bucket_name}/khiops-cicd/tmp"
            runner.root_temp_dir = f"gs://{bucket_name}/khiops-cicd/tmp"

    @classmethod
    def tearDownClass(cls):
        """Sets back the runner defaults"""
        if gcs_config_exists():
            kh.get_runner().__init__()

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
              launches the service and makes sure it is operational before excuting the
              test case.  Otherwise it skips the test case.
            - Then it copies ``samples`` to a shared directory accessible to both the
              local Khiops runner service and the process using Khiops Python.
            - Finnaly it creates create the `.KhiopsDockerRunner` client for the
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
            docker_runner.khiops_temp_dir = os.path.join(shared_dir, "tmp")
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
            shutil.rmtree(kh.get_runner().samples_dir)
            shutil.rmtree(kh.get_runner().khiops_temp_dir)

        # Reset the Khiops Python runner to the initial one
        kh.set_runner(cls.initial_runner)

    def config_exists(self):
        return docker_runner_config_exists()

    def remote_access_test_case(self):
        return "KhiopsDockerRunner"

    def results_dir_root(self):
        return kh.get_runner().khiops_temp_dir
