###############################################################################
# Copyright (c) 2022 Orange - All Rights Reserved
# * This software is the confidential and proprietary information of Orange.
# * You shall not disclose such Restricted Information and shall use it only in
#   accordance with the terms of the license agreement you entered into with
#   Orange named the "Khiops - Python Library Evaluation License".
# * Unauthorized copying of this file, via any medium is strictly prohibited.
# * See the "LICENSE.md" file for more details.
##############################################################################
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

import pykhiops.core as pk
import pykhiops.core.filesystems as fs
from pykhiops.extras.docker import PyKhiopsDockerRunner
from pykhiops.sklearn import KhiopsClassifier, KhiopsCoclustering


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


class PyKhiopsRemoteAccessTestsContainer:
    """Container class to allow unittest.TestCase inheritance"""

    class PyKhiopsRemoteAccessTests(unittest.TestCase):
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
            iris_data_file_res = (
                fs.create_resource(pk.get_runner().samples_dir)
                .create_child("Iris")
                .create_child("Iris.txt")
            )
            iris_kdic_file_res = (
                fs.create_resource(pk.get_runner().samples_dir)
                .create_child("Iris")
                .create_child("Iris.kdic")
            )
            pk.train_predictor(
                iris_kdic_file_res.uri,
                dictionary_name="Iris",
                data_table_path=iris_data_file_res.uri,
                target_variable="Class",
                results_dir=self.results_dir_root()
                + f"/test_{self.remote_access_test_case()}_remote_files",
                trace=True,
            )

        def test_khiops_classifier_with_remote_access(self):
            """Test the training of a khiops_classifier with remote resources"""
            # Setup paths
            output_dir_uri = (
                pk.get_runner().khiops_tmp_dir
                + f"/KhiopsClassifier_output_dir_{uuid.uuid4()}/"
            )
            output_dir_res = fs.create_resource(output_dir_uri)
            iris_data_file_res = (
                fs.create_resource(pk.get_runner().samples_dir)
                .create_child("Iris")
                .create_child("Iris.txt")
            )
            iris_dataset = {
                "tables": {"Iris": (iris_data_file_res.uri, None)},
                "format": ("\t", True),
            }

            # Test if the 'fit' output files were created
            classifier = KhiopsClassifier(output_dir=output_dir_uri)
            classifier.fit(iris_dataset, "Class")
            self.assertTrue(output_dir_res.create_child("AllReports.khj").exists())
            self.assertTrue(output_dir_res.create_child("Modeling.kdic").exists())

            # Test if the 'predict' output file was created
            with io.BytesIO(iris_data_file_res.read()) as iris_data_file_contents:
                iris_df = pd.read_csv(iris_data_file_contents, sep="\t")
                iris_df.pop("Class")
            classifier.predict(iris_df)
            self.assertTrue(output_dir_res.create_child("transformed.txt").exists())

            # Cleanup
            for filename in output_dir_res.list_dir():
                output_dir_res.create_child(filename).remove()

        def test_khiops_coclustering_with_remote_access(self):
            """Test the training of a khiops_coclustering with remote resources"""
            # Setup paths
            output_dir_uri = (
                pk.get_runner().khiops_tmp_dir
                + f"/KhiopsCoclustering_output_dir_{uuid.uuid4()}/"
            )
            output_dir_res = fs.create_resource(output_dir_uri)

            splice_data_file_res = (
                fs.create_resource(pk.get_runner().samples_dir)
                .create_child("SpliceJunction")
                .create_child("SpliceJunctionDNA.txt")
            )

            # Read the splice junction secondary datatable
            with io.BytesIO(splice_data_file_res.read()) as splice_data_file_contents:
                splice_df = pd.read_csv(splice_data_file_contents, sep="\t")

            # Fit the coclustering
            pkcc = KhiopsCoclustering(output_dir=output_dir_uri)
            pkcc.fit(splice_df, id_column="SampleId")

            # Test if the 'fit' files were created
            self.assertTrue(output_dir_res.create_child("Coclustering.kdic").exists())
            self.assertTrue(output_dir_res.create_child("Coclustering.khcj").exists())

        def test_train_predictor_fail_and_log_with_remote_access(self):
            """Test train_predictor failure and access to a remote log"""
            log_file_res = fs.create_resource(
                f"{pk.get_runner().khiops_tmp_dir}/khiops_log_{uuid.uuid4()}.log"
            )
            iris_data_file_res = (
                fs.create_resource(pk.get_runner().samples_dir)
                .create_child("Iris")
                .create_child("Iris.txt")
            )
            iris_fake_kdic_file_res = (
                fs.create_resource(pk.get_runner().samples_dir)
                .create_child("Iris")
                .create_child("INEXISTENT_KDIC.kdic")
            )
            with self.assertRaises(pk.PyKhiopsRuntimeError):
                pk.train_predictor(
                    iris_fake_kdic_file_res.uri,
                    dictionary_name="Iris",
                    data_table_path=iris_data_file_res.uri,
                    target_variable="Class",
                    results_dir=self.results_dir_root()
                    + f"/test_{self.remote_access_test_case()}_remote_files",
                    log_file_path=log_file_res.uri,
                )
            # Check and remove log file
            self.assertTrue(log_file_res.exists())
            log_file_res.remove()


class PyKhiopsS3RemoteFileTests(
    PyKhiopsRemoteAccessTestsContainer.PyKhiopsRemoteAccessTests
):
    """Integration tests with Amazon S3 filesystems"""

    @classmethod
    def setUpClass(cls):
        """Sets up remote directories in runner"""
        if s3_config_exists():
            runner = pk.get_runner()
            bucket_name = os.environ["S3_BUCKET_NAME"]
            runner.samples_dir = f"s3://{bucket_name}/project/pykhiops-cicd/samples"
            runner.khiops_tmp_dir = f"s3://{bucket_name}/project/pykhiops-cicd/tmp"
            runner.root_temp_dir = f"s3://{bucket_name}/project/pykhiops-cicd/tmp"

    @classmethod
    def tearDownClass(cls):
        """Sets back the runner defaults"""
        if s3_config_exists():
            pk.get_runner().__init__()

    def config_exists(self):
        return s3_config_exists()

    def remote_access_test_case(self):
        return "S3"


class PyKhiopsGCSRemoteFileTests(
    PyKhiopsRemoteAccessTestsContainer.PyKhiopsRemoteAccessTests
):
    """Integration tests with Google Cloud Storage filesystems"""

    @classmethod
    def setUpClass(cls):
        """Sets up remote directories in runner"""
        if gcs_config_exists():
            runner = pk.get_runner()
            bucket_name = os.environ["GCS_BUCKET_NAME"]
            runner.samples_dir = f"gs://{bucket_name}/pykhiops-cicd/samples"
            runner.khiops_tmp_dir = f"gs://{bucket_name}/pykhiops-cicd/tmp"
            runner.root_temp_dir = f"gs://{bucket_name}/pykhiops-cicd/tmp"

    @classmethod
    def tearDownClass(cls):
        """Sets back the runner defaults"""
        if gcs_config_exists():
            pk.get_runner().__init__()

    def config_exists(self):
        return gcs_config_exists()

    def remote_access_test_case(self):
        return "GCS"


class PyKhiopsDockerRunnerTests(
    PyKhiopsRemoteAccessTestsContainer.PyKhiopsRemoteAccessTests
):
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
              local Khiops runner service and the process using pyKhiops.
            - Finnaly it creates create the `.PyKhiopsDockerRunner` client for the
              Khiops service and set it as current runner.
        """
        # Save the initial pyKhiops runner
        cls.initial_runner = pk.get_runner()

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

            # copy the samples directory
            source_samples_dir = cls.initial_runner.samples_dir
            target_samples_dir = os.path.join(shared_dir, "samples")
            target_samples_dir_res = fs.create_resource(target_samples_dir)
            if not target_samples_dir_res.exists():
                target_samples_dir_res.make_dir()
            for dir_name, _, file_names in os.walk(source_samples_dir):
                target_dir_name_res = fs.create_resource(
                    os.path.join(target_samples_dir, os.path.split(dir_name)[1])
                )
                if not target_dir_name_res.exists():
                    target_dir_name_res.make_dir()
                for file_name in file_names:
                    target_dir_name_res.copy_from_local(
                        os.path.join(dir_name, file_name)
                    )

            # create khiops service runner
            docker_runner = PyKhiopsDockerRunner(
                url=os.environ["KHIOPS_DOCKER_RUNNER_URL"],
                shared_dir=shared_dir,
                insecure=True,
            )
            docker_runner.khiops_tmp_dir = os.path.join(shared_dir, "tmp")
            docker_runner.samples_dir = target_samples_dir

            # set current runner to the created Khiops service runner
            pk.set_runner(docker_runner)

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
            shutil.rmtree(pk.get_runner().samples_dir)
            shutil.rmtree(pk.get_runner().khiops_tmp_dir)

        # Reset the pyKhiops runner to the initial one
        pk.set_runner(cls.initial_runner)

    def config_exists(self):
        return docker_runner_config_exists()

    def remote_access_test_case(self):
        return "PyKhiopsDockerRunner"

    def results_dir_root(self):
        return pk.get_runner().khiops_tmp_dir
