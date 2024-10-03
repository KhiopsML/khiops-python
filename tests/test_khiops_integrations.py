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
import unittest

import khiops.core as kh
from khiops.core.exceptions import KhiopsEnvironmentError
from khiops.core.internals.runner import KhiopsLocalRunner
from khiops.extras.docker import KhiopsDockerRunner
from khiops.sklearn.estimators import KhiopsClassifier
from tests.test_helper import KhiopsTestHelper

# Eliminate protected-access check from these tests
# pylint: disable=protected-access


def _get_linux_distribution_from_os_release_file(os_release_file_path):
    # The `NAME` variable is always defined according to the freedesktop.org
    # standard:
    # https://www.freedesktop.org/software/systemd/man/latest/os-release.html
    with open(os_release_file_path, encoding="ascii") as os_release_info_file:
        for entry in os_release_info_file:
            if entry.startswith("NAME"):
                linux_distribution = entry.split("=")[-1].strip('"\n')
                break
    return linux_distribution


def _get_linux_distribution_name():
    """Detect Linux distribution name

    Parses the `NAME` variable defined in the  `/etc/os-release` or
    `/usr/lib/os-release` files and converts it to lowercase.

    Returns
    -------
    str
        Name of the Linux distribution, converted to lowecase

    Raises
    ------
    OSError
        If neither `/etc/os-release` nor `/usr/lib/os-release` are found
    """

    assert platform.system() == "Linux"

    # If Python version >= 3.10, use standard library support; see
    # https://docs.python.org/3/library/platform.html#platform.freedesktop_os_release
    python_ver_major, python_ver_minor, _ = platform.python_version_tuple()
    if int(python_ver_major) >= 3 and int(python_ver_minor) >= 10:
        linux_distribution = platform.freedesktop_os_release()["NAME"]

    # If Python version < 3.10, determine the Linux distribution manually,
    # but mimic the behavior of Python >= 3.10 standard library support
    else:
        # First try to parse /etc/os-release
        try:
            linux_distribution = _get_linux_distribution_from_os_release_file(
                os.path.join(os.sep, "etc", "os-release")
            )
        except FileNotFoundError:
            # Fallback on parsing /usr/lib/os-release
            try:
                linux_distribution = _get_linux_distribution_from_os_release_file(
                    os.path.join(os.sep, "usr", "lib", "os-release")
                )
            # Mimic `platform.freedesktop_os_release` function behavior
            except FileNotFoundError as error:
                raise OSError from error
    return linux_distribution.lower()


class KhiopsRunnerEnvironmentTests(unittest.TestCase):
    """Test that runners in different environments work"""

    @unittest.skipIf(
        platform.system() != "Linux", "Skipping test for non-Linux platform"
    )
    def test_runner_has_mpiexec_on_linux(self):
        """Test that local runner has executable mpiexec on Linux if MPI is installed"""
        # Check package is installed on supported platform:
        # Check /etc/os-release for Linux version
        linux_distribution = _get_linux_distribution_name()
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
            mpiexec_path = shutil.which(runner.mpi_command_args[0])
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

        # Create a fresh local runner
        runner = KhiopsLocalRunner()

        # Check that MODL* files as set in the runner exist and are executable
        self.assertTrue(os.path.isfile(runner.khiops_path))
        self.assertTrue(os.access(runner.khiops_path, os.X_OK))
        self.assertTrue(os.path.isfile(runner.khiops_coclustering_path))
        self.assertTrue(os.access(runner.khiops_coclustering_path, os.X_OK))

        # Check that mpiexec is set correctly in the runner:
        mpi_command_args = runner.mpi_command_args
        self.assertTrue(len(mpi_command_args) > 0)
        mpiexec_path = shutil.which(runner.mpi_command_args[0])
        self.assertTrue(os.path.exists(mpiexec_path))
        self.assertTrue(os.path.isfile(mpiexec_path))
        self.assertTrue(os.access(mpiexec_path, os.X_OK))


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
