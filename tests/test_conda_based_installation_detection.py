######################################################################################
# Copyright (c) 2023-2026 Orange. All rights reserved.                               #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Test the detection of Conda-based installations"""

import ntpath
import posixpath
import tempfile
import unittest
from pathlib import Path, PurePosixPath, PureWindowsPath
from unittest import mock

from khiops.core.internals import runner

# pylint: disable=protected-access

MODULE_PARTS = ("khiops", "core", "internals", "runner.py")


class WindowsModulePath(PureWindowsPath):
    """Windows path semantics on any host, without altering contents on the disk"""

    def resolve(self):
        return self


class PosixModulePath(PurePosixPath):
    """POSIX path semantics on any host, without altering contents on the disk"""

    def resolve(self):
        return self


class RunnerDetectionTests(unittest.TestCase):
    """Test the Conda-based installation detection"""

    def test_base_dir_inference_on_windows(self):
        """Test the base directory inference with Windows paths"""
        for module_dir, expected_base_dir in [
            (
                r"C:\Users\runneradmin\miniconda3\envs\test\lib\python\site-packages",
                r"C:\Users\runneradmin\miniconda3\envs\test",
            ),
            (r"C:\envs\test\Lib\site-packages", r"C:\envs\test"),
        ]:
            module_file = ntpath.join(module_dir, *MODULE_PARTS)
            with self.subTest(module_dir=module_dir):
                with (
                    mock.patch.object(runner, "__file__", module_file),
                    mock.patch.object(runner, "Path", WindowsModulePath),
                    mock.patch.object(runner.os, "path", ntpath),
                ):
                    self.assertEqual(
                        runner._infer_base_dir_for_conda_based_or_pip_installations(),
                        expected_base_dir,
                    )

    def test_base_dir_inference_on_posix(self):
        """Test the base directory inference with POSIX paths"""
        for module_dir, expected_base_dir in [
            ("/env/lib/python3.15/site-packages", "/env"),
            ("/env/lib/python3.15t/site-packages", "/env"),
            ("/env/Lib/python3.15/dist-packages", "/env"),
            ("/usr/lib/python3/dist-packages", "/usr"),
            # Peculiar installation: no site-packages directory
            ("/", ""),
        ]:
            module_file = posixpath.join(module_dir, *MODULE_PARTS)
            with self.subTest(module_dir=module_dir):
                with (
                    mock.patch.object(runner, "__file__", module_file),
                    mock.patch.object(runner, "Path", PosixModulePath),
                ):
                    self.assertEqual(
                        runner._infer_base_dir_for_conda_based_or_pip_installations(),
                        expected_base_dir,
                    )

    def test_conda_based_installation_is_detected(self):
        """Test that a complete Conda environment is not detected as `pip`"""
        for system, site_packages_parts, binary_dir_parts, script_name in [
            ("Windows", ("Lib", "site-packages"), ("Library", "bin"), "khiops_env.cmd"),
            (
                "Windows",
                ("lib", "python", "site-packages"),
                ("Library", "bin"),
                "khiops_env.cmd",
            ),
            ("Linux", ("lib", "python3.15", "site-packages"), ("bin",), "khiops_env"),
            ("Linux", ("lib", "python3.15t", "site-packages"), ("bin",), "khiops_env"),
            ("Darwin", ("lib", "python3.15", "site-packages"), ("bin",), "khiops_env"),
        ]:
            with (
                self.subTest(system=system, site_packages_parts=site_packages_parts),
                tempfile.TemporaryDirectory() as temp_dir,
            ):
                # Create a Conda environment which contains the `khiops_env`
                # script
                prefix = Path(temp_dir).resolve()
                (prefix / "conda-meta").mkdir()
                binary_dir = prefix.joinpath(*binary_dir_parts)
                binary_dir.mkdir(parents=True)
                (binary_dir / script_name).touch()
                module_file = str(prefix.joinpath(*site_packages_parts, *MODULE_PARTS))
                with (
                    mock.patch.object(runner.platform, "system", return_value=system),
                    mock.patch.object(runner, "__file__", module_file),
                ):
                    # No CONDA_PREFIX defined: the environment is not activated
                    self.assertEqual(
                        runner._infer_khiops_installation_method({}), "conda-based"
                    )


if __name__ == "__main__":
    unittest.main()
