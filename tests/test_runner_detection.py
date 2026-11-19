######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Focused installation detection tests, without native Khiops executables."""

import ast
import ntpath
import os
import subprocess
import sys
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path, PurePosixPath, PureWindowsPath
from unittest.mock import patch

from khiops.core.exceptions import KhiopsEnvironmentError
from khiops.core.internals import runner

# pylint: disable=protected-access


class WindowsPath(PureWindowsPath):
    """Use Windows path semantics without requiring a Windows host."""

    def resolve(self):
        return self

    def is_dir(self):
        return self in self.directories

    def is_file(self):
        return self in self.regular_files

    def read_text(self):
        return self.configuration


class PosixPath(PurePosixPath):
    """Use POSIX path semantics on every host."""

    def resolve(self):
        return self


class RunnerDetectionTests(unittest.TestCase):
    """Exercise the shared root helper, classification and script selection."""

    prefix = r"C:\Conda Envs\test"
    layouts = (r"Lib\site-packages", r"lib\python\site-packages")

    @contextmanager
    def windows_installation(self, layout, missing=None, configuration=None):
        binary_dir = WindowsPath(self.prefix) / "Library" / "bin"
        metadata_dir = WindowsPath(self.prefix) / "conda-meta"
        script = binary_dir / "khiops_env.cmd"
        directories = {binary_dir, metadata_dir} - {missing}
        regular_files = {script} - {missing}
        if configuration is not None:
            regular_files.add(WindowsPath(r"C:\other\pyvenv.cfg"))
        with ExitStack() as stack:
            for name, value in (
                ("Path", WindowsPath),
                (
                    "__file__",
                    str(
                        WindowsPath(self.prefix)
                        / layout
                        / "khiops"
                        / "core"
                        / "internals"
                        / "runner.py"
                    ),
                ),
                ("os.path", ntpath),
                ("platform.system", lambda: "Windows"),
                ("sys.prefix", r"C:\other"),
            ):
                stack.enter_context(patch(f"{runner.__name__}.{name}", value))
            stack.enter_context(
                patch.object(WindowsPath, "directories", directories, create=True)
            )
            stack.enter_context(
                patch.object(WindowsPath, "regular_files", regular_files, create=True)
            )
            stack.enter_context(
                patch.object(WindowsPath, "configuration", configuration, create=True)
            )
            stack.enter_context(
                patch.object(
                    ntpath, "exists", lambda path: WindowsPath(path) in regular_files
                )
            )
            stack.enter_context(
                patch.object(
                    ntpath, "isfile", lambda path: WindowsPath(path) in regular_files
                )
            )
            stack.enter_context(
                patch.object(runner.KhiopsRunner, "_initialize_root_temp_dir")
            )
            yield

    def test_windows_roots(self):
        for layout in (
            *self.layouts,
            r"LiB\PyThOn\SiTe-PaCkAgEs",
            r"Lib\dist-packages",
        ):
            with self.subTest(layout=layout), self.windows_installation(layout):
                self.assertEqual(
                    runner._infer_base_dir_for_conda_based_or_pip_installations(),
                    self.prefix,
                )

    def test_short_windows_paths(self):
        for filename, expected in (
            (r"C:\khiops\core\internals\runner.py", ""),
            (r"C:\lib\python\site-packages\khiops\core\internals\runner.py", "C:\\"),
        ):
            with (
                self.subTest(filename=filename),
                self.windows_installation(self.layouts[0]),
                patch.object(runner, "__file__", filename),
            ):
                self.assertEqual(
                    runner._infer_base_dir_for_conda_based_or_pip_installations(),
                    expected,
                )

    def test_posix_roots(self):
        for system in ("Linux", "Darwin"):
            for layout in (
                "lib/python3.15/site-packages",
                "Lib/python3.15t/dist-packages",
            ):
                filename = f"/env/{layout}/khiops/core/internals/runner.py"
                with (
                    self.subTest(system=system, layout=layout),
                    patch.object(runner.platform, "system", return_value=system),
                    patch.object(runner, "__file__", filename),
                    patch.object(runner, "Path", PosixPath),
                ):
                    self.assertEqual(
                        runner._infer_base_dir_for_conda_based_or_pip_installations(),
                        "/env",
                    )

    def test_classification_and_environment_precedence(self):
        for layout in self.layouts:
            with (
                self.subTest(layout=layout),
                self.windows_installation(layout),
                patch.dict(os.environ, {"CONDA_PREFIX": self.prefix}),
            ):
                before = os.environ.copy()
                environment = {}
                self.assertEqual(
                    runner._infer_khiops_installation_method(environment), "conda-based"
                )
                self.assertEqual(runner._infer_khiops_installation_method(), "conda")
                self.assertEqual(environment, {})
                self.assertEqual(dict(os.environ), before)

    def test_incomplete_conda_and_virtual_environment_fallbacks(self):
        for missing in ("conda-meta", r"Library\bin", r"Library\bin\khiops_env.cmd"):
            for configuration, expected in (
                (None, "pip"),
                ("command = python -m venv", "pip"),
                ("uv = 0.9", "uv"),
            ):
                with (
                    self.subTest(missing=missing, configuration=configuration),
                    self.windows_installation(
                        self.layouts[1],
                        missing=WindowsPath(self.prefix) / missing,
                        configuration=configuration,
                    ),
                ):
                    self.assertEqual(
                        runner._infer_khiops_installation_method({}), expected
                    )

    def test_interpreter_prefix_does_not_override_package_location(self):
        with (
            self.windows_installation(self.layouts[1]),
            patch.object(runner.sys, "prefix", self.prefix),
            patch.object(
                runner,
                "__file__",
                r"C:\separate\lib\python\site-packages\khiops\core\internals\runner.py",
            ),
        ):
            self.assertEqual(runner._infer_khiops_installation_method({}), "pip")

    def test_pip_and_uv_script_selection(self):
        for method in ("pip", "uv"):
            with (
                self.subTest(method=method),
                self.windows_installation(self.layouts[0]),
                patch.object(
                    runner, "_infer_khiops_installation_method", return_value=method
                ),
                patch.object(runner, "distribution") as distribution,
                patch.object(
                    runner.sysconfig, "get_path", return_value=r"C:\other\Scripts"
                ),
                patch.object(ntpath, "exists", return_value=True),
                patch.object(runner.subprocess, "Popen") as popen,
                patch.object(runner.KhiopsLocalRunner, "_check_tools"),
                patch.object(
                    runner.KhiopsLocalRunner, "_initialize_default_samples_dir"
                ),
            ):
                process = popen.return_value.__enter__.return_value
                process.communicate.return_value = ("", "")
                process.returncode = 0
                runner.KhiopsLocalRunner(environment={})
                distribution.assert_called_once_with("khiops-core")
                self.assertEqual(
                    popen.call_args.args[0],
                    [r"C:\other\Scripts\khiops_env.cmd", "--env"],
                )

    def test_conda_script_initialization(self):
        for layout in self.layouts:
            for returncode in (0, 1):
                with (
                    self.subTest(layout=layout, returncode=returncode),
                    self.windows_installation(layout),
                    patch.object(
                        runner,
                        "distribution",
                        side_effect=AssertionError("Unexpected Pip branch"),
                    ),
                    patch.object(runner.subprocess, "Popen") as popen,
                    patch.object(runner.KhiopsLocalRunner, "_check_tools"),
                    patch.object(
                        runner.KhiopsLocalRunner, "_initialize_default_samples_dir"
                    ),
                ):
                    process = popen.return_value.__enter__.return_value
                    process.communicate.return_value = (
                        "TEST_VARIABLE value\n",
                        "script diagnostic",
                    )
                    process.returncode = returncode
                    before = os.environ.copy()
                    environment = {}
                    if returncode:
                        with self.assertRaisesRegex(
                            KhiopsEnvironmentError, "script diagnostic"
                        ):
                            runner.KhiopsLocalRunner(environment=environment)
                    else:
                        local_runner = runner.KhiopsLocalRunner(environment=environment)
                        self.assertEqual(
                            local_runner._environment["TEST_VARIABLE"], "value"
                        )
                    self.assertEqual(
                        popen.call_args.args[0],
                        [self.prefix + r"\Library\bin\khiops_env.cmd", "--env"],
                    )
                    self.assertEqual(popen.call_args.kwargs["env"], {})
                    self.assertEqual(environment, {})
                    self.assertEqual(dict(os.environ), before)

    def test_fresh_process_system_fallback(self):
        # Execute only the production imports and detector in a -S process: importing
        # unittest or the whole Core module can preload importlib.util and hide the bug.
        source = Path(runner.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = [
            node
            for node in tree.body
            if isinstance(node, ast.Import)
            and any(alias.name.startswith("importlib") for alias in node.names)
        ]
        detector = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_infer_khiops_installation_method"
        )
        code = "\n".join(
            ast.get_source_segment(source, node) for node in [*imports, detector]
        )
        code += """
class Path:
    def __init__(self, value): pass
    def is_file(self): return False
class Sys:
    prefix = '/other'
class Os:
    class path:
        @staticmethod
        def join(*parts): return '/'.join(parts)
sys = Sys()
os = Os()
_current_environment = lambda environment: environment
_infer_env_bin_dir_for_conda_based_installations = lambda: '/env/bin'
_check_conda_env_bin_dir = lambda directory: False
assert _infer_khiops_installation_method({}) == 'pip'
importlib.util.find_spec = lambda name: None
assert _infer_khiops_installation_method({}) == 'unknown'
"""
        subprocess.run(
            [sys.executable, "-S", "-c", code],
            check=True,
            capture_output=True,
            text=True,
        )
