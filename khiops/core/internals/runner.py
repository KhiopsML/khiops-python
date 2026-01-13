######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################

# Important note:
#  To detect the installation environment, this module makes use of the path in
#  `__file__`. If you move to another place make sure you properly update any line using
#  Path(__file__).

"""Classes implementing Khiops Python' backend runners"""

import io
import os
import platform
import shlex
import shutil
import site
import subprocess
import sys
import tempfile
import uuid
import warnings
from abc import ABC, abstractmethod
from importlib.metadata import PackageNotFoundError, files
from pathlib import Path

import khiops
import khiops.core.internals.filesystems as fs
from khiops.core.exceptions import KhiopsEnvironmentError, KhiopsRuntimeError
from khiops.core.internals.common import (
    CommandLineOptions,
    SystemSettings,
    invalid_keys_message,
    is_string_like,
    type_error_message,
)
from khiops.core.internals.io import KhiopsOutputWriter
from khiops.core.internals.task import KhiopsTask
from khiops.core.internals.version import KhiopsVersion


def _isdir_without_all_perms(dir_path):
    """Returns True if the path is a directory but missing of any of rwx permissions"""
    return os.path.isdir(dir_path) and not os.access(
        dir_path, os.R_OK | os.W_OK | os.X_OK
    )


def get_default_samples_dir():
    """Returns the default samples directory

    The default samples directory is computed according to the following priorities:
        - all systems: ``KHIOPS_SAMPLES_DIR/khiops_data/samples`` if set
        - Windows:
            - ``%PUBLIC%\\khiops_data\\samples`` if ``%PUBLIC%`` is defined
            - ``%USERPROFILE%\\khiops_data\\samples`` otherwise
        - Linux/macOS: ``$HOME/khiops_data/samples``
    """
    if "KHIOPS_SAMPLES_DIR" in os.environ and os.environ["KHIOPS_SAMPLES_DIR"]:
        samples_dir = os.environ["KHIOPS_SAMPLES_DIR"]
    elif platform.system() == "Windows" and "PUBLIC" in os.environ:
        samples_dir = os.path.join(os.environ["PUBLIC"], "khiops_data", "samples")
    else:
        # The filesystem abstract layer is used here
        # as the path can be either local or remote
        samples_dir = fs.get_child_path(
            fs.get_child_path(os.environ["HOME"], "khiops_data"), "samples"
        )
    return samples_dir


def _get_dir_status(a_dir):
    """Returns the status of a local or remote directory

    Against a local directory a real check is performed. A remote directory is detected
    but not checked.
    """
    if fs.is_local_resource(a_dir):
        a_dir_res = fs.create_resource(os.path.normpath(a_dir))

        # a_dir_res is a LocalFilesystemResource already
        a_dir_path = a_dir_res.path
        if not os.path.exists(a_dir_path):
            status = "non-existent"
        elif not os.path.isdir(a_dir_path):
            status = "not-a-dir"
        else:
            status = "ok"
    else:
        status = "remote-path"

    assert status in ["non-existent", "not-a-dir", "ok", "remote-path"]
    return status


def _check_samples_dir(samples_dir):
    # Warn if there are problems with the samples_dir
    samples_dir_status = _get_dir_status(samples_dir)
    download_msg = (
        "Execute the kh-download-datasets script or "
        "the khiops.tools.download_datasets function to download them."
    )
    if samples_dir_status == "non-existent":
        warnings.warn(
            "Sample datasets location does not exist "
            f"({samples_dir}). {download_msg}",
            stacklevel=3,
        )
    elif samples_dir_status == "not-a-dir":
        warnings.warn(
            "Sample datasets location is not a directory "
            f"({samples_dir}). {download_msg}",
            stacklevel=3,
        )


def _khiops_env_file_exists(env_dir):
    """Check ``khiops_env`` exists relative to the specified environment dir"""
    khiops_env_path = os.path.join(env_dir, "khiops_env")
    if platform.system() == "Windows":
        khiops_env_path += ".cmd"
    return os.path.exists(khiops_env_path) and os.path.isfile(khiops_env_path)


def _infer_env_bin_dir_for_conda_based_installations():
    """Infers the bin directory of
    *supposed* Conda-based Khiops installations

    Returns
    -------
    str
        absolute path of the 'bin' dir where the khiops binaries are installed

        .. note::
            Borderline case : if no Conda-based Khiops installation is found
            this function will return 'bin'
    """
    conda_env_dir = _infer_base_dir_for_conda_based_or_pip_installations()

    # Conda env binary dir is:
    # - on Windows: conda_env_dir\Library\bin
    # - on Linux/macOS: conda_env_dir\bin
    if platform.system() == "Windows":
        env_bin_dir = os.path.join(str(conda_env_dir), "Library", "bin")
    else:
        env_bin_dir = os.path.join(str(conda_env_dir), "bin")

    return env_bin_dir


def _infer_base_dir_for_conda_based_or_pip_installations():
    """Infers reference directory (base directory)
     for Khiops installations

    This function detects
    - 'conda' and 'conda-based' installations
    - system-wide pure python installation (in a dist-packages folder)
    - pure python virtual environment installation (in a site-packages folder)
    Any installation in an unexpected location is regarded as borderline

    Returns
    -------
    str
        An absolute path to the base directory

        .. note::
            It returns an empty string if it detects a borderline installation
    """
    assert os.path.basename(Path(__file__).parents[2]) == "khiops", (
        "Please fix the `Path.parents` in this method "
        "so it finds environment directory of this module"
    )

    # Obtain a normalized (OS-dependent and without symlinks) full path
    # of the current file
    current_file_path = Path(__file__).resolve()

    # Windows: Match either
    # %CONDA_PREFIX%\Lib\site-packages\khiops\core\internals\runner.py
    # or {python lib root}\Lib\dist-packages\khiops\core\internals\runner.py
    # or {virtual env root}\Lib\site-packages\khiops\core\internals\runner.py
    if platform.system() == "Windows":
        # safeguard to prevent an IndexError on borderline installations
        if len(current_file_path.parents) < 6:
            base_dir = ""
        else:
            base_dir = str(current_file_path.parents[5])
    # Linux/macOS: Match either
    # $CONDA_PREFIX/[Ll]ib/python3.X/site-packages/khiops/core/internals/runner.py
    # or {python lib root}/
    #       [Ll]ib/python3.X/dist-packages/khiops/core/internals/runner.py
    # or {virtual env root}/
    #       [Ll]ib/python3.X/site-packages/khiops/core/internals/runner.py
    else:
        # safeguard to prevent an IndexError on borderline installations
        if len(current_file_path.parents) < 7:
            base_dir = ""
        else:
            base_dir = str(current_file_path.parents[6])

    return base_dir


def _check_conda_env_bin_dir(conda_env_bin_dir):
    """Checks inferred Conda environment binary directory really is one

    A real Conda environment binary directory:
    - should exist
    - should not be directly under the root directory
    - should coexist with `conda-meta` directory under the same parent
    """
    conda_env_bin_dir_path = Path(conda_env_bin_dir)

    # Conda env bin dir should end with `/bin`
    assert conda_env_bin_dir_path.parts[-1] == "bin"

    is_conda_env_bin_dir = False

    # Conda env dir is not equal to its root dir
    # Conda env bin dir exists, along with the `conda-meta` dir
    # Note: On Windows, Conda env bin dir equals conda env dir\Library\bin
    conda_env_dir_path = conda_env_bin_dir_path.parent
    if platform.system() == "Windows":
        conda_env_dir_path = conda_env_dir_path.parent
    if (
        str(conda_env_dir_path) != conda_env_dir_path.root  # `.root` is an `str`
        and conda_env_bin_dir_path.is_dir()
        and conda_env_dir_path.joinpath("conda-meta").is_dir()
    ):
        is_conda_env_bin_dir = True
    return is_conda_env_bin_dir


def _infer_khiops_installation_method(trace=False):
    """Returns the Khiops installation method

    Definitions :
    - 'conda' environment contains binaries, shared libraries and python libraries
    - 'conda-based' environment is similar to 'conda' except that
       it was not activated previously nor during the execution
       and thus the CONDA_PREFIX environment variable is undefined
       and the path to the `bin` directory inside the conda environment is not in PATH
    - 'binary+pip' installs the binaries and the shared libraries system-wide
      but will keep the python libraries
      in the python system folder
      or in the Python folder inside the home directory of the user,
      or in a virtual environment (if one is used)

    """
    # We are in a Conda environment if
    # - the CONDA_PREFIX environment variable exists and,
    # - the khiops_env script exists within:
    #   - `%CONDA_PREFIX\Library\bin%` on Windows
    #   - `$CONDA_PREFIX/bin` on Linux and MacOS
    # Note: The check that the Khiops binaries are actually executable is done
    #       afterwards by the initializations method.
    installation_method = "unknown"
    if "CONDA_PREFIX" in os.environ:
        conda_env_dir = os.environ["CONDA_PREFIX"]
        if platform.system() == "Windows":
            conda_binary_dir = os.path.join(conda_env_dir, "Library", "bin")
        else:
            conda_binary_dir = os.path.join(conda_env_dir, "bin")
        if _khiops_env_file_exists(conda_binary_dir):
            installation_method = "conda"
    # Otherwise (installation_method is still "unknown"), we choose between
    # conda-based and local (default choice)
    if installation_method == "unknown":
        env_bin_dir = _infer_env_bin_dir_for_conda_based_installations()
        if trace:
            print(f"Environment binary dir: '{env_bin_dir}'")
        if _check_conda_env_bin_dir(env_bin_dir) and _khiops_env_file_exists(
            env_bin_dir
        ):
            installation_method = "conda-based"
        else:
            installation_method = "binary+pip"
    if trace:
        print(f"Installation method: '{installation_method}'")
    assert installation_method in ("conda", "conda-based", "binary+pip")
    return installation_method


def _check_executable(bin_path):
    if not os.path.isfile(bin_path):
        raise KhiopsEnvironmentError(f"Non-regular executable file. Path: {bin_path}")
    elif not os.access(bin_path, os.X_OK):
        raise KhiopsEnvironmentError(
            f"Executable has no execution rights. Path: {bin_path}"
        )


def _get_current_library_installer():
    """Returns the installer of the python library

    Returns
    -------
    str
        installer name among : 'pip', 'conda' or 'unknown'
    """

    try:
        # Each time a python library is installed a 'dist-info' folder is created
        # Normalized files can be found in this folder
        installer_files = [path for path in files("khiops") if path.name == "INSTALLER"]
        if len(installer_files) > 0:
            try:
                return installer_files[0].read_text().strip()
            except FileNotFoundError:
                # At this step a FileNotFoundError exception can still occur
                # because the files list is read first from a RECORD file
                # before the filesystem is actually accessed.
                # The exception is ignored here because a warning
                # for the general case of a missing INSTALLER file
                # will be created below.
                pass
        # No "INSTALLER" file is found inside the package metadata
        warnings.warn(
            "The python library metadata exists ('khiops-*.dist-info') "
            "but seems corrupted as no INSTALLER file can be found. "
            "Please re-install using the same tool ('conda' or 'pip').",
            stacklevel=3,
        )
        return "unknown"
    except PackageNotFoundError:
        # The python library is not installed via standard tools like conda, pip...
        return "unknown"


def _build_khiops_process_environment():
    """Build a specific environment used for the execution of khiops in a process

    This environment can be modified freely without interfering
    with the global one.
    """
    khiops_env = os.environ.copy()

    # Ensure HOME is always set for OpenMPI 5+
    # (using KHIOPS_MPI_HOME if it exists)
    if "HOME" not in khiops_env:
        khiops_env["HOME"] = khiops_env.get("KHIOPS_MPI_HOME", "")
    return khiops_env


class KhiopsRunner(ABC):
    """Abstract Khiops Python runner to be re-implemented"""

    def __init__(self):
        """See class docstring"""
        self._initialize_root_temp_dir()
        self._khiops_version = None
        self._samples_dir = None

        # Whether to write the Khiops Python library version of the scenarios
        # For development uses only
        self._write_version = True

    def _initialize_root_temp_dir(self):
        """Initializes the runner's root temporary directory

        It tries to set a proper root temporary directory. It tries the following
        strategies in order:
        - Check that ``$TEMP/khiops/python`` exists and use it
        - Try to create ``$TEMP/khiops/python` and use it
        - Create a ``$TEMP/khiops_<HASH>/python`` and use it
        """

        # Create the directory if it doesn't exists
        self._root_temp_dir = os.path.join(tempfile.gettempdir(), "khiops", "python")
        if not os.path.exists(self._root_temp_dir):
            os.makedirs(self._root_temp_dir)
        # Create the dir with a hash name if it is a dir but it doesn't have all
        # permissions or if it is a file
        elif os.path.isfile(self._root_temp_dir) or _isdir_without_all_perms(
            self._root_temp_dir
        ):
            self._root_temp_dir = os.path.join(
                tempfile.mkdtemp(prefix="khiops_"), "python"
            )
            os.makedirs(self._root_temp_dir, exist_ok=True)

    @property
    def root_temp_dir(self):
        r"""str: The runner's temporary directory

        The temporary scenarios/templates and dictionary files created by
        the Khiops Python library are stored here.

        Default value:
            - Windows: ``%TEMP%\khiops\python``
            - Linux: ``$TMP/khiops/python``

        When set to a local path it tries to create the specified directory if it
        doesn't exist.

        Raises
        ------
        `.KhiopsEnvironmentError`
            If set to a local path: if it is a file or if it does not have ``+rwx``
            permissions.
        """
        return self._root_temp_dir

    @root_temp_dir.setter
    def root_temp_dir(self, dir_path):
        # Check existence, directory status and permissions for local paths
        if fs.is_local_resource(dir_path):
            real_dir_path = fs.create_resource(dir_path).path
            if os.path.exists(real_dir_path):
                if os.path.isfile(real_dir_path):
                    raise KhiopsEnvironmentError(
                        f"File at temporary directory os.path. Path: {real_dir_path}"
                    )
                elif _isdir_without_all_perms(real_dir_path):
                    raise KhiopsEnvironmentError(
                        "Temporary directory must have +rwx permissions. "
                        f"Path: {real_dir_path}"
                    )
            else:
                os.makedirs(real_dir_path)
        # There are no checks for non-local filesystems (no `else` statement)
        self._root_temp_dir = dir_path

    def create_temp_file(self, prefix, suffix):
        """Creates a unique temporary file in the runner's root temporary directory

        .. note::
            For remote filesystems no actual file is created, just a (highly probable)
            unique path is returned.

        Parameters
        ----------
        prefix : str
            Prefix for the file's name.

        suffix : str
            Suffix for the file's name.

        Returns
        -------
        str
            A unique path within the root temporary directory. The file is created only
            in the case of a local filesystem.
        """
        # Local resource: Effectively create the file with the python file API
        if fs.is_local_resource(self.root_temp_dir):
            # Extract the path from the potential URI
            root_temp_dir_path = fs.create_resource(self.root_temp_dir).path

            # Create the temporary file
            tmp_file_fd, tmp_file_path = tempfile.mkstemp(
                prefix=prefix, suffix=suffix, dir=root_temp_dir_path
            )
            os.close(tmp_file_fd)
        # Remote resource: Just return a highly probable unique path
        else:
            tmp_file_path = fs.get_child_path(
                self.root_temp_dir, f"{prefix}{uuid.uuid4()}{suffix}"
            )

        return tmp_file_path

    def create_temp_dir(self, prefix):
        """Creates a unique directory in the runner's root temporary directory

        Parameters
        ----------
        prefix : str
            Prefix for the directory's name.

        Returns
        -------
        str
            A unique directory path within the root temporary directory. The directory
            is created only in the case of a local filesystem.
        """
        # Local resource: Effectively create the directory with the python file API
        if fs.is_local_resource(self.root_temp_dir):
            root_temp_dir_path = fs.create_resource(self.root_temp_dir).path
            temp_dir = tempfile.mkdtemp(prefix=prefix, dir=root_temp_dir_path)
        # Remote resource: Just return a highly probable unique path
        else:
            temp_dir = fs.get_child_path(self.root_temp_dir, f"{prefix}{uuid.uuid4()}")
        return temp_dir

    @property
    def samples_dir(self):
        r"""str: Location of the Khiops' sample datasets directory. May be an URL/URI"""
        return self._get_samples_dir()

    def _get_samples_dir(self):
        """To be overridden by subclasses"""
        return self._samples_dir

    @samples_dir.setter
    def samples_dir(self, samples_dir):
        if not is_string_like(samples_dir):
            raise TypeError(
                type_error_message("samples_dir", samples_dir, "string-like")
            )
        self._set_samples_dir(samples_dir)

    def _set_samples_dir(self, samples_dir):
        """To be overridden by child classes to add additional checks"""
        self._samples_dir = samples_dir

    @property
    def khiops_version(self):
        """`.KhiopsVersion`: The version of the Khiops backend of this runner"""
        return self._get_khiops_version()

    def _get_khiops_version(self):
        """khiops_version getter to be overridden by subclasses"""
        return self._khiops_version

    def _build_status_message(self):
        """Constructs the status message

        Descendant classes can add additional information.

        Returns
        -------
        tuple
            A 3-tuple containing in this order :
            - The status message
            - A list of error messages (str)
            - A list of warning messages (str)
        """
        # Capture the status of the samples dir
        warning_list = []
        with warnings.catch_warnings(record=True) as caught_warnings:
            samples_dir_path = self.samples_dir
        if caught_warnings is not None:
            # caught_warnings contains a list of WarningMessage
            warning_list.extend([w.message for w in caught_warnings])

        # the following path is accurate only if the current file
        # is still in the 'khiops.core.internals' package
        assert (
            os.path.basename(Path(__file__).parents[2]) == "khiops"
        ), "Please fix the `Path.parents` in this method "
        library_root_dir = Path(__file__).parents[2]

        status_msg = "Khiops Python library settings\n"
        status_msg += f"version             : {khiops.__version__}\n"
        status_msg += f"runner class        : {self.__class__.__name__}\n"
        status_msg += f"root temp dir       : {self.root_temp_dir}\n"
        status_msg += f"sample datasets dir : {samples_dir_path}\n"
        status_msg += f"library root dir    : {library_root_dir}\n"

        error_list = []

        return status_msg, error_list, warning_list

    def print_status(self):
        """Prints the status of the runner to stdout"""
        # Obtains the status_msg, errors and warnings
        status_msg, error_list, warning_list = self._build_status_message()

        # Print status details
        print(status_msg, end="")

        if error_list or warning_list:
            print("Installation issues detected:\n")
            print("---\n")

        # Print the errors (if any)
        if error_list:
            print("Errors:")
            for error in error_list:
                print(f"\tError: {error}\n")

        # Print the warnings (if any)
        if warning_list:
            print("Warnings:")
            for warning in warning_list:
                print(f"\tWarning: {warning}\n")

        # The exit code is non-zero if there are errors
        if len(error_list) == 0:
            return 0
        return 1

    @abstractmethod
    def _initialize_khiops_version(self):
        """Initialization of `khiops_version` to be implemented in child classes"""

    def run(
        self,
        task,
        task_args,
        command_line_options=None,
        trace=False,
        system_settings=None,
        stdout_file_path="",
        stderr_file_path="",
        force_ansi_scenario=False,
        **kwargs,
    ):
        """Runs a Khiops Task

        Parameters
        ----------
        task : `.KhiopsTask`
            Khiops task to be run.
        task_args : dict
            Arguments for the task.
        command_line_options : `.CommandLineOptions`, optional
            Command line options for all tasks. If not set the default values are used.
            See the `.CommandLineOptions` for more information.
        trace : bool, default ``False``
            If True prints the command line executed of the process and does not delete
            any temporary files created.
        system_settings : `.SystemSettings`, optional
            *Advanced:* System settings for all tasks. See the `.SystemSettings`
            class for more information.
        stdout_file_path : str, default ""
            *Advanced* Path to a file where the Khiops process writes its stdout stream.
            Normally Khiops should not write to this stream but MPI, filesystems plugins
            or debug versions may do it. The stream is captured with a UTF-8 encoding
            and replacing encoding errors. If equal to "" then it writes no file.
        stderr_file_path : str, default ""
            *Advanced* Path to a file where the Khiops process writes its stderr stream.
            Normally Khiops should not write to this stream but MPI, filesystems plugins
            or debug versions may do it. The stream is captured with a UTF-8 encoding
            and replacing encoding errors. If equal to "" then it writes no file.
        force_ansi_scenario : bool, default ``False``
            *Advanced:* If True the internal scenario generated by Khiops will force
            characters such as accentuated ones to be decoded with the UTF8->ANSI khiops
            transformation.

        Raises
        ------
        `ValueError`
            - Unknown keyword argument
            - Files or executable not found
            - Errors in the execution of the Khiops tool

        `TypeError`
            - Invalid type of a keyword argument
            - When the search/replace pairs are not strings
        """
        # Warn if there are still kwargs: At this point any keyword argument is invalid
        if kwargs:
            warnings.warn(invalid_keys_message(kwargs), stacklevel=3)
            kwargs.clear()

        # Use the default command line options if not specified
        if command_line_options is None:
            command_line_options = CommandLineOptions()

        # Check the call arguments
        if not isinstance(trace, bool):
            raise TypeError(type_error_message("trace", trace, bool))
        command_line_options.check()
        if not isinstance(stdout_file_path, str):
            raise TypeError(
                type_error_message("stdout_file_path", stdout_file_path, str)
            )
        if not isinstance(stderr_file_path, str):
            raise TypeError(
                type_error_message("stderr_file_path", stderr_file_path, str)
            )

        # Write the scenarios file
        scenario_path = self._write_task_scenario_file(
            task,
            task_args,
            system_settings,
            force_ansi_scenario=force_ansi_scenario,
        )

        # If no log file specified: Use a temporary file
        tmp_log_file_path = None
        if not command_line_options.log_file_path:
            tmp_log_file_path = self.create_temp_file("_run_", ".log")
            command_line_options.log_file_path = tmp_log_file_path

        # Execute Khiops
        try:
            # Disable pylint warning about abstract method _run returning None
            # pylint: disable=assignment-from-no-return
            return_code, stdout, stderr = self._run(
                task.tool_name,
                scenario_path,
                command_line_options,
                trace,
            )
            # pylint: enable=assignment-from-no-return
        # Catch an OS level error if any
        except OSError as error:
            raise KhiopsRuntimeError("Khiops execution failed.") from error
        # Report any errors raised by Khiops
        else:
            # Write the stdout and stderr streams if specified
            if stdout_file_path:
                fs.write(stdout_file_path, bytes(stdout, encoding="utf8"))
            if stderr_file_path:
                fs.write(stderr_file_path, bytes(stderr, encoding="utf8"))

            # Report the exit status
            self._report_exit_status(
                task.tool_name,
                return_code,
                stdout,
                stderr,
                command_line_options.log_file_path,
            )
        # Clean files unless trace mode is activated
        finally:
            if trace:
                print(f"Khiops execution scenario: {scenario_path}")
                print(f"Khiops log file: {command_line_options.log_file_path}")
            else:
                fs.remove(scenario_path)
                if tmp_log_file_path is not None:
                    fs.remove(tmp_log_file_path)

    def _report_exit_status(
        self, tool_name, return_code, stdout, stderr, log_file_path
    ):
        """Reports the exit status of a Khiops execution"""
        # Note:
        #   We report stdout and stderr below because we use a log file and thus
        #   normally Khiops doesn't write anything to these streams. In practice MPI and
        #   the remote filesystems plugins may write to them to report anomalies.

        # Report messages:
        # - The warnings in the log
        # - The errors and/or fatal errors in the log
        # - The stdout if not empty
        # - The stderr if not empty
        #
        # If there were any errors (fatal or not) or the return code is non-zero the
        # reporting is via an exception. Otherwise we show the message as a warning.
        #

        # Create the message reporting the errors and warnings
        error_msg = ""
        # If the log file exists: Collect the errors and warnings messages
        if fs.exists(log_file_path):
            errors, fatal_errors, warning_messages = self._collect_errors(log_file_path)
            if warning_messages:
                error_msg += "Warnings in log:\n" + "".join(warning_messages)
            if errors:
                if error_msg:
                    error_msg += "\n"
                error_msg += "Errors in log:\n" + "".join(errors)
            if fatal_errors:
                if error_msg:
                    error_msg += "\n"
                error_msg += "Fatal errors in log:\n" + "".join(fatal_errors)
        # Otherwise warn that the log file is missing
        else:
            warnings.warn(
                f"Log file not found after {tool_name} execution."
                f"Path: {log_file_path}"
            )
            errors = fatal_errors = []

        # Add stdout to the warning message if non empty
        if stdout:
            if error_msg:
                error_msg += "\n"
            error_msg += f"Contents of stdout:\n{stdout}"

        # Add stderr to the warning message if non empty
        if stderr:
            if error_msg:
                error_msg += "\n"
            error_msg += f"Contents of stderr:\n{stderr}"

        # Report the message to the user if there were any
        if error_msg:
            # Raise an exception if there were errors
            if errors or fatal_errors or return_code != 0:
                raise KhiopsRuntimeError(
                    f"{tool_name} execution had errors (return code {return_code}):\n"
                    f"{error_msg}"
                )
            # Otherwise show the message as a warning
            else:
                error_msg = (
                    f"Khiops ended correctly but there were minor issues:\n{error_msg}"
                )
                warnings.warn(error_msg.rstrip())
        # Raise an exception anyway for a non-zero return code without any message
        else:
            if return_code != 0:
                raise KhiopsRuntimeError(
                    f"{tool_name} execution had errors (return code {return_code}) "
                    "but no message is available"
                )

    def _collect_errors(self, log_file_path):
        # Collect errors any errors found in the log
        errors = []
        fatal_errors = []
        warning_messages = []

        # Look in the log for error lines
        log_file_lines = None
        try:
            log_file_contents = fs.read(log_file_path)
            log_file_lines = io.TextIOWrapper(
                io.BytesIO(log_file_contents), encoding="utf8", errors="replace"
            )
            for line_number, line in enumerate(log_file_lines, start=1):
                if line.startswith("warning : "):
                    warning_messages.append(f"Line {line_number}: {line}")
                elif line.startswith("error : "):
                    errors.append(f"Line {line_number}: {line}")
                elif line.startswith("fatal error : "):
                    fatal_errors.append(f"Line {line_number}: {line}")

        # Warn on error for remote file handling. Replace with empty log file.
        except ImportError:
            warnings.warn(
                "Could not read remote log file and errors may not be "
                "reported. Make sure you have installed the extra "
                "dependencies for remote filesystems.",
                stacklevel=3,
            )

        return errors, fatal_errors, warning_messages

    def _create_scenario_file(self, task):
        assert isinstance(task, KhiopsTask)
        return self.create_temp_file(f"{task.name}_", "._kh")

    def _write_task_scenario_file(
        self, task, task_args, system_settings, force_ansi_scenario=False
    ):
        scenario_path = self._create_scenario_file(task)
        with io.BytesIO() as scenario_stream:
            writer = KhiopsOutputWriter(scenario_stream, force_ansi=force_ansi_scenario)
            if self._write_version:
                writer.writeln(f"// Generated by khiops-python {khiops.__version__}")
            self._write_task_scenario(writer, task, task_args, system_settings)
            fs.write(scenario_path, scenario_stream.getvalue())

        return scenario_path

    def _write_task_scenario(self, writer, task, task_args, system_settings):
        assert isinstance(task, KhiopsTask)
        assert isinstance(task_args, dict)
        assert isinstance(system_settings, SystemSettings)

        # Write the task scenario
        self._write_scenario_prologue(writer, system_settings)
        task.write_execution_scenario(writer, task_args)
        self._write_scenario_exit_statement(writer)

    def _write_scenario_prologue(self, writer, system_settings):
        # Write the system settings if any
        if (
            system_settings.max_cores
            or system_settings.memory_limit_mb
            or system_settings.temp_dir
        ):
            writer.writeln("// System settings")
            if system_settings.max_cores:
                writer.write("AnalysisSpec.SystemParameters.MaxCoreNumber ")
                writer.writeln(str(system_settings.max_cores))
            if system_settings.memory_limit_mb:
                writer.write("AnalysisSpec.SystemParameters.MemoryLimit ")
                writer.writeln(str(system_settings.memory_limit_mb))
            if system_settings.temp_dir:
                writer.write("AnalysisSpec.SystemParameters.TemporaryDirectoryName ")
                writer.writeln(system_settings.temp_dir)
            writer.writeln("")

        # Write the user defined prologue
        if system_settings.scenario_prologue:
            writer.writeln("// User-defined prologue")
            for line in system_settings.scenario_prologue.split("\n"):
                writer.writeln(line)
            writer.writeln("")

    def _write_scenario_exit_statement(self, writer):
        writer.writeln("")
        writer.writeln("// Exit Khiops")
        writer.writeln("ClassManagement.Quit")
        writer.writeln("OK")

    @abstractmethod
    def _run(
        self,
        tool_name,
        scenario_path,
        command_line_options,
        trace,
    ):
        """Abstract run method to be implemented in child classes

        Returns
        -------
        tuple
            A 3-tuple containing the return code, the stdout and the stderr of the
            Khiops process.

        Raises
        ------
        `.KhiopsRuntimeError`
            If there were any errors in the Khiops execution.
        """


class KhiopsLocalRunner(KhiopsRunner):
    r"""Implementation of a local Khiops runner

    Requires either:

    - This library installed through Conda and run from a Conda environment, or
    - the ``khiops-core`` Linux native library installed on the local machine, or
    - the Windows Khiops desktop application installed on the local machine

    .. rubric:: Samples directory settings

    Default values for the ``samples_dir`` attribute:

    - The value of the ``KHIOPS_SAMPLES_DIR`` environment variable (path to the Khiops
      sample datasets directory).
    - Otherwise:

      - Windows:

        - ``%PUBLIC%\khiops_data\samples%`` if ``%PUBLIC%`` is defined
        - ``%USERPROFILE%\khiops_data\samples%`` otherwise

      - Linux and macOS:

        - ``$HOME/khiops_data/samples``

    """

    def __init__(self):
        # Define specific attributes
        self._mpi_command_args = None
        self._khiops_path = None
        self._khiops_coclustering_path = None
        self._khiops_version = None
        self._samples_dir = None
        self._samples_dir_checked = False

        # Call parent constructor
        super().__init__()

        # Initialize Khiops environment
        self._initialize_khiops_environment()

    def _initialize_khiops_environment(self):
        # Check the `khiops_env` script
        # On Windows native installations, rely on the `KHIOPS_HOME` environment
        # variable set by the Khiops Desktop Application installer
        installation_method = _infer_khiops_installation_method()
        if platform.system() == "Windows" and installation_method == "binary+pip":
            # KHIOPS_HOME variable by default
            if "KHIOPS_HOME" in os.environ:
                khiops_env_path = os.path.join(
                    os.environ["KHIOPS_HOME"], "bin", "khiops_env.cmd"
                )
            # Raise error if KHIOPS_HOME is not set
            else:
                raise KhiopsEnvironmentError(
                    "No environment variable named 'KHIOPS_HOME' found. "
                    "Make sure you have installed Khiops >= 10.2.3. "
                    "Go to https://khiops.org for more information."
                )

        # In Conda-based environments, `khiops_env` might not be in the PATH,
        # hence its path must be inferred
        elif installation_method == "conda-based":
            khiops_env_path = os.path.join(
                _infer_env_bin_dir_for_conda_based_installations(), "khiops_env"
            )
            if platform.system() == "Windows":
                khiops_env_path += ".cmd"

        # On UNIX or Conda, khiops_env is always in path for a proper installation
        else:
            khiops_env_path = shutil.which("khiops_env")
            if khiops_env_path is None:
                raise KhiopsEnvironmentError(
                    "The 'khiops_env' script not found for the current "
                    f"'{installation_method}' installation method. Make sure "
                    "you have installed khiops >= 10.2.3. "
                    "Go to https://khiops.org for more information."
                )

        with subprocess.Popen(
            [khiops_env_path, "--env"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        ) as khiops_env_process:
            stdout, stderr = khiops_env_process.communicate()
            if khiops_env_process.returncode != 0:
                raise KhiopsEnvironmentError(
                    "Error initializing the environment for Khiops from the "
                    f"{khiops_env_path} script. Contents of stderr:\n{stderr}"
                )
            for line in stdout.split("\n"):
                tokens = line.rstrip().split(maxsplit=1)
                if len(tokens) == 2:
                    var_name, var_value = tokens
                elif len(tokens) == 1:
                    var_name = tokens[0]
                    var_value = ""
                else:
                    continue

                # Special var to export in order
                # to prepend to or to set to HOME for OpenMPI 5+
                # when running khiops core
                if var_name == "KHIOPS_MPI_HOME":
                    os.environ["KHIOPS_MPI_HOME"] = var_value
                # Set paths to Khiops binaries
                elif var_name == "KHIOPS_PATH":
                    self.khiops_path = var_value
                    os.environ["KHIOPS_PATH"] = var_value
                elif var_name == "KHIOPS_COCLUSTERING_PATH":
                    self.khiops_coclustering_path = var_value
                    os.environ["KHIOPS_COCLUSTERING_PATH"] = var_value
                # Set MPI command
                elif var_name == "KHIOPS_MPI_COMMAND":
                    self._mpi_command_args = shlex.split(var_value)
                    os.environ["KHIOPS_MPI_COMMAND"] = var_value
                # Propagate all the other environment variables to Khiops binaries
                else:
                    os.environ[var_name] = var_value

                # Set KHIOPS_API_MODE to `true`
                os.environ["KHIOPS_API_MODE"] = "true"

        # Check the tools exist and are executable
        self._check_tools()

        # Initialize the default samples dir
        self._initialize_default_samples_dir()

    def _initialize_default_samples_dir(self):
        """See class docstring"""
        samples_dir = get_default_samples_dir()
        _check_samples_dir(samples_dir)
        self._samples_dir = samples_dir
        assert self._samples_dir is not None

    def _check_tools(self):
        """Checks the that the tool binaries exist and are executable"""
        for tool_name in ["khiops", "khiops_coclustering"]:
            _check_executable(self._tool_path(tool_name))

    def _initialize_khiops_version(self):
        # Run khiops with the -v
        stdout, stderr, return_code = self.raw_run("khiops", ["-v"], use_mpi=False)

        # On success parse and save the version
        if return_code == 0:
            # Skip potential non-version lines
            # (ex: Completed loading of file driver, debug info ...)
            for line in stdout.split("\n"):
                line = line.strip("\r")  # remove Windows-specific Carriage-Return
                if line.startswith("Khiops"):
                    khiops_version_str = line.rstrip().split(" ")[1]
                    break
        # If MODL -v fails we raise an informative error
        else:
            error_msg = (
                f"Could not obtain the Khiops version."
                f"\nError executing '{self._tool_path('khiops')} -v': "
                f"return code {return_code}."
            )
            error_msg += f"\nstdout: {stdout}" if stdout else ""
            error_msg += f"\nstderr: {stderr}" if stderr else ""
            raise KhiopsRuntimeError(error_msg)

        # Khiops core version
        self._khiops_version = KhiopsVersion(khiops_version_str)

        # Library version
        compatible_khiops_version = khiops.get_compatible_khiops_version()

        operating_system = platform.system()
        installation_method = _infer_khiops_installation_method()

        # Fail immediately if the major versions differ
        # Note: the installation status will not show at all
        if self.khiops_version.major != compatible_khiops_version.major:
            raise KhiopsRuntimeError(
                f"Major version '{self.khiops_version.major}' of the Khiops "
                "executables does not match the Khiops Python library major version "
                f"'{compatible_khiops_version.major}'. "
                "To avoid any compatibility error, "
                "please update either the Khiops "
                f"executables package for your '{operating_system}' operating "
                "system, or the Khiops Python library, "
                f"according to your '{installation_method}' environment. "
                "See https://khiops.org for more information.",
            )

        # Warn if the khiops minor and patch versions do not match
        # the Khiops Python library ones
        # KhiopsVersion implements the equality operator, which however also
        # takes pre-release tags into account.
        # The restriction here does not apply to pre-release tags
        if (
            self.khiops_version.minor,
            self.khiops_version.patch,
        ) != (
            compatible_khiops_version.minor,
            compatible_khiops_version.patch,
        ):
            warnings.warn(
                f"Version '{self._khiops_version}' of the Khiops executables "
                "does not match the Khiops Python library version "
                f"'{khiops.__version__}' (different minor.patch version). "
                "There may be compatibility errors and "
                "we recommend to update either the Khiops "
                f"executables package for your '{operating_system}' operating "
                "system, or the Khiops Python library, "
                f"according to your '{installation_method}' environment. "
                "See https://khiops.org for more information.",
                stacklevel=3,
            )

    def _detect_library_installation_incompatibilities(self, library_root_dir):
        """Detects known incompatible installations of this library
        in the 3 installation modes see `_infer_khiops_installation_method`
        (binary+pip, conda, conda-based)

        The error_list or warning_list collections
        are not empty if an issue is detected

        Parameters
        ----------
        library_root_dir : PosixPath
            path to this current library


        Returns
        -------
        tuple
            A 2-tuple containing:
                - a list of error messages
                - a list of warning messages
        """

        error_list = []
        warning_list = []

        installation_method = _infer_khiops_installation_method()
        # activated 'conda' installation
        if installation_method == "conda":

            # under Windows, the system-wide msmpi
            # will take precedence over the one installed
            # in the current 'conda' environment.
            # Because msmpi will not be updated that much
            # this case can be ignored
            windows_msmpi_path = os.path.join("c:", "Windows", "System32", "msmpi.dll")
            if platform.system() == "Windows" and os.path.exists(windows_msmpi_path):
                warning = (
                    "You have a system wide installation of MSMPI in "
                    f"{windows_msmpi_path}. "
                    "You conda environment will raise a warning "
                    "about a possible overshadowing. "
                    "You can ignore this warning.\n"
                )
                warning_list.append(warning)

            # the conda environment must match the library installation
            if not str(library_root_dir).startswith(os.environ["CONDA_PREFIX"]):
                error = (
                    f"Khiops Python library installation path '{library_root_dir}' "
                    "does not match the current Conda environment "
                    f"'{os.environ['CONDA_PREFIX']}'. "
                    "Either deactivate the current Conda environment "
                    "or use the Khiops Python library "
                    "belonging to the current Conda environment. "
                    "Go to https://khiops.org for instructions.\n"
                )
                error_list.append(error)
            # the khiops executable path must also match the conda environment one
            # meaning khiops core was installed using conda
            if not self.khiops_path.startswith(os.environ["CONDA_PREFIX"]):
                error = (
                    f"Khiops binary path '{self.khiops_path}' "
                    "does not match the current Conda environment "
                    f"'{os.environ['CONDA_PREFIX']}'. "
                    "We recommend installing the Khiops binary "
                    "in the current Conda environment. "
                    "Go to https://khiops.org for instructions.\n"
                )
                error_list.append(error)
        # 'binary+pip', 'conda-based' or borderline installations
        else:

            # ensure a known installer was used otherwise unexpected issues can occur
            # if the installer is unknown we face a borderline installation
            # (for example a run inside a cloned git repo)
            with warnings.catch_warnings(record=True) as caught_warnings:
                current_library_installer = _get_current_library_installer()
            if caught_warnings is not None:
                # caught_warnings contains a list of WarningMessage
                warning_list.extend([w.message for w in caught_warnings])
            if current_library_installer not in ("conda", "pip"):
                warning = (
                    "Khiops Python library "
                    "was not installed with 'conda' or 'pip' "
                    f"but with '{current_library_installer}' installer. "
                    "This will probably lead to unexpected errors. "
                    "Go to https://khiops.org for instructions to re-install it.\n"
                )
                warning_list.append(warning)

            # we consider only the 'binary+pip' and 'conda-based' installations here
            # - 'conda-based' installation (similar to a non-activated virtual env)
            # - 'binary+pip' installation under a virtual env
            # - User site 'binary+pip' installation (without virtual env)
            # - system-wide 'binary+pip' installation (without virtual env)...
            # (an empty string means a borderline installation was found,
            # no further check cannot be performed)
            base_dir = _infer_base_dir_for_conda_based_or_pip_installations()
            if len(base_dir) > 0:
                # within a virtual env, sys.prefix is set to the virtual env folder
                # whereas sys.base_prefix remains unchanged.
                # Please be aware that if a python executable of a virtual env is used
                # the corresponding virtual env is activated and sys.prefix updated
                if sys.base_prefix != sys.prefix:
                    # the python executable location
                    # (within the virtual env or the conda-based env)
                    # must match the library installation
                    if (
                        platform.system() == "Windows"
                        and
                        # Under Windows, there are two cases :
                        (
                            # for conda-based installations python is inside 'base_dir'
                            str(Path(sys.executable).parents[0]) != base_dir
                            and
                            # for 'binary+pip' installations (within a virtual env)
                            # python is inside 'base_dir'/Scripts
                            str(Path(sys.executable).parents[1]) != base_dir
                        )
                        # Under Linux or MacOS a bin/ folder exists
                        or str(Path(sys.executable).parents[1]) != base_dir
                    ):
                        error = (
                            "Khiops Python library installation path "
                            f"'{library_root_dir}' "
                            "does not match the current python environment "
                            f"('{sys.executable}'). "
                            "Go to https://khiops.org for instructions "
                            "to re-install it "
                            "(preferably in a virtual environment).\n"
                        )
                        error_list.append(error)
                else:
                    # the installation is not within a virtual env
                    # (sys.base_prefix == sys.prefix)
                    if not sys.executable.startswith(sys.base_prefix):
                        # the executable is not the expected one
                        # (the system-wide python)
                        error = (
                            "Khiops Python library installed in "
                            f"'{library_root_dir}' "
                            "is run with an unexpected executable "
                            f"'{sys.executable}'. "
                            "The system-wide python located in "
                            f"'{sys.base_prefix}' "
                            "should have been used. "
                            "Go to https://khiops.org for instructions "
                            "to re-install it "
                            "(preferably in a virtual environment).\n"
                        )
                        error_list.append(error)
                    # fetch the 'User site' site-packages path
                    # which is already adapted for each OS (Windows, MacOS, Linux)
                    user_site_packages_dir = site.getusersitepackages()
                    if not str(library_root_dir).startswith(user_site_packages_dir):
                        # the library is not installed on the 'User site'
                        if not str(library_root_dir).startswith(sys.base_prefix):
                            # the library is supposed to be installed system-wide,
                            # but it seems that the location is wrong
                            error = (
                                "Khiops Python library installation path "
                                f"'{library_root_dir}' "
                                "does not match the system-wide Python prefix in "
                                f"'{sys.base_prefix}'. "
                                "Go to https://khiops.org for instructions "
                                "to re-install it "
                                "(preferably in a virtual environment).\n"
                            )
                            error_list.append(error)

        return error_list, warning_list

    def _build_status_message(self):
        # Call the parent's method
        status_msg, error_list, warning_list = super()._build_status_message()

        library_root_dir = Path(__file__).parents[2]

        installation_errors, installation_warnings = (
            self._detect_library_installation_incompatibilities(library_root_dir)
        )

        # Build the messages for install type and mpi
        install_type_msg = _infer_khiops_installation_method()
        if self._mpi_command_args:
            mpi_command_args_msg = " ".join(self._mpi_command_args)
        else:
            mpi_command_args_msg = "<empty>"

        # Build the message
        with warnings.catch_warnings(record=True) as caught_warnings:
            status_msg += "\n\n"
            status_msg += "khiops local installation settings\n"
            status_msg += f"version             : {self.khiops_version}\n"
            status_msg += f"Khiops path         : {self.khiops_path}\n"
            status_msg += f"Khiops CC path      : {self.khiops_coclustering_path}\n"
            status_msg += f"install type        : {install_type_msg}\n"
            status_msg += f"MPI command         : {mpi_command_args_msg}\n"

            # Add output of khiops -s which gives the MODL_* binary status
            status_msg += "\n\n"
            khiops_executable = os.path.join(
                os.path.dirname(self.khiops_path), "khiops"
            )
            status_msg += (
                f"Khiops executable status (output of '{khiops_executable} -s')\n"
            )
            stdout, stderr, return_code = self.raw_run("khiops", ["-s"], use_mpi=True)

            # On success retrieve the status and added to the message
            if return_code == 0:
                status_msg += stdout
            else:
                error_list.append(stderr)
            status_msg += "\n"

        if caught_warnings is not None:
            # caught_warnings contains a list of WarningMessage
            warning_list.extend([w.message for w in caught_warnings])

        return (
            status_msg,
            error_list + installation_errors,
            warning_list + installation_warnings,
        )

    def _get_khiops_version(self):
        # Initialize the first time it is called
        if self._khiops_version is None:
            self._initialize_khiops_version()
        assert isinstance(self._khiops_version, KhiopsVersion), type_error_message(
            self._khiops_version, "khiops_version", KhiopsVersion
        )
        return self._khiops_version

    @property
    def mpi_command_args(self):
        return self._mpi_command_args

    @property
    def khiops_path(self):
        """str: Path to the ``MODL*`` Khiops binary

        Set by the ``khiops_env`` script from the ``khiops-core`` package.

        """
        return self._khiops_path

    @khiops_path.setter
    def khiops_path(self, modl_path):
        # Check that the path is a directory and it exists
        if not os.path.exists(modl_path):
            raise KhiopsEnvironmentError(f"Inexistent Khiops path: '{modl_path}'")
        if not os.path.isfile(modl_path):
            raise KhiopsEnvironmentError(
                f"Khiops file path is a directory: {modl_path}"
            )

        # Set the MODL path
        self._khiops_path = modl_path

    @property
    def khiops_coclustering_path(self):
        """str: Path to the ``MODL_Coclustering`` Khiops Coclustering binary

        Set by the ``khiops_env`` script from the ``khiops-core`` package.

        """
        return self._khiops_coclustering_path

    @khiops_coclustering_path.setter
    def khiops_coclustering_path(self, modl_coclustering_path):
        # Check that the path is a directory and it exists
        if not os.path.exists(modl_coclustering_path):
            raise KhiopsEnvironmentError(
                f"Inexistent Khiops coclustering path: '{modl_coclustering_path}'"
            )
        if not os.path.isfile(modl_coclustering_path):
            raise KhiopsEnvironmentError(
                "Khiops coclustering file path is a directory: "
                f"{modl_coclustering_path}"
            )

        # Set the MODL_Coclustering path
        self._khiops_coclustering_path = modl_coclustering_path

    def _tool_path(self, tool_name):
        """Full path of a Khiops tool binary"""
        assert (
            self.khiops_path is not None and self.khiops_coclustering_path is not None
        )
        tool_name = tool_name.lower()
        if tool_name not in ["khiops", "khiops_coclustering"]:
            raise ValueError(f"Invalid tool name: {tool_name}")
        modl_binaries = {
            "khiops": self.khiops_path,
            "khiops_coclustering": self.khiops_coclustering_path,
        }
        bin_path = modl_binaries[tool_name]

        return bin_path

    def _set_samples_dir(self, samples_dir):
        """Checks and sets the samples directory"""
        _check_samples_dir(samples_dir)
        super()._set_samples_dir(samples_dir)

    def _get_samples_dir(self):
        # Check the samples dir once (the check emits only warnings)
        if not self._samples_dir_checked:
            _check_samples_dir(self._samples_dir)
            self._samples_dir_checked = True
        return self._samples_dir

    def raw_run(self, tool_name, command_line_args=None, use_mpi=True, trace=False):
        """Execute a Khiops tool with given command line arguments

        Parameters
        ----------
        tool_name : {"khiops", "khiops_coclustering"}
            Name of the tool to execute.
        command_line_args : list of str, optional
            Command line arguments of Khiops.
        use_mpi : bool, optional
            Whether to execute the application with MPI
        trace : bool, default False
            If ``True`` print the trace of the process.

        Examples
        --------
        >>> raw_run("khiops", ["-b", "-i" , "scenario._kh"])

        is equivalent to execute in a shell::

            > khiops -b -i scenario._kh
        """
        # Check command_line_args type
        if command_line_args and not isinstance(command_line_args, list):
            raise TypeError(
                type_error_message("command_line_args", command_line_args, list)
            )

        # Build command line arguments
        khiops_process_args = []
        if use_mpi:
            khiops_process_args += self._mpi_command_args
        khiops_process_args += [self._tool_path(tool_name)]
        if command_line_args:
            khiops_process_args += command_line_args

        # If trace is on: display call arguments
        if trace:
            quote = '"' if platform.system() == "Windows" else "'"
            khiops_call = khiops_process_args[0]
            for arg in khiops_process_args[1:]:
                if isinstance(arg, bytes):
                    arg = arg.decode("utf8", errors="replace")
                if arg.startswith("-"):
                    khiops_call += f" {arg}"
                else:
                    khiops_call += f" {quote}{arg}{quote}"
            print(f"Khiops execution call: {khiops_call}")

        # Build custom Khiops process environment
        # which makes sure HOME is defined and set
        # according to khiops_env's KHIOPS_MPI_HOME
        khiops_env = _build_khiops_process_environment()

        # Execute the process
        with subprocess.Popen(
            khiops_process_args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf8",
            errors="replace",
            universal_newlines=True,
            env=khiops_env,
        ) as khiops_process:
            stdout, stderr = khiops_process.communicate()

        return stdout, stderr, khiops_process.returncode

    def _run(
        self,
        tool_name,
        scenario_path,
        command_line_options,
        trace,
    ):
        # Execute the tool
        khiops_args = command_line_options.build_command_line_options(scenario_path)
        stdout, stderr, return_code = self.raw_run(
            tool_name, command_line_args=khiops_args, trace=trace
        )

        return return_code, stdout, stderr


#########################
# Current Runner Access #
#########################

# Disable pylint UPPER_CASE convention: _khiops_runner is non-constant
# pylint: disable=invalid-name

# Runner (backend) of Khiops Python, by default None for lazy initialization
_khiops_runner = None


def set_runner(runner):
    """Sets the current KhiopsRunner of the module"""
    if not isinstance(runner, KhiopsRunner):
        raise TypeError(type_error_message("runner", runner, KhiopsRunner))
    global _khiops_runner
    _khiops_runner = runner


def get_runner():
    """Returns the current KhiopsRunner of the module

    Returns
    -------
    `.KhiopsRunner`
        The current Khiops Python runner of the module.
    """
    #  Define and initialize a runner for a local Khiops installation
    global _khiops_runner
    if _khiops_runner is None:
        _khiops_runner = KhiopsLocalRunner()
    return _khiops_runner


# pylint: enable=invalid-name
