######################################################################################
# Copyright (c) 2024 Orange. All rights reserved.                                    #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Classes implementing Khiops Python' backend runners"""

import io
import os
import platform
import re
import shlex
import shutil
import subprocess
import tempfile
import uuid
import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import khiops
import khiops.core.internals.filesystems as fs
from khiops.core.exceptions import KhiopsEnvironmentError, KhiopsRuntimeError
from khiops.core.internals.common import (
    CommandLineOptions,
    GeneralOptions,
    deprecation_message,
    invalid_keys_message,
    is_string_like,
    removal_message,
    renaming_message,
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


def get_dir_status(a_dir):
    """Returns the status of a local or remote directory

    Against a local directory a real check is performed. A remote directory is detected
    but not checked.
    """
    if fs.is_local_resource(a_dir):
        # Remove initial slash on windows systems
        # urllib's url2pathname does not work properly
        a_dir_res = fs.create_resource(os.path.normpath(a_dir))
        a_dir_path = a_dir_res.uri_info.path
        if platform.system() == "Windows":
            if a_dir_path.startswith("/"):
                a_dir_path = a_dir_path[1:]

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


def check_samples_dir(samples_dir):
    # Warn if there are problems with the samples_dir
    samples_dir_status = get_dir_status(samples_dir)
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


def _extract_path_from_uri(uri):
    res = fs.create_resource(uri)
    if platform.system() == "Windows":
        # Case of file:///<LETTER>:/<REST_OF_PATH>:
        #   Eliminate first slash ("/") from path if the first component
        if (
            res.uri_info.scheme == ""
            and res.uri_info.path[0] == "/"
            and res.uri_info.path[1].isalpha()
            and res.uri_info.path[2] == ":"
        ):
            path = res.uri_info.path[1:]
        # Case of C:/<REST_OF_PATH>:
        #   Just use the original path
        elif len(res.uri_info.scheme) == 1:
            path = uri
        # Otherwise return URI path as-is
        else:
            path = res.uri_info.path

    else:
        path = res.uri_info.path
    return path


def _get_system_cpu_cores():
    """Portably obtains the number of cpu cores (no hyperthreading)"""
    # Set the cpu info command and arguments for each platform
    if platform.system() == "Linux":
        cpu_system_info_args = ["lscpu", "-b", "-p=Core,Socket"]
    elif platform.system() == "Windows":
        cpu_system_info_args = [
            "powershell.exe",
            "-Command",
            "$OutputEncoding = [System.Text.Encoding]::UTF8;",
            "Get-CimInstance -ClassName 'Win32_Processor' "
            "| Select-Object -Property 'NumberOfCores'",
            "| Format-Table -HideTableHeaders ",
            "| Write-Output",
        ]
    elif platform.system() == "Darwin":
        cpu_system_info_args = ["sysctl", "-n", "hw.physicalcpu"]
    else:
        raise KhiopsEnvironmentError(
            f"Unsupported OS {platform.system()}. "
            "Check the supported OSes at https://khiops.org."
        )

    # Execute the cpu info process
    with subprocess.Popen(
        cpu_system_info_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    ) as cpu_system_info_process:
        cpu_system_info_output, _ = cpu_system_info_process.communicate()

    # Count the cpus for each system
    if platform.system() == "Linux":
        # Ignore the comment lines starting with '#' tha lscpu puts in the message
        # Each non commented line is for a cpu
        cpu_entries = {
            entry
            for entry in cpu_system_info_output.splitlines()
            if not entry.startswith("#")
        }
        cpu_core_count = len(cpu_entries)
    elif platform.system() == "Windows":
        # Each line of the cpu count command contains a number of cores of a socket
        cores_per_socket = [
            int(line.strip()) for line in cpu_system_info_output.strip().splitlines()
        ]
        cpu_core_count = sum(cores_per_socket)
    elif platform.system() == "Darwin":
        cpu_core_count = int(cpu_system_info_output.strip())
    else:
        raise KhiopsEnvironmentError(
            f"Unsupported OS {platform.system()}. "
            "Check the supported OSes at https://khiops.org."
        )

    return cpu_core_count


def _compute_max_cores_from_proc_number(proc_number):
    # if KHIOPS_PROC_NUMBER is 0 we set max_cores to the system's core number
    if proc_number == 0:
        max_cores = _get_system_cpu_cores()
    # Otherwise we set max_cores to KHIOPS_PROC_NUMBER
    else:
        max_cores = proc_number

    return max_cores


def _mpiexec_path(bin_dir):
    mpiexec_exe = "mpiexec"
    if platform.system() == "Windows":
        mpiexec_exe += ".exe"
    return os.path.join(bin_dir, mpiexec_exe)


def _modl_and_mpiexec_executables_exist(bin_dir):
    """Check MODL* executables exist relative to the specified binary dir"""
    modl_path = os.path.join(bin_dir, "MODL")
    modl_cc_path = os.path.join(bin_dir, "MODL_Coclustering")
    mpiexec_path = _mpiexec_path(bin_dir)
    if platform.system() == "Windows":
        modl_path += ".exe"
        modl_cc_path += ".exe"
        if not os.path.exists(mpiexec_path):
            mpiexec_path = _mpiexec_path(
                Path(bin_dir).parent.joinpath("Library", "bin").as_posix()
            )
    return all(
        os.path.exists(exe_path) for exe_path in (modl_path, modl_cc_path, mpiexec_path)
    )


def _infer_env_bin_dir_for_conda_based_installations():
    """Infer reference directory for Conda-based Khiops installations"""
    reference_file = khiops.__file__
    assert reference_file is not None
    # Windows: Match %CONDA_PREFIX%\Lib\site-packages\khiops\__init__.py
    if platform.platform() == "Windows":
        conda_env_dir = Path(reference_file).parents[3]
    # Linux/macOS:
    # Match $CONDA_PREFIX/[Ll]ib/python3.X/site-packages/khiops/__init__.py
    else:
        conda_env_dir = Path(reference_file).parents[4]
    env_bin_dir = os.path.join(conda_env_dir.as_posix(), "bin")
    return env_bin_dir


def _check_conda_env_bin_dir(conda_env_bin_dir):
    """Check inferred Conda environment binary directory really is one

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
    conda_env_dir_path = conda_env_bin_dir_path.parent
    if (
        conda_env_dir_path != conda_env_dir_path.root
        and conda_env_bin_dir_path.is_dir()
        and conda_env_dir_path.joinpath("conda-meta").is_dir()
    ):
        is_conda_env_bin_dir = True
    return is_conda_env_bin_dir


def _infer_khiops_installation_method(trace=False):
    """Return the Khiops installation method"""
    # We are in a conda environment if
    # - if the CONDA_PREFIX environment variable exists and,
    # - if MODL, MODL_Coclustering and mpiexec files exists in
    # `$CONDA_PREFIX/bin`
    #
    # Note: The check that MODL and MODL_Coclustering are actually executable is done
    #       afterwards by the initializations method.
    # We are in a conda env if the Khiops binaries exists within `$CONDA_PREFIX/bin`
    if "CONDA_PREFIX" in os.environ and _modl_and_mpiexec_executables_exist(
        os.path.join(os.environ["CONDA_PREFIX"], "bin")
    ):
        installation_method = "conda"
    # Otherwise, we choose between conda-based and local (default choice)
    else:
        env_bin_dir = _infer_env_bin_dir_for_conda_based_installations()
        if trace:
            print(f"Environment binary dir: '{env_bin_dir}'")
        if _check_conda_env_bin_dir(
            env_bin_dir
        ) and _modl_and_mpiexec_executables_exist(env_bin_dir):
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


def get_linux_distribution_name():
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

    def get_linux_distribution_from_os_release_file(os_release_file_path):
        # The `NAME` variable is always defined according to the freedesktop.org
        # standard:
        # https://www.freedesktop.org/software/systemd/man/latest/os-release.html
        with open(os_release_file_path, encoding="ascii") as os_release_info_file:
            for entry in os_release_info_file:
                if entry.startswith("NAME"):
                    linux_distribution = entry.split("=")[-1].strip('"\n')
                    break
        return linux_distribution

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
            linux_distribution = get_linux_distribution_from_os_release_file(
                os.path.join(os.sep, "etc", "os-release")
            )
        except FileNotFoundError:
            # Fallback on parsing /usr/lib/os-release
            try:
                linux_distribution = get_linux_distribution_from_os_release_file(
                    os.path.join(os.sep, "usr", "lib", "os-release")
                )
            # Mimic `platform.freedesktop_os_release` function behavior
            except FileNotFoundError as error:
                raise OSError from error
    return linux_distribution.lower()


class KhiopsRunner(ABC):
    """Abstract Khiops Python runner to be re-implemented"""

    def __init__(self):
        """See class docstring"""
        self.general_options = GeneralOptions()
        self._initialize_root_temp_dir()
        self._khiops_version = None
        self._samples_dir = None

        # Whether to write the khiops-python version of the scenarios
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
        khiops-python are stored here.

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
            real_dir_path = _extract_path_from_uri(dir_path)
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
        # There are no checks for non local filesystems (no `else` statement)
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
            root_temp_dir_path = _extract_path_from_uri(self.root_temp_dir)

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
            root_temp_dir_path = _extract_path_from_uri(self.root_temp_dir)
            temp_dir = tempfile.mkdtemp(prefix=prefix, dir=root_temp_dir_path)
        # Remote resource: Just return a highly probable unique path
        else:
            temp_dir = fs.get_child_path(self.root_temp_dir, f"{prefix}{uuid.uuid4()}")
        return temp_dir

    @property
    def scenario_prologue(self):
        """str: Prologue applicable to prepend to all execution scenarios

        Raises
        ------
        `TypeError`
            If set to a non str object.
        """
        return self.general_options.scenario_prologue

    @scenario_prologue.setter
    def scenario_prologue(self, prologue):
        self.general_options.user_scenario_prologue = prologue
        self.general_options.check()

    @property
    def max_cores(self):
        """int: Maximum number of cores for Khiops executions

        If set to 0 it uses the system's default.

        Raises
        ------
        `TypeError`
            If it is set to a non int object.
        `ValueError`
            If it is set to a negative int.
        """
        return self.general_options.max_cores

    @max_cores.setter
    def max_cores(self, core_number):
        self._set_max_cores(core_number)

    def _set_max_cores(self, core_number):
        self.general_options.max_cores = core_number
        self.general_options.check()

    @property
    def max_memory_mb(self):
        """int: Maximum amount of memory (in MB) for Khiops executions

        If set to 0 it uses the maximum available in the system.

        Raises
        ------
        `TypeError`
            If it is set to a non int object.
        `ValueError`
            If it is set to a negative int.
        """
        return self.general_options.max_memory_mb

    @max_memory_mb.setter
    def max_memory_mb(self, memory_mb):
        self.general_options.max_memory_mb = memory_mb
        self.general_options.check()

    @property
    def khiops_temp_dir(self):
        """str: Temporary directory for Khiops executions

        If equal to ``""`` it uses the system's temporary directory.

        Raises
        ------
        `TypeError`
            If set to a non str object.
        """
        return self.general_options.khiops_temp_dir

    @khiops_temp_dir.setter
    def khiops_temp_dir(self, temp_dir):
        """Setter of khiops_temp_dir"""
        self.general_options.khiops_temp_dir = temp_dir
        self.general_options.check()

    @property
    def samples_dir(self):
        r"""str: Location of the Khiops' sample datasets directory. May be an URL/URI"""
        return self._get_samples_dir()

    def _get_samples_dir(self):
        """To be overriden by subclasses"""
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
        """khiops_version getter to be overriden by subclasses"""
        return self._khiops_version

    def _build_status_message(self):
        """Constructs the status message

        Descendant classes can add additional information.

        Returns
        -------
        tuple
            A 2-tuple containing:
            - The status message
            - A list of warning messages
        """
        # Capture the status of the the samples dir
        warning_list = []
        with warnings.catch_warnings(record=True) as caught_warnings:
            samples_dir_path = self.samples_dir
        if caught_warnings is not None:
            warning_list += caught_warnings

        status_msg = "khiops-python settings\n"
        status_msg += f"version             : {khiops.__version__}\n"
        status_msg += f"runner class        : {self.__class__.__name__}\n"
        status_msg += f"max cores           : {self.max_cores}"
        if self.max_cores == 0:
            status_msg += " (no limit)"
        status_msg += "\n"
        status_msg += f"max memory (MB)     : {self.max_memory_mb}"
        if self.max_memory_mb == 0:
            status_msg += " (no limit)"
        status_msg += "\n"
        status_msg += f"temp dir            : {self.root_temp_dir}\n"
        status_msg += f"sample datasets dir : {samples_dir_path}\n"
        status_msg += f"package dir         : {os.path.dirname(khiops.__file__)}\n"
        return status_msg, warning_list

    def print_status(self):
        """Prints the status of the runner to stdout"""
        # Obtain the status_msg, errors and warnings
        try:
            status_msg, warning_list = self._build_status_message()
        except (KhiopsEnvironmentError, KhiopsRuntimeError) as error:
            print(f"khiops-python status KO: {error}")
            return 1

        # Print status details
        print(status_msg, end="")

        # Print status
        print("khiops-python status OK", end="")
        if warning_list:
            print(", with warnings:")
            for warning in warning_list:
                print(f"warning: {warning.message}")
        else:
            print("")
        return 0

    @abstractmethod
    def _initialize_khiops_version(self):
        """Initialization of `khiops_version` to be implemented in child classes"""

    def run(
        self,
        task,
        task_args,
        command_line_options=None,
        trace=False,
        general_options=None,
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
        general_options : `.GeneralOptions`, optional
            *Advanced:* General options for all tasks. If not set then the runner's
            values are used. Unless you know what are you doing, prefer setting these
            options with the runner's accessors. See the `.GeneralOptions` class for
            more information.
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
        # Handle renamed parameters
        if "batch" in kwargs:
            warnings.warn(
                renaming_message("batch", "batch_mode", "10.0.0"), stacklevel=3
            )
            del kwargs["batch"]
        if "output_script" in kwargs:
            warnings.warn(
                renaming_message("output_script", "output_scenario_path", "10.0.0"),
                stacklevel=3,
            )
            del kwargs["output_script"]
        if "log" in kwargs:
            warnings.warn(
                renaming_message("log", "log_file_path", "10.0.0"), stacklevel=3
            )
            del kwargs["log"]
        if "task" in kwargs:
            warnings.warn(
                renaming_message("task", "task_file_path", "10.0.0"), stacklevel=3
            )
            del kwargs["task"]

        # Handle removed parameters
        if "search_replace" in kwargs:
            warnings.warn(
                removal_message("search_replace", "10.1.2", None), stacklevel=3
            )
            del kwargs["search_replace"]

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
            task, task_args, general_options, force_ansi_scenario=force_ansi_scenario
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
        self, task, task_args, general_options, force_ansi_scenario=False
    ):
        scenario_path = self._create_scenario_file(task)
        with io.BytesIO() as scenario_stream:
            writer = KhiopsOutputWriter(scenario_stream, force_ansi=force_ansi_scenario)
            if self._write_version:
                writer.writeln(f"// Generated by khiops-python {khiops.__version__}")
            self._write_task_scenario(
                writer,
                task,
                task_args,
                (
                    general_options
                    if general_options is not None
                    else self.general_options
                ),
            )
            fs.write(scenario_path, scenario_stream.getvalue())

        return scenario_path

    def _write_task_scenario(self, writer, task, task_args, general_options):
        assert isinstance(task, KhiopsTask)
        assert isinstance(task_args, dict)
        assert isinstance(general_options, GeneralOptions)

        # Write the task scenario
        self._write_scenario_prologue(writer, general_options)
        task.write_execution_scenario(writer, task_args)
        self._write_scenario_exit_statement(writer)

    def _write_scenario_prologue(self, writer, general_options):
        # Write the system settings if any
        if (
            general_options.max_cores
            or general_options.max_memory_mb
            or general_options.khiops_temp_dir
        ):
            writer.writeln("// System settings")
            if general_options.max_cores:
                writer.write("AnalysisSpec.SystemParameters.MaxCoreNumber ")
                writer.writeln(str(general_options.max_cores))
            if general_options.max_memory_mb:
                writer.write("AnalysisSpec.SystemParameters.MemoryLimit ")
                writer.writeln(str(general_options.max_memory_mb))
            if general_options.khiops_temp_dir:
                writer.write("AnalysisSpec.SystemParameters.TemporaryDirectoryName ")
                writer.writeln(general_options.khiops_temp_dir)
            writer.writeln("")

        # Write the user defined prologue
        if general_options.user_scenario_prologue:
            writer.writeln("// User-defined prologue")
            for line in general_options.user_scenario_prologue.split("\n"):
                writer.writeln(line)
            writer.writeln("")

    def _write_scenario_exit_statement(self, writer):
        # Set the exit statement depending on the version
        if self.khiops_version >= KhiopsVersion("10.0.0"):
            exit_statement = "ClassManagement.Quit"
        else:
            exit_statement = "Exit"

        # Write the scenario exit code
        writer.writeln("")
        writer.writeln("// Exit Khiops")
        writer.writeln(exit_statement)
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

    - This package installed through Conda and run from a Conda environment, or
    - the Khiops desktop app installed on the local machine

    .. rubric:: Environment variables taken into account by the runner:

    - part of the Khiops API:

      - ``KHIOPS_PROC_NUMBER``: number of processes launched by Khiops
      - ``KHIOPS_MEMORY_LIMIT``: memory limit of the Khiops executables in megabytes;
        ignored if set above the system memory limit
      - ``KHIOPS_TMP_DIR``: path to Khiops' temporary directory

    - other environment variables:

      - ``KHIOPS_HOME``: *Windows only* path to the Khiops installation directory
      - ``KHIOPS_SAMPLES_DIR``: path to the Khiops sample datasets directory

    - advanced configuration variables. Most of the time the user does not need modify
      them:

      - ``KHIOPS_MPI_COMMAND_ARGS``: arguments to the ``mpiexec`` command
      - ``KHIOPS_MPIEXEC_PATH``: path to the ``mpiexec`` command
      - ``KHIOPS_MPI_LIB``: *Linux and MacOS only* path to the MPI library; added to
        the beginning of ``LD_LIBRARY_PATH``

    .. rubric:: Samples directory settings

    Default values for the ``samples_dir``:

    - The value of the ``KHIOPS_SAMPLES_DIR`` environment variable
    - Otherwise:

      - Windows:

        - ``%PUBLIC%\khiops_data\samples%`` if it exists and is a directory
        - ``%USERPROFILE%\khiops_data\samples%`` otherwise

      - Linux and macOS:

        - ``$HOME/khiops_data/samples``

    """

    def __init__(self):
        # Define specific attributes
        self.mpi_command_args = None
        self._khiops_bin_dir = None
        self._khiops_version = None
        self._samples_dir = None
        self._samples_dir_checked = False

        # Call parent constructor
        super().__init__()

        # Initialize Khiops environment
        self._start_khiops_environment_initialization()

    def _set_max_cores(self, core_number):
        super()._set_max_cores(core_number)
        self._initialize_mpi_command_args()

    def _start_khiops_environment_initialization(self):
        # Set the Khiops process number according to the `KHIOPS_PROC_NUMBER` env var
        if "KHIOPS_PROC_NUMBER" in os.environ:
            self.max_cores = _compute_max_cores_from_proc_number(
                int(os.environ["KHIOPS_PROC_NUMBER"])
            )
        # If not defined, set it to the number of system cores
        else:
            self.max_cores = _get_system_cpu_cores()
            os.environ["KHIOPS_PROC_NUMBER"] = str(self.max_cores)

        # Set the Khiops memory limit
        if "KHIOPS_MEMORY_LIMIT" in os.environ:
            self.max_memory_mb = int(os.environ["KHIOPS_MEMORY_LIMIT"])
        else:
            self.max_memory_mb = 0

        # Set MPI command
        self._initialize_mpi_command_args()

        # Add custom path to MPI library to LD_LIBRARY_PATH, to be used in priority
        if "KHIOPS_MPI_LIB" in os.environ:
            custom_mpi_lib_path = os.environ["KHIOPS_MPI_LIB"]
            if "LD_LIBRARY_PATH" in os.environ:
                old_ld_library_path = os.environ["LD_LIBRARY_PATH"]
                os.environ["LD_LIBRARY_PATH"] = (
                    custom_mpi_lib_path + os.pathsep + old_ld_library_path
                )
            else:
                os.environ["LD_LIBRARY_PATH"] = custom_mpi_lib_path

        # Set the default Khiops temporary directory ("" means system's default)
        if "KHIOPS_TMP_DIR" in os.environ:
            self.khiops_temp_dir = os.environ["KHIOPS_TMP_DIR"]
        else:
            self.khiops_temp_dir = ""

        installation_method = _infer_khiops_installation_method()
        if platform.system() == "Linux" and installation_method == "binary+pip":
            # Set the OpenMPI variable OMPI_MCA_plm_rsh_agent to the empty string
            # if not set
            # This avoids errors on systems without ssh (eg. simple Docker containers)
            if "OMPI_MCA_plm_rsh_agent" not in os.environ:
                os.environ["OMPI_MCA_plm_rsh_agent"] = ""

            # Set the OpenMPI variable OMPI_MCA_btl_vader_single_copy_mechanism
            # to the "none" string value to remove the mpi message
            # "Read -1, expected 65536, errno = 1" that appears on Docker
            if "OMPI_MCA_btl_vader_single_copy_mechanism" not in os.environ:
                os.environ["OMPI_MCA_btl_vader_single_copy_mechanism"] = "none"

            # Set the OpenMPI variable PSM3_DEVICES to the "self" string value to
            # fix issue https://github.com/KhiopsML/khiops/issues/307 on Rocky
            if (
                get_linux_distribution_name() == "rocky linux"
                and "PSM3_DEVICES" not in os.environ
            ):
                os.environ["PSM3_DEVICES"] = "self"

        # Initialize the default samples dir
        self._initialize_default_samples_dir()

    def _initialize_mpi_command_args(self):
        """Creates the mpiexec call arguments for each platform"""
        # Note: Unless enforced by `KHIOPS_MPIEXEC_PATH`, the mpiexec
        # accessible in the path takes precedence over any other mpiexec install.
        installation_method = _infer_khiops_installation_method()
        # In Conda-based, but non-Conda environment, specify mpiexec path
        if installation_method == "conda-based":
            # Python `os.path.realpath` resolves symlinks recursively, like GNU
            # `readlink -f`; Python `os.readlink` does not
            mpiexec_path = os.environ.get("KHIOPS_MPIEXEC_PATH") or os.path.realpath(
                os.path.join(
                    _infer_env_bin_dir_for_conda_based_installations(), "mpiexec"
                )
            )
            if platform.system() == "Windows" and not os.path.splitext(mpiexec_path):
                mpiexec_path += ".exe"
                # If mpiexec[.exe] is not in the Conda environment's bin path,
                # update mpiexec's path to the expected Windows path to mpiexec:
                # - replace "bin" with "Library/bin" in the binary directory
                # - re-add mpiexec[.exe] to the path: look it up in the original
                #   path (so that extension calculation is not repeated)
                if not os.path.exists(mpiexec_path):
                    mpiexec_path = (
                        Path(mpiexec_path)
                        .parents[1]
                        .joinpath("Library", "bin", Path(mpiexec_path).parts[-1])
                        .as_posix()
                    )
        # In Conda or local installations, expect mpiexec in the PATH
        else:
            link_to_mpiexec = shutil.which("mpiexec")
            mpiexec_path = (
                os.environ.get("KHIOPS_MPIEXEC_PATH")
                or link_to_mpiexec
                and os.path.realpath(link_to_mpiexec)
            )
        # If mpiexec is not in the path, and the installation method is local,
        # then try to load MPI environment module so that mpiexec is in the path
        if mpiexec_path is None and installation_method == "binary+pip":
            # If environment modules are installed, then load the MPI module
            module_init_script_path = os.path.join(
                os.path.sep, "etc", "profile.d", "modules.sh"
            )

            # List available environment modules
            if os.path.exists(module_init_script_path):
                shell_command = shlex.split(
                    f"sh -c 'source {module_init_script_path} && module avail'"
                )
                with subprocess.Popen(
                    shell_command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                ) as env_modules_init_process:
                    stdout, stderr = env_modules_init_process.communicate()

                # If environment modules listing worked, look-up the MPI module
                if env_modules_init_process.returncode == 0:
                    for line in sorted(
                        re.sub(
                            "\\s+", "\\n", (stdout + stderr).decode("utf-8")
                        ).splitlines(),
                        reverse=True,
                    ):
                        # If MPI environment module is found, attempt to load it
                        if f"openmpi-{platform.machine()}" in line:
                            mpi_module = line
                            # Use 'type -P' to get the path to executable,
                            # as 'which' is non-portable
                            shell_command = shlex.split(
                                f"sh -c 'source {module_init_script_path} && "
                                f"module unload mpi && module load {mpi_module} && "
                                "type -P mpiexec'"
                            )
                            with subprocess.Popen(
                                shell_command,
                                stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL,
                            ) as mpi_module_load_process:
                                stdout, _ = mpi_module_load_process.communicate()

                            # If MPI module is loaded, get the path to mpiexec
                            if mpi_module_load_process.returncode == 0:
                                for line in stdout.decode("utf-8").splitlines():
                                    if "mpiexec" in line:
                                        mpiexec_path = line
                                    break
                                if mpiexec_path is not None:
                                    self._set_mpi_command_args_with_mpiexec(
                                        mpiexec_path, installation_method
                                    )
                            break

        # If MPI is found, then set the path to mpiexec accordingly
        if mpiexec_path is not None:
            self._set_mpi_command_args_with_mpiexec(mpiexec_path, installation_method)
        # If MPI is still not found, then do not use MPI and warn the user
        else:
            self.mpi_command_args = []
            warnings.warn(
                "mpiexec is not in PATH, Khiops will run with just one CPU. "
                "We recommend you to reinstall khiops. "
                "Go to https://khiops.org for more information."
            )

    def _set_mpi_command_args_with_mpiexec(self, mpiexec_path, installation_method):
        assert mpiexec_path is not None
        # User-specified MPI command args take precendence over automatic setting
        if "KHIOPS_MPI_COMMAND_ARGS" in os.environ:
            self.mpi_command_args = [mpiexec_path] + shlex.split(
                os.environ["KHIOPS_MPI_COMMAND_ARGS"]
            )
        # With only 1 or 2 processes run sequentially (without MPI)
        elif self.max_cores in (1, 2):
            self.mpi_command_args = []
            warnings.warn(
                f"Too few cores: {self.max_cores}. "
                "To efficiently run Khiops in parallel at least 3 processes "
                "are needed. Khiops will run in a single process."
            )
        # Otherwise, build the mpiexec command arguments
        else:
            self.mpi_command_args = [mpiexec_path]
            if platform.system() == "Windows":
                self.mpi_command_args += [
                    "-al",
                    "spr:P",
                    "-n",
                    str(self.max_cores),
                    "/priority",
                    "1",
                ]
            elif platform.system() == "Linux":
                # For Linux native installations we use OpenMPI
                if installation_method == "binary+pip":
                    self.mpi_command_args += ["--allow-run-as-root", "--quiet"]
                self.mpi_command_args += [
                    "-n",
                    str(self.max_cores),
                ]
            elif platform.system() == "Darwin":
                self.mpi_command_args += [
                    "-host",
                    "localhost",
                    "-n",
                    str(self.max_cores),
                ]
            else:
                raise KhiopsEnvironmentError(
                    f"Unsupported OS {platform.system()}. "
                    "Check the supported OSes at https://khiops.org."
                )

    def _initialize_default_samples_dir(self):
        """See class docstring"""
        # Set the fallback value for the samples directory
        home_samples_dir = Path.home() / "khiops_data" / "samples"

        # Take the value of an environment variable in priority
        if "KHIOPS_SAMPLES_DIR" in os.environ:
            self._samples_dir = os.environ["KHIOPS_SAMPLES_DIR"]

        # The samples location of Windows systems is:
        # - %PUBLIC%\khiops_data\samples if %PUBLIC% exists
        # - %USERPROFILE%\khiops_data\samples otherwise
        elif platform.system() == "Windows":
            if "PUBLIC" in os.environ:
                public_samples_dir = os.path.join(
                    os.environ["PUBLIC"], "khiops_data", "samples"
                )
            else:
                public_samples_dir = None

            ok_statuses = ["ok", "remote"]
            if (
                public_samples_dir is not None
                and get_dir_status(public_samples_dir) in ok_statuses
            ):
                self._samples_dir = public_samples_dir
            else:
                self._samples_dir = str(home_samples_dir)

        # The default samples location on Unix systems is:
        # $HOME/khiops/samples on Linux and Mac OS
        else:
            self._samples_dir = str(home_samples_dir)

        assert self._samples_dir is not None

    def _finish_khiops_environment_initialization(self):
        # Initialize Khiops binary directory
        self._initialize_khiops_bin_dir()

        # Check the tools exist and are executable
        self._check_tools()

        # Initialize the khiops version
        self._initialize_khiops_version()

    def _initialize_khiops_bin_dir(self):
        # Initialize Khiops bin dir, according to the environment
        installation_method = _infer_khiops_installation_method()

        # Conda case
        if installation_method == "conda":
            self._khiops_bin_dir = os.path.join(os.environ["CONDA_PREFIX"], "bin")
        # Conda "based" (fallback)
        elif installation_method == "conda-based":
            self._khiops_bin_dir = _infer_env_bin_dir_for_conda_based_installations()
        # System-wide installations
        else:
            self._initialize_default_system_wide_khiops_bin_dir()
        assert self.khiops_bin_dir is not None

    def _initialize_default_system_wide_khiops_bin_dir(self):
        # Warn if both KHIOPS_HOME and KhiopsHome are set
        if "KHIOPS_HOME" in os.environ and "KhiopsHome" in os.environ:
            warnings.warn(
                "Both KHIOPS_HOME and KhiopsHome environment variables "
                "are set. Only the KHIOPS_HOME will be used."
            )
        # Windows: KHIOPS_HOME value
        if platform.system() == "Windows":
            # KHIOPS_HOME variable by default
            if "KHIOPS_HOME" in os.environ:
                self._khiops_bin_dir = os.path.join(os.environ["KHIOPS_HOME"], "bin")
            # Look for KhiopsHome to support Khiops 9
            elif "KhiopsHome" in os.environ:
                self._khiops_bin_dir = os.path.join(os.environ["KhiopsHome"], "bin")
            # Raise error if KHIOPS_HOME is not set
            else:
                raise KhiopsEnvironmentError(
                    "No environment variable named 'KHIOPS_HOME' or "
                    "'KhiopsHome' found. We recommend you to reinstall Khiops. "
                    "Go to https://khiops.org for more information."
                )
        # MacOS: /usr/local/bin
        elif platform.system() == "Darwin":
            self._khiops_bin_dir = os.path.join(os.path.sep, "usr", "local", "bin")
        # Linux/Unix: /usr/bin
        elif platform.system() == "Linux":
            self._khiops_bin_dir = os.path.join(os.path.sep, "usr", "bin")
        # Raise an error for unknown platforms
        else:
            raise KhiopsEnvironmentError(
                f"Unsupported OS {platform.system()}. "
                "Check the supported OSes at https://khiops.org."
            )

    def _check_tools(self):
        """Checks the that the tool binaries exist and are executable"""
        assert self.khiops_bin_dir is not None
        for tool_name in ["khiops", "khiops_coclustering"]:
            if not os.path.exists(self._tool_path(tool_name)):
                raise KhiopsEnvironmentError(
                    f"Inexistent Khiops executable path: {self._tool_path(tool_name)}"
                )
            _check_executable(self._tool_path(tool_name))

    def _initialize_khiops_version(self):
        # Run khiops with the -v
        stdout, _, return_code = self.raw_run("khiops", ["-v"], use_mpi=False)

        # On success parse and save the version
        if return_code == 0:
            # Skip potential non-version lines (ex: Completed loading of file driver...)
            for line in stdout.split(os.linesep):
                if line.startswith("Khiops"):
                    khiops_version_str = line.rstrip().split(" ")[1]
                    break
        # If -v fails it means it is Khiops 9 or lower so we try the old way
        else:
            khiops_version_str, _, _, _ = _get_tool_info_khiops9(self, "khiops")
            warnings.warn(
                "Khiops version is earlier than 10.0; khiops-python will "
                f"run in legacy mode. Khiops path: {self._tool_path('khiops')}",
                stacklevel=3,
            )
        self._khiops_version = KhiopsVersion(khiops_version_str)

        # Warn if the khiops version is too far from the khiops-python version
        compatible_khiops_version = khiops.get_compatible_khiops_version()
        if self._khiops_version.major > compatible_khiops_version.major:
            warnings.warn(
                f"Khiops version '{self._khiops_version}' is ahead of "
                f"the khiops-python version '{khiops.__version__}'. "
                "There may be compatibility errors and "
                "we recommend you to update to the latest khiops-python version."
                "Go to https://khiops.org for more information.",
                stacklevel=3,
            )

    def _build_status_message(self):
        # Initialize if necessary
        with warnings.catch_warnings(record=True) as warning_list:
            if self.khiops_bin_dir is None:
                self._finish_khiops_environment_initialization()

        # Call the parent's method
        status_msg, parent_warning_list = super()._build_status_message()
        warning_list += parent_warning_list

        # Build the messages for temp_dir, install type and mpi
        if self.khiops_temp_dir:
            khiops_temp_dir_msg = self.khiops_temp_dir
        else:
            khiops_temp_dir_msg = "<empty> (system's default)"
        install_type_msg = _infer_khiops_installation_method()
        if self.mpi_command_args:
            mpi_command_args_msg = " ".join(self.mpi_command_args)
        else:
            mpi_command_args_msg = "<empty>"

        # Build the message
        status_msg += "\n\n"
        status_msg += "khiops local installation settings\n"
        status_msg += f"version         : {self._khiops_version}\n"
        status_msg += f"executables dir : {self._khiops_bin_dir}\n"
        status_msg += f"temp dir        : {khiops_temp_dir_msg}\n"
        status_msg += f"install type    : {install_type_msg}\n"
        status_msg += f"MPI command     : {mpi_command_args_msg}\n"
        status_msg += "\n"

        return status_msg, warning_list

    def _get_khiops_version(self):
        # Initialize the first time it is called
        if self.khiops_bin_dir is None:
            self._finish_khiops_environment_initialization()
        assert isinstance(self._khiops_version, KhiopsVersion), type_error_message(
            self._khiops_version, "khiops_version", KhiopsVersion
        )
        return self._khiops_version

    @property
    def khiops_bin_dir(self):
        r"""str: Path of the directory containing Khiops' binaries

        Default values:

            - conda installation: ``$CONDA_PREFIX/bin``
            - system-wide installations :

                - Windows:

                    - ``%KHIOPS_HOME%\bin``
                    - ``%KhiopsHome%\bin`` (deprecated)

                - Linux: ``/usr/bin``
                - Mac OS: ``/usr/local/bin``

        """
        return self._khiops_bin_dir

    @khiops_bin_dir.setter
    def khiops_bin_dir(self, bin_dir):
        # Check that the path is a directory and it exists
        if not os.path.exists(bin_dir):
            raise KhiopsEnvironmentError(
                f"Inexistent Khiops binaries directory {bin_dir}"
            )
        if not os.path.isdir(bin_dir):
            raise KhiopsEnvironmentError(
                f"Khiops binaries directory is a file: {bin_dir}"
            )

        # Set the directory, check and initialize the version
        self._khiops_bin_dir = bin_dir
        self._check_tools()
        self._initialize_khiops_version()

    def _tool_path(self, tool_name):
        """Full path of a Khiops tool binary"""
        assert self.khiops_bin_dir is not None
        tool_name = tool_name.lower()
        if tool_name not in ["khiops", "khiops_coclustering"]:
            raise ValueError(f"Invalid tool name: {tool_name}")
        modl_binaries = {
            "khiops": "MODL",
            "khiops_coclustering": "MODL_Coclustering",
        }
        bin_path = os.path.join(self.khiops_bin_dir, modl_binaries[tool_name])
        if platform.system() == "Windows":
            bin_path += ".exe"

        return bin_path

    def _set_samples_dir(self, samples_dir):
        """Checks and sets the samples directory"""
        check_samples_dir(samples_dir)
        super()._set_samples_dir(samples_dir)

    def _get_samples_dir(self):
        # Check the samples dir once (the check emmits only warnings)
        if not self._samples_dir_checked:
            check_samples_dir(self._samples_dir)
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
        # Nota: Khiops Coclustering is executed without MPI
        khiops_process_args = []
        if tool_name == "khiops" and use_mpi:
            khiops_process_args += self.mpi_command_args
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

        # Execute the process
        with subprocess.Popen(
            khiops_process_args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf8",
            errors="replace",
            universal_newlines=True,
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
        # Initialize if necessary (lazy initialization)
        if self.khiops_bin_dir is None:
            self._finish_khiops_environment_initialization()

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

# Runner (backend) of Khiops Python, by default one for a local Khiops installation
_khiops_runner = KhiopsLocalRunner()


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
    return _khiops_runner


# pylint: enable=invalid-name

##################################
# Deprecated Methods and Classes #
##################################


def _get_tool_info_khiops10(runner, tool_name):
    """Returns a Khiops tool (version 10) license information

    *This method is deprecated and kept only for backwards compatibility*

    Parameters
    ----------
    runner : `.KhiopsLocalRunner`
        A Khiops Python local runner instance.
    tool_name : "khiops" or "khiops_coclustering"
        Name of the tool.

    Returns
    -------
    tuple
        A 4-tuple containing:

        - The tool version
        - The name of the machine
        - The ID of the machine
        - The number of remaining days for the license

    Raises
    ------
    `.KhiopsEnvironmentError`
        If the current khiops runner is not of class `.KhiopsLocalRunner`.
    """
    # Get the version
    stdout, _, _ = runner.raw_run(tool_name, ["-v"], use_mpi=False)
    version = stdout.rstrip().split(" ")[1]

    # Get the license information
    stdout, _, _ = runner.raw_run(tool_name, ["-l"], use_mpi=False)
    lines = stdout.split("\n")
    computer_name = lines[1].split("\t")[1]
    machine_id = lines[2].split("\t")[1]
    remaining_days = int(lines[3].split("\t")[1])

    return version, computer_name, machine_id, remaining_days


def _get_tool_info_khiops9(runner, tool_name):
    """Returns a Khiops tool (version 9) license information

    *This method is deprecated and kept only for backwards compatibility*

    Parameters
    ----------
    runner : `.KhiopsLocalRunner`
        A Khiops Python local runner instance.
    tool_name : "khiops" or "khiops_coclustering"
        Name of the tool.

    Returns
    -------
    tuple
        A 4-tuple containing:

        - The tool version
        - The name of the machine
        - The ID of the machine
        - The number of remaining days for the license

    Raises
    ------
    `.KhiopsEnvironmentError`
        If the current khiops runner is not of class `.KhiopsLocalRunner`.
    """
    # Create a temporary file for the log
    tmp_log_file_path = runner.create_temp_file("_get_tool_info", ".log")
    tmp_scenario_path = runner.create_temp_file("_get_tool_info", "._kh")
    with open(tmp_scenario_path, "w", encoding="ascii") as tmp_scenario:
        tmp_scenario.write("// Show license information\n")
        tmp_scenario.write(
            "LearningTools.LicenseManager.ActionShowLicenseFullInformation\n\n"
        )
        tmp_scenario.write("// Exit\n")
        tmp_scenario.write("Exit\n")
        tmp_scenario.write("OK\n")

    # Run Khiops tool
    _, stderr, return_code = runner.raw_run(
        tool_name,
        ["-i", tmp_scenario_path, "-e", tmp_log_file_path, "-b"],
        use_mpi=False,
    )

    # If tool executed successfully:
    if return_code == 0:
        # Since pylint 3.2.5 an used before assignment error appears with no reason
        # This code is going to be eliminated in Khiops 11 so no need to fix it
        # pylint: disable=used-before-assignment,possibly-used-before-assignment
        # Parse the contents of the log file
        tmp_log_file_contents = io.BytesIO(fs.read(tmp_log_file_path))
        with io.TextIOWrapper(tmp_log_file_contents, encoding="ascii") as tmp_log_file:
            for line in tmp_log_file:
                if line.startswith("Khiops"):
                    fields = line.strip().split(" ")
                    if fields[1] == "Coclustering":
                        version = fields[2]
                    else:
                        version = fields[1]
                else:
                    fields = line.strip().split("\t")
                    if len(fields) == 2:
                        if fields[0] == "Computer name":
                            computer_name = fields[1]
                        if fields[0] == "Machine ID":
                            machine_id = fields[1]
                    else:
                        if "Perpetual license" in line:
                            remaining_days = float("inf")
                        elif "License expire at " in line:
                            fields = line.split(" ")
                            remaining_days = int(fields[-2])
        # Clean temporary file
        fs.remove(tmp_log_file_path)

        return version, computer_name, machine_id, remaining_days
    # else, raise KhiopsRuntimeError, as Khiops failed for another reason
    raise KhiopsRuntimeError(stderr)


class PyKhiopsRunner(KhiopsRunner):
    """Deprecated

    See `KhiopsRunner`.
    """

    def __init__(self):
        super().__init__()
        warnings.warn(deprecation_message("PyKhiopsRunner", "KhiopsRunner", "11.0.0"))


class PyKhiopsLocalRunner(KhiopsLocalRunner):
    """Deprecated

    See `KhiopsLocalRunner`.
    """

    def __init__(self):
        super().__init__()
        warnings.warn(
            deprecation_message("PyKhiopsLocalRunner", "KhiopsLocalRunner", "11.0.0")
        )
