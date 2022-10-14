######################################################################################
# Copyright (c) 2023 Orange. All rights reserved.                                    #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Classes implementing pyKhiops' backend runners"""

import getpass
import io
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import uuid
import warnings
from abc import ABC, abstractmethod

import pykhiops
import pykhiops.core.internals.filesystems as fs
from pykhiops.core.exceptions import PyKhiopsEnvironmentError, PyKhiopsRuntimeError
from pykhiops.core.internals.common import (
    CommandLineOptions,
    GeneralOptions,
    invalid_keys_message,
    is_string_like,
    removal_message,
    renaming_message,
    type_error_message,
)
from pykhiops.core.internals.io import PyKhiopsOutputWriter
from pykhiops.core.internals.task import KhiopsTask
from pykhiops.core.internals.version import KhiopsVersion


def _isdir_without_all_perms(dir_path):
    """Returns True if the path is a directory but missing of any of rwx permissions"""
    return os.path.isdir(dir_path) and not os.access(
        dir_path, os.R_OK | os.W_OK | os.X_OK
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


class PyKhiopsRunner(ABC):
    """Abstract pyKhiops runner to be re-implemented"""

    def __init__(self):
        """See class docstring"""
        self.general_options = GeneralOptions()
        self._init_root_temp_dir()

        # Version and samples directory are set to None for lazy initialization
        self._khiops_version = None
        self._samples_dir = None

    def _init_root_temp_dir(self):
        """Initializes the runner's root temporary directory

        It tries to set a proper root temporary directory. It tries the following
        strategies in order:
        - Check that ``$TEMP/pykhiops`` exists and use it
        - Try to create ``$TEMP/pykhiops` and use it
        - Create a ``$TEMP/pykhiops_<HASH>`` and use it
        """

        # Create the directory if it doesn't exists
        self._root_temp_dir = os.path.join(tempfile.gettempdir(), "pykhiops")
        if not os.path.exists(self._root_temp_dir):
            os.makedirs(self._root_temp_dir)

        # Create the dir with a hash name if it is a dir but it doesn't have all
        # permissions or if it is a file
        elif os.path.isfile(self._root_temp_dir) or _isdir_without_all_perms(
            self._root_temp_dir
        ):
            self._root_temp_dir = tempfile.mkdtemp(prefix="pykhiops_")

    @property
    def root_temp_dir(self):
        r"""str: The runner's temporary directory

        The temporary scenarios/templates and dictionary files created by pykhiops are
        stored here.

        Default value:
            - Windows: ``%TEMP%\pykhiops``
            - Linux: ``$TMP/pykhiops``

        When set to a local path it tries to create the specified directory if it
        doesn't exist.

        Raises
        ------
        `.PyKhiopsEnvironmentError`
            If set to a local path: if it is a file or if it does not have ``+rwx``
            permissions.
        """
        return self._root_temp_dir

    @root_temp_dir.setter
    def root_temp_dir(self, dir_path):
        # Check existence, directory status and permissions for local paths
        if fs.is_local_resource(dir_path):
            real_dir_path = _extract_path_from_uri(dir_path)
            if not os.path.exists(real_dir_path):
                os.makedirs(real_dir_path)
            elif os.path.isfile(real_dir_path):
                raise PyKhiopsEnvironmentError(
                    f"File at temporary directory os.path. Path: {real_dir_path}"
                )
            elif _isdir_without_all_perms(real_dir_path):
                raise PyKhiopsEnvironmentError(
                    "Temporary directory must have +rwx permissions. "
                    f"Path: {real_dir_path}"
                )
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
            A unique path within the temporary directory. The file is effectively
            created file only in the case of a local filesystem.
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
            The path of the created directory.
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
        if not isinstance(prologue, str):
            raise TypeError(type_error_message("scenario_prologue", prologue, str))
        self.general_options.user_scenario_prologue = prologue

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
        if not isinstance(core_number, int):
            raise TypeError(type_error_message("max_cores", core_number, int))
        if core_number < 0:
            raise ValueError("'max_cores' must be non-negative")
        self.general_options.max_cores = core_number

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
        if not isinstance(memory_mb, int):
            raise TypeError(type_error_message("max_memory_mb", memory_mb, int))
        if memory_mb < 0:
            raise ValueError("'max_memory_mb' must be non-negative")
        self.general_options.max_memory_mb = memory_mb

    @property
    def khiops_temp_dir(self):
        """str: Temporary directory for Khiops executions

        Raises
        ------
        `TypeError`
            If set to a non str object.
        """
        return self.general_options.khiops_temp_dir

    @khiops_temp_dir.setter
    def khiops_temp_dir(self, temp_dir):
        """Setter of khiops_temp_dir"""
        if not isinstance(temp_dir, str):
            raise TypeError(type_error_message("khiops_temp_dir", temp_dir, str))
        self.general_options.khiops_temp_dir = temp_dir

    @property
    def samples_dir(self):
        r"""str: Location of the Khiops' sample datasets directory. May be an URL/URI"""
        # Lazy initialization of the environment if not set
        if self._samples_dir is None:
            self._initialize_samples_dir()
        return self._samples_dir

    @samples_dir.setter
    def samples_dir(self, samples_dir):
        if not is_string_like(samples_dir):
            raise TypeError(
                type_error_message("samples_dir", samples_dir, "string-like")
            )
        self._set_samples_dir(samples_dir)

    def _initialize_samples_dir(self):
        """To be overridden by child classes if necessary"""
        self._samples_dir = ""

    def _set_samples_dir(self, samples_dir):
        """To be overridden by child classes to add additional checks"""
        self._samples_dir = samples_dir

    @property
    def khiops_version(self):
        """`.KhiopsVersion`: The version of the Khiops backend of this runner

        Its initialization is lazy: it is delayed to its first access.
        """
        return self._get_khiops_version()

    def _get_khiops_version(self):
        """khiops_version getter"""
        # Initialize the first time it is called
        if self._khiops_version is None:
            self._initialize_khiops_version()
            assert isinstance(self._khiops_version, KhiopsVersion), type_error_message(
                self._khiops_version, "khiops_version", KhiopsVersion
            )
            compatible_khiops_version = pykhiops.get_compatible_khiops_version()
            if (
                self._khiops_version.major > compatible_khiops_version.major
                or self._khiops_version.minor > compatible_khiops_version.minor
            ):
                warnings.warn(
                    f"Khiops version '{self._khiops_version}' is very ahead "
                    f"from pyKhiops version '{pykhiops.__version__}'. "
                    "There may be compatibility errors and "
                    "it is recommended to update to the latest pyKhiops version.",
                    stacklevel=3,
                )
        return self._khiops_version

    def _build_status_message(self):
        """Constructs the status message

        Descendant classes can add additional information
        """
        status_msg = f"pyKhiops {pykhiops.__version__} info and defaults\n"
        status_msg += f"pyKhiops runner class    : {self.__class__.__name__}\n"
        status_msg += f"pyKhiops max cores       : {self.max_cores}"
        if self.max_cores == 0:
            status_msg += " (no limit)"
        status_msg += "\n"
        status_msg += f"pyKhiops max memory (MB) : {self.max_memory_mb}"
        if self.max_memory_mb == 0:
            status_msg += " (no limit)"
        status_msg += "\n"
        status_msg += f"pyKhiops temp dir        : {self.root_temp_dir}"
        return status_msg

    def print_status(self):
        """Prints the status of the runner to stdout"""
        print(self._build_status_message())

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
            values are used. Unless you know what are you doing, prefer setting this
            options with the runners accessors. See the `.GeneralOptions` class for more
            information.
        force_ansi_scenario : bool, default ``False``
            *Advanced:* If True the internal scenario generated by pyKhiops will force
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
            warnings.warn(renaming_message("batch", "batch_mode", "10.0"), stacklevel=3)
            del kwargs["batch"]
        if "output_script" in kwargs:
            warnings.warn(
                renaming_message("output_script", "output_scenario_path", "10.0"),
                stacklevel=3,
            )
            del kwargs["output_script"]
        if "log" in kwargs:
            warnings.warn(
                renaming_message("log", "log_file_path", "10.0"), stacklevel=3
            )
            del kwargs["log"]
        if "task" in kwargs:
            warnings.warn(
                renaming_message("task", "task_file_path", "10.0"), stacklevel=3
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
            return_code, stderr = self._run(
                task.tool_name,
                scenario_path,
                command_line_options,
                trace,
            )
            # pylint: enable=assignment-from-no-return
        # Catch an OS level error if any
        except OSError as error:
            raise PyKhiopsRuntimeError("Khiops execution failed.") from error
        # Report any errors raised by Khiops
        else:
            self._report_exit_status(
                task.tool_name,
                return_code,
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

    def _report_exit_status(self, tool_name, return_code, stderr, log_file_path):
        """Reports the exit status of a Khiops execution

        - If there were fatal errors it raises a PyKhiopsRuntimeError
        - If there were only errors it warns them
        - If the process ended ok but there was stderr output it warns as well
        """
        # If there were no errors warn if:
        # - stderr was not empty
        # - There were warnings in the log
        if return_code == 0:
            # Add Khiops log warnings to the warning message if any
            warning_msg = ""
            _, _, warning_messages = self._collect_errors(log_file_path)
            if warning_messages:
                warning_msg += "\nWarnings in log:\n" + "".join(warning_messages)

            # Add stderr to the warning message if non empty
            if stderr:
                warning_msg += f"\nContents of stderr:\n{stderr}"

            # Report the message if there were any
            if warning_msg:
                warning_msg = (
                    "Khiops ended correctly but there were minor issues" + warning_msg
                )
                warnings.warn(warning_msg.rstrip(), stacklevel=4)
        # If there were errors or fatal errors collect them and report
        else:
            # Collect errors and warnings
            errors, fatal_errors, warning_messages = self._collect_errors(log_file_path)

            # Create the message reporting the errors
            error_msg = f"{tool_name} ended with return code {return_code}"
            if warning_messages:
                error_msg += "\nWarnings in log:\n" + "".join(warning_messages)
            if errors:
                error_msg += "\nErrors in log:\n" + "".join(errors)
            if fatal_errors:
                error_msg += "\nFatal errors in log:\n" + "".join(fatal_errors)
            if stderr:
                error_msg += f"\nContents of stderr:\n{stderr}"

            # Raise an exception with the errors
            raise PyKhiopsRuntimeError(error_msg)

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
            writer = PyKhiopsOutputWriter(
                scenario_stream, force_ansi=force_ansi_scenario
            )
            self._write_task_scenario(
                writer,
                task,
                task_args,
                general_options
                if general_options is not None
                else self.general_options,
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
        if self.khiops_version >= KhiopsVersion("10"):
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
            A 2-tuple containing the return code and the stderr of the Khiops process

        Raises
        ------
        `.PyKhiopsRuntimeError`
            If there were any errors in the Khiops execution.
        """


class PyKhiopsLocalRunner(PyKhiopsRunner):
    r"""Implementation of a local Khiops runner

    Requires the Khiops desktop app installed in the local machine

    Default values for ``samples_dir``:

    - The value of the ``KHIOPS_SAMPLES_DIR`` environment variable
    - Otherwise:
        - Windows: ``%KHIOPS_HOME%\samples`` or ``%KhiopsHome%\samples``
        - Linux: ``/opt/khiops/samples``
    """

    def __init__(self):
        # Parent constructor
        super().__init__()

        # Lazy initialization until the first run to avoid errors
        # in environments without a local installation
        self.execute_with_modl = None
        self.mpi_command_args = None
        self._khiops_bin_dir = None
        self._run_with_vendored_khiops = None

    def _cpu_count_command(self):
        if platform.system() == "Linux":
            return ["lscpu", "-b", "-p=Core,Socket"]
        elif platform.system() == "Windows":
            return [
                "powershell.exe",
                "-Command",
                "$OutputEncoding = [System.Text.Encoding]::UTF8;",
                "Get-CimInstance -ClassName 'Win32_Processor' "
                "| Select-Object -Property 'NumberOfCores' | Write-Output",
            ]
        elif platform.system() == "Darwin":
            return ["sysctl", "-a"]
        else:
            raise RuntimeError(f"The '{platform.system()}' is not supported.")

    def _cpu_count(self, command_output):
        if platform.system() == "Linux":
            return (
                len(
                    {
                        cpu_entry
                        for cpu_entry in command_output.splitlines()
                        if not cpu_entry.startswith("#")
                    }
                )
                + 1
            )
        elif platform.system() == "Windows":
            n_cores = 1  # number of cores + 1
            for out in command_output.splitlines():
                try:
                    n_cores += int(out)
                except ValueError:
                    continue
            return n_cores

        elif platform.system() == "Darwin":
            for out in command_output.splitlines():
                if out.startswith("hw.physicalcpu:"):
                    return int(out.split(" ")[-1]) + 1
        else:
            raise RuntimeError(f"The '{platform.system()}' is not supported.")

    def _mpi_exec_command(self):
        assert hasattr(self, "max_cores") and self.max_cores is not None
        if platform.system() == "Linux":
            # Get path to `mpiexec`
            mpiexec_path = shutil.which("mpiexec")
            return (
                f"{mpiexec_path} -bind-to hwthread -map-by core -n {self.max_cores + 1}"
                if mpiexec_path is not None
                else ""
            )
        elif platform.system() == "Darwin":
            # Get path to `mpiexec`
            mpiexec_path = shutil.which("mpiexec")
            return (
                f"{mpiexec_path} -n {self.max_cores + 1}"
                if mpiexec_path is not None
                else ""
            )

        elif platform.system() == "Windows":
            # Get path to `mpiexec`
            mpiexec_path = os.path.join(
                sys.exec_prefix, os.environ["MSMPI_BIN"].strip(os.path.sep), "mpiexec"
            )
            return (
                f"{mpiexec_path} -al spr:P -n {self.max_cores + 1} /priority 1"
                if mpiexec_path is not None
                else ""
            )
        else:
            raise RuntimeError(f"The '{platform.system()}' is not supported.")

    def _initialize_environment_from_vendored_khiops(self):
        # Set the environment variables, for Linux for now
        # set Khiops proc number
        with subprocess.Popen(
            self._cpu_count_command(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        ) as cpu_informer:
            cpu_count_command_output, _ = cpu_informer.communicate()
        cpu_count = self._cpu_count(cpu_count_command_output)
        self.max_cores = cpu_count - 1
        os.environ["KHIOPS_PROC_NUMBER"] = str(cpu_count)
        # set MPI command
        mpi_command = self._mpi_exec_command()
        if platform.system() == "Windows":
            is_posix = False
        else:
            # Linux and Mac OS
            is_posix = True
        self.mpi_command_args = shlex.split(mpi_command, posix=is_posix)
        os.environ["KHIOPS_MPI_COMMAND"] = mpi_command
        # set Khiops temporary directory
        self.khiops_temp_dir = os.path.join(
            tempfile.gettempdir(), "khiops", getpass.getuser()
        )
        os.environ["KHIOPS_TMP_DIR"] = self.khiops_temp_dir

    def _check_vendored_khiops(self):
        if platform.system() in ("Linux", "Windows", "Darwin") and (
            # inside Pip virtualenv
            sys.exec_prefix != sys.base_exec_prefix
            or
            # inside Conda environment
            (
                any(
                    os.environ.get(conda_env_var) is not None
                    for conda_env_var in ("CONDA_PREFIX", "CONDA_DEFAULT_ENV")
                )
                and sys.exec_prefix
                == sys.base_exec_prefix
                == os.environ.get("CONDA_PREFIX")
                and os.environ.get("CONDA_PREFIX").endswith(
                    os.environ["CONDA_DEFAULT_ENV"]
                )
            )
        ):
            vendored_path = os.path.join(
                sys.exec_prefix,
                "bin",
            )
            for dirname, _, filenames in os.walk(vendored_path):
                # recursively look for MODL and MODL_Coclustering and set khiops
                # path to the directory where they are found
                for filename in filenames:
                    if filename.startswith("MODL") and os.access(
                        os.path.join(dirname, filename), os.X_OK
                    ):
                        # MODL executable file found
                        self._khiops_bin_dir = dirname
                        self._run_with_vendored_khiops = True
                if self._run_with_vendored_khiops:
                    break
        else:
            # Khiops vendoring not supported for other OSes
            self._run_with_vendored_khiops = False

    def _initialize_execution_configuration(self):
        if self._run_with_vendored_khiops:
            self._initialize_environment_from_vendored_khiops()
            self.execute_with_modl = True
        else:
            # Set the environment script name
            if platform.system() == "Windows":
                khiops_env_script_path = os.path.join(
                    self.khiops_bin_dir, "khiops_env.cmd"
                )
            else:
                khiops_env_script_path = os.path.join(self.khiops_bin_dir, "khiops-env")

            # If the environment script exists then obtain the execution environment
            if os.path.exists(khiops_env_script_path):
                # Execute khiops environment script
                with subprocess.Popen(
                    [khiops_env_script_path, "--env"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                ) as khiops_process:
                    stdout, _ = khiops_process.communicate()

                # Parse the output of the khiops environment script and save the
                # settings
                path_additions = ["KHIOPS_PATH", "KHIOPS_JAVA_PATH"]
                for line in stdout.split("\n"):
                    tokens = line.rstrip().split(maxsplit=1)
                    if len(tokens) == 2:
                        var_name, var_value = tokens
                    elif len(tokens) == 1:
                        var_name = tokens[0]
                        var_value = ""
                    else:
                        continue

                    if var_name in path_additions:
                        os.environ["PATH"] = var_value + os.pathsep + os.environ["PATH"]
                    elif var_name == "KHIOPS_CLASSPATH":
                        if "CLASSPATH" in os.environ:
                            os.environ["CLASSPATH"] = (
                                var_value + os.pathsep + os.environ["CLASSPATH"]
                            )
                        else:
                            os.environ["CLASSPATH"] = var_value
                    elif var_name == "KHIOPS_MPI_LIB":
                        if "LD_LIBRARY_PATH" in os.environ:
                            os.environ["LD_LIBRARY_PATH"] = (
                                var_value + os.pathsep + os.environ["LD_LIBRARY_PATH"]
                            )
                        else:
                            os.environ["LD_LIBRARY_PATH"] = var_value
                    elif var_name == "KHIOPS_MPI_COMMAND":
                        self.mpi_command_args = shlex.split(var_value)
                    elif var_name == "KHIOPS_PROC_NUMBER" and var_value:
                        self.max_cores = int(var_value) - 1
                        os.environ["KHIOPS_PROC_NUMBER"] = var_value
                    elif var_name == "KHIOPS_MEMORY_LIMIT" and var_value:
                        self.max_memory_mb = int(var_value)
                        os.environ["KHIOPS_MEMORY_LIMIT"] = var_value
                    elif var_name == "KHIOPS_TMP_DIR" and var_value:
                        self.khiops_temp_dir = var_value
                        os.environ["KHIOPS_TMP_DIR"] = var_value
                    else:
                        os.environ[var_name] = var_value
                self.execute_with_modl = True
            # If there is no environment script then just the `khiops` script
            else:
                self.execute_with_modl = False

            # Check the tool binaries
            self._check_tools()

    def _build_status_message(self):
        status_msg = super()._build_status_message()
        status_msg += "\n\n"
        status_msg += "Khiops local installation\n"
        status_msg += f"Khiops version           : {self.khiops_version}\n"
        status_msg += f"Khiops binaries dir      : {self.khiops_bin_dir}\n"
        status_msg += f"Khiops samples dir       : {self.samples_dir}\n"
        status_msg += f"Khiops temp dir          : {self.khiops_temp_dir}"
        if self.khiops_temp_dir == "":
            status_msg += "<empty> (system's default)"
        return status_msg

    @property
    def khiops_bin_dir(self):
        r"""str: Path of the directory containing Khiops' binaries

        Default values:

            - Windows: ``%KHIOPS_HOME%\bin`` or ``%KhiopsHome%\bin``
            - Linux: ``/usr/bin``


        Raises
        ------
        `.PyKhiopsEnvironmentError`
            - When set to a path that does not exist.
            - When set to a path that is actually a file.
        """
        # Lazy initialization of the environment
        if self._khiops_bin_dir is None:
            self._initialize_khiops_bin_dir()
        return self._khiops_bin_dir

    def _initialize_khiops_bin_dir(self):
        # Warn if both KHIOPS_HOME and KhiopsHome are set
        if "KHIOPS_HOME" in os.environ and "KhiopsHome" in os.environ:
            warnings.warn(
                "Both KHIOPS_HOME and KhiopsHome environment variables "
                "are set. Only the KHIOPS_HOME will be used."
            )

        # First look for a KHIOPS_HOME variable
        # Nota: KhiopsHome is to support Khiops 9 in Windows
        if "KHIOPS_HOME" in os.environ:
            self.khiops_bin_dir = os.path.join(os.environ["KHIOPS_HOME"], "bin")
        elif "KhiopsHome" in os.environ:
            self.khiops_bin_dir = os.path.join(os.environ["KhiopsHome"], "bin")
        # Otherwise try default locations in *nix systems
        else:
            # Error on windows: KHIOPS_HOME mandatory
            if platform.system() == "Windows":
                raise PyKhiopsEnvironmentError(
                    "No environment variable named 'KHIOPS_HOME' or 'KhiopsHome' found,"
                    " verify your Khiops installation."
                )
            # MacOS: /usr/local/bin
            elif platform.system() == "Darwin":
                self.khiops_bin_dir = os.path.join(os.path.sep, "usr", "local", "bin")
            # Linux/Unix: /usr/bin
            else:
                self.khiops_bin_dir = os.path.join(os.path.sep, "usr", "bin")

    @khiops_bin_dir.setter
    def khiops_bin_dir(self, bin_dir):
        # Check that the path is a directory and it exists
        if not os.path.exists(bin_dir):
            raise PyKhiopsEnvironmentError(
                f"Inexistent Khiops binaries directory {bin_dir}"
            )
        if not os.path.isdir(bin_dir):
            raise PyKhiopsEnvironmentError(
                f"Khiops binaries directory is a file: {bin_dir}"
            )
        # Set the khiops version to None so it will be reinitialized on access
        self._khiops_version = None

        self._khiops_bin_dir = bin_dir

    def tool_path(self, tool_name):
        """Full path of a Khiops tool binary"""
        tool_name = tool_name.lower()
        if tool_name not in ["khiops", "khiops_coclustering"]:
            raise ValueError(f"Invalid tool name: {tool_name}")

        if self.execute_with_modl:
            modl_binaries = {
                "khiops": "MODL",
                "khiops_coclustering": "MODL_Coclustering",
            }
            bin_path = os.path.join(self.khiops_bin_dir, modl_binaries[tool_name])
            if platform.system() == "Windows":
                bin_path += ".exe"
        else:
            bin_path = os.path.join(self.khiops_bin_dir, tool_name)
            if platform.system() == "Windows":
                bin_path += ".cmd"

        return bin_path

    def _check_tools(self):
        """Checks the that the tool binaries exist and are executable"""
        for tool_name in ["khiops", "khiops_coclustering"]:
            tool_bin_path = self.tool_path(tool_name)
            if not os.path.exists(tool_bin_path):
                raise PyKhiopsEnvironmentError(
                    f"Inexistent Khiops binary path: {tool_bin_path}"
                )
            if not os.path.isfile(tool_bin_path):
                raise PyKhiopsEnvironmentError(
                    f"Non-regular binary file. Path: {tool_bin_path}"
                )
            if not os.access(tool_bin_path, os.X_OK):
                raise PyKhiopsEnvironmentError(
                    f"Tool has no execution rights. Path: {tool_bin_path}"
                )

    def _initialize_samples_dir(self):
        """See class docstring"""
        # Take the value of an environment variable in priority
        if "KHIOPS_SAMPLES_DIR" in os.environ:
            self._set_samples_dir(os.environ["KHIOPS_SAMPLES_DIR"])
        # Take the default value for windows systems ("KhiopsHome" to support Khiops 9)
        elif platform.system() == "Windows":
            if "KHIOPS_HOME" in os.environ:
                self._set_samples_dir(
                    os.path.join(os.environ["KHIOPS_HOME"], "samples")
                )
            elif "KhiopsHome" in os.environ:
                self._set_samples_dir(os.path.join(os.environ["KhiopsHome"], "samples"))
            else:
                raise PyKhiopsEnvironmentError(
                    "No environment variable named 'KHIOPS_HOME' or 'KhiopsHome' found,"
                    " verify your Khiops installation"
                )
        # The default samples location in Linux is /opt/khiops/samples
        else:
            self._set_samples_dir("/opt/khiops/samples")

    def _set_samples_dir(self, samples_dir):
        # Check existence samples directory if it is a local path
        if fs.is_local_resource(samples_dir):
            # Remove initial slash on windows systems
            # urllib's url2pathname does not work properly
            samples_dir_res = fs.create_resource(os.path.normpath(samples_dir))
            samples_dir_path = samples_dir_res.uri_info.path
            if platform.system() == "Windows":
                if samples_dir_path.startswith("/"):
                    samples_dir_path = samples_dir_path[1:]

            if not os.path.exists(samples_dir_path):
                warnings.warn(
                    "Sample datasets local directory does not exist. "
                    f"Make sure it is located here: {samples_dir_path}",
                    stacklevel=3,
                )
            elif not os.path.isdir(samples_dir_path):
                warnings.warn(
                    "Sample datasets local directory path is not a directory. "
                    f"Make sure it is located here: {samples_dir_path}",
                    stacklevel=3,
                )
        # There are no checks for non local filesystems (no `else` statement)

        # Call parent method
        super()._set_samples_dir(samples_dir)

    def _initialize_khiops_version(self):
        # Run khiops with the -v flag
        # Determine which Khiops to run, according to whether one has a
        # vendored Khiops or not
        stdout, _, return_code = self.raw_run("khiops", ["-v"])

        # On success parse and save the version
        if return_code == 0:
            # Skip potential non-version lines (ex: Completed loading of file driver...)
            for line in stdout.split(os.linesep):
                if line.startswith("Khiops"):
                    khiops_version_str = line.rstrip().split(" ")[1]
                    break
        # If -v fails it means it is khiops 9 or lower we try the old way
        else:
            khiops_version_str, _, _, _ = _get_tool_info_khiops9(self, "khiops")
            warnings.warn(
                "Khiops version is earlier than 10.0; pyKhiops will "
                f"run in legacy mode. Khiops path: {self.tool_path('khiops')}",
                stacklevel=3,
            )
        self._khiops_version = KhiopsVersion(khiops_version_str)

    def raw_run(self, tool_name, command_line_args=None, trace=False):
        """Execute a Khiops tool with given command line arguments

        Parameters
        ----------
        tool_name : {"khiops", "khiops_coclustering"}
            Name of the tool to execute.
        command_line_args : list of str, optional
            Command line arguments of Khiops.
        trace : bool, default False
            If ``True`` print the trace of the process.

        Examples
        --------
        >>> raw_run("khiops", ["-b", "-i" , "scenario._kh"])

        is equivalent to execute in a shell::

            > khiops -b -i scenario._kh
        """
        if command_line_args and not isinstance(command_line_args, list):
            raise TypeError(
                type_error_message("command_line_args", command_line_args, list)
            )
        if self._run_with_vendored_khiops is None:
            # Lazy check if installing with vendored Khiops
            self._check_vendored_khiops()

        # Lazy initialization of the execution configuration
        if self.execute_with_modl is None:
            self._initialize_execution_configuration()
        # Build command line arguments
        # Nota: Khiops Coclustering is executed without MPI
        khiops_process_args = []
        if self.execute_with_modl and tool_name == "khiops":
            khiops_process_args += self.mpi_command_args
        khiops_process_args += [self.tool_path(tool_name)]
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
        # Execute the tool
        khiops_args = command_line_options.build_command_line_options(scenario_path)
        _, stderr, return_code = self.raw_run(
            tool_name, command_line_args=khiops_args, trace=trace
        )

        return return_code, stderr


#########################
# Current Runner access #
#########################

# Disable pylint UPPER_CASE convention: _pykhiops_runner is non-constant
# pylint: disable=invalid-name

# Runner (backend) of pyKhiops, by default one for a local Khiops installation
_pykhiops_runner = PyKhiopsLocalRunner()


def set_runner(runner):
    """Sets the current PyKhiopsRunner of the module"""
    if not isinstance(runner, PyKhiopsRunner):
        raise TypeError(type_error_message("runner", runner, PyKhiopsRunner))
    global _pykhiops_runner
    _pykhiops_runner = runner


def get_runner():
    """Returns the current PyKhiopsRunner of the module

    Returns
    -------
    `.PyKhiopsRunner`
        The current pyKhiops runner of the module.
    """
    return _pykhiops_runner


# pylint: enable=invalid-name

######################
# Deprecated Methods #
######################


def _get_tool_info_khiops10(runner, tool_name):
    """Returns a Khiops tool (version 10) license information

    *This method is deprecated and kept only for backwards compatibility*

    Parameters
    ----------
    runner : `.PyKhiopsLocalRunner`
        A pyKhiops local runner instance.
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
    `.PyKhiopsEnvironmentError`
        If the current pykhiops runner is not of class `.PyKhiopsLocalRunner`.
    """
    # Get the version
    stdout, _, _ = runner.raw_run(tool_name, ["-v"])
    version = stdout.rstrip().split(" ")[1]

    # Get the license information
    stdout, _, _ = runner.raw_run(tool_name, ["-l"])
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
    runner : `.PyKhiopsLocalRunner`
        A pyKhiops local runner instance.
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
    `.PyKhiopsEnvironmentError`
        If the current pykhiops runner is not of class `.PyKhiopsLocalRunner`.
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
    runner.raw_run(tool_name, ["-i", tmp_scenario_path, "-e", tmp_log_file_path, "-b"])

    # Parse the contents
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
