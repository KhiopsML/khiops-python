######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Common utility functions and classes"""
import os
from collections.abc import Iterable, Mapping, Sequence
from urllib.parse import urlparse


class SystemSettings:
    """Khiops system settings

    .. note::
        These settings are not available in the `CommandLineOptions`.
    """

    def __init__(
        self,
        max_cores=None,
        memory_limit_mb=None,
        temp_dir="",
        scenario_prologue="",
    ):
        """See class docstring"""
        self.max_cores = max_cores
        self.memory_limit_mb = memory_limit_mb
        self.temp_dir = temp_dir
        self.scenario_prologue = scenario_prologue
        self.check()

    def __repr__(self):
        return (
            f"System settings: max_cores = {self.max_cores}, "
            f"memory_limit_mb = {self.memory_limit_mb}, "
            f"temp_dir = {self.temp_dir}, "
            f"scenario_prologue = {self.scenario_prologue}"
        )

    def check(self):
        """Checks the types of the system settings

        Raises
        ------
        `TypeError`
            If any of the system settings does not have the proper type.
        `ValueError`
            If ``max_cores`` or ``memory_limit_mb`` are set to negative numbers.
        """
        # Check the field types and ranges where applicable
        if self.max_cores is not None:
            if not isinstance(self.max_cores, int):
                raise TypeError(type_error_message("max_cores", self.max_cores, int))
            if self.max_cores < 0:
                raise ValueError(
                    f"max_cores must be non-negative (it is {self.max_cores})"
                )
        if self.memory_limit_mb is not None:
            if not isinstance(self.memory_limit_mb, int):
                raise TypeError(
                    type_error_message("memory_limit_mb", self.memory_limit_mb, int)
                )
            if self.memory_limit_mb < 0:
                raise ValueError(
                    "memory_limit_mb must be non-negative "
                    f"(it is {self.memory_limit_mb})"
                )
        if not is_string_like(self.temp_dir):
            raise TypeError(
                type_error_message(
                    "temp_dir",
                    self.temp_dir,
                    str,
                    bytes,
                )
            )
        if not is_string_like(self.scenario_prologue):
            raise TypeError(
                type_error_message(
                    "scenario_prologue", self.scenario_prologue, str, bytes
                )
            )


class CommandLineOptions:
    """Khiops command line options

    Attributes
    ----------
    log_file_path : str, default ""
        Path of the log file for the Khiops process (command line option ``-e`` of the
        desktop app). If equal to "" then it writes no log file.
    output_scenario_path : str, default ""
        Path of the output Khiops scenario file (command line option ``-o`` of the
        desktop app). If the empty string is specified no output scenario file is
        generated.
    task_file_path : str, default ""
        Path of the task file for the Khiops process (command line option ``-p`` of the
        desktop app). If equal to "" then it writes no task file.
    batch_mode : bool, default True
        *Deprecated* Will be removed in Khiops 11. If ``True`` activates batch mode
        (command line option ``-b`` of the app).
    """

    def __init__(
        self,
        batch_mode=True,
        log_file_path="",
        task_file_path="",
        output_scenario_path="",
    ):
        """See class docstring"""
        self.batch_mode = batch_mode
        self.log_file_path = log_file_path
        self.task_file_path = task_file_path
        self.output_scenario_path = output_scenario_path
        self.check()

    def __repr__(self):
        def to_str(string):
            if isinstance(string, str):
                repr_str = string
            else:
                assert isinstance(string, bytes)
                repr_str = string.decode("utf8", errors="replace")
            return repr_str

        command_line_options = []
        if self.batch_mode:
            command_line_options += ["-b"]
        if self.output_scenario_path:
            command_line_options += ["-o", to_str(self.output_scenario_path)]
        if self.log_file_path:
            command_line_options += ["-e", to_str(self.log_file_path)]
        if self.task_file_path:
            command_line_options += ["-p", to_str(self.task_file_path)]
        print(command_line_options)
        return "Khiops command line options: " + " ".join(command_line_options)

    def build_command_line_options(self, scenario_path):
        command_line_options = []
        if self.batch_mode:
            command_line_options += ["-b"]
        command_line_options += ["-i", scenario_path]
        if self.output_scenario_path:
            command_line_options += ["-o", self.output_scenario_path]
        if self.log_file_path:
            command_line_options += ["-e", self.log_file_path]
        if self.task_file_path:
            command_line_options += ["-p", self.task_file_path]

        return command_line_options

    def check(self):
        """Checks the types of the command line options

        Raises
        ------
        `TypeError`
            If any of the command line options does not have the proper type.
        """
        if not isinstance(self.batch_mode, bool):
            raise TypeError(type_error_message("batch_mode", self.batch_mode, bool))
        if self.output_scenario_path and not is_string_like(self.output_scenario_path):
            raise TypeError(
                type_error_message(
                    "output_scenario_path", self.output_scenario_path, str, bytes
                )
            )
        if self.log_file_path and not is_string_like(self.log_file_path):
            raise TypeError(
                type_error_message("log_file_path", self.log_file_path, str, bytes)
            )
        if self.task_file_path and not is_string_like(self.task_file_path):
            raise TypeError(
                type_error_message("task_file_path", self.task_file_path, str, bytes)
            )


def create_unambiguous_khiops_path(path):
    """Creates a path that is unambiguous for Khiops

    Khiops needs that a non absolute path starts with "." so that it does not use the
    path of an internally saved state as reference point.

    For example: if we open the data table "/some/path/to/data.txt" and then set the
    results directory simply as "results" the effective location of the results
    directory will be "/some/path/to/results" instead of "$CWD/results". This behavior
    is a feature in the Khiops GUI but it is undesirable when using it as a library.

    This function returns a path so that the library behaves as expected: a path
    relative to the $CWD if it is a non absolute path.
    """
    # Check for string
    if not isinstance(path, (str, bytes)):
        raise TypeError(type_error_message("path", path, str, bytes))

    # Empty path returned as-is
    if not path:
        return path

    # Add a "." to a local path if necessary. It is *not* necessary when:
    # - `path` is an URI
    # - `path` is an absolute path
    # - `path` is a path starting with "."
    dot = "."
    empty = ""
    if isinstance(path, bytes):
        dot = bytes(dot, encoding="ascii")
        empty = bytes(empty, encoding="ascii")
    uri_info = urlparse(path, allow_fragments=False)
    if os.path.isabs(path) or path.startswith(dot) or uri_info.scheme != empty:
        return path
    else:
        return os.path.join(dot, path)


############
# Messages #
############


def type_error_message(variable_name, variable, *target_types):
    """Formats a type error message

    Parameters
    ----------
    variable_name : str
        Name of the variable for whom the type error is signaled.
    variable : any
        Actual variable for whom the type error is signaled.
    target_types : list
        Expected types for ``variable``, either as a type or as a string.

    Returns
    -------
    str
        The type error message.

    """
    assert len(target_types) > 0, "At least one target type must be provided"
    assert all(
        isinstance(target_type, (type, str)) for target_type in target_types
    ), "All target types must be 'type' or 'str'"
    assert isinstance(variable_name, str), "'variable_name' must be 'str'"
    assert len(variable_name) > 0, "'variable_name' should not be empty"

    # Transform to 'type' the string arguments
    typed_target_types = []
    for target_type in target_types:
        if isinstance(target_type, str):
            typed_target_types.append(type(target_type, (), {}))
        else:
            typed_target_types.append(target_type)

    # Build the type error message
    if len(typed_target_types) == 1:
        target_type_str = f"'{typed_target_types[0].__name__}'"
    elif len(typed_target_types) == 2:
        target_type_str = (
            f"either '{typed_target_types[0].__name__}' "
            f"or '{typed_target_types[1].__name__}'"
        )
    else:
        target_types_str = " or ".join(
            f"'{target_type.__name__}'" for target_type in typed_target_types
        )
        target_type_str = f"one of {target_types_str}"

    if len(variable_name.strip().split(" ")) == 1:
        variable_name_str = f"'{variable_name}'"
    else:
        variable_name_str = variable_name

    return (
        f"{variable_name_str} type must be {target_type_str}, "
        f"not '{type(variable).__name__}'"
    )


def removal_message(removed_feature, since, replacement=None):
    """Formats a feature removal message"""
    message = f"'{removed_feature}' removed since {since}. "
    if replacement:
        message += f"Use '{replacement}'."
    else:
        message += "There is no replacement."
    return message


def renaming_message(renamed_feature, new_name, since):
    """Formats a feature renaming message"""
    return f"Ignoring '{renamed_feature}': renamed to '{new_name}' since {since}."


def invalid_keys_message(kwargs):
    """Formats an invalid keyword parameter message"""
    return f"Ignoring invalid parameter(s): {','.join(kwargs.keys())}."


def deprecation_message(
    deprecated_feature, deadline_version, replacement=None, quote=True
):
    """Formats a deprecation message"""
    if quote:
        message = f"'{deprecated_feature}' is deprecated "
    else:
        message = f"{deprecated_feature} is deprecated "
    message += f"and will be removed by version {deadline_version}."
    if replacement is not None:
        if quote:
            message += f" Prefer '{replacement}'."
        else:
            message += f" Prefer {replacement}."
    else:
        message += " There will be no replacement when removed."
    return message


###############
# Type checks #
###############


def is_string_like(test_object):
    """Returns True if an object is a valid Python string or sequence of bytes"""
    return isinstance(test_object, (str, bytes))


def is_list_like(list_like):
    """Returns True if an object is list-like

    An object is ``list-like`` if and only if inherits from `collections.abc.Sequence`
    and it is not `string-like <is_string_like>`
    """
    return isinstance(list_like, Sequence) and not is_string_like(list_like)


def is_dict_like(test_object):
    """Returns True if an object is dict-like

    An object is ``dict-like`` if and only if inherits from the
    `collections.abc.Mapping`.
    """
    return isinstance(test_object, Mapping)


def is_iterable(test_object):
    """Return True if a container object is iterable, but not string-like"""
    return isinstance(test_object, (Sequence, Iterable)) and not is_string_like(
        test_object
    )
