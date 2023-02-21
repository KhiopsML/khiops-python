######################################################################################
# Copyright (c) 2018 - 2023 Orange - All Rights Reserved                             #
# * This software is the confidential and proprietary information of Orange.         #
# * You shall not disclose such Restricted Information and shall use it only in      #
#   accordance with the terms of the license agreement you entered into with Orange  #
#   named the "Khiops - Python Library Evaluation License".                          #
# * Unauthorized copying of this file, via any medium is strictly prohibited.        #
# * See the "LICENSE.md" file for more details.                                      #
######################################################################################
"""Common options for Khiops tasks"""
from pykhiops.core.common import is_string_like, type_error_message


class GeneralOptions:
    """Khiops general options

    .. note::
        These options are not available in the `CommandLineOptions`.
    """

    def __init__(
        self,
        max_cores=0,
        max_memory_mb=0,
        khiops_temp_dir="",
        user_scenario_prologue="",
    ):
        """See class docstring"""
        self.max_cores = max_cores
        self.max_memory_mb = max_memory_mb
        self.khiops_temp_dir = khiops_temp_dir
        self.user_scenario_prologue = user_scenario_prologue
        self.check()

    def __repr__(self):
        return (
            f"General options: max_cores = {self.max_cores}, "
            f"max_memory_mb = {self.max_memory_mb}, "
            f"khiops_temp_dir = {self.khiops_temp_dir}"
        )

    def check(self):
        """Checks the types of the general options

        Raises
        ------
        `TypeError`
            If any of the command line options does not have the proper type.
        `ValueError`
            If ``max_cores`` or ``max_memory_mb`` are negative.
        """
        # Check the field types
        if not isinstance(self.max_cores, int):
            raise TypeError(type_error_message("max_cores", self.max_cores, int))
        if not isinstance(self.max_memory_mb, int):
            raise TypeError(
                type_error_message("max_memory_mb", self.max_memory_mb, int)
            )
        if not is_string_like(self.khiops_temp_dir):
            raise TypeError(
                type_error_message("khiops_temp_dir", self.khiops_temp_dir, str, bytes)
            )
        if not is_string_like(self.user_scenario_prologue):
            raise TypeError(
                type_error_message(
                    "user_scenario_prologue", self.user_scenario_prologue, str, bytes
                )
            )

        # Check the field range
        if self.max_cores < 0:
            raise ValueError(f"max_cores must be non-negative (it is {self.max_cores})")
        if self.max_memory_mb < 0:
            raise ValueError(
                f"max_memory_mb must be non-negative (it is {self.max_memory_mb})"
            )


class CommandLineOptions:
    """Khiops command line options

    Attributes
    ----------
    log_file_path : str, default ""
        Path of the log file (command line option ``-e`` of the app). If the empty
        string is specified no log file is generated.
    output_scenario_path : str, default ""
        Path of the output Khiops scenario file (command line option ``-o`` of the app).
        If the empty string is specified no output scenario file is generated.
    task_file_path : str, default ""
        Path of the task file (command line option ``-p`` of the desktop app). If the
        empty string is specified no task file is generated.
    batch_mode : bool, default True
        *Deprecated* Will be removed in pyKhiops 11. If ``True`` activates batch mode
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
