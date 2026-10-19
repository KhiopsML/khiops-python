######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Core API functions abstractions"""
import textwrap

from khiops.core.internals.common import snake_to_camel, type_error_message
from khiops.core.internals.io import encode_file_path
from khiops.core.internals.scenario import ConfigurableKhiopsScenario
from khiops.core.internals.types import (
    AbstractDictType,
    AbstractListType,
    DictType,
    KhiopsTaskArgumentType,
    StringLikeType,
    default_from_declaration,
    name_to_type,
)
from khiops.core.internals.version import KhiopsVersion


def encode_path_valued_arg(arg, arg_name, arg_type):
    """Encodes an argument containing paths (str or dict)
    Adds trailing comment for ensuring URI parsing by Khiops.

    Parameters
    ----------
    arg : str or dict
        The value of a path valued argument.
    arg_name : str
        The name of the argument.
    arg_type : KhiopsTaskArgumentType
        The type of the argument. Must be either [StringLikeType][] or
        [DictType][][  [StringLikeType][] , [StringLikeType][]] .

    See Also
    --------
    .encode_path_valued_arg : The underlying transformation function.

    """
    if not issubclass(arg_type, StringLikeType) and not issubclass(
        arg_type, DictType(StringLikeType, StringLikeType)
    ):
        raise TypeError(type_error_message(arg_name, arg, str, bytes, "dict-str-str"))
    if issubclass(arg_type, StringLikeType):
        encoded_arg = encode_file_path(arg) + b" //"
    else:
        encoded_arg = {}
        for key in arg.keys():
            encoded_arg[key] = encode_file_path(arg[key]) + b" //"
    return encoded_arg


class KhiopsTask:
    """Specification of a Khiops Task

    Parameters
    ----------
    name : str
        Name of the task.
    tool_name : str
        Name of the tool to execute the task (ex. "khiops", "khiops_coclustering").
    args_signature : list
        A list of 2-tuples for each mandatory parameter of the task, containing:
            - its name (str)
            - its type ([KhiopsTaskArgumentType][])
    kwargs_signature : list
        A list of 3-tuples for each optional parameter of the task, containing:
            - its name (str)
            - its type ([KhiopsTaskArgumentType][])
            - its default value
    path_valued_arg_names : list of str
        A list of the parameters that contain paths. They must be contained either in
        args_signature or kwargs_signature. The only accepted types are
        [StringLikeType][] or containers thereof.
    scenario_template : str
        A Khiops scenario template. The template mini-language is described above.

    Attributes
    ----------
    scenario : ConfigurableKhiopsScenario
        Scenario object built from `scenario_template`.
    """

    def __init__(
        self,
        name,
        tool_name,
        args_signature,
        kwargs_signature,
        path_valued_arg_names,
        scenario_template,
    ):
        # Pre-initialization checks
        # Check the basic input types
        if not isinstance(name, str):
            raise TypeError(type_error_message("name", name, str))
        if not isinstance(tool_name, str):
            raise TypeError(type_error_message("tool_name", tool_name, str))
        if not isinstance(args_signature, list):
            raise TypeError(type_error_message("args_signature", args_signature, list))
        if not isinstance(kwargs_signature, list):
            raise TypeError(
                type_error_message("kwargs_signature", kwargs_signature, list)
            )
        if not isinstance(path_valued_arg_names, list):
            raise TypeError(
                type_error_message("path_valued_arg_names", path_valued_arg_names, list)
            )
        if not isinstance(scenario_template, str):
            raise TypeError(
                type_error_message("scenario_template", scenario_template, str)
            )

        # Check the types of the contents of args
        for arg_index, arg_tuple in enumerate(args_signature):
            if not isinstance(arg_tuple, tuple):
                raise TypeError(
                    type_error_message(f"arg[{arg_index}] tuple", arg_tuple, tuple)
                )
            if len(arg_tuple) != 2:
                raise ValueError(f"arg[{arg_index}] tuple must have length 2")
            arg_name, arg_type = arg_tuple
            if not isinstance(arg_name, str):
                raise TypeError(type_error_message(f"arg[{arg_index}])", arg_name, str))
            if not issubclass(arg_type, KhiopsTaskArgumentType):
                raise TypeError(
                    type_error_message(
                        f"arg[{arg_index}]'s type", arg_type, KhiopsTaskArgumentType
                    )
                )

        # Check the types of the contents of kwargs
        for kwarg_index, kwarg_tuple in enumerate(kwargs_signature):
            if not isinstance(kwarg_tuple, tuple):
                raise TypeError(
                    type_error_message(
                        f"kwarg[{kwarg_index}] tuple", kwarg_tuple, tuple
                    )
                )
            if len(kwarg_tuple) != 3:
                raise ValueError(f"kwarg[{kwarg_index}] tuple must have length 3")
            kwarg_name, kwarg_type, kwarg_default = kwarg_tuple
            if not isinstance(kwarg_name, str):
                raise TypeError(
                    type_error_message(f"kwarg[{kwarg_index}])", kwarg_name, str)
                )
            if not issubclass(kwarg_type, KhiopsTaskArgumentType):
                raise TypeError(
                    type_error_message(
                        f"kwarg[{kwarg_index}]) type",
                        kwarg_type,
                        KhiopsTaskArgumentType,
                    )
                )
            if kwarg_default is not None:
                if not kwarg_type.is_of_this_type(kwarg_default):
                    raise TypeError(
                        type_error_message(
                            f"kwarg[{kwarg_index}]) default",
                            kwarg_default,
                            kwarg_type.short_name(),
                            "None",
                        )
                    )
                # We do not admit bytes default values for KhString
                if issubclass(kwarg_type, StringLikeType) and isinstance(
                    kwarg_default, bytes
                ):
                    raise TypeError(
                        f"kwarg[{kwarg_index}]) string-like default value "
                        "cannot be of type 'bytes'"
                    )

        # Check the types of path_valued_arg_names
        for arg_index, arg_name in enumerate(path_valued_arg_names):
            if not isinstance(arg_name, str):
                raise TypeError(
                    type_error_message(
                        f"path_valued_arg_names[{arg_index}]", arg_name, str
                    )
                )

        # Initialize the public members
        self.name = name
        self.tool_name = tool_name
        self.args_signature = args_signature
        self.kwargs_signature = kwargs_signature
        self.scenario = ConfigurableKhiopsScenario(textwrap.dedent(scenario_template))
        self.path_valued_arg_names = path_valued_arg_names

        # Index the arguments by name
        self._args_signature_by_name = {
            arg_signature[0]: arg_signature for arg_signature in self.args_signature
        }
        self._kwargs_signature_by_name = {
            kwarg_signature[0]: kwarg_signature
            for kwarg_signature in self.kwargs_signature
        }

        # Post-initialization checks
        # Check that the path_valued_arg_names are contained in either args or kwargs
        all_arg_names = list(self._args_signature_by_name.keys()) + list(
            self._kwargs_signature_by_name.keys()
        )
        for path_valued_arg_name in path_valued_arg_names:
            if path_valued_arg_name not in all_arg_names:
                raise ValueError(
                    f"Path argument '{path_valued_arg_name}' not found "
                    "in args nor kwargs"
                )

        # Check the keyword coherence between the declaration and the template
        all_scenario_arg_names = [
            f"__{snake_to_camel(arg_name)}__" for arg_name in all_arg_names
        ]
        self.scenario.check_keyword_completeness(all_scenario_arg_names)

        # Check the coherence of path_valued_arg_names
        for arg_name in self.path_valued_arg_names:
            if (
                arg_name not in self._args_signature_by_name
                and not self._kwargs_signature_by_name
            ):
                raise ValueError(f"Path argument '{arg_name}' not found in signature")
            if arg_name in self._args_signature_by_name:
                _, arg_type = self._args_signature_by_name[arg_name]
            else:
                _, arg_type, _ = self._kwargs_signature_by_name[arg_name]

            if not issubclass(arg_type, StringLikeType) and not issubclass(
                arg_type, DictType(StringLikeType, StringLikeType)
            ):
                raise TypeError(
                    f"Path argument '{arg_name}' should be string-like or "
                    "dict of string-like pairs"
                )

    def write_execution_scenario(self, writer, args):
        """Writes a task's execution scenario file contents for the specified arguments

        Parameters
        ----------
        writer : KhiopsOutputWriter
            Writer object for the scenario file.
        args : dict
            Task arguments. Includes mandatory and optional arguments.
        """
        # Check that all the arguments are known
        for arg_name, arg in args.items():
            if (
                # HACK: construction_rules_not_empty is an added argument
                arg_name != "construction_rules_not_empty"
                and not arg_name in self._args_signature_by_name
                and not arg_name in self._kwargs_signature_by_name
            ):
                raise ValueError(f"Unknown argument '{arg_name}'")

        # Check that all mandatory arguments are present
        for arg_name in self._args_signature_by_name.keys():
            if not arg_name in args:
                raise ValueError(f"Missing mandatory argument '{arg_name}'")

        # Check the types of the arguments
        for arg_name, arg in args.items():
            if arg_name in self._args_signature_by_name:
                _, arg_type = self._args_signature_by_name[arg_name]
            else:
                # HACK: construction_rules_not_empty is an added argument
                if arg_name == "construction_rules_not_empty":
                    continue
                assert arg_name in self._kwargs_signature_by_name
                _, arg_type, _ = self._kwargs_signature_by_name[arg_name]
            if (
                not issubclass(arg_type, (AbstractDictType, AbstractListType))
                or arg is not None
            ):
                # HACK: "use_complement_as_test" declared as a bool is mutated to str
                # Let's skip it for the moment
                if arg_name != "use_complement_as_test":
                    arg_type.check(arg, arg_name)

        # Save the non-specified optional arguments
        absent_kwarg_names = []
        for arg_name in self._kwargs_signature_by_name:
            if arg_name not in args:
                absent_kwarg_names.append(arg_name)

        # Transform to string-like parameters
        # Path-valued parameters are encoded differently depending on the platform
        scenario_args = {}
        for arg_name, arg in args.items():
            if arg_name in self._args_signature_by_name:
                _, arg_type = self._args_signature_by_name[arg_name]
                scenario_arg = arg_type.to_scenario_arg(arg)
            else:
                # HACK: construction_rules_not_empty is an added argument
                if arg_name == "construction_rules_not_empty":
                    continue
                assert arg_name in self._kwargs_signature_by_name
                _, arg_type, arg_default = self._kwargs_signature_by_name[arg_name]
                if arg is not None:
                    scenario_arg = arg_type.to_scenario_arg(arg)
                else:
                    scenario_arg = arg
            if arg_name in self.path_valued_arg_names and arg is not None:
                scenario_arg = encode_path_valued_arg(scenario_arg, arg_name, arg_type)
            scenario_args[f"__{arg_name}__"] = scenario_arg

        # Transform absent parameters (with their default value)
        for arg_name in absent_kwarg_names:
            _, arg_type, arg_default = self._kwargs_signature_by_name[arg_name]
            if arg_default is not None:
                scenario_arg = arg_type.to_scenario_arg(arg_default)
            else:
                scenario_arg = None
            scenario_args[f"__{arg_name}__"] = scenario_arg

        # Write the scenario file
        writer.writeln(f"// Scenario for task {self.name}")
        self.scenario.write(writer)
        writer.writeln("")
        writer.writeln(f"// End of scenario for task {self.name}")


class KhiopsTaskRegistry:
    """Registry of Khiops tasks

    The registry structure is the following :
        'name' -> 'declaration'
              --> 'template'
    """

    def __init__(self):
        self.registry = {}

    def register_task(self, task_name, declaration_data, template_data):
        self.registry[task_name] = {
            "declaration": declaration_data,
            "template": template_data,
        }

    def get_task(self, task_name, target_version):
        """Retrieves the latest task object for the specified task and version

        Parameters
        ----------
        task_name : str
            The name of the task to be retrieved.
        target_version : str or KhiopsVersion
            The target Khiops version.

        Returns
        -------
        KhiopsTask
            The latest task object for the specified task and version.

        Raises
        ------
        ValueError
            If there are no tasks registered with the specified name.
        """
        # Check the argument type
        if not isinstance(target_version, (str, KhiopsVersion)):
            raise TypeError(type_error_message(target_version, str, KhiopsVersion))

        # Check that the task exists
        if task_name not in self.registry:
            raise ValueError(f"There are no task named '{task_name}' registered")

        mandatory_parameters = []
        optional_parameters = []
        for param in self.registry[task_name]["declaration"]["arguments"]:
            if param["name"] == "kwargs":
                # this special argument must be skipped
                continue
            if param["default"] is None:
                # mandatory
                mandatory_parameters.append(
                    (param["name"], name_to_type(param["arg_type"]))
                )
            else:
                # optional
                optional_parameters.append(
                    (
                        param["name"],
                        name_to_type(param["arg_type"]),
                        default_from_declaration(param["arg_type"], param["default"]),
                    )
                )

        return KhiopsTask(
            task_name,
            self.registry[task_name]["declaration"]["tool"],
            mandatory_parameters,
            optional_parameters,
            [],
            self.registry[task_name]["template"],
        )

    def get_tasks(self, target_version):
        """Retrieves the latest Khiops tasks compatible with the specified version

        Parameters
        ----------
        target_version : str or KhiopsVersion
            The target Khiops version for the tasks to be retrieved.

        Returns
        -------
        list of KhiopsTask
            The list of compatible tasks with `target_version`.
        """
        # Check the argument type
        if not isinstance(target_version, (str, KhiopsVersion)):
            raise TypeError(type_error_message(target_version, str, KhiopsVersion))

        # Transform the version to KhiopsVersion if necessary
        if isinstance(target_version, str):
            target_version = KhiopsVersion(target_version)

        # Fill the task list with the latests versions
        compatible_tasks = []
        for task_name in self.task_names:
            # Skip the task if the target version is ahead of its end version
            task_end_version = self.task_families[task_name].end_version
            if task_end_version is not None and task_end_version < target_version:
                continue

            # Add the task with the latest compatible version
            compatible_tasks.append(self.get_task(task_name, target_version))

        return compatible_tasks

    @property
    def task_names(self):
        """list of str: Names of all tasks"""
        return list(self.task_families.keys())


# _REGISTRY is global but it is not assigned because it is already initialized
# pylint: disable=global-variable-not-assigned
_REGISTRY = KhiopsTaskRegistry()


def get_task_registry():
    """Direct access to the registry"""
    global _REGISTRY
    return _REGISTRY
