######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Classes for creating Khiops scenario files"""
import re

from khiops.core.internals.common import is_string_like


class ConfigurableKhiopsScenario:
    """A configurable Khiops scenario

    This class encapsulates a template Khiops scenario and its parameters. It allows to
    replace the template keyword to write an executable scenario.

    Parameters
    ----------
    template : str
        The template scenario.

    Attributes
    ----------
    template : str
        The template scenario file.
    """

    def __init__(self, template):
        """See class docstring"""
        # Save and parse the template
        self.template = template
        self._parse_template()

    def _parse_template(self):
        # Initialize private fields
        self._parsed_template = []

        # Matches a template entry
        template_entry_re = re.compile(r"\s*(([a-zA-Z.]+)\s+)?(__\w+__)")

        # Iterate over the template lines
        line_iter = iter(self.template.split("\n"))
        line = next(line_iter, None)
        line_num = 1
        while line is not None:
            entry = None
            match = template_entry_re.fullmatch(line)
            # If there is no match add the line as-is
            if match is None:
                entry = line
            # Otherwise apply the detect process depending of the entry type
            else:
                field_name = match.group(2)
                keyword = match.group(3)
                if field_name is not None:
                    param_name = keyword
                    entry = (param_name, field_name)
                else:
                    if keyword == "__OPT__":
                        entry = self._parse_opt(line_iter)
                        _, _, statements = entry
                        line_num += 2 + len(statements)
                    elif keyword == "__LIST__":
                        entry = self._parse_list(line_iter)
                        _, _, prologue, _, entry_values = entry
                        line_num += 3 + len(prologue) + len(entry_values)
                    elif keyword == "__DICT__":
                        entry = self._parse_dict(line_iter)
                        _, _, prologue, *_ = entry
                        line_num += 4 + len(prologue)
                    else:
                        raise ValueError(
                            "Expected keyword __DICT__, __OPT__ "
                            f"or __LIST__ not '{keyword}'"
                        )
            assert entry is not None, "entry must not be None"
            self._parsed_template.append(entry)

            # Update the template line
            line = next(line_iter, None)
            line_num += 1

        # Eliminate empty lines at the beginning and end if any
        if self._parsed_template[0] == "":
            self._parsed_template.pop(0)
        if self._parsed_template[-1] == "":
            self._parsed_template.pop()

        assert isinstance(self._parsed_template, list)

    def _parse_section(self, section_keyword, line_iter):
        # Obtain the end keyword
        end_section_keyword = f"__END_{section_keyword[2:]}"
        end_section_re = re.compile(r"\s*" + end_section_keyword)

        # Obtain the section lines
        section_spec = []
        match = None
        while match is None:
            try:
                line = next(line_iter)
                match = end_section_re.fullmatch(line)
            except StopIteration as error:
                raise ValueError(
                    f"{section_keyword} section has "
                    f"no matching {end_section_keyword}"
                ) from error
            section_spec.append(line)

        # Eliminate the last line which is necessarily the end keyword
        section_spec.pop()

        # Separate the parameter name from the spec
        section_param_name = section_spec.pop(0).lstrip()

        # Check that the parameter name conforms to the parameter naming
        param_name_re = re.compile(r"__\w+__")
        if param_name_re.match(section_param_name) is None:
            raise ValueError(
                f"{section_keyword} template parameter name does not conform "
                f"to the __param__ notation: '{section_param_name}'"
            )
        # Check that the section spec conforms to the scenario commands naming
        statement_re = re.compile(r"\s*[a-zA-Z][a-zA-Z.]*(\s+.*)?$")
        for statement in section_spec:
            if statement_re.match(statement) is None:
                raise ValueError(
                    "Statement must contain only alphabetic characters and '.' "
                    f"(no '.' at the beginning): '{statement}'"
                )

        return section_param_name, section_spec

    def _parse_opt(self, line_iter):
        opt_param_name, opt_section_spec = self._parse_section("__OPT__", line_iter)
        return (opt_param_name, "__OPT__", opt_section_spec)

    def _parse_list(self, line_iter):
        list_param_name, list_spec = self._parse_section("__LIST__", line_iter)

        # Check the section length
        if len(list_spec) < 2:
            raise ValueError("__LIST__ section must have at least 3 statements")

        # Look for the InsertItemAfter statement index
        list_value_start = None
        for i, line in enumerate(list_spec):
            if line.endswith("InsertItemAfter"):
                list_value_start = line
                list_value_start_index = i
                break
        if list_value_start is None:
            raise ValueError(
                "__LIST__ section does not contain "
                "list statement ending with '.InsertItemAfter'"
            )
        if list_value_start_index == len(list_spec) - 1:
            raise ValueError("__LIST__ section does not contain any value statement")

        # Separate into:
        # - the prologue
        # - the tuple start statement
        # - the tuple value statements
        list_prologue = list_spec[:list_value_start_index]
        list_value_start = list_spec[list_value_start_index]
        list_value_names = list_spec[list_value_start_index + 1 :]

        return (
            list_param_name,
            "__LIST__",
            list_prologue,
            list_value_start,
            list_value_names,
        )

    def _parse_dict(self, line_iter):
        # Obtain the parsed elements from the line iterator
        dict_param_name, dict_section = self._parse_section("__DICT__", line_iter)

        # Separate into
        # - The prologue statements
        # - The key statement
        # - The value statement
        if len(dict_section) < 2:
            raise ValueError("__DICT__ section must have at least 2 statements")
        dict_prologue = dict_section[:-2]
        dict_key_field_name = dict_section[-2]
        dict_value_field_name = dict_section[-1]
        if not dict_key_field_name.endswith(".Key"):
            raise ValueError("__DICT__ key statement must end with '.Key'")

        return (
            dict_param_name,
            "__DICT__",
            dict_prologue,
            dict_key_field_name,
            dict_value_field_name,
        )

    def __str__(self):
        return str(self._parsed_template)

    def check_keyword_completeness(self, call_keywords):
        """Check if a list of keys is exactly that of the template of this scenario"""
        # Transform keywords to set
        call_keyword_set = set(call_keywords)
        template_keyword_set = set(
            entry[0] for entry in self._parsed_template if isinstance(entry, tuple)
        )

        # Check if there are unknown keys in the input keywords
        extra_call_keyword_set = call_keyword_set - template_keyword_set
        if extra_call_keyword_set:
            raise ValueError(
                "Call parameter keyword(s) not in template keywords '"
                + ",'".join(extra_call_keyword_set)
                + "'"
            )

        # Check if there are missing keys in the input keywords
        extra_template_keyword_set = template_keyword_set - call_keyword_set
        if extra_template_keyword_set:
            raise ValueError(
                "Template parameter keyword(s) not in call keywords '"
                + ",'".join(extra_template_keyword_set)
                + "'"
            )

    def write(self, writer, scenario_args):
        """Writes the de-templatized scenario with the specified arguments

        Parameters
        ----------
        writer : `.KhiopsOutputWriter`
            The output writer for the scenario file.
        scenario_args : dict
            Values of the scenario template arguments.
        """
        for entry in self._parsed_template:
            if isinstance(entry, str):
                writer.writeln(entry)
            else:
                assert isinstance(entry, tuple)
                assert len(entry) >= 2
                entry_arg_name, entry_type, *_ = entry
                arg_value = scenario_args[entry_arg_name]
                if entry_type not in ["__DICT__", "__OPT__", "__LIST__"]:
                    self._write_str(writer, entry, arg_value)
                elif entry_type == "__OPT__":
                    self._write_opt(writer, entry, arg_value)
                elif entry_type == "__DICT__" and arg_value is not None:
                    self._write_dict(writer, entry, arg_value)
                elif entry_type == "__LIST__" and arg_value is not None:
                    self._write_list(writer, entry, arg_value)

    def _write_str(self, writer, entry, arg_value):
        assert is_string_like(arg_value)
        assert len(entry) == 2
        _, entry_name = entry
        writer.write(f"{entry_name} ")
        writer.writeln(arg_value)

    def _write_opt(self, writer, entry, arg_value):
        assert isinstance(arg_value, str)
        assert len(entry) == 3
        _, entry_type, opt_actions = entry
        assert entry_type == "__OPT__"
        if arg_value == "true":
            for action in opt_actions:
                writer.writeln(action)

    def _write_dict(self, writer, entry, arg_values):
        assert isinstance(arg_values, (dict, list))
        assert len(entry) == 5

        # Unroll the entry
        _, entry_type, entry_prologue, entry_key, entry_value = entry
        assert entry_type == "__DICT__"

        # Write the prologue
        for line in entry_prologue:
            writer.writeln(line)

        # Transform list arguments to "true"-valued dictionary
        if isinstance(arg_values, list):
            dict_arg_values = {key: "true" for key in arg_values}
        else:
            dict_arg_values = arg_values

        # Write the argument values
        for key, value in dict_arg_values.items():
            writer.write(f"{entry_key} ")
            writer.writeln(key)
            writer.write(f"{entry_value} ")
            writer.writeln(value)

    def _write_list(self, writer, entry, arg_values):
        assert isinstance(arg_values, list)
        for value in arg_values:
            assert is_string_like(value) or isinstance(value, tuple)

        # Unroll the entry
        _, entry_type, entry_prologue, entry_header, entry_value_names = entry
        assert entry_type == "__LIST__"

        # Write the prologue
        for line in entry_prologue:
            writer.writeln(line)

        # Write the contained values of the list argument
        for value in arg_values:
            writer.writeln(entry_header)

            # Tuple list case
            if isinstance(value, tuple):
                assert len(entry_value_names) == len(value)
                for entry_value_name, subvalue in zip(entry_value_names, value):
                    writer.write(f"{entry_value_name} ")
                    writer.writeln(subvalue)
            # String list case
            else:
                assert len(entry_value_names) == 1
                assert is_string_like(value)
                writer.write(f"{entry_value_names[0]} ")
                writer.writeln(value)
