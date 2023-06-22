######################################################################################
# Copyright (c) 2018 - 2022 Orange - All Rights Reserved                             #
# * This software is the confidential and proprietary information of Orange.         #
# * You shall not disclose such Restricted Information and shall use it only in      #
#   accordance with the terms of the license agreement you entered into with Orange  #
#   named the "Khiops - Python Library Evaluation License".                          #
# * Unauthorized copying of this file, via any medium is strictly prohibited.        #
# * See the "LICENSE.md" file for more details.                                      #
######################################################################################
"""Classes for creating Khiops scenario files"""
import glob
import io
import os
import platform
import warnings
from abc import ABC, abstractmethod

from pykhiops.core import common
from pykhiops.core import filesystems as fs
from pykhiops.core.common import (
    KhiopsVersion,
    PyKhiopsOutputWriter,
    is_list_like,
    is_string_like,
    type_error_message,
)


class ConfigurableKhiopsScenario:
    """A configurable Khiops scenario

    This class encapsulates a template Khiops scenario and its parameters. It allows to
    replace the template keyword to write an executable scenario.

    Parameters
    ----------
    params : dict
        A dictionary indexed by template keywords. Its values are the parameters to be
        replaced in the template, either string-like or a  `KhiopsScenarioParameter`.
    template_path : str
        Path of a Khiops scenario file.

    Attributes
    ----------
    template_path : str
        Path of the template scenario file.
    template_name : str
        File name of the scenario template.
    template_ext : str
        File extension of the scenario template.
    params : dict[str, Union[KhiopsScenarioParam, StringLike]
        See constructor parameters.
    """

    def __init__(self, template_path, params):
        """See class docstring"""
        self.template_path = template_path
        self.params = params

        # Save the template name and extension
        self.template_name, self.template_ext = os.path.splitext(
            os.path.basename(template_path)
        )

        # Load template lines to memory
        with open(self.template_path, encoding="ascii") as template:
            self._template_lines = [line.rstrip() for line in template.readlines()]

        # Check the types of the scenario parameters
        self.check()

    def __str__(self):
        stream = io.BytesIO()
        writer = PyKhiopsOutputWriter(stream)
        self.write(writer)
        return str(stream.getvalue(), encoding="utf8", errors="replace")

    def add_prologue(
        self, user_prologue="", max_cores=0, max_memory_mb=0, khiops_temp_dir=""
    ):
        """Adds a prologue containing a user-defined prologue and system settings

        Parameters
        ----------
        user_prologue : str, default ""
            A user defined scenario prologue (written in Khiops scenario language).
        max_cores : int, default 0
            Maximum number of cores.
        max_memory_mb : int, default 0
            Maximum memory (in MB).
        khiops_temp_dir : str, default ""
            Path of the Khiops temporary directory.
        """
        prologue = []

        # Fill system settings prologue
        if max_cores or max_memory_mb or khiops_temp_dir:
            prologue.append("// System settings")
            if max_cores:
                prologue.append(
                    f"AnalysisSpec.SystemParameters.MaxCoreNumber {max_cores}"
                )
            if max_memory_mb:
                prologue.append(
                    f"AnalysisSpec.SystemParameters.MemoryLimit {max_memory_mb}"
                )
            if khiops_temp_dir:
                prologue.append(
                    f"AnalysisSpec.SystemParameters.TemporaryDirectoryName "
                    f"{khiops_temp_dir}"
                )
            prologue.append("")

        # Fill user defined prologue
        if user_prologue:
            prologue += ["// User-defined prologue"] + user_prologue.split("\n") + [""]
        self._template_lines = prologue + self._template_lines

    def write_file(self, scenario_path, force_ansi=False):
        """Writes the scenario to a file

        Parameters
        ----------
        scenario_path : str
            Path of the output scenario file.
        """
        with io.BytesIO() as scenario_stream:
            writer = PyKhiopsOutputWriter(scenario_stream)
            self.write(writer, force_ansi=force_ansi)
            scenario_res = fs.create_resource(scenario_path)
            scenario_res.write(scenario_stream.getvalue())

    def write(self, writer, force_ansi=False):
        """Writes the scenario to a file objet

        It replaces the scenario keywords specified in ``database_params``,
        ``path_params``, ``dict_params``, ``list_params`` and ``string_params``. Each
        type of parameter has its own way to be replaced.

        Parameters
        ----------
        writer : `.PyKhiopsOutputWriter`
            The output writer for the scenario file.
        """
        remaining_keywords = set(self.params.keys())
        for line in self._template_lines:
            replacement_happened = False

            # Write comments and empty lines without substitution
            if line.startswith("//") or line == "":
                writer.writeln(line)
                continue

            for keyword, param in self.params.items():
                if keyword in line:
                    if not isinstance(param, (DatabaseParameter, PathParameter)):
                        writer.force_ansi = force_ansi

                    if is_string_like(param) or isinstance(param, PathParameter):
                        keyword_index = line.find(keyword)
                        before_keyword = line[:keyword_index]
                        after_keyword = line[keyword_index + len(keyword) :]
                        writer.write(before_keyword)
                        if is_string_like(param):
                            writer.write(param)
                        else:
                            param.write(writer)
                        writer.writeln(after_keyword)
                    else:
                        param.write(writer)

                    writer.force_ansi = False
                    replacement_happened = True
                    remaining_keywords.remove(keyword)
                    break

            if not replacement_happened:
                writer.writeln(line)

        if remaining_keywords:
            warnings.warn(
                "Not all parameters were written to the scenario."
                " Remaining parameter keywords: {', '.join(remaining_keywords)}",
                stacklevel=6,
            )

    def check(self):
        """Check the types of the parameters

        Raises
        ------
        TypeError
            - If any of the keywords is not str
            - If any of the parameters is not StringLike
        """
        for keyword, param in self.params.items():
            if not isinstance(keyword, str):
                raise TypeError(type_error_message("Parameter keywords", keyword, str))

            if isinstance(param, KhiopsScenarioParameter):
                param.check()

            elif not is_string_like(param):
                raise TypeError(
                    type_error_message(
                        "Parameters", param, "StringLike", KhiopsScenarioParameter
                    )
                )


class KhiopsScenarioParameter(ABC):
    """Abstract scenario parameter"""

    @abstractmethod
    def write(self, writer):
        """Abstract method for parameter writing."""

    @abstractmethod
    def check(self):
        """Abstract method for parameter checking."""


class DatabaseParameter(KhiopsScenarioParameter):
    """A database parameter

    Parameters
    ----------
    name : str
        Name of the database.
    data_tables : dict
        A dictionary whose entries are the data path - file path pairs of the database
        tables. Either the key and the value can be of type str, bytes or bytearray.
        See :doc:`/multi_table_tasks` for more details.

    Attributes
    ----------
    name : str
        See constructor parameters.
    data_tables : dict
        See constructor parameters.
    """

    def __init__(self, name, data_tables):
        """See class docstring"""
        self.name = name
        self.data_tables = data_tables

    def write(self, writer):
        for data_path, file_path in self.data_tables.items():
            writer.write(f"{self.name}.DatabaseFiles.List.Key ")
            writer.writeln(data_path)
            writer.write(f"{self.name}.DatabaseFiles.DataTableName ")
            writer.writeln(encode_file_path(file_path))

    def check(self):
        # No checks by default
        pass


class PathParameter(KhiopsScenarioParameter):
    """A file path parameter

    Parameters
    ----------
    file_path : string-like
        An URI or local file path.

    Attributes
    ----------
    file_path : string-like
        See constructor parameters.
    """

    def __init__(self, file_path):
        """See class docstring"""
        self.file_path = file_path

    def write(self, writer):
        """Writes the path parameter to a Khiops scenario file

        The file path is encoded differently depending on the platform. See
        `encode_file_path`.

        Parameters
        ----------
        writer : `~pykhiops.core.common.PyKhiopsOutputWriter`
            The output writer for the scenario file.
        """
        encoded_file_path = encode_file_path(self.file_path)
        writer.write(encoded_file_path)

    def check(self):
        if not is_string_like(self.file_path):
            raise TypeError(
                type_error_message("file_path", self.file_path, "StringLike")
            )


class KeyValueListParameter(KhiopsScenarioParameter):
    """A parameter that is a list of key-value pairs

    Parameters
    ----------
    name : str
        Name of the parameter.
    value_field_name : str
        Name of the "value" field.
    keyvalues : dict
        The key-value pairs. Keys and values must be string-like.

    Attributes
    ----------
    name : str
        See constructor parameters.
    value_field_name : str
        See constructor parameters.
    keyvalues : dict
        See constructor parameters.
    """

    def __init__(self, name, value_field_name, keyvalues):
        """See class docstring"""
        self.name = name
        self.value_field_name = value_field_name
        self.keyvalues = keyvalues

    def write(self, writer):
        """Writes the key value list to a Khiops scenario file

        For each ``key`` and ``value`` in ``self.keyvalues`` it writes::

            <self.name>.List.Key <key>
            <self.name>.<self.value_name> <value>

        Parameters
        ----------
        writer : `~pykhiops.core.common.PyKhiopsOutputWriter`
            The output writer for the scenario file.
        """
        for key, value in self.keyvalues.items():
            writer.write(f"{self.name}.List.Key ")
            writer.writeln(key)
            writer.write(f"{self.name}.{self.value_field_name} ")
            writer.writeln(value)

    def check(self):
        """Checks the types of the contained object

        Raises
        ------
        `TypeError`
            If any of the contained objects has an invalid type.
        """
        if not isinstance(self.name, str):
            raise TypeError(type_error_message("name", self.name, str))
        if not isinstance(self.value_field_name, str):
            raise TypeError(
                type_error_message("value_field_name", self.value_field_name, str)
            )

        if not isinstance(self.keyvalues, dict):
            raise TypeError(type_error_message("keyvalues", self.keyvalues, dict))

        for key, value in self.keyvalues.items():
            if not is_string_like(key):
                raise TypeError(
                    type_error_message("key in 'keyvalues'", key, "StringLike")
                )
            if not is_string_like(value):
                raise TypeError(
                    type_error_message("value in 'keyvalues'", value, "StringLike")
                )


class RecordListParameter(KhiopsScenarioParameter):
    """A parameter that is a list of fixed-size records

    Parameters
    ----------
    name : str
        Name of the parameters in the scenario.
    records_header : str or tuple
        Name(s) of the record fields.
    records : list
        The records. The list elements can be either string-like or tuples of
        string-like.

    Attributes
    ----------
    name : str
        See constructor parameters.
    records_header : tuple of string-like
        See constructor parameters.
    records : list of tuple
        See constructor parameters.
    """

    def __init__(self, name, records_header, records):
        """See class docstring"""
        self.name = name
        if isinstance(records_header, tuple):
            self.records_header = records_header
            self.records = records
        else:
            self.records_header = (records_header,)
            self.records = [(record,) for record in records]

    def write(self, writer):
        """Writes the record list to a Khiops scenario file

        For each ``record`` in ``records`` it writes::

            <self.name>.InsertItemAfter
            <self.name>.<self.records_header[0]> <record[0]>
            <self.name>.<self.records_header[1]> <record[1]>
            ...
            <self.name>.<self.records_header[n]> <record[n]>

        Parameters
        ----------
        writer : `~pykhiops.core.common.PyKhiopsOutputWriter`
            The output writer for the scenario file.
        """
        for record in self.records:
            writer.writeln(f"{self.name}.InsertItemAfter")
            for record_field_name, record_field in zip(self.records_header, record):
                writer.write(f"{self.name}.{record_field_name} ")
                writer.writeln(record_field)

    def check(self):
        """Checks the types of the contained objects

        Raises
        ------
        `TypeError`
            If any of the contained objects has an invalid type.
        """

        if not isinstance(self.name, str):
            raise TypeError(type_error_message("'name' attribute", self.name, str))
        if not isinstance(self.records_header, tuple):
            raise TypeError(
                type_error_message(
                    "'records_header' attribute", self.records_header, tuple
                )
            )

        for field_name in self.records_header:
            if not isinstance(field_name, str):
                raise TypeError(
                    type_error_message("'records_header' fields", field_name, str)
                )

        if not is_list_like(self.records):
            raise TypeError(type_error_message("records", self.records, "list-like"))

        for record in self.records:
            if not isinstance(record, tuple):
                raise TypeError(type_error_message("'records' content", record, tuple))
            if len(record) != len(self.records_header):
                raise ValueError(
                    "records content must have length "
                    f"{len(self.records_header)} not {len(self.records_header)}"
                )


def encode_file_path(file_path):
    """Encodes a file path

    This is custom path encoding for Khiops scenarios that is platform dependent. The
    encoding is done only if file_path is of type str.

    Parameters
    ----------
    file_path : string-like
        The path of a file.

    Returns
    -------
    `bytes`
        If ``file_path`` is str
            - In Windows : The path decoded to UTF-8 excepting the "ANSI" Unicode
              characters.
            - In Linux/Unix/Mac : The path decoded to UTF-8.
        If ``file_path`` is `bytes`:
            It just returns the input ``file_path``

    Raises
    ------
    `TypeError`
        If ``file_path`` is not string-like.
    """
    # Check input type
    if not is_string_like(file_path):
        raise TypeError(type_error_message("file_path", file_path, "string-like"))
    if not isinstance(file_path, str):
        return file_path

    # Platform *nix: return UTF-8 encoded path
    if platform.system() != "Windows":
        return bytes(file_path, encoding="utf8")

    # Platform Windows:
    # - Return ANSI encoded chars if they over the 128-255 window
    # - Return UTF8 encoded chars otherwise
    decoded_bytes = bytearray()
    for char in file_path:
        if char in common.all_ansi_unicode_chars:
            decoded_bytes.extend(common.all_ansi_unicode_chars_to_ansi[char])
        else:
            decoded_bytes.extend(bytearray(char, "utf8"))
    return bytes(decoded_bytes)


def _get_pykhiops_core_dir():
    """Access the pykhiops.core location directory"""
    # Return the directory containing this file
    return os.path.dirname(os.path.realpath(__file__))


def get_scenario_file_path(api_call_name, target_version, coclustering=False):
    """Returns a compatible template scenario given an API call and a Khiops version"""
    assert isinstance(target_version, KhiopsVersion), type_error_message(
        target_version, "target_version", KhiopsVersion
    )

    # Look all scenario files matching the api call
    scenario_ext = "_khc" if coclustering else "_kh"
    glob_pattern = os.path.join(
        _get_pykhiops_core_dir(), "scenarios", "*", f"_{api_call_name}.{scenario_ext}"
    )
    scenario_files = glob.glob(glob_pattern)

    # Extract the versions from the file paths and sort them in reverse
    available_versions = [
        KhiopsVersion(os.path.basename(os.path.dirname(scenario_file)))
        for scenario_file in scenario_files
    ]
    available_versions.sort(reverse=True)

    # Look for the latest available version that is before the target version
    latest_version = None
    for version in available_versions:
        if target_version >= version:
            latest_version = version
            break

    # If no version was found raise an error
    if latest_version is None:
        raise ValueError(
            f"No compatible {'khiops coclustering' if coclustering else 'khiops'} "
            f"scenario for api call '{api_call_name}' version {target_version}"
        )

    return os.path.join(
        _get_pykhiops_core_dir(),
        "scenarios",
        str(latest_version),
        f"_{api_call_name}.{scenario_ext}",
    )
