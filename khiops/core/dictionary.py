######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Classes to manipulate Khiops Dictionary files

.. note::
    To have a complete illustration of the access to the information of all classes in
    this module look at their ``write`` methods which write them in Khiops Dictionary
    file format (``.kdic``).

"""
import io
import math
import os
import re
import warnings

import khiops.core.internals.filesystems as fs
from khiops.core import api
from khiops.core.exceptions import KhiopsJSONError
from khiops.core.internals.common import (
    deprecation_message,
    is_dict_like,
    is_string_like,
    type_error_message,
)
from khiops.core.internals.io import (
    KhiopsJSONObject,
    KhiopsOutputWriter,
    flexible_json_load,
)
from khiops.core.internals.runner import get_runner


def _format_name(name):
    """Formats a name of a dictionary or variable to a valid ``.kdic`` file identifier

    Returns unchanged the names that contain only "identifier" characters:
      - underscore
      - alphanumeric

    Otherwise, it returns the name between backquoted (backquotes within are doubled)
    """
    # Check if the name is an identifier
    # Python isalnum is not used because of utf-8 encoding (accentuated chars
    # are considered alphanumeric)
    # Return original name if is an identifier, otherwise between backquotes
    identifier_pattern = r"^[a-zA-Z_][a-zA-Z0-9_]*"
    str_identifier_regex = re.compile(identifier_pattern)
    bytes_identifier_regex = re.compile(bytes(identifier_pattern, encoding="ascii"))
    if isinstance(name, str):
        if str_identifier_regex.fullmatch(name) is not None:
            formatted_name = name
        else:
            formatted_name = "`" + name.replace("`", "``") + "`"
    else:
        assert isinstance(name, bytes)
        if bytes_identifier_regex.fullmatch(name) is not None:
            formatted_name = name
        else:
            formatted_name = b"`" + name.replace(b"`", b"``") + b"`"
    return formatted_name


def _quote_value(value):
    """Double-quotes a string

    Categorical, Text and metadata values are quoted with this method.
    """
    if isinstance(value, str):
        quoted_value = '"' + value.replace('"', '""') + '"'
    else:
        assert isinstance(value, bytes)
        quoted_value = b'"' + value.replace(b'"', b'""') + b'"'
    return quoted_value


def _check_name(name):
    """Ensures the variable name is consistent with the Khiops core name constraints

    Plain string or bytes are both accepted as input. The Khiops core forbids a name:

        - with a length outside the [1,128] interval
        - containing a simple (Unix) carriage-return (\n)
        - with leading and trailing spaces.

    This function must check these constraints.

    Parameters
    ----------
        name : str
            Name to be validated.
    Raises
    ------
        `ValueError`
            If the provided name does not comply with the formatting constraints.
    """
    # Check that the type of name is string or bytes
    if not is_string_like(name):
        raise TypeError(type_error_message("name", name, "string-like"))

    # Check the name complies with the Khiops core constraints
    if isinstance(name, str):
        contains_carriage_return = "\n" in name
    else:
        assert isinstance(name, bytes)
        contains_carriage_return = b"\n" in name
    if len(name) > 128 or contains_carriage_return or name != name.strip():
        raise ValueError(
            f"Variable name '{name}' cannot be accepted "
            "(invalid length or characters)"
        )


def _is_valid_type(type_str):
    """Checks whether the type is known"""
    return (
        _is_native_type(type_str)
        or _is_object_type(type_str)
        or type_str in ["TextList", "Structure"]
    )  # internal types


def _is_native_type(type_str):
    """Checks whether the type is native (not internal or relational)"""
    return type_str in [
        "Categorical",
        "Numerical",
        "Time",
        "Date",
        "Timestamp",
        "TimestampTZ",
        "Text",
    ]


def _is_object_type(type_str):
    """Checks whether the type is an object one (relational)"""
    return type_str in ["Entity", "Table"]


class DictionaryDomain(KhiopsJSONObject):
    """Main class containing the information of a Khiops dictionary file

    A DictionaryDomainain is a collection of `Dictionary` objects. These dictionaries
    usually represent either a database schema or a predictor model.

    Parameters
    ----------
    json_data : dict, optional
        Python dictionary representing the data of a Khiops Dictionary JSON file. If not
        specified it returns an empty instance.

        .. note::
            Prefer the `.read_dictionary_file` function from the core API to obtain an
            instance of this class from a Khiops Dictionary file (``kdic`` or
            ``kdicj``).

    Attributes
    ----------
    tool : str
        Name of the Khiops tool that generated the dictionary file.
    version : str
        Version of the Khiops tool that generated the dictionary file.
    dictionaries : list of `Dictionary`
        The domain's dictionaries.
    """

    def __init__(self, json_data=None):
        """See class docstring"""
        # Check the type of json_data
        if json_data is not None and not isinstance(json_data, dict):
            raise TypeError(type_error_message("json_data", json_data, dict))

        # Initialize base attributes
        super().__init__(json_data=json_data)

        # Transform to an empty dictionary if json_data is not specified
        if json_data is None:
            json_data = {}
        # Otherwise check if the tool field is the proper one
        else:
            if self.tool != "Khiops Dictionary":
                raise KhiopsJSONError(
                    f"'tool' value must be 'Khiops Dictionary' not '{self.tool}'"
                )

        # Initialize the Khiops dictionary objects
        self.dictionaries = []
        self._dictionaries_by_name = {}
        for json_dictionary in json_data.get("dictionaries", []):
            dictionary = Dictionary(json_dictionary)
            self.add_dictionary(dictionary)

    def __repr__(self):
        """Returns a human readable string representation"""
        if len(self.dictionaries) == 0:
            return "Dictionaries ()"
        if len(self.dictionaries) == 1:
            return f"Dictionaries ({self.dictionaries[0].name})"
        return f"Dictionaries ({self.dictionaries[0].name},...)"

    def __str__(self):
        stream = io.BytesIO()
        writer = KhiopsOutputWriter(stream)
        self.write(writer)
        return str(stream.getvalue(), encoding="utf8", errors="replace")

    def copy(self):
        """Copies this domain instance

        Returns
        -------
        `DictionaryDomain`
            A copy of this instance.
        """
        dictionary_domain_copy = DictionaryDomain()
        dictionary_domain_copy.tool = self.tool
        dictionary_domain_copy.version = self.version
        dictionary_domain_copy.khiops_encoding = self.khiops_encoding
        dictionary_domain_copy.ansi_chars = self.ansi_chars
        dictionary_domain_copy.colliding_utf8_chars = self.colliding_utf8_chars
        for dictionary in self.dictionaries:
            dictionary_domain_copy.add_dictionary(dictionary.copy())
        return dictionary_domain_copy

    def get_dictionary(self, dictionary_name):
        """Returns the specified dictionary

        Parameters
        ----------
        dictionary_name : str
            Name of the dictionary.

        Returns
        -------
        `Dictionary`
            The specified dictionary. ``None`` is returned if the dictionary name
            is not found.
        """
        return self._dictionaries_by_name.get(dictionary_name)

    def add_dictionary(self, dictionary):
        """Adds a dictionary to this domain

        Parameters
        ----------
        dictionary : `Dictionary`
            The dictionary to be added.

        Raises
        ------
        `TypeError`
            If ``dictionary`` is not of type ``Dictionary``.
        """
        if not isinstance(dictionary, Dictionary):
            raise TypeError(type_error_message("dictionary", dictionary, Dictionary))
        self.dictionaries.append(dictionary)
        self._dictionaries_by_name[dictionary.name] = dictionary

    def remove_dictionary(self, dictionary_name):
        """Removes a dictionary from the domain

        Returns
        -------
        `Dictionary`
            The removed dictionary.

        Raises
        ------
        `KeyError`
            If no dictionary with the specified name exists.
        """
        dictionary = self._dictionaries_by_name.pop(dictionary_name)
        self.dictionaries.remove(dictionary)
        return dictionary

    def extract_data_paths(self, source_dictionary_name):
        """Extracts the data paths for a dictionary in a multi-table schema

        See :doc:`/multi_table_primer` for more details about data paths.

        Parameters
        ----------
        source_dictionary_name : str
            Name of a dictionary.

        Returns
        -------
        list of str
            The additional data paths for the secondary tables of the specified
            dictionary.
        """
        # List of entity names found in the exploration of _extract_data_paths
        entity_dictionary_names = []

        # List of data paths found in the exploration of _extract_data_paths
        data_paths = []

        def _extract_data_paths(
            current_dictionary, current_data_path, current_dictionary_alias=None
        ):
            """Builds the path for secondary tables and updates the entity list

            `current_dictionary_alias` contains:
            - in the traversal, the name of the dictionary as it was named by
              the variable that referenced it;
            - or, otherwise, the name of an external dictionary (for Entity tables).
            """

            # Update the data paths
            if current_dictionary_alias:
                current_data_path.append(current_dictionary_alias)
            else:
                current_data_path.append(current_dictionary.name)
            data_paths.append(current_data_path)

            # Analyze variables to extract additional data paths
            for variable in current_dictionary.variables:
                if variable.is_relational():
                    # Case of a table: Deep-first exploration of the referenced dicts.
                    #                  Explore only non rule tables
                    if variable.rule == "" and variable.variable_block is None:
                        _extract_data_paths(
                            self.get_dictionary(variable.object_type),
                            current_data_path.copy(),
                            variable.name,
                        )
                    # Case of an entity: update the list of unique entity dictionaries
                    elif variable.is_reference_rule():
                        if variable.object_type not in entity_dictionary_names:
                            entity_dictionary_names.append(variable.object_type)

        # == End of inner function _extract_data_paths ==

        # Extract all the data paths from the source dictionary
        source_dictionary = self.get_dictionary(source_dictionary_name)
        _extract_data_paths(source_dictionary, [])

        # Remove the source dictionary from the found data paths
        for i, data_path in enumerate(data_paths):
            data_paths[i] = data_path[1:]

        # Remove source dictionary from the entity list
        if source_dictionary.name in entity_dictionary_names:
            entity_dictionary_names.remove(source_dictionary.name)

        # Extract the data paths recursively for the entity dictionaries found during
        # the first extraction
        # Recall that _extract_data_paths modifies the 'entity_dictionary_names' list;
        # that's why we loop with a 'while' statement
        name_index = 0
        while name_index < len(entity_dictionary_names):
            entity_dictionary_name = entity_dictionary_names[name_index]
            entity_dictionary = self.get_dictionary(entity_dictionary_name)
            name_index += 1

            # Provide custom dictionary alias for Entity tables
            _extract_data_paths(entity_dictionary, [], f"/{entity_dictionary.name}")
        # Remove first data path (that of the source dictionary) before returning
        return ["/".join(data_path) for data_path in data_paths[1:]]

    def get_dictionary_at_data_path(self, data_path):
        """Returns the dictionary name for the specified data path

        Parameters
        ----------
        data_path : str
            A data path for the specified table. Usually the output of
            `extract_data_paths`.

        Returns
        -------
        `Dictionary`
            The dictionary object pointed by this data path.

        Raises
        ------
        `ValueError`
            If the path is not found.
        """
        # If data_path includes "`" and starts with an existing dictionary,
        # assume legacy data path
        if "`" in data_path:
            data_path_parts = data_path.split("`")
            source_dictionary_name = data_path_parts[0]
            if any(kdic.name == source_dictionary_name for kdic in self.dictionaries):
                warnings.warn(
                    deprecation_message(
                        "'`'-based dictionary data path convention",
                        "11.0.1",
                        replacement="'/'-based dictionary data path convention",
                        quote=False,
                    )
                )
                return self._get_dictionary_at_data_path_legacy(data_path)
        return self._get_dictionary_at_data_path(data_path)

    def _get_dictionary_at_data_path_legacy(self, data_path):
        # Legacy data-path convention support
        data_path_parts = data_path.split("`")
        source_dictionary_name = data_path_parts[0]

        dictionary = self.get_dictionary(source_dictionary_name)
        if dictionary is None:
            raise ValueError(f"Source dictionary not found: '{source_dictionary_name}'")

        for table_variable_name in data_path_parts[1:]:
            table_variable = dictionary.get_variable(table_variable_name)
            if table_variable is None:
                raise ValueError(
                    f"Table variable '{table_variable_name}' in data path not found"
                )

            if table_variable.type not in ["Table", "Entity"]:
                raise ValueError(
                    f"Table variable  '{table_variable_name}' "
                    f"in data path is of type '{table_variable.type}'"
                )

            dictionary = self.get_dictionary(table_variable.object_type)
            if dictionary is None:
                raise ValueError(
                    f"Table variable '{table_variable_name}' in data path "
                    f"points to unknown dictionary '{table_variable.object_type}'"
                )
        return dictionary

    def _get_dictionary_at_data_path(self, data_path):
        # Obtain the parts of the data path
        data_path_parts = data_path.lstrip("/").split("/")

        # Attempt to get the first dictionary from the data path:
        # - either it is found as such,
        # - or it is a Table or Entity variable whose table needs to be looked-up
        first_table_variable_name = data_path_parts[0]

        dictionary = self.get_dictionary(first_table_variable_name)
        if dictionary is None:
            for a_dictionary in self.dictionaries:
                try:
                    table_variable = a_dictionary.get_variable(
                        first_table_variable_name
                    )
                    if table_variable is not None:
                        if table_variable.type not in ["Table", "Entity"]:
                            raise ValueError(
                                f"Variable '{table_variable}' "
                                "must be of type 'Table' or 'Entity'"
                            )
                        dictionary = self.get_dictionary(table_variable.object_type)
                        if dictionary is not None:
                            break
                except ValueError:
                    continue
            else:
                raise ValueError(f"Dictionary not found in data path: '{data_path}'")

        for table_variable_name in data_path_parts[1:]:
            table_variable = dictionary.get_variable(table_variable_name)
            if table_variable is None:
                raise ValueError(
                    f"Table variable '{table_variable_name}' in data path not found"
                )

            if table_variable.type not in ["Table", "Entity"]:
                raise ValueError(
                    f"Table variable  '{table_variable_name}' "
                    f"in data path is of type '{table_variable.type}'"
                )

            dictionary = self.get_dictionary(table_variable.object_type)
            if dictionary is None:
                raise ValueError(
                    f"Table variable '{table_variable_name}' in data path "
                    f"points to unknown dictionary '{table_variable.object_type}'"
                )
        return dictionary

    def export_khiops_dictionary_file(self, kdic_file_path):
        """Exports the domain in ``.kdic`` format

        Parameters
        ----------
        kdic_file_path : str
            Path of the output dictionary file (``.kdic``).
        """
        with io.BytesIO() as kdic_contents_stream:
            kdic_file_writer = self.create_output_file_writer(kdic_contents_stream)
            self.write(kdic_file_writer)
            fs.write(kdic_file_path, kdic_contents_stream.getvalue())

    def write(self, stream_or_writer):
        """Writes the domain to a file writer in ``.kdic`` format

        Parameters
        ----------
        stream_or_writer : `io.IOBase` or `.KhiopsOutputWriter`
            Output stream or writer.
        """
        if isinstance(stream_or_writer, io.IOBase):
            writer = self.create_output_file_writer(stream_or_writer)
        elif isinstance(stream_or_writer, KhiopsOutputWriter):
            writer = stream_or_writer
        else:
            raise TypeError(
                type_error_message(
                    "stream_or_writer",
                    stream_or_writer,
                    io.IOBase,
                    KhiopsOutputWriter,
                )
            )
        writer.write("#Khiops ")
        writer.writeln(self.version)
        for dictionary in self.dictionaries:
            dictionary.write(writer)


def read_dictionary_file(dictionary_file_path):
    """Reads a Khiops dictionary file

    Parameters
    ----------
    dictionary_file : str
        Path of the file to be imported. The file can be either Khiops Dictionary
        (extension ``kdic``) or Khiops JSON Dictionary (extension ``.json`` or
        ``.kdicj``).

    Returns
    -------
    `.DictionaryDomain`
        An dictionary domain representing the information in the dictionary file.

    Raises
    ------
    `ValueError`
        When the file has an extension other than ``.kdic``, ``.kdicj`` or ``.json``.

    Examples
    --------
    See the following functions of the ``samples.py`` documentation script:
        - `samples.export_dictionary_files()`
        - `samples.train_predictor_with_cross_validation()`
        - `samples.multiple_train_predictor()`
        - `samples.deploy_model_expert()`
    """
    # Check the extension of the input dictionary file
    extension = os.path.splitext(dictionary_file_path)[1].lower()
    if extension not in [".kdic", ".kdicj", ".json"]:
        raise ValueError(
            f"Input file must have extension 'kdic', 'kdicj' or 'json'."
            f"It has extension: '{extension}'."
        )

    # Import dictionary file: Translate to JSON first if it is 'kdic'
    if extension == ".kdic":
        # Create a temporary file
        tmp_dictionary_file_path = get_runner().create_temp_file(
            "_read_dictionary_file_", ".kdicj"
        )
        # Transform the .kdic file to .kdicj (JSON)
        api.export_dictionary_as_json(dictionary_file_path, tmp_dictionary_file_path)
        json_dictionary_file_path = tmp_dictionary_file_path
    else:
        json_dictionary_file_path = dictionary_file_path

    # Read the JSON dictionary file into a dictionary domain object
    domain = DictionaryDomain(json_data=flexible_json_load(json_dictionary_file_path))

    # Clean the temporary file if the input file was .kdic
    if extension == ".kdic":
        fs.remove(tmp_dictionary_file_path)

    return domain


class Dictionary:
    """A Khiops Dictionary

    A Khiops Dictionary is a description of a table transformation. Common uses in the
    Khiops framework are :

    - Describing the schema of an input table: In this case it is the identity
      transformation of the table(s).
    - Describing a predictor (classifier or regressor): In this case it is the
      transformation between the original table(s) and the prediction values or
      probabilities.

    Parameters
    ----------
    json_data : dict, optional
        Python dictionary representing an element of the list at the ``dictionaries``
        field of a Khiops Dictionary JSON file. If not specified returns an empty
        instance.

    Attributes
    ----------
    name : str
        Dictionary name.
    root : bool
        True if the dictionary is the root of an dictionary hierarchy.
    key : list of str
        Names of the key variables.
    variables : list of `Variable`
        The dictionary variables.
    variable_blocks : list of `VariableBlock`
        The dictionary variable blocks.
    label : str
        Dictionary label.
    comments : list of str
        List of dictionary comments.
    internal_comments : list of str
        List of internal dictionary comments.
    meta_data : `MetaData`
        MetaData object of the dictionary.
    """

    def __init__(self, json_data=None):
        """See class docstring"""
        # Check the type of json_data
        if json_data is not None and not isinstance(json_data, dict):
            raise TypeError(type_error_message("json_data", json_data, dict))

        # Transform to an empty dictionary if json_data is not specified
        if json_data is None:
            json_data = {}
        # Otherwise check the type of the json data and its integrity
        else:
            if "name" not in json_data:
                raise KhiopsJSONError("'name' key not found")

        # Initialize main attributes
        self.name = json_data.get("name", "")
        self.label = json_data.get("label", "")
        self.comments = json_data.get("comments", [])
        self.internal_comments = json_data.get("internalComments", [])
        self.root = json_data.get("root", False)

        # Initialize names of key variable
        self.key = json_data.get("key", [])

        # Initialize the metadata
        json_meta_data = json_data.get("metaData")
        if json_meta_data is None:
            self.meta_data = MetaData()
        else:
            self.meta_data = MetaData(json_meta_data)

        # Initialize variables and blocks
        self.variables = []
        self.variable_blocks = []
        self._variables_by_name = {}
        self._variable_blocks_by_name = {}
        for json_variable in json_data.get("variables", []):
            # Case of a simple variable
            if "name" in json_variable:
                variable = Variable(json_variable)
                self.add_variable(variable)
            # Case of a variable block
            elif "blockName" in json_variable:
                variable_block = VariableBlock(json_variable)
                self.add_variable_block(variable_block)
            else:
                raise KhiopsJSONError(
                    f"Variable/block name not found. JSON data: {json_variable}"
                )

    def __repr__(self):
        """Returns a human readable string representation"""
        return f"Dictionary ({self.name})"

    def __str__(self):
        stream = io.BytesIO()
        writer = KhiopsOutputWriter(stream)
        self.write(writer)
        return str(stream.getvalue(), encoding="utf8", errors="replace")

    def copy(self):
        """Returns a copy of this instance

        Returns
        -------
        `Dictionary`
            A copy of this instance.
        """
        # Create an empty dictionary
        dictionary_copy = Dictionary()

        # Copy dictionary main features
        dictionary_copy.name = self.name
        dictionary_copy.label = self.label
        dictionary_copy.comments = self.comments.copy()
        dictionary_copy.internal_comments = self.internal_comments.copy()
        dictionary_copy.root = self.root
        dictionary_copy.key = self.key.copy()
        dictionary_copy.meta_data = self.meta_data.copy()

        # Copy variables
        i = 0
        while i < len(self.variables):
            variable = self.variables[i]
            # Simple variable case
            if variable.variable_block is None:
                variable_copy = variable.copy()
                dictionary_copy.add_variable(variable_copy)
                i += 1
            # Variable block case
            else:
                variable_block = variable.variable_block
                variable_block_copy = VariableBlock()
                variable_block_copy.name = variable.variable_block.name
                variable_block_copy.label = variable.variable_block.label
                variable_block_copy.comments = variable.variable_block.comments.copy()
                variable_block_copy.internal_comments = (
                    variable.variable_block.internal_comments.copy()
                )
                variable_block_copy.rule = variable.variable_block.rule
                variable_block_copy.meta_data = variable_block.meta_data.copy()

                for variable in variable_block.variables:
                    variable_block_copy.add_variable(variable.copy())

                dictionary_copy.add_variable_block(variable_block_copy)
                i += len(variable_block.variables)
        return dictionary_copy

    def get_value(self, key):
        """Returns the metadata value associated to the specified key

        Returns
        -------
        `MetaData`
            Metadata value associated to the specified key. ``None`` is returned
            if the metadata key is not found.
        """
        return self.meta_data.get_value(key)

    def use_all_variables(self, is_used):
        """Sets the ``used`` flag of all dictionary variables to the specified value

        Parameters
        ----------
        is_used : bool
            Sets the ``used`` field to ``is_used`` for all the `Variable` objects in
            this dictionary.
        """
        for variable in self.variables:
            variable.used = is_used

    def get_variable(self, variable_name):
        """Returns the specified variable

        Parameters
        ----------
        variable_name : str
            A name of a variable.

        Returns
        -------
        `Variable`
            The specified variable. ``None`` is returned if the variable name is
            not found.
        """
        return self._variables_by_name.get(variable_name)

    def get_variable_block(self, variable_block_name):
        """Returns the specified variable block

        Parameters
        ----------
        variable_block_name : str
            A name of a variable block.

        Returns
        -------
        `VariableBlock`
            The specified variable block. ``None`` is returned if the variable
            block name is not found.
        """
        return self._variable_blocks_by_name.get(variable_block_name)

    def add_variable(self, variable):
        """Adds a variable to this dictionary

        Parameters
        ----------
        variable : `Variable`
            The variable to be added.

        Raises
        ------
        `TypeError`
            If variable is not of type `Variable`

        `ValueError`
            If the name is empty or if there is already a variable with that name.
        """
        if not isinstance(variable, Variable):
            raise TypeError(type_error_message("variable", variable, Variable))
        if not variable.name:
            raise ValueError(
                "Cannot add to dictionary unnamed variable "
                f"(variable.name = '{variable.name}')"
            )
        if variable.name in self._variables_by_name:
            raise ValueError(
                f"Dictionary already has a variable named '{variable.name}'"
            )
        self.variables.append(variable)
        self._variables_by_name[variable.name] = variable

    def add_variable_from_spec(
        self,
        name,
        type,
        label="",
        used=True,
        object_type=None,
        structure_type=None,
        rule=None,
        meta_data=None,
    ):
        """Adds a variable to this dictionary using a complete specification

        Parameters
        ----------
        name : str
            Variable name.
        type : str
            Variable type. See `Variable`.
        label : str, default ""
            Label of the variable.
        used : bool, default ``True``
            Usage status of the variable.
        object_type : str, optional
            Object type. Ignored if variable type not in ["Entity", "Table"].
        structure_type : str, optional
            Structure type. Ignored if variable type is not "Structure".
        rule : str, optional
            String representation of a variable rule.
        meta_data : dict, optional
            A Python dictionary which holds the metadata specification.
            The dictionary keys are str. The values can be str, bool, float or int.

        Raises
        ------
        `ValueError`
            - If the variable name is empty or does not comply
              with the formatting constraints.
            - If there is already a variable with the same name.
            - If the given variable type is unknown.
            - If a native type is given 'object_type' or 'structure_type'.
            - If the 'meta_data' is not a dictionary.
        """
        # Values and Types checks
        if not name:
            raise ValueError(
                "Cannot add to dictionary unnamed variable " f"(name = '{name}')"
            )
        if name in self._variables_by_name:
            raise ValueError(f"Dictionary already has a variable named '{name}'")
        if not _is_valid_type(type):
            raise ValueError(f"Invalid type '{type}'")
        if _is_native_type(type):
            if object_type or structure_type:
                raise ValueError(
                    f"Native type '{type}' "
                    "cannot have 'object_type' or 'structure_type'"
                )
        if _is_object_type(type) and object_type is None:
            raise ValueError(f"'object_type' must be provided for type '{type}'")
        if type == "Structure" and structure_type is None:
            raise ValueError(f"'structure_type' must be provided for type '{type}'")
        if meta_data is not None:
            if not is_dict_like(meta_data):
                raise TypeError(type_error_message("meta_data", meta_data, "dict-like"))
        if object_type is not None:
            if not is_string_like(object_type):
                raise TypeError(
                    type_error_message("object_type", object_type, "string-like")
                )
        if structure_type is not None:
            if not is_string_like(structure_type):
                raise TypeError(
                    type_error_message("structure_type", structure_type, "string-like")
                )
        if rule is not None:
            if not isinstance(rule, str):
                raise TypeError(type_error_message("rule", rule, str))

        # Variable initialization
        variable = Variable()
        variable.name = name
        variable.type = type
        variable.used = used
        if meta_data is not None:
            for key, value in meta_data.items():
                variable.meta_data.add_value(key, value)
        variable.label = label
        if object_type is not None:
            variable.object_type = object_type
        if structure_type is not None:
            variable.structure_type = structure_type
        if rule is not None:
            variable.rule = str(rule)
        self.add_variable(variable)

    def remove_variable(self, variable_name):
        """Removes the specified variable from this dictionary

        Parameters
        ----------
        variable_name : str
            Name of the variable to be removed.

        Returns
        -------
        `Variable`
            The removed variable.

        Raises
        ------
        `KeyError`
            If no variable with the specified name exists.
        """
        variable = self._variables_by_name.pop(variable_name)
        self.variables.remove(variable)
        if variable.variable_block is not None:
            variable.variable_block.remove_variable(variable)
            if not variable.variable_block.variables:
                self.remove_variable_block(variable.variable_block.name)
        return variable

    def add_variable_block(self, variable_block):
        """Adds a variable block to this dictionary

        Parameters
        ----------
        variable_block : `VariableBlock`
            The variable block to be added.

        Raises
        ------
        `TypeError`
            If variable is not of type `VariableBlock`

        `ValueError`
            If the name is empty or if there is already a variable block with that name.

        """
        if not isinstance(variable_block, VariableBlock):
            raise TypeError(
                type_error_message("variable_block", variable_block, VariableBlock)
            )
        if variable_block.name is None or variable_block.name == "":
            raise ValueError(
                "Cannot add to dictionary unnamed variable block; "
                f"block.name = '{variable_block.name}'"
            )
        if variable_block.name in self._variable_blocks_by_name:
            raise ValueError(
                f"Dictionary already has a variable block named '{variable_block.name}'"
            )
        self.variable_blocks.append(variable_block)
        self._variable_blocks_by_name[variable_block.name] = variable_block
        for variable in variable_block.variables:
            self.add_variable(variable)

    def remove_variable_block(
        self, variable_block_name, keep_native_block_variables=True
    ):
        """Removes the specified variable block from this dictionary

        .. note::
            Non-native block variables (those created from block rules) are never kept
            in the dictionary.

        Parameters
        ----------
        variable_name : str
            Name of the variable block to be removed.
        keep_native_block_variables : bool, default ``True``
            If ``True`` and the block is native then  only the block structure is
            removed from the dictionary but the variables are kept in it; neither the
            variables point to the block nor the removed block points to the variables.
            If ``False`` the variables are removed from the dictionary; the block
            preserves the references to their variables.

        Returns
        -------
        `VariableBlock`
            The removed variable block.

        Raises
        ------
        `KeyError`
            If no variable block with the specified name exists.
        """
        removed_block = self.get_variable_block(variable_block_name)

        # Only eliminate variable->block and block->variable references when:
        # - It is a native block and
        # - keep_native_block_variables is True
        if removed_block.rule == "" and keep_native_block_variables:
            for variable in removed_block.variables:
                variable.variable_block = None
            removed_block.variables = []
        # Otherwise: Eliminate variables from the dictionary and keep refs. on block
        else:
            for variable in removed_block.variables:
                self._variables_by_name.pop(variable.name)
                self.variables.remove(variable)

        # Remove block and its indexing
        del self._variable_blocks_by_name[removed_block.name]
        self.variable_blocks.remove(removed_block)

        return removed_block

    def is_key_variable(self, variable):
        """Returns ``True`` if a variable belongs to this dictionary's key

        Parameters
        ----------
        variable : `Variable`
            The variable for the query.

        Returns
        -------
        bool
            ``True`` if the variable belong to the key.
        """
        return variable.name in self.key

    def write(self, writer):
        """Writes the dictionary to a file writer in ``.kdic`` format

        Parameters
        ----------
        writer : `.KhiopsOutputWriter`
            Output dictionary file.
        """
        # Check file object type
        if not isinstance(writer, KhiopsOutputWriter):
            raise TypeError(type_error_message("writer", writer, KhiopsOutputWriter))

        # Write dictionary header
        writer.writeln("")
        if self.label:
            writer.write("// ")
            writer.writeln(self.label)
        if self.comments:
            for comment in self.comments:
                writer.write("// ")
                writer.writeln(comment)
        if self.root:
            writer.write("Root\t")
        writer.write("Dictionary\t")
        writer.write(_format_name(self.name))
        if self.key:
            writer.write("\t(")
            for i, variable_name in enumerate(self.key):
                if i > 0:
                    writer.write(", ")
                writer.write(_format_name(variable_name))
            writer.write(")")
        writer.writeln("")

        # Write metadata if available
        if not self.meta_data.is_empty():
            self.meta_data.write(writer)
            writer.writeln("")

        # Write variables and variable blocks
        writer.writeln("{")
        i = 0
        while i < len(self.variables):
            variable = self.variables[i]
            if variable.variable_block is None:
                variable.write(writer)
                i += 1
            else:
                variable.variable_block.write(writer)
                i += len(variable.variable_block.variables)

        # Write internal comments if available
        for comment in self.internal_comments:
            writer.write("// ")
            writer.writeln(comment)
        writer.writeln("};")


class Variable:
    """A variable of a Khiops dictionary

    Parameters
    ----------
    json_data : dict, optional
        Python dictionary representing an element of the list at the ``variables`` field
        of dictionaries found in a Khiops Dictionary JSON file. If not specified it
        returns an empty instance.

    Attributes
    ----------
    name : str
        Variable name.
    used : bool
        True if the variable is used.
    type : str
        Variable type.
        It can be either native (``Categorical``, ``Numerical``, ``Time``,
        ``Date``, ``Timestamp``, ``TimestampTZ``, ``Text``),
        internal (``TextList``, ``Structure``)

            - See https://khiops.org/11.0.0-b.0/api-docs/kdic/text-list-rules/
            - See https://khiops.org/11.0.0-b.0/api-docs/kdic/structures-introduction/

        or relational (``Entity`` - 0-1 relationship, ``Table`` - 0-n relationship)

            - See https://khiops.org/11.0.0-b.0/tutorials/kdic_multi_table/

    object_type : str
        Type complement for the ``Table`` and ``Entity`` types.
    structure_type : str
        Type complement for the ``Structure`` type. Set to "" for other types.
    rule : str
        Derivation rule or external table reference. Set to "" if there is no
        rule associated to this variable. Examples:

            - standard rule: "Sum(Var1, Var2)"
            - reference rule: "[TableName]"

    variable_block : `VariableBlock`
        Block to which the variable belongs. Not set if the variable does not belong to
        a block.
    label : str
        Variable label.
    comments : list of str
        List of variable comments.
    meta_data : `MetaData`
        Variable metadata.

    Examples
    --------
    See the following function of the ``samples.py`` documentation script:
        - `samples.create_dictionary_domain()`
    """

    def __init__(self, json_data=None):
        """See class docstring"""
        # Check the type of json_data
        if json_data is not None and not isinstance(json_data, dict):
            raise TypeError(type_error_message("json_data", json_data, dict))

        # Main attributes
        # The variable name is protected attribute accessible only via a property
        # to ensure it is always valid
        self._name = ""
        self.label = ""
        self.comments = []
        self.used = True
        self.type = ""

        # Type complement attributes
        self.object_type = ""
        self.structure_type = ""

        # Derivation rule
        self._rule = ""

        # Metadata
        self.meta_data = MetaData()

        # Reference to parent variable block
        self.variable_block = None

        # Return empty instance if no JSON data
        if json_data is None:
            return

        # Check the type of the json data and its integrity
        if not isinstance(json_data, dict):
            raise KhiopsJSONError(
                type_error_message("json data for variable", json_data, dict)
            )
        if "name" not in json_data:
            raise KhiopsJSONError("'name' key not found")
        if "type" not in json_data:
            raise KhiopsJSONError("'type' key not found")

        # Initialize main attributes
        self.name = json_data.get("name")
        self.label = json_data.get("label", "")
        self.comments = json_data.get("comments", [])
        self.used = json_data.get("used", True)
        self.type = json_data.get("type")

        # Initialize complement of the type
        if _is_object_type(self.type):
            self.object_type = json_data.get("objectType")
        elif self.type == "Structure":
            self.structure_type = json_data.get("structureType")

        # Initialize derivation rule
        self._rule = json_data.get("rule", "")

        # Initialize metadata
        json_meta_data = json_data.get("metaData")
        if json_meta_data is not None:
            self.meta_data = MetaData(json_meta_data)

    def __repr__(self):
        """Returns a human-readable string representation"""
        return f"Variable ({self.name})"

    def __str__(self):
        stream = io.BytesIO()
        writer = KhiopsOutputWriter(stream)
        self.write(writer)
        return str(stream.getvalue(), encoding="utf8", errors="replace")

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        _check_name(value)
        self._name = value

    @property
    def rule(self):
        return self._rule

    @rule.setter
    def rule(self, value):
        if not is_string_like(value):
            raise TypeError(type_error_message("rule", value, "string-like"))
        self._rule = value

    def copy(self):
        """Copies this variable instance

        Returns
        -------
        `Variable`
            A copy of this instance.
        """
        variable = Variable()
        variable.name = self.name
        variable.label = self.label
        variable.comments = self.comments.copy()
        variable.used = self.used
        variable.type = self.type
        variable.object_type = self.object_type
        variable.structure_type = self.structure_type
        variable.rule = self.rule
        variable.meta_data = self.meta_data.copy()
        return variable

    def get_value(self, key):
        """Returns the metadata value associated to the specified key

        Returns
        -------
        `MetaData`
            Metadata value associated to the specified key. ``None`` is returned
            if the metadata key is not found.
        """
        return self.meta_data.get_value(key)

    def is_native(self):
        """Returns ``True`` if the variable comes directly from a data column

        Variables are **not native** if they come from a derivation rule, an external
        entity, a sub-table or structures.

        Returns
        -------
        bool
            ``True`` if a variables comes directly from a data column.

        """
        base_types = [
            "Categorical",
            "Numerical",
            "Time",
            "Date",
            "Timestamp",
            "TimestampTZ",
            "Text",
            "TextList",
        ]
        if self.variable_block is None:
            return self.rule == "" and self.type in base_types
        return self.variable_block.rule == ""

    def is_relational(self):
        """Returns ``True`` if the variable is of relational type

        Relational variables reference other tables or external entities.

        Returns
        -------
        bool
            True if the variable is of relational type.
        """
        return self.type in ["Entity", "Table"]

    def is_reference_rule(self):
        """Returns ``True`` if the special reference rule is used

        The reference rule is used to make reference to an external entity.

        Returns
        -------
        bool
            ``True`` if the special reference rule is used.
        """
        if self.rule:
            if isinstance(self.rule, str):
                if self.rule.startswith("[") and self.rule.endswith("]"):
                    return True
            else:
                assert isinstance(self.rule, bytes)
                if self.rule.startswith(b"[") and self.rule.endswith(b"]"):
                    return True
        return False

    def full_type(self):
        """Returns the variable's full type

        Returns
        -------
        str
            The full type is the variable type plus its complement if the type is not
            basic.
        """
        full_type = self.type
        if _is_object_type(self.type):
            full_type += f"({self.object_type})"
        elif self.type == "Structure":
            full_type += f"({self.structure_type})"
        return full_type

    def write(self, writer):
        """Writes the domain to a file writer in ``.kdic`` format

        Parameters
        ----------
        writer : `.KhiopsOutputWriter`
            Output writer.
        """
        # Check file object type
        if not isinstance(writer, KhiopsOutputWriter):
            raise TypeError(type_error_message("writer", writer, KhiopsOutputWriter))

        # Write comments if available
        for comment in self.comments:
            writer.write("\t// ")
            writer.writeln(comment)

        # Write "Unused" flag if variable not used
        if not self.used:
            writer.write("Unused")

        # Write variable's full type
        writer.write("\t" + self.full_type())

        # Write external name
        writer.write("\t")
        writer.write(_format_name(self.name))
        writer.write("\t")

        # Write derivation rule if available
        if self.rule:
            if not self.is_reference_rule():
                writer.write(" = ")
            writer.write(self.rule)
        writer.write("\t;")

        # Write metadata if available
        if not self.meta_data.is_empty():
            writer.write(" ")
            self.meta_data.write(writer)
        writer.write("\t")

        # Write label if available
        if self.label:
            writer.write("// ")
            writer.write(self.label)
        writer.writeln("")


class VariableBlock:
    """A variable block of a Khiops dictionary

    Parameters
    ----------
    json_data : dict, optional
        Python dictionary representing an element of the list at the ``variables`` field
        of a dictionary object in a Khiops Dictionary JSON file. The element must have a
        ``blockName`` field.  If not specified it returns an empty instance.

    Attributes
    ----------
    name : str
        Block name.
    rule :
        Block derivation rule.
    variables :
        List of the Variable objects of the block.
    label : str
        Block label.
    comments : list of str
        List of block comments.
    internal_comments : list of str
        List of internal block comments.
    meta_data :
        Metadata object of the block.
    """

    def __init__(self, json_data=None):
        """See class docstring"""
        # Check the type of json_data
        if json_data is not None and not isinstance(json_data, dict):
            raise TypeError(type_error_message("json_data", json_data, dict))

        # Transform to an empty dictionary if json_data is not specified
        if json_data is None:
            json_data = {}
        # Otherwise fail if 'blockName' is present in json_data
        else:
            if "blockName" not in json_data:
                raise KhiopsJSONError("'blockName' key not found")

        # Initialize main attributes
        self.name = json_data.get("blockName", "")
        self.label = json_data.get("label", "")
        self.comments = json_data.get("comments", [])
        self.internal_comments = json_data.get("internalComments", [])

        # Initialize derivation rule
        self._rule = json_data.get("rule", "")

        # Initialize metadata if available
        json_meta_data = json_data.get("metaData")
        if json_meta_data is None:
            self.meta_data = MetaData()
        else:
            self.meta_data = MetaData(json_meta_data)

        # Initialize variables
        self.variables = []
        for json_variable in json_data.get("variables", []):
            variable = Variable(json_variable)
            self.add_variable(variable)

    def __repr__(self):
        """Returns a human readable string representation"""
        return f"Variable block ({self.name})"

    def __str__(self):
        stream = io.BytesIO()
        writer = KhiopsOutputWriter(stream)
        self.write(writer)
        return str(stream.getvalue(), encoding="utf8", errors="replace")

    @property
    def rule(self):
        return self._rule

    @rule.setter
    def rule(self, value):
        if not is_string_like(value):
            raise TypeError(type_error_message("rule", value, "string-like"))
        self._rule = value

    def add_variable(self, variable):
        """Adds a variable to this block

        Parameters
        ----------
        variable : `Variable`
            The variable to be added.

        Raises
        ------
        `TypeError`
            If the variable is not of type `Variable`.
        """
        if not isinstance(variable, Variable):
            raise TypeError(type_error_message("variable", variable, Variable))
        if variable in self.variables:
            raise ValueError(
                f"Block already has the input variable named {variable.name}"
            )
        self.variables.append(variable)
        variable.variable_block = self

    def remove_variable(self, variable):
        """Removes a variable from this block

        Parameters
        ----------
        variable : `Variable`
            The variable to be removed.

        Raises
        ------
        `TypeError`
            If the variable is not of type `Variable`.
        """
        # Check input
        if not isinstance(variable, Variable):
            raise TypeError(type_error_message("variable", variable, Variable))
        if variable not in self.variables:
            raise ValueError(f"Variable not in block (name: {variable.name})")
        self.variables.remove(variable)

    def get_value(self, key):
        """Returns the metadata value associated to the specified key

        Returns
        -------
        `MetaData`
            Metadata value associated to the specified key. ``None`` is returned
            if the metadata key is not found.
        """
        return self.meta_data.get_value(key)

    def write(self, writer):
        """Writes the variable block to a file writer in ``.kdic`` format

        Parameters
        ----------
        writer : `.KhiopsOutputWriter`
            Output writer.
        """
        # Check file object type
        if not isinstance(writer, KhiopsOutputWriter):
            raise TypeError(type_error_message("writer", writer, KhiopsOutputWriter))

        # Write comments if available
        for comment in self.comments:
            writer.write("\t// ")
            writer.writeln(comment)

        # Write variables
        writer.writeln("\t{")
        for variable in self.variables:
            variable.write(writer)

        # Write internal comments if available
        for comment in self.internal_comments:
            writer.write("\t// ")
            writer.writeln(comment)
        writer.write("\t}")

        # Write block's name
        writer.write("\t")
        writer.write(_format_name(self.name))
        writer.write("\t")

        # Write derivation rule if available
        if self.rule:
            writer.write(" = ")
            writer.write(self.rule)
        writer.write("\t;")

        # Write metadata if available
        if not self.meta_data.is_empty():
            writer.write(" ")
            self.meta_data.write(writer)
        writer.write("\t")

        # Write label if available
        if self.label:
            writer.write("// ")
            writer.write(self.label)
        writer.writeln("")


class Rule:
    """A rule of a variable or variable block in a Khiops dictionary

    This object is a convenience feature which eases rule creation and
    serialization, especially in complex cases (rule operands which are
    variables or rules themselves, sometimes upper-scoped). A `Rule` instance
    must be converted to `str` before setting it in a `Variable` or
    `VariableBlock` instance.

    `Rule` instances can be created either from full operand specifications, or
    from verbatim rules. The latter is useful when the rule is retrieved from an
    existing variable or variable block and is used as an operand in another
    rule.

    Parameters
    ----------
    name_and_operands : tuple
        Each tuple member can have one of the following types:

            - str
            - bytes
            - int
            - float
            - `Variable`
            - `Rule`
            - upper-scoped `Variable`
            - upper-scoped `Rule`

        The first element of the ``name_and_operands`` tuple is the name of the
        rule and must be str or bytes and non-empty for a standard rule, i.e. if
        ``is_reference`` is not set.
    verbatim : str or bytes, optional
        Verbatim representation of an entire rule. If set, then ``names_and_operands``
        must be empty.
    is_reference : bool, default ``False``
        If set to ``True``, then the rule is serialized as a reference rule:
        ``Rule(Operand1, Operand2, ...)`` is serialized as
        ``[Operand1, Operand2, ...]``.

    Attributes
    ----------
    name : str or bytes or ``None``
        Name of the rule. It is ``None`` for reference rules.
    operands : tuple of operands
        Each operand has one of the following types:

            - str
            - bytes
            - int
            - float
            - `Variable`
            - `Rule`
            - upper-scoped `Variable`
            - upper-scoped `Rule`

    is_reference : bool
        The reference status of the rule.

        .. note::
            This attribute cannot be changed on a `Rule` instance.

    Examples
    --------
        - basic rule, with variables as operands:
            - verbatim:
                .. code-block::

                    Product(PetalLength, PetalWidth)

            - object construction:
                .. highlight:: python
                .. code-block:: python

                    petal_length_var = kh.Variable()
                    petal_length_var.name = "PetalLength"
                    petal_length_var.type = "Numerical"
                    petal_width_var = kh.Variable()
                    petal_width_var.name = "PetalWidth"
                    petal_width_var.type = "Numerical"
                    rule = kh.Rule("Product", petal_length_var, petal_width_var)

        - multi-table rule:
            - verbatim:
                .. code-block::

                    TableCount(
                        TableSelection(
                            Vehicles,
                            EQ(PassengerNumber, 1)
                        )
                    )

            - object construction:
                .. highlight:: python
                .. code-block:: python

                    vehicles_var = accidents_dictionary.get_variable("Vehicles")
                    passenger_number_var = vehicles_dictionary.get_variable(
                        "PassengerNumber"
                    )
                    rule = kh.Rule(
                        "TableCount",
                        kh.Rule(
                            "TableSelection",
                            vehicles_var,
                            kh.Rule("EQ", passenger_number_var, 1)
                        )
                    )

        - multi-table rule with upper-scoped operands (advanced usage):
            - verbatim:
                .. code-block::

                    TableSelection(
                        Vehicles,
                        EQ(
                            PassengerNumber,
                            .TableMax(Vehicles, PassengerNumber)
                        )
                    )

            - object construction:
                .. highlight:: python
                .. code-block:: python

                    vehicles_var = accidents_dictionary.get_variable("Vehicles")
                    passenger_number_var = vehicles_dictionary.get_variable(
                        "PassengerNumber"
                    )
                    rule = kh.Rule(
                        "TableSelection",
                        vehicles_var,
                        kh.Rule(
                            "EQ",
                            passenger_number_var,
                            kh.upper_scope(
                                kh.Rule(
                                    "TableMax",
                                    vehicle_var,
                                    passenger_number_var
                                )
                            )
                        )
                    )

    """

    def __init__(self, *name_and_operands, verbatim=None, is_reference=False):
        """See class docstring"""
        # Check input parameters and initialize rule fragments accordigly
        if not isinstance(is_reference, bool):
            raise TypeError(type_error_message("is_reference", is_reference, bool))

        # Rule provided as name plus operands
        if verbatim is None:
            if not name_and_operands:
                raise ValueError("A name must be provided to a standard rule")
            if is_reference:
                self.name = None
                self.operands = name_and_operands
            else:
                name, *operands = name_and_operands
                if not is_string_like(name):
                    raise TypeError(type_error_message("name", name, "string-like"))
                if not name:
                    raise ValueError("'name' must be a non-empty string")
                self.name = name
                self.operands = operands
        # Rule provided as verbatim
        else:
            if not is_string_like(verbatim):
                raise TypeError(type_error_message("verbatim", verbatim, "string-like"))
            if not verbatim:
                raise ValueError("'verbatim' must be a non-empty string")
            if name_and_operands:
                raise ValueError(
                    "Rule name and operands must not be provided for verbatim rules"
                )
            self.name = None
            self.operands = ()

        # Check operand types
        for operand in self.operands:
            if not is_string_like(operand) and not isinstance(
                operand, (int, float, Variable, Rule, _ScopedOperand)
            ):
                raise TypeError(
                    type_error_message(
                        f"Operand '{operand}'",
                        operand,
                        "string-like",
                        int,
                        float,
                        Variable,
                        Rule,
                        "upper-scoped Variable",
                        "upper-scoped Rule",
                    )
                )

        # Initialize private attributes
        self._verbatim = verbatim
        self._is_reference = is_reference

    @property
    def is_reference(self):
        return self._is_reference

    def __repr__(self):
        stream = io.BytesIO()
        writer = KhiopsOutputWriter(stream)
        self.write(writer)
        return str(stream.getvalue(), encoding="utf8", errors="replace")

    def copy(self):
        """Copies this rule instance

        Returns
        -------
        `Rule`
            A copy of this instance.
        """
        return Rule(self.name, *self.operands)

    def write(self, writer):
        """Writes the rule to a file writer in the ``.kdic`` format

        This method ensures proper `Rule` serialization, automatically handling:

            - back-quote recoding in variable names
            - double-quote recoding in categorical constants
            - missing data (``inf``, ``-inf``, ``NaN``) serialization as ``#Missing``
            - upper-scope operator serialization as ``.``

        Parameters
        ----------
        writer : `.KhiopsOutputWriter`
            Output writer.

            .. note::
                ``self.name`` is not included in the serialization of reference rules.
        """
        # Check the type of the writer
        if not isinstance(writer, KhiopsOutputWriter):
            raise TypeError(type_error_message("writer", writer, KhiopsOutputWriter))

        # Write standard rule
        rule_name_pattern = r"^[A-Z]([a-zA-Z]*)$"
        rule_name_regex = re.compile(rule_name_pattern)
        bytes_rule_name_regex = re.compile(bytes(rule_name_pattern, encoding="ascii"))
        if self.operands:
            if self.is_reference:
                writer.write("[")
            else:
                writer.write(_format_name(self.name))
                writer.write("(")

            # Write operand, according to its type
            # Variable operands have their name written only
            for i, operand in enumerate(self.operands):
                if isinstance(operand, (Rule, _ScopedOperand)):
                    operand.write(writer)
                elif isinstance(operand, Variable):
                    writer.write(_format_name(operand.name))
                elif is_string_like(operand):
                    writer.write(_quote_value(operand))
                elif isinstance(operand, float) and not math.isfinite(operand):
                    writer.write("#Missing")
                # int or finite float cases
                else:
                    writer.write(str(operand))
                if i < len(self.operands) - 1:
                    writer.write(", ")
            if self.is_reference:
                writer.write("]")
            else:
                writer.write(")")
        # Write no-operand rule
        elif (
            isinstance(self.name, str)
            and rule_name_regex.match(self.name)
            or isinstance(self.name, bytes)
            and bytes_rule_name_regex.match(self.name)
        ):
            writer.write(_format_name(self.name))

            # Add parentheses automatically
            writer.write("()")
        # Write verbatim-given rule
        elif self._verbatim:
            writer.write(self._verbatim)


class _ScopedOperand:
    def __init__(self, operand):
        assert type(operand) in (Variable, Rule, _ScopedOperand), type_error_message(
            "operand", operand, Variable, Rule, "upper-scoped Variable or Rule"
        )
        self.operand = operand

    def write(self, writer):
        assert isinstance(writer, KhiopsOutputWriter), type_error_message(
            "writer", writer, KhiopsOutputWriter
        )
        writer.write(".")
        if isinstance(self.operand, Variable):
            writer.write(_format_name(self.operand.name))
        else:
            self.operand.write(writer)

    def __repr__(self):
        stream = io.BytesIO()
        writer = KhiopsOutputWriter(stream)
        self.write(writer)
        return str(stream.getvalue(), encoding="utf8", errors="replace")


def upper_scope(operand):
    """Applies the upper-scope operator ``.`` to an operand

    Parameters
    ----------
    operand : `Variable`, `Rule`, upper-scoped `Variable` or upper-scoped `Rule`
        Operand that is upper-scoped.

    Raises
    ------
    `TypeError`
        If the type of ``operand`` is not `Variable`, `Rule`, upper-scoped `Variable`
        or upper-scoped `Rule`.

    Returns
    -------
    upper-scoped operand
        The upper-scoped operand, as if the upper-scope operator ``.`` were
        applied to an operand in a rule in the ``.kdic`` dictionary language.

    """
    if not isinstance(operand, (Variable, Rule, _ScopedOperand)):
        raise TypeError(
            type_error_message(
                "operand", operand, Variable, Rule, "upper-scoped Variable or Rule"
            )
        )
    return _ScopedOperand(operand)


class MetaData:
    """A metadata container for a dictionary, a variable or variable block

    The metadata for both dictionaries and variables is a list of key-value pairs. The
    values can be set either to a string, to a number, or to the boolean value True. The
    latter represents flag metadata: they are either present (``True``) or absent.

    Parameters
    ----------
    json_data : dict, optional
        Python dictionary representing the object at a ``metaData`` field of a
        dictionary domain, dictionary or variable in a Khiops Dictionary JSON file. If
        None it returns an empty instance.

    Attributes
    ----------
    keys : list of str
        The metadata keys.
    values : list
        Metadata values for each key in ``keys`` (synchronized lists). They can be
        either str, int or float.
    """

    def __init__(self, json_data=None):
        """See class docstring"""
        # Check the type of json_data
        if json_data is not None and not isinstance(json_data, dict):
            raise TypeError(type_error_message("json_data", json_data, dict))

        # Transform to an empty dictionary if json_data is not specified
        if json_data is None:
            json_data = {}

        # Initialize the key-value pairs
        # They are stored in two separate lists to keep the key ordering
        self.keys = []
        self.values = []
        for key, value in json_data.items():
            self.add_value(key, value)

    def __contains__(self, key):
        return key in self.keys

    def copy(self):
        """Copies this metadata instance

        Returns
        -------
        `MetaData`
            A copy of this instance.
        """
        new_meta_data = MetaData()
        new_meta_data.keys = self.keys.copy()
        new_meta_data.values = self.values.copy()
        return new_meta_data

    def is_empty(self):
        """Returns ``True`` if the meta-data is empty

        Returns
        -------
        bool
            Returns ``True`` if the meta-data is empty
        """
        return len(self.keys) == 0

    def get_value(self, key):
        """Returns the value at the specified key

        Returns
        -------
        int, str or float
            The value at the specified key. ``None`` is returned if the key is
            not found.

        Raises
        ------
        `TypeError`
            If ``key`` is not str.
        """
        # Check the argument types
        if not is_string_like(key):
            raise TypeError(type_error_message("key", key, "string-like"))

        # Linear search for the key
        for i, stored_key in enumerate(self.keys):
            if stored_key == key:
                return self.values[i]
        return None

    def add_value(self, key, value):
        """Adds a value at the specified key

        Parameters
        ----------
        key : str
            Key to be added.
            A valid key is a sequence of non-accented alphanumeric characters
            which starts with a non-numeric character.
        value : bool, int, float or str
            Value to be added.

        Raises
        ------
        `TypeError`
            - If the key is not a valid string
            - If the value is not a valid string or if is not bool, int, float.
        `ValueError`
            If the key is already stored.
        """
        # Check that the type of key is string-like
        if not is_string_like(key):
            raise TypeError(type_error_message("name", key, "string-like"))

        # Check that the type of the value is a valid-one
        if not (isinstance(value, (bool, int, float)) or is_string_like(value)):
            raise TypeError(
                type_error_message("value", value, bool, "string-like", int, float)
            )

        # Linear search to check if the key already exists
        for stored_key in self.keys:
            if stored_key == key:
                raise ValueError(f"Cannot add value to existent key '{key}'")
        self.keys.append(key)
        self.values.append(value)

    def remove_key(self, key):
        """Removes the value at the specified key

        Parameters
        ----------
        key : str
            The key to be removed.

        Returns
        -------
        bool, int, float, str
            The value associated to the key removed.

        Raises
        ------
        `TypeError`
            If the key is not str.
        `KeyError`
            If the key is not contained in this metadata.
        """
        # Check that the type of key is string-like
        if not is_string_like(key):
            raise TypeError(type_error_message("name", key, "string-like"))

        # Linear search and elimination of the key and its value if they exist
        for i, stored_key in enumerate(self.keys):
            if stored_key == key:
                self.keys.pop(i)
                return self.values.pop(i)

        # Raise key error if not found
        raise KeyError(key)

    def write(self, writer):
        """Writes the metadata to a file writer in ``.kdic`` format

        Parameters
        ----------
        writer : `.KhiopsOutputWriter`
            Output writer.
        """
        if not isinstance(writer, KhiopsOutputWriter):
            raise TypeError(type_error_message("writer", writer, KhiopsOutputWriter))

        for i, key in enumerate(self.keys):
            value = self.values[i]
            if i > 0:
                writer.write(" ")

            # Write the key-value bracket, values are:
            #  quoted strings, unquoted ints/floats and nothing for booleans.
            writer.write("<")
            writer.write(key)
            if is_string_like(value):
                writer.write("=")
                writer.write(_quote_value(value))
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                writer.write("=")
                writer.write(str(value))
            writer.write(">")
