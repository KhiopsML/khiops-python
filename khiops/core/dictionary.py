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
import os
import re
import warnings

import khiops.core.internals.filesystems as fs
from khiops.core import api
from khiops.core.exceptions import KhiopsJSONError
from khiops.core.internals.common import (
    deprecation_message,
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
    # Check that the type of name is string or bytes
    if not is_string_like(name):
        raise TypeError(type_error_message("name", name, "string-like"))

    # Check if the name is an identifier
    # Python isalnum is not used because of utf-8 encoding (accentuated chars
    # are considered alphanumeric)
    # Return original name if is an identifier, otherwise between backquotes
    identifier_pattern = r"^[a-zA-Z][a-zA-Z0-9_]*"
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
            The specified dictionary.

        Raises
        ------
        `KeyError`
            If no dictionary with the specified name exist.

        """
        return self._dictionaries_by_name[dictionary_name]

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

        try:
            dictionary = self.get_dictionary(source_dictionary_name)
        except KeyError as error:
            raise ValueError(
                f"Source dictionary not found: '{source_dictionary_name}'"
            ) from error

        for table_variable_name in data_path_parts[1:]:
            try:
                table_variable = dictionary.get_variable(table_variable_name)
            except KeyError as error:
                raise ValueError(
                    f"Table variable '{table_variable_name}' in data path not found"
                ) from error

            if table_variable.type not in ["Table", "Entity"]:
                raise ValueError(
                    f"Table variable  '{table_variable_name}' "
                    f"in data path is of type '{table_variable.type}'"
                )

            try:
                dictionary = self.get_dictionary(table_variable.object_type)
            except KeyError as error:
                raise ValueError(
                    f"Table variable '{table_variable_name}' in data path "
                    f"points to unknown dictionary '{table_variable.object_type}'"
                ) from error
        return dictionary

    def _get_dictionary_at_data_path(self, data_path):
        # Obtain the parts of the data path
        data_path_parts = data_path.lstrip("/").split("/")

        # Attempt to get the first dictionary from the data path:
        # - either it is found as such,
        # - or it is a Table or Entity variable whose table needs to be looked-up
        first_table_variable_name = data_path_parts[0]
        try:
            dictionary = self.get_dictionary(first_table_variable_name)
        except KeyError as error:
            for a_dictionary in self.dictionaries:
                try:
                    table_variable = a_dictionary.get_variable(
                        first_table_variable_name
                    )
                    if table_variable.type not in ["Table", "Entity"]:
                        raise ValueError from error
                    dictionary = self.get_dictionary(table_variable.object_type)
                    break
                except (KeyError, ValueError):
                    continue
            else:
                raise ValueError(
                    f"Dictionary not found in data path: '{data_path}'"
                ) from error

        for table_variable_name in data_path_parts[1:]:
            try:
                table_variable = dictionary.get_variable(table_variable_name)
            except KeyError as error:
                raise ValueError(
                    f"Table variable '{table_variable_name}' in data path not found"
                ) from error

            if table_variable.type not in ["Table", "Entity"]:
                raise ValueError(
                    f"Table variable  '{table_variable_name}' "
                    f"in data path is of type '{table_variable.type}'"
                )

            try:
                dictionary = self.get_dictionary(table_variable.object_type)
            except KeyError as error:
                raise ValueError(
                    f"Table variable '{table_variable_name}' in data path "
                    f"points to unknown dictionary '{table_variable.object_type}'"
                ) from error
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
    label : str
        Dictionary label/comment.
    root : bool
        True if the dictionary is the root of an dictionary hierarchy.
    key : list of str
        Names of the key variables.
    meta_data : `MetaData`
        MetaData object of the dictionary.
    variables : list of `Variable`
        The dictionary variables.
    variable_blocks : list of `VariableBlock`
        The dictionary variable blocks.
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
                variable_block_copy.rule = variable.variable_block.rule
                variable_block_copy.meta_data = variable_block.meta_data.copy()

                for variable in variable_block.variables:
                    variable_block_copy.add_variable(variable.copy())

                dictionary_copy.add_variable_block(variable_block_copy)
                i += len(variable_block.variables)
        return dictionary_copy

    def get_value(self, key):
        """Returns the metadata value associated to the specified key

        Raises
        ------
        `KeyError`
            If the key is not found
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
            The specified variable.

        Raises
        ------
        `KeyError`
            If no variable with the specified name exists.
        """
        return self._variables_by_name[variable_name]

    def get_variable_block(self, variable_block_name):
        """Returns the specified variable block

        Parameters
        ----------
        variable_block_name : str
            A name of a variable block.

        Returns
        -------
        `VariableBlock`
            The specified variable.

        Raises
        ------
        `KeyError`
            If no variable block with the specified name exists.

        """
        return self._variable_blocks_by_name[variable_block_name]

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
    label : str
        Variable label/comment.
    used : bool
        True if the variable is used.
    type : str
        Variable type.
    object_type : str
        Type complement for the ``Table`` and ``Entity`` types.
    structure_type : str
        Type complement for the ``Structure`` type. Set to "" for other types.
    rule : str
        Derivation rule. Set to "" if there is no rule associated to this variable.
    meta_data : `MetaData`
        Variable metadata.
    variable_block : `VariableBlock`
        Block to which the variable belongs. Not set if the variable does not belong to
        a block.
    """

    def __init__(self, json_data=None):
        """See class docstring"""
        # Check the type of json_data
        if json_data is not None and not isinstance(json_data, dict):
            raise TypeError(type_error_message("json_data", json_data, dict))

        # Main attributes
        self.name = ""
        self.label = ""
        self.used = True
        self.type = ""

        # Type complement attributes
        self.object_type = ""
        self.structure_type = ""

        # Derivation rule
        self.rule = ""

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
        self.used = json_data.get("used", True)
        self.type = json_data.get("type")

        # Initialize complement of the type
        if self.type in ("Entity", "Table"):
            self.object_type = json_data.get("objectType")
        elif self.type == "Structure":
            self.structure_type = json_data.get("structureType")

        # Initialize derivation rule
        self.rule = json_data.get("rule", "")

        # Initialize metadata
        json_meta_data = json_data.get("metaData")
        if json_meta_data is not None:
            self.meta_data = MetaData(json_meta_data)

    def __repr__(self):
        """Returns a human readable string representation"""
        return f"Variable ({self.name})"

    def __str__(self):
        stream = io.BytesIO()
        writer = KhiopsOutputWriter(stream)
        self.write(writer)
        return str(stream.getvalue(), encoding="utf8", errors="replace")

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
        variable.used = self.used
        variable.type = self.type
        variable.object_type = self.object_type
        variable.structure_type = self.structure_type
        variable.rule = self.rule
        variable.meta_data = self.meta_data.copy()
        return variable

    def get_value(self, key):
        """Returns the metadata value associated to the specified key

        Raises
        ------
        `KeyError`
            If no metadata has this key.
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

        return self.rule and self.rule[0] == "["

    def full_type(self):
        """Returns the variable's full type

        Returns
        -------
        str
            The full type is the variable type plus its complement if the type is not
            basic.
        """
        full_type = self.type
        if self.type in ("Entity", "Table"):
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

        # Write label/commentary if available
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
    name :
        Block name.
    label :
        Block label/commentary.
    rule :
        Block derivation rule.
    meta_data :
        Metadata object of the block.
    variables :
        List of the Variable objects of the block.
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

        # Initialize derivation rule
        self.rule = json_data.get("rule", "")

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

        Raises
        ------
        `KeyError`
            If ``key`` is not found
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
        # Write variables
        writer.writeln("\t{")
        for variable in self.variables:
            variable.write(writer)
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

        # Write label/commentary if available
        if self.label:
            writer.write("// ")
            writer.write(self.label)
        writer.writeln("")


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
            The value at the specified key

        Raises
        ------
        `TypeError`
            If ``key`` is not str.
        `KeyError`
            If ``key`` is not found.
        """
        # Check the argument types
        if not is_string_like(key):
            raise TypeError(type_error_message("key", key, "string-like"))

        # Linear search for the key
        for i, stored_key in enumerate(self.keys):
            if stored_key == key:
                return self.values[i]
        raise KeyError(key)

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
