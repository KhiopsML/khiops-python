######################################################################################
# Copyright (c) 2018 - 2022 Orange - All Rights Reserved                             #
# * This software is the confidential and proprietary information of Orange.         #
# * You shall not disclose such Restricted Information and shall use it only in      #
#   accordance with the terms of the license agreement you entered into with Orange  #
#   named the "Khiops - Python Library Evaluation License".                          #
# * Unauthorized copying of this file, via any medium is strictly prohibited.        #
# * See the "LICENSE.md" file for more details.                                      #
######################################################################################
"""Classes for handling diverse data tables"""
import csv
import io
import warnings
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

import pykhiops.core as pk
import pykhiops.core.filesystems as fs
from pykhiops.core.common import (
    deprecation_message,
    is_dict_like,
    is_list_like,
    type_error_message,
)

# Disable PEP8 variable names because of scikit-learn X,y conventions
# To capture invalid-names other than X,y run:
#   pylint --disable=all --enable=invalid-names tables.py
# pylint: disable=invalid-name


def get_khiops_type(numpy_type):
    """Translates a numpy type to a Khiops dictionary type

    Parameters
    ----------
    numpy_type : `numpy.dtype`:
        Numpy type of the column

    Returns
    -------
    str
        Khiops type name. Either "Categorical", "Numerical" or "Timestamp"
    """
    lower_numpy_type = str(numpy_type).lower()

    # timedelta64 and datetime64 types
    if "time" in lower_numpy_type:
        return "Timestamp"
    # float<x>, int<x>, uint<x> types
    elif "int" in lower_numpy_type or "float" in lower_numpy_type:
        return "Numerical"
    # bool_ and object, character, bytes_, str_, void, record and other types
    else:
        return "Categorical"


def read_internal_data_table(file_path_or_stream):
    """Reads into a DataFrame a data table file with the internal format settings

    The table is read with the following settings:

    - Use tab as separator
    - Read the column names from the first line
    - Use '"' as quote character
    - Use `csv.QUOTE_MINIMAL`
    - double quoting enabled (quotes within quotes can be escaped with '""')
    - UTF-8 encoding

    Parameters
    ----------
    file_path_or_stream : str or file object
        The path of the internal data table file to be read or a readable file
        object.

    Returns
    -------
    `pandas.DataFrame`
        The dataframe representation.
    """
    return pd.read_csv(
        file_path_or_stream,
        sep="\t",
        header=0,
        quotechar='"',
        quoting=csv.QUOTE_MINIMAL,
        doublequote=True,
        encoding="utf-8",
    )


def write_internal_data_table(dataframe, file_path_or_stream):
    """Writes a DataFrame to data table file with the internal format settings

    The table is written with the following settings:

    - Use tab as separator
    - Write the column names on the first line
    - Use '"' as quote character
    - Use `csv.QUOTE_MINIMAL`
    - double quoting enabled (quotes within quotes can be escaped with '""')
    - UTF-8 encoding
    - The index is not written

    Parameters
    ----------
    dataframe : `pandas.DataFrame`
        The dataframe to write.
    file_path_or_stream : str or file object
        The path of the internal data table file to be written or a writable file
        object.
    """
    dataframe.to_csv(
        file_path_or_stream,
        sep="\t",
        header=True,
        quotechar='"',
        quoting=csv.QUOTE_MINIMAL,
        doublequote=True,
        encoding="utf-8",
        index=False,
    )


class Dataset:
    """A representation of a dataset

    Parameters
    ----------
    X : `pandas.DataFrame` or dict (**Deprecated types**: tuple and list)
        Either:
          - A single dataframe
          - A ``dict`` dataset specification
    y : `pandas.Series` or str, optional
        The target column.
    categorical_target : bool, default True
        ``True`` if the vector ``y`` should be considered as a categorical variable. If
        ``False`` it is considered as numeric. Ignored if ``y`` is ``None``.
    key : str
        The name of the key column.
        **Deprecated:** Will be removed in pyKhiops 11.
    """

    def __init__(self, X, y=None, categorical_target=True, key=None):
        # Initialize members
        self.main_table = None
        self.secondary_tables = None
        self.relations = None
        self.sep = None
        self.header = None

        # Initialization from different types of input "X"
        if isinstance(X, pd.DataFrame):
            self._init_tables_from_dataframe(
                X, y, categorical_target=categorical_target
            )
        elif isinstance(X, tuple):
            warnings.warn(
                deprecation_message(
                    "Tuple dataset input",
                    "11.0.0",
                    replacement="dict dataset input",
                    quote=False,
                ),
                stacklevel=3,
            )
            self._init_tables_from_tuple(X, y, categorical_target=categorical_target)
        elif is_list_like(X):
            warnings.warn(
                deprecation_message(
                    "List dataset input",
                    "11.0.0",
                    replacement="dict dataset input",
                    quote=False,
                ),
                stacklevel=3,
            )
            self._init_tables_from_sequence(X, y, key=key)
        elif is_dict_like(X):
            self._init_tables_from_mapping(X, y)
        else:
            raise TypeError(
                type_error_message("X", X, pd.DataFrame, tuple, Sequence, Mapping)
            )

        assert self.main_table is not None, "'main_table' is 'None' after init"
        assert isinstance(
            self.secondary_tables, list
        ), "'secondary_tables' is not a list after init"
        assert not self.is_multitable() or len(
            self.secondary_tables
        ), "'secondary_tables' is empty in a multi-table dataset"

    def _init_tables_from_dataframe(self, X, y=None, categorical_target=True):
        """Initializes the dataset from a 'X' of type pandas.DataFrame"""
        assert isinstance(X, pd.DataFrame), "'X' must be a pandas.DataFrame"

        if y is not None and not isinstance(y, pd.Series):
            raise TypeError(
                type_error_message("y", y, pd.Series)
                + " (must be coherent with X of type pandas.DataFrame)"
            )

        self.main_table = PandasTable(
            "main_table", X, target_column=y, categorical_target=categorical_target
        )
        self.secondary_tables = []

    def _init_tables_from_tuple(self, X, y=None, categorical_target=True):
        """Initializes the spec from a 'X' of type tuple"""
        assert isinstance(X, tuple), "'X' must be a tuple"

        # Check the input tuple
        self._check_input_tuple(X, y)

        # Obtain path and separator
        path, sep = X

        # Initialization
        self.main_table = FileTable(
            "main_table",
            categorical_target=categorical_target,
            target_column_id=y,
            path=path,
            sep=sep,
        )
        self.secondary_tables = []

    def _check_input_tuple(self, X, y=None):
        if len(X) != 2:
            raise ValueError(f"'X' tuple input must have length 2 not {len(X)}")
        if not isinstance(X[0], str):
            raise TypeError(type_error_message("X[0]", X[0], str))
        if not isinstance(X[1], str):
            raise TypeError(type_error_message("X[1]", X[1], str))
        if y is not None and not isinstance(y, str):
            raise TypeError(type_error_message("y", y, str))

    def _init_tables_from_sequence(self, X, y=None, categorical_target=True, key=None):
        """Initializes the spec from a list-like 'X'"""
        assert is_list_like(X), "'X' must be a list-like"

        # Check the input sequence
        self._check_input_sequence(X, y, key=key)

        # Initialize the tables
        if isinstance(X[0], pd.DataFrame):
            self.main_table = PandasTable(
                "main_table",
                X[0],
                target_column=y,
                categorical_target=categorical_target,
                key=key,
            )
            self.secondary_tables = []
            for index, dataframe in enumerate(X[1:], start=1):
                self.secondary_tables.append(
                    PandasTable(f"secondary_table_{index:02d}", dataframe, key=key)
                )
        else:
            self.main_table = FileTable(
                "main_table",
                X[0],
                target_column_id=y,
                categorical_target=categorical_target,
                key=key,
            )
            self.secondary_tables = []
            for index, table_path in enumerate(X[1:], start=1):
                self.secondary_tables.append(
                    FileTable(f"secondary_table_{index:02d}", table_path, key=key)
                )
        # Create a list of relations
        main_table_name = self.main_table.name
        self.relations = [
            (main_table_name, table.name) for table in self.secondary_tables
        ]

    def _check_input_sequence(self, X, y=None, key=None):
        # Check the first table
        if len(X) == 0:
            raise ValueError("'X' must be a non-empty sequence")
        if not isinstance(X[0], (str, pd.DataFrame)):
            raise TypeError(type_error_message("X[0]", X[0], str, pd.DataFrame))

        # Check that the secondary table types are coherent with that of the first
        main_table_type = type(X[0])
        for i, secondary_X in enumerate(X[1:], start=1):
            if not isinstance(secondary_X, main_table_type):
                raise TypeError(
                    type_error_message(f"X[{i}]", X[i], main_table_type)
                    + " as the first table in X"
                )

        # Check the type of y
        if y is not None:
            if isinstance(X[0], str) and not isinstance(y, str):
                raise TypeError(
                    type_error_message("y", y, str)
                    + " (must be coherent with X's type(s))"
                )
            elif isinstance(X[0], pd.DataFrame) and not isinstance(y, pd.Series):
                raise TypeError(
                    type_error_message("y", y, pd.Series)
                    + " (must be coherent with X's type(s))"
                )

        # Check the type of key
        if not is_list_like(key) and not isinstance(key, str):
            raise TypeError(type_error_message("key", key, "list-like", str))
        if is_list_like(key):
            for column_index, column_name in enumerate(key):
                if not isinstance(column_name, str):
                    raise TypeError(
                        type_error_message(
                            f"key[{column_index}]", key[column_index], str
                        )
                    )

    def _init_tables_from_mapping(self, X, y=None, categorical_target=True):
        """Initializes the table spec from a dict-like 'X'"""
        assert is_dict_like(X), "'X' must be dict-like"

        # Check the input mapping
        self._check_input_mapping(X, y)

        # Initialize tables
        if len(X["tables"]) == 1:
            main_table_name = list(X["tables"])[0]
            main_table_source, main_table_key = list(X["tables"].values())[0]
            if isinstance(main_table_key, str):
                main_table_key = [main_table_key]
        else:
            main_table_name = X["main_table"]
            main_table_source, main_table_key = X["tables"][main_table_name]

        # Case of paths
        if isinstance(main_table_source, str):
            warnings.warn(
                deprecation_message(
                    "File-path dataset input",
                    "11.0.0",
                    "dataframe-based dataset or pykhiops.core API",
                    quote=False,
                ),
                stacklevel=4,
            )
            if "format" in X:
                self.sep, self.header = X["format"]
            else:
                self.sep = "\t"
                self.header = True
            self.main_table = FileTable(
                main_table_name,
                main_table_source,
                target_column_id=y,
                categorical_target=categorical_target,
                key=main_table_key,
                sep=self.sep,
                header=self.header,
            )
            self.secondary_tables = []
            for table_name, (table_source, table_key) in X["tables"].items():
                if isinstance(table_key, str):
                    table_key = [table_key]
                if table_name != main_table_name:
                    self.secondary_tables.append(
                        FileTable(
                            table_name,
                            table_source,
                            key=table_key,
                            sep=self.sep,
                            header=self.header,
                        )
                    )
        # Case of dataframes
        else:
            self.main_table = PandasTable(
                main_table_name,
                main_table_source,
                key=main_table_key,
                target_column=y,
                categorical_target=categorical_target,
            )
            self.secondary_tables = []
            for table_name, (table_source, table_key) in X["tables"].items():
                if table_name != main_table_name:
                    self.secondary_tables.append(
                        PandasTable(table_name, table_source, key=table_key)
                    )

        if self.secondary_tables:
            if "relations" not in X:
                main_table_name = self.main_table.name
                self.relations = [
                    (main_table_name, table.name) for table in self.secondary_tables
                ]
            else:
                self.relations = X["relations"]

    def _check_cycle_exists(self, relations, main_table_name):
        """Check existence of a cycle into 'relations'"""
        tables_to_visit = [main_table_name]
        tables_visited = set()
        while tables_to_visit:
            current_table = tables_to_visit.pop(0)
            tables_visited.add(current_table)
            for relation in relations:
                parent_table, child_table = relation
                if parent_table == current_table:
                    tables_to_visit.append(child_table)
                    if tables_visited.intersection(tables_to_visit):
                        raise ValueError(
                            f"Relations at X['relations'] contain a cycle which"
                            f" includes the relation '{relation}'"
                        )

    def _check_relation_keys(self, X, left_table_name, right_table_name):
        """Check coherence of keys"""
        _, left_table_key = X["tables"][left_table_name]
        _, right_table_key = X["tables"][right_table_name]
        table_key_error = False
        if isinstance(left_table_key, str) and isinstance(right_table_key, str):
            table_key_error = right_table_key != left_table_key
        elif isinstance(left_table_key, str) and is_list_like(right_table_key):
            table_key_error = left_table_key not in right_table_key
        elif is_list_like(left_table_key) and is_list_like(right_table_key):
            table_key_error = not set(left_table_key).issubset(set(right_table_key))
        elif is_list_like(left_table_key) and isinstance(right_table_key, str):
            table_key_error = True

        if table_key_error:
            if isinstance(right_table_key, str):
                right_table_key_msg = f"[{right_table_key}]"
            else:
                right_table_key_msg = f"[{', '.join(right_table_key)}]"
            if isinstance(left_table_key, str):
                left_table_key_msg = f"[{left_table_key}]"
            else:
                left_table_key_msg = f"[{', '.join(left_table_key)}]"
            raise ValueError(
                f"key for table '{right_table_name}' "
                f"{right_table_key_msg} is incompatible with "
                f"that of table "
                f"'{left_table_name}' {left_table_key_msg}"
            )

    def _check_relations(self, X, relations):
        """Check relations"""
        main_table_name = X["main_table"]
        if not is_list_like(relations):
            raise TypeError(
                type_error_message(
                    "Relations at X['tables']['relations']", relations, "list-like"
                )
            )
        else:
            for relation in relations:
                if not isinstance(relation, tuple):
                    raise TypeError(type_error_message("Relation", relation, tuple))
                if len(relation) != 2:
                    raise ValueError(
                        f"A relation must be of size 2 not {len(relation)}"
                    )
                for table in relation:
                    if not isinstance(table, str):
                        raise TypeError(
                            type_error_message("Table of a relation", table, str)
                        )
                parent_table, child_table = relation
                if parent_table == child_table:
                    raise ValueError(
                        f"Tables in relation '{relation}' are the same. "
                        f"They must be different."
                    )
                if relations.count(relation) != 1:
                    raise ValueError(
                        f"Relation '{relation}' appears "
                        f"'{relations.count(relation)}' times. "
                        f"Each relation must be unique."
                    )
                if not parent_table in X["tables"].keys():
                    raise ValueError(
                        f"X['tables'] does not contain a table named '{parent_table}'. "
                        f"All tables in X['relations'] must be declared in X['tables']"
                    )
                if not child_table in X["tables"].keys():
                    raise ValueError(
                        f"X['tables'] does not contain a table named '{child_table}'. "
                        f"All tables in X['relations'] must be declared in X['tables']."
                    )
                self._check_relation_keys(X, parent_table, child_table)
                self._check_cycle_exists(relations, main_table_name)

    def _check_input_mapping(self, X, y=None):
        # Check the "tables" field (basic)
        if "tables" not in X:
            raise ValueError("Mandatory key 'tables' missing from dict 'X'")
        if not is_dict_like(X["tables"]):
            raise TypeError(type_error_message('X["tables"]', X["tables"], Mapping))
        if len(X["tables"]) == 0:
            raise ValueError("X['tables'] cannot be empty")

        # Check coherence of each table specification
        for table_name, table_input in X["tables"].items():
            if not isinstance(table_input, tuple):
                raise TypeError(
                    type_error_message(
                        f"Table input at X['tables']['{table_name}']",
                        table_input,
                        tuple,
                    )
                )
            if len(table_input) != 2:
                raise ValueError(
                    f"Table input tuple at X['tables']['{table_name}'] "
                    f"must have size 2 not {len(table_input)}"
                )
            table_source, table_key = table_input
            if not isinstance(table_source, (pd.DataFrame, str)):
                raise TypeError(
                    type_error_message(
                        f"Table source at X['tables']['{table_name}']",
                        table_source,
                        pd.DataFrame,
                        str,
                    )
                )
            if (
                table_key is not None
                and not is_list_like(table_key)
                and not isinstance(table_key, str)
            ):
                raise TypeError(
                    type_error_message(
                        f"Table key at X['tables']['{table_name}']",
                        table_key,
                        str,
                        Sequence,
                    )
                )

            if table_key is not None:
                for column_name in table_key:
                    if not isinstance(column_name, str):
                        raise TypeError(
                            type_error_message(
                                "Column name of table key "
                                f"at X['tables']['{table_name}']",
                                column_name,
                                str,
                            )
                        )

        # Multi-table specific table checks
        if len(X["tables"]) > 1:
            # Check the "main_table" field
            if "main_table" not in X:
                raise ValueError(
                    "'main_table' must be specified for multi-table datasets"
                )
            if not isinstance(X["main_table"], str):
                raise TypeError(
                    type_error_message("X['main_table']", X["main_table"], str)
                )
            if X["main_table"] not in X["tables"]:
                raise ValueError(
                    f"X['main_table'] ({X['main_table']}) "
                    f"must be present in X['tables']"
                )
            main_table_source, main_table_key = X["tables"][X["main_table"]]
            if main_table_key is None:
                raise ValueError("key of the root table is 'None'")
            if len(main_table_key) == 0:
                raise ValueError(
                    "key of the root table must be non-empty for multi-table datasets"
                )

            # Check that there are no repeated table names
            table_name_counts = Counter(X["tables"].keys())
            for table_name, count in table_name_counts.items():
                if count > 1:
                    raise ValueError(f"There are {count} tables named '{table_name}'")

            # Check that all secondary tables have non-None keys
            for table_name, (_, table_key) in X["tables"].items():
                if table_name != X["main_table"] and table_key is None:
                    raise ValueError(
                        f"key of the secondary table '{table_name}' is 'None':"
                        " table keys must be specified in multitable datasets"
                    )

            if "relations" not in X:
                # the schema is by default 'star'
                # create a list of relations [(main_table, secondary_table), ...]
                main_table_name = X["main_table"]
                table_names = X["tables"].keys()
                relations = [
                    (main_table_name, name)
                    for name in table_names
                    if name != main_table_name
                ]
                # check the created 'relations'
                self._check_relations(X, relations)
            else:
                # check the 'relations' field
                # the schema could be 'star' or 'snowflake'
                relations = X["relations"]
                self._check_relations(X, relations)

        # Check the 'format' field
        if "format" in X:
            if not isinstance(X["format"], tuple):
                raise TypeError(type_error_message("X['format']", X["format"], tuple))
            if not isinstance(X["format"][0], str):
                raise TypeError(
                    type_error_message("X['format'] first element", X["format"][0], str)
                )
            if len(X["format"][0]) != 1:
                raise ValueError(
                    'X["format"] first element must be a single character. '
                    f'Value: {X["format"][0]}'
                )
            if not isinstance(X["format"][1], bool):
                raise TypeError(
                    type_error_message(
                        "X['format'] first element", X["format"][0], bool
                    )
                )

        # Check the target coherence with X's tables
        if y is not None:
            if len(X["tables"]) == 1:
                main_table_source, _ = list(X["tables"].values())[0]
            else:
                main_table_source, _ = X["tables"][X["main_table"]]
            if isinstance(main_table_source, pd.DataFrame) and not isinstance(
                y, pd.Series
            ):
                raise TypeError(
                    type_error_message("y", y, pd.Series)
                    + " (X's tables are of type pandas.DataFrame)"
                )
            if isinstance(main_table_source, str) and not isinstance(y, str):
                raise TypeError(
                    type_error_message("y", y, str)
                    + " (X's tables are of type str [file paths])"
                )

    def is_dataframe_based(self):
        """Tests whether the dataset is constituted of pandas.DataFrame tables

        Returns
        -------
        bool
            `True` if the dataset is constituted of pandas.DataFrame tables.
        """
        return isinstance(self.main_table, PandasTable)

    def is_multitable(self):
        """Tests whether the dataset is a multi-table one

        Returns
        -------
        bool
            ``True`` if the dataset is multi-table.
        """
        return self.secondary_tables is not None and len(self.secondary_tables) > 0

    def copy(self):
        """Creates a copy of the dataset

        Referenced dataframes in tables are copied as references
        """
        dataset_spec = {}
        dataset_spec["main_table"] = self.main_table.name
        dataset_spec["tables"] = {}
        if self.is_dataframe_based():
            dataset_spec["tables"][self.main_table.name] = (
                self.main_table.dataframe,
                self.main_table.key,
            )
            for table in self.secondary_tables:
                dataset_spec["tables"][table.name] = (table.dataframe, table.key)
        else:
            dataset_spec["tables"][self.main_table.name] = (
                self.main_table.path,
                self.main_table.key,
            )
            for table in self.secondary_tables:
                dataset_spec["tables"][table.name] = (table.path, table.key)
            dataset_spec["format"] = (self.sep, self.header)
        return Dataset(dataset_spec)

    def create_khiops_dictionary_domain(self):
        """Creates a Khiops dictionary domain representing this dataset

        Returns
        -------
        `.DictionaryDomain`
            The dictionary domain object representing this dataset
        """
        assert self.main_table is not None, "'main_table' must be initialized"

        # Create root dictionary and add it to the domain
        dictionary_domain = pk.DictionaryDomain()
        root_dictionary = self.main_table.create_khiops_dictionary()
        dictionary_domain.add_dictionary(root_dictionary)

        # Create the dictionaries for each secondary table and the table variables in
        # root dictionary that point to each secondary table
        # This is performed using a breadth-first-search over the graph of relations
        # Note: In general 'name' and 'object_type' fields of Variable can be different
        if self.secondary_tables:
            root_dictionary.root = True
            table_names = [table.name for table in self.secondary_tables]
            tables_to_visit = [self.main_table.name]
            for current_table in tables_to_visit:
                for relation in self.relations:
                    parent_table, child_table = relation
                    if parent_table == current_table:
                        tables_to_visit.append(child_table)
                        parent_table_name = parent_table
                        index_table = table_names.index(child_table)
                        table = self.secondary_tables[index_table]
                        parent_table_dictionary = dictionary_domain.get_dictionary(
                            parent_table_name
                        )
                        dictionary = table.create_khiops_dictionary()
                        dictionary_domain.add_dictionary(dictionary)
                        table_variable = pk.Variable()
                        table_variable.type = "Table"
                        table_variable.name = table.name
                        table_variable.object_type = table.name
                        parent_table_dictionary.add_variable(table_variable)

        return dictionary_domain

    def create_table_files_for_khiops(self, directory_res, sort=True):
        """Prepares the tables of the dataset to be used by Khiops

        If this is a multi-table dataset it will create sorted copies the tables.

        Parameters
        ----------
        directory_res : `pykhiops.core.filesystems.FilesystemResource`
            A working directory for the sorted tables to be created

        Returns
        -------
        tuple
            A tuple containing:

            - The path of the main table
            - A dictionary containing the relation [table-name -> file-path] for the
              secondary tables. The dictionary is empty for monotable datasets.
        """
        # Sort the main table unless:
        # - The caller specifies not to do it (sort = False)
        # - The dataset is mono-table and the main table has no key
        sort_main_table = sort and (
            self.is_multitable() or self.main_table.key is not None
        )
        main_table_path = self.main_table.create_table_file_for_khiops(
            directory_res.uri, sort=sort_main_table
        )

        # Create a copy of each secondary table
        secondary_table_paths = {}
        for table in self.secondary_tables:
            secondary_table_paths[table.name] = table.create_table_file_for_khiops(
                directory_res.uri, sort=sort
            )
        return main_table_path, secondary_table_paths

    @property
    def target_column_type(self):
        """The target column's type"""
        if self.main_table.target_column_id is None:
            raise ValueError("Target column is not set")
        if self.is_dataframe_based():
            return self.main_table.target_column.dtype
        else:
            return self.main_table.dtypes[self.main_table.target_column_id]

    def __repr__(self):
        return str(self.create_khiops_dictionary_domain())


class DatasetTable(ABC):
    """A generic dataset table"""

    @abstractmethod
    def __init__(self, name, categorical_target=True, key=None):
        # Check input
        if not isinstance(name, str):
            raise TypeError(type_error_message("name", name, str))
        if not name:
            raise ValueError("'name' is the empty string")
        if key is not None:
            if not is_list_like(key) and not isinstance(key, str):
                raise TypeError(type_error_message("key", key, str, "list-like"))
            if is_list_like(key):
                for column_index, column_name in enumerate(key):
                    if not isinstance(column_name, str):
                        raise TypeError(
                            type_error_message(f"key[{column_index}]", column_name, str)
                            + f" at table '{name}'"
                        )

        # Initialization (must be completed by concrete sub-classes)
        self.name = name
        self.categorical_target = categorical_target
        if is_list_like(key) or key is None:
            self.key = key
        else:
            self.key = [key]
        self.target_column_id = None
        self.column_ids = None
        self.dtypes = None

    def check_key(self):
        """Checks that the key columns exist"""
        if self.key is not None:
            if not is_list_like(self.key):
                raise ValueError("'key' must be list-like")
            for column_name in self.key:
                if column_name not in self.column_ids:
                    raise ValueError(
                        f"Column '{column_name}' not present in table '{self.name}'"
                    )

    @abstractmethod
    def create_table_file_for_khiops(self, output_dir, sort=True):
        """Creates a copy of the table at the specified directory"""

    def create_khiops_dictionary(self):
        """Creates a Khiops dictionary representing this table

        Returns
        -------
        `.Dictionary`:
            The Khiops Dictionary object describing this table's schema

        Raises
        ------
        `ValueError`
            If ``key`` is specified but there is no column with that name
        """
        assert self.column_ids is not None, "Dataset column list is None"
        assert self.key is None or is_list_like(self.key), "'key' is not list-like"

        # Create dictionary object
        dictionary = pk.Dictionary()
        dictionary.name = self.name
        if self.key is not None:
            dictionary.key = list(self.key)

        # For each column add a Khiops variable to the dictionary
        for column_id in self._get_all_column_ids():
            variable = pk.Variable()

            # Set the variable name for string and integer column indexes
            if isinstance(column_id, str):
                variable.name = str(column_id)
            else:
                assert isinstance(column_id, np.int64)
                variable.name = f"Var{column_id}"

            # Set the type of the column/variable
            # Case of a column in the key : Set to categorical
            if self.key is not None and column_id in self.key:
                variable.type = "Categorical"
            # Case of the target column: Set to specified type
            elif column_id == self.target_column_id:
                assert self.target_column_id is not None
                if self.categorical_target:
                    variable.type = "Categorical"
                else:
                    variable.type = "Numerical"
            # The rest of columns: Obtain the type from dtypes
            else:
                variable.type = get_khiops_type(self.dtypes[column_id])
            dictionary.add_variable(variable)
        return dictionary

    @abstractmethod
    def _get_all_column_ids(self):
        """Returns the column ids including the target"""

    def __repr__(self):
        dtypes_str = str(self.dtypes).replace("\n", ", ")[:-16].replace("    ", ":")
        return (
            f"<{self.__class__.__name__}; cols={list(self.column_ids)}; "
            f"dtypes={dtypes_str}; target={self.target_column_id}>"
        )


class PandasTable(DatasetTable):
    """Table encapsulating (X,y) pair with types (pandas.DataFrame, pandas.Series)

    Parameters
    ----------
    name : str
        Name for the table.
    dataframe : `pandas.DataFrame`
        The data frame to be encapsulated.
    key : list-like of str, optional
        The names of the columns composing the key
    target_column : `pandas.Series`, optional
        The series representing the target column.
    categorical_target : bool, default ``True``.
        ``True`` if the target column is categorical.
    """

    def __init__(
        self, name, dataframe, key=None, target_column=None, categorical_target=True
    ):
        # Call the parent method
        super().__init__(name, categorical_target=categorical_target, key=key)

        # Check inputs specific to this sub-class
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(type_error_message("dataframe", dataframe, pd.DataFrame))
        if dataframe.shape[0] == 0:
            raise ValueError("'dataframe' is empty")
        if target_column is not None:
            if not isinstance(target_column, pd.Series):
                raise TypeError(
                    type_error_message("target_column", target_column, pd.Series)
                )
            if (
                target_column.name is not None
                and target_column.name in dataframe.columns
            ):
                raise ValueError(
                    f"Target series name '{target_column.name}'"
                    f"is already present in dataframe : {list(dataframe.columns)}"
                )

        # Initialize the attributes
        self.dataframe = dataframe
        self.dtypes = self.dataframe.dtypes

        # Initialize feature columns and verify their types
        self.column_ids = self.dataframe.columns.values
        if not np.issubdtype(self.column_ids.dtype, np.integer):
            if np.issubdtype(self.column_ids.dtype, object):
                for i, column_id in enumerate(self.column_ids):
                    if not isinstance(column_id, str):
                        raise TypeError(
                            f"Dataframe column ids must be integer or string. "
                            f"Column id at index {i} ('{column_id}') is of type "
                            f"'{type(column_id).__name__}'"
                        )
            else:
                raise TypeError(
                    f"Dataframe column ids must be integer or string. "
                    f"The column has dtype '{self.column_ids.dtype}'"
                )

        # Initialize target column (if any)
        self.target_column = target_column
        if self.target_column is not None:
            if self.target_column.name is None:
                if pd.api.types.is_integer_dtype(self.column_ids):
                    self.target_column_id = self.column_ids[-1] + 1
                else:
                    assert pd.api.types.is_string_dtype(self.column_ids)
                    self.target_column_id = "UnknownTargetColumn"
            else:
                self.target_column_id = target_column.name

        # Check key integrity
        self.check_key()

    def _get_all_column_ids(self):
        if self.target_column is not None:
            all_column_ids = list(self.column_ids) + [self.target_column_id]
        else:
            all_column_ids = list(self.column_ids)
        return all_column_ids

    def get_khiops_variable_name(self, column_id):
        """Return the khiops variable name associated to a column id"""
        assert column_id == self.target_column_id or column_id in self.column_ids
        if isinstance(column_id, str):
            variable_name = column_id
        else:
            assert isinstance(column_id, np.int64)
            variable_name = f"Var{column_id}"
        return variable_name

    def _create_dataframe_copy(self):
        """Creates an in memory copy of the dataframe with the target column"""
        # Create a copy of the dataframe and add a copy of the target column (if any)
        if self.target_column is not None:
            if self.target_column.name is None:
                output_target_column = pd.Series(
                    self.target_column, name=self.target_column_id
                )
            else:
                output_target_column = self.target_column.reset_index(drop=True)
            output_dataframe = pd.concat(
                [self.dataframe.reset_index(drop=True), output_target_column],
                axis=1,
            )
        else:
            output_dataframe = self.dataframe.copy()

        # Rename the columns
        output_dataframe_column_names = {}
        for column_id in self._get_all_column_ids():
            output_dataframe_column_names[column_id] = self.get_khiops_variable_name(
                column_id
            )
        output_dataframe.rename(
            output_dataframe_column_names, axis="columns", inplace=True
        )

        return output_dataframe

    def create_table_file_for_khiops(self, output_dir, sort=True):
        assert not sort or self.key is not None, "Cannot sort table without a key"
        assert not sort or is_list_like(
            self.key
        ), "Cannot sort table with a key is that is not list-like"
        assert not sort or len(self.key) > 0, "Cannot sort table with an empty key"

        # Create the output table resource object
        output_dir_res = fs.create_resource(output_dir)
        table_res = output_dir_res.create_child(f"{self.name}.txt")

        # Write the output dataframe
        output_dataframe = self._create_dataframe_copy()

        # Sort by key if requested (as string)
        if sort:
            output_dataframe.sort_values(
                by=self.key,
                key=lambda array: array.astype("str"),
                inplace=True,
                kind="mergesort",
            )

        # Write the dataframe to an internal table file
        with io.StringIO() as output_dataframe_stream:
            write_internal_data_table(output_dataframe, output_dataframe_stream)
            table_res.write(output_dataframe_stream.getvalue().encode("utf-8"))

        return table_res.uri


class FileTable(DatasetTable):
    """A table representing a delimited text file

    Parameters
    ----------
    name : str
        Name for the table.
    path : str
        Path of the file containing the table.
    sep : str, optional
        Field separator character. If not specified it will be inferred from the file.
    header : bool, optional
        Indicates if the table
    key : list-like of str, optional
        The names of the columns composing the key
    target_column_id : str, optional
        Name of the target variable column.
    categorical_target : bool, default ``True``.
        ``True`` if the target column is categorical.
    """

    def __init__(
        self,
        name,
        path,
        target_column_id=None,
        categorical_target=True,
        key=None,
        sep="\t",
        header=True,
    ):
        # Initialize parameters
        super().__init__(name=name, categorical_target=categorical_target, key=key)

        # Check inputs specific to this sub-class
        if not isinstance(path, str):
            raise TypeError(type_error_message("path", path, str))
        table_file_res = fs.create_resource(path)
        if not table_file_res.exists():
            raise ValueError(f"Non-existent data table file: {path}")

        # Initialize members specific to this sub-class
        self.path = path
        self.sep = sep
        self.header = header
        self.target_column_id = target_column_id

        # Obtain the columns and their types from a sample of the data table
        # We build the sample by reading the first 100 rows / 4MB of the file
        table_file_head_contents = table_file_res.read(size=4096 * 1024 - 1)
        with io.BytesIO(table_file_head_contents) as table_file_head_contents_stream:
            table_sample_dataframe = pd.read_csv(
                table_file_head_contents_stream,
                nrows=100,
                sep=self.sep,
                header=0 if self.header else None,
            )

            # Raise error if there is no data in the table
            if table_sample_dataframe.shape[0] == 0:
                raise ValueError(f"Empty data table file: {path}")

            # Save the columns and their types
            self.column_ids = table_sample_dataframe.columns.values
            self.dtypes = table_sample_dataframe.dtypes

        # Check key integrity
        self.check_key()

    def _get_all_column_ids(self):
        return list(self.column_ids)

    def get_khiops_variable_name(self, column_id):
        assert column_id in self._get_all_column_ids()
        return column_id

    def create_table_file_for_khiops(self, output_dir, sort=True):
        assert not sort or self.key is not None, "key is 'None'"

        # Create the input and output file resources
        output_dir_res = fs.create_resource(output_dir)
        table_file_res = fs.create_resource(self.path)
        if sort:
            output_table_file_res = output_dir_res.create_child(
                f"sorted_{self.name}.txt"
            )
        else:
            output_table_file_res = output_dir_res.create_child(f"copy_{self.name}.txt")

        # Fail if they have the same path
        if output_table_file_res.uri_info == table_file_res.uri_info:
            raise ValueError(
                f"Cannot overwrite this table's path: {table_file_res.uri_info}"
            )

        # Create a sorted copy if requested
        if sort:
            # Create the sorting dictionary domain
            sort_dictionary_domain = pk.DictionaryDomain()
            sort_dictionary_domain.add_dictionary(self.create_khiops_dictionary())

            # Delegate the sorting and copy to pykhiops.core.sort_data_table
            # We use the same input format of the original table
            pk.sort_data_table(
                sort_dictionary_domain,
                self.name,
                self.path,
                output_table_file_res.uri,
                self.key,
                field_separator=self.sep,
                header_line=self.header,
                output_field_separator=self.sep,
                output_header_line=self.header,
            )

        # Otherwise copy the contents to the output file
        else:
            output_table_file_res.write(table_file_res.read())

        return output_table_file_res.uri
