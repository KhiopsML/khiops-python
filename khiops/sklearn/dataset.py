######################################################################################
# Copyright (c) 2024 Orange. All rights reserved.                                    #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Classes for handling diverse data tables"""
import csv
import io
import json
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.utils import check_array
from sklearn.utils.validation import column_or_1d

import khiops.core as kh
import khiops.core.internals.filesystems as fs
from khiops.core.dictionary import VariableBlock
from khiops.core.exceptions import KhiopsRuntimeError
from khiops.core.internals.common import (
    deprecation_message,
    is_dict_like,
    is_list_like,
    type_error_message,
)

# Disable PEP8 variable names because of scikit-learn X,y conventions
# To capture invalid-names other than X,y run:
#   pylint --disable=all --enable=invalid-names dataset.py
# pylint: disable=invalid-name


def check_dataset_spec(ds_spec):
    """Checks that a dataset spec is valid

    Parameters
    ----------
    ds_spec : dict
        A specification of a multi-table dataset (see :doc:`/multi_table_primer`).

    Raises
    ------
    TypeError
        If there are objects of the spec with invalid type.
    ValueError
        If there are objects of the spec with invalid values.
    """
    # Check the spec type
    if not is_dict_like(ds_spec):
        raise TypeError(type_error_message("ds_spec", ds_spec, Mapping))

    # Check the "tables" field
    if "tables" not in ds_spec:
        raise ValueError("'tables' entry missing from dataset dict spec")
    if not is_dict_like(ds_spec["tables"]):
        raise TypeError(
            type_error_message("'tables' entry", ds_spec["tables"], Mapping)
        )
    if len(ds_spec["tables"]) == 0:
        raise ValueError("'tables' dictionary cannot be empty")
    for table_name, table_entry in ds_spec["tables"].items():
        _check_table_entry(table_name, table_entry)

    # Multi-table specific table checks
    if len(ds_spec["tables"]) > 1:
        _check_multitable_spec(ds_spec)

    # Check the 'format' field
    if "format" in ds_spec:
        _check_format_entry(ds_spec["format"])


def _check_table_entry(table_name, table_spec):
    if not isinstance(table_spec, tuple):
        raise TypeError(
            type_error_message(f"'{table_name}' table entry", table_spec, tuple)
        )
    if len(table_spec) != 2:
        raise ValueError(
            f"'{table_name}' table entry must have size 2, not {len(table_spec)}"
        )
    source, key = table_spec
    if not isinstance(source, (pd.DataFrame, sp.spmatrix, str)) and not hasattr(
        source, "__array__"
    ):
        raise TypeError(
            type_error_message(
                f"'{table_name}' table's source",
                source,
                "array-like",
                "scipy.sparse.spmatrix",
                str,
            )
        )
    _check_table_key(table_name, key)


def _check_table_key(table_name, key):
    if key is not None:
        if not is_list_like(key) and not isinstance(key, str):
            raise TypeError(
                type_error_message(f"'{table_name}' table's key", key, str, Sequence)
            )
        if len(key) == 0:
            raise ValueError(f"'{table_name}' table's key is empty")
        for column_name in key:
            if not isinstance(column_name, str):
                raise TypeError(
                    type_error_message(
                        f"'{table_name}' table's key column name",
                        column_name,
                        str,
                    )
                )


def _check_multitable_spec(ds_spec):
    # Check the main table
    if "main_table" not in ds_spec:
        raise ValueError(
            "'main_table' entry must be specified for multi-table datasets"
        )
    if not isinstance(ds_spec["main_table"], str):
        raise TypeError(
            type_error_message("'main_table' entry", ds_spec["main_table"], str)
        )
    if ds_spec["main_table"] not in ds_spec["tables"]:
        raise ValueError(
            f"A table entry with the main table's name ('{ds_spec['main_table']}') "
            f"must be present in the 'tables' dictionary"
        )

    # Check that all tables have non-None keys
    for table_name, (_, table_key) in ds_spec["tables"].items():
        if table_key is None:
            table_kind = "main" if ds_spec["main_table"] == table_name else "secondary"
            raise ValueError(
                f"key of {table_kind} table '{table_name}' is 'None': "
                "table keys must be specified in multi-table datasets"
            )

    # Check that all the tables have the same type as the main
    main_table_type = type(ds_spec["tables"][ds_spec["main_table"]][0])
    for table_name, (table_source, _) in ds_spec["tables"].items():
        if table_name != ds_spec["main_table"]:
            if not isinstance(table_source, main_table_type):
                raise ValueError(
                    f"Secondary table '{table_name}' has type "
                    f"'{type(table_source).__name__}' which is different from the "
                    f"main table's type '{main_table_type.__name__}'."
                )

    # If the 'relations' entry exists check it
    if "relations" in ds_spec:
        relations_spec = ds_spec["relations"]
    # Otherwise build a star schema relations spec and check it
    else:
        relations_spec = [
            (ds_spec["main_table"], table)
            for table in ds_spec["tables"].keys()
            if table != ds_spec["main_table"]
        ]
    _check_relations_entry(ds_spec["main_table"], ds_spec["tables"], relations_spec)


def _check_relations_entry(main_table_name, tables_spec, relations_spec):
    # Check the types and size of the relation entries
    if not is_list_like(relations_spec):
        raise TypeError(
            type_error_message("'relations' entry", relations_spec, "list-like")
        )
    for i, relation in enumerate(relations_spec, 1):
        # Check that the relation is a 2 or 3 tuple
        if not isinstance(relation, tuple):
            raise TypeError(type_error_message("Relation", relation, tuple))
        if len(relation) not in (2, 3):
            raise ValueError(f"A relation must be of size 2 or 3, not {len(relation)}")

        # Check the types of the tuple contents
        parent_table, child_table = relation[:2]
        if not isinstance(parent_table, str):
            raise TypeError(
                type_error_message(f"Relation #{i}'s parent table", parent_table, str)
            )
        if not isinstance(child_table, str):
            raise TypeError(
                type_error_message(f"Relation #{i}'s child table", child_table, str)
            )
        if len(relation) == 3 and not isinstance(relation[2], bool):
            raise TypeError(
                type_error_message(
                    f"Relation #{i} ({parent_table}, {child_table}) 1-1 flag",
                    relation[2],
                    bool,
                )
            )

    # Check structure and coherence with the rest of the spec
    parents_and_children = [relation[:2] for relation in relations_spec]
    for i, relation in enumerate(relations_spec, 1):
        parent_table, child_table = relation[:2]
        if parent_table == child_table:
            raise ValueError(
                f"Relation #{i}'s tables are equal: ({parent_table}, {child_table}). "
                "They must be different."
            )
        for table in (parent_table, child_table):
            if not table in tables_spec.keys():
                raise ValueError(
                    f"Relation #{i} ({parent_table}, {child_table}) contains "
                    f"non-existent table '{table}'. All relation tables must exist "
                    "in the 'tables' entry."
                )
        if parents_and_children.count(relation[:2]) > 1:
            raise ValueError(
                f"Relation #{i} ({parent_table}, {child_table}) occurs "
                f"{parents_and_children.count(relation[:2])} times. "
                f"Each relation must be unique."
            )

        # Check hierarchical keys
        _check_hierarchical_keys(
            i,
            parent_table,
            tables_spec[parent_table][1],
            child_table,
            tables_spec[child_table][1],
        )

    # Check there are no cycles
    _check_no_cycles(relations_spec, main_table_name)


def _check_hierarchical_keys(
    relation_id, parent_table, parent_table_key, child_table, child_table_key
):
    """Check that the parent table's key is contained in the child table's key"""
    # Perform the check and save the error status
    error_found = False
    if isinstance(parent_table_key, str) and isinstance(child_table_key, str):
        error_found = child_table_key != parent_table_key
    elif isinstance(parent_table_key, str) and is_list_like(child_table_key):
        error_found = parent_table_key not in child_table_key
    elif is_list_like(parent_table_key) and is_list_like(child_table_key):
        error_found = not set(parent_table_key).issubset(child_table_key)
    elif is_list_like(parent_table_key) and isinstance(child_table_key, str):
        error_found = (
            len(parent_table_key) != 1 or child_table_key not in parent_table_key
        )

    # Report any error found
    if error_found:
        if isinstance(child_table_key, str):
            child_table_key_msg = f"[{child_table_key}]"
        else:
            child_table_key_msg = f"[{', '.join(child_table_key)}]"
        if isinstance(parent_table_key, str):
            parent_table_key_msg = f"[{parent_table_key}]"
        else:
            parent_table_key_msg = f"[{', '.join(parent_table_key)}]"
        raise ValueError(
            f"Relation #{relation_id} child table '{child_table}' "
            f"key ({child_table_key_msg}) does not contain that of parent table "
            f"'{parent_table}' ({parent_table_key_msg})."
        )


def _check_no_cycles(relations_spec, main_table_name):
    """Check that there are no cycles in the 'relations' entry"""
    tables_to_visit = [main_table_name]
    tables_visited = set()
    while tables_to_visit:
        current_table = tables_to_visit.pop(0)
        tables_visited.add(current_table)
        for relation in relations_spec:
            parent_table, child_table = relation[:2]
            if parent_table == current_table:
                tables_to_visit.append(child_table)
                if tables_visited.intersection(tables_to_visit):
                    raise ValueError(
                        "'relations' entry contains a cycle that includes "
                        f"the relation ({parent_table}, {child_table})."
                    )


def _check_format_entry(format_spec):
    if not isinstance(format_spec, tuple):
        raise TypeError(type_error_message("'format' entry", format_spec, tuple))
    if len(format_spec) != 2:
        raise ValueError(
            f"'format' entry must be a tuple of size 2, not {len(format_spec)}"
        )
    sep, header = format_spec
    if not isinstance(sep, str):
        raise TypeError(
            type_error_message("'format' tuple's 1st element (separator)", sep, str)
        )
    if not isinstance(header, bool):
        raise TypeError(
            type_error_message("'format' tuple's 2nd element (header)", header, bool)
        )
    if len(sep) != 1:
        raise ValueError(f"'format' separator must be a single char, got '{sep}'")


def get_khiops_type(numpy_type):
    """Translates a numpy dtype to a Khiops dictionary type

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
    if "datetime64" in lower_numpy_type or "timedelta64" in lower_numpy_type:
        khiops_type = "Timestamp"
    # float<x>, int<x>, uint<x> types
    elif "int" in lower_numpy_type or "float" in lower_numpy_type:
        khiops_type = "Numerical"
    # bool_ and object, character, bytes_, str_, void, record and other types
    else:
        khiops_type = "Categorical"

    return khiops_type


def get_khiops_variable_name(column_id):
    """Return the khiops variable name associated to a column id"""
    if isinstance(column_id, str):
        variable_name = column_id
    else:
        assert isinstance(column_id, np.int64)
        variable_name = f"Var{column_id}"
    return variable_name


def read_internal_data_table(file_path_or_stream, column_dtypes=None):
    """Reads into a DataFrame a data table file with the internal format settings

    The table is read with the following settings:

    - Use tab as separator
    - Read the column names from the first line
    - Use '"' as quote character
    - Use `csv.QUOTE_MINIMAL`
    - double quoting enabled (quotes within quotes can be escaped with '""')
    - UTF-8 encoding
    - User-specified dtypes (optional)

    Parameters
    ----------
    file_path_or_stream : str or file object
        The path of the internal data table file to be read or a readable file
        object.
    column_dtypes : dict, optional
        Dictionary linking column names with dtypes. See ``dtype`` parameter of the
        `pandas.read_csv` function. If not set, then the column types are detected
        automatically by pandas.

    Returns
    -------
    `pandas.DataFrame`
        The dataframe representation of the data table.
    """
    # Change the 'U' types (Unicode strings) to 'O' because pandas does not support them
    # in read_csv
    if column_dtypes is not None:
        execution_column_dtypes = {}
        for column_name, dtype in column_dtypes.items():
            if hasattr(dtype, "kind") and dtype.kind == "U":
                execution_column_dtypes[column_name] = np.dtype("O")
    else:
        execution_column_dtypes = None

    # Read and return the dataframe
    return pd.read_csv(
        file_path_or_stream,
        sep="\t",
        header=0,
        quotechar='"',
        quoting=csv.QUOTE_MINIMAL,
        doublequote=True,
        encoding="utf-8",
        dtype=execution_column_dtypes,
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
        The name of the key column for all tables.
        **Deprecated:** Will be removed in khiops-python 11.
    """

    def __init__(self, X, y=None, categorical_target=True, key=None):
        # Initialize members
        self.main_table = None
        self.secondary_tables = None
        self.relations = None
        self.categorical_target = categorical_target
        self.target_column = None
        self.target_column_id = None
        self.sep = None
        self.header = None

        # Initialization from different types of input "X"
        # A single pandas dataframe
        if isinstance(X, pd.DataFrame):
            self.main_table = PandasTable("main_table", X)
            self.secondary_tables = []
        # A single numpy array (or compatible object)
        elif hasattr(X, "__array__"):
            self.main_table = NumpyTable("main_table", X)
            self.secondary_tables = []
        # A scipy.sparse.spmatrix
        elif isinstance(X, sp.spmatrix):
            self.main_table = SparseTable("main_table", X)
            self.secondary_tables = []
        # Special rejection for scipy.sparse.sparray (to pass the sklearn tests)
        # Note: We don't use scipy.sparse.sparray because it is not implemented in scipy
        # 1.10 which is the latest supporting py3.8
        elif isinstance(
            X,
            (
                sp.bsr_array,
                sp.coo_array,
                sp.csc_array,
                sp.csr_array,
                sp.dia_array,
                sp.dok_array,
                sp.lil_array,
            ),
        ):
            check_array(X, accept_sparse=False)
        # A tuple spec
        elif isinstance(X, tuple):
            warnings.warn(
                deprecation_message(
                    "Tuple dataset input",
                    "11.0.0",
                    replacement="dict dataset spec",
                    quote=False,
                ),
                stacklevel=3,
            )
            # Check the input tuple
            self._check_input_tuple(X)

            # Obtain path and separator
            path, sep = X

            # Initialization
            self.main_table = FileTable("main_table", path=path, sep=sep)
            self.secondary_tables = []

        # A dataset sequence spec
        # We try first for compatible python arrays then the deprecated sequences spec
        elif is_list_like(X):
            # Try to transform to a numerical array with sklearn's check_array
            # On failure we try the old deprecated sequence interface
            # When the old list interface is eliminated this will considerably reduce
            # this branch's code
            try:
                X_checked = check_array(X, ensure_2d=True, force_all_finite=False)
                self.main_table = NumpyTable("main_table", X_checked)
                self.secondary_tables = []
            except ValueError:
                warnings.warn(
                    deprecation_message(
                        "List dataset input",
                        "11.0.0",
                        replacement="dict dataset spec",
                        quote=False,
                    ),
                    stacklevel=3,
                )
                self._init_tables_from_sequence(X, key=key)
        # A a dataset dict spec
        elif is_dict_like(X):
            self._init_tables_from_mapping(X)
        # Fail if X is not recognized
        else:
            raise TypeError(
                type_error_message("X", X, "array-like", tuple, Sequence, Mapping)
            )

        # Initialization of the target column if any
        if y is not None:
            self._init_target_column(y)

        # Index the tables by name
        self._tables_by_name = {
            table.name: table for table in [self.main_table] + self.secondary_tables
        }

        # Deprecation warning for file-based datasets
        if isinstance(self.main_table, FileTable):
            warnings.warn(
                deprecation_message(
                    "File-based dataset spec",
                    "11.0.0",
                    "dataframe-based dataset or khiops.core API",
                    quote=False,
                ),
            )

        # Post-conditions
        assert self.main_table is not None, "'main_table' is 'None' after init"
        assert isinstance(
            self.secondary_tables, list
        ), "'secondary_tables' is not a list after init"
        assert not self.is_multitable or len(
            self.secondary_tables
        ), "'secondary_tables' is empty in a multi-table dataset"
        assert (
            y is None or self.target_column is not None
        ), "'y' is set but 'target_column' is None"

    def _check_input_tuple(self, X):
        if len(X) != 2:
            raise ValueError(f"'X' tuple input must have length 2 not {len(X)}")
        if not isinstance(X[0], str):
            raise TypeError(type_error_message("X[0]", X[0], str))
        if not isinstance(X[1], str):
            raise TypeError(type_error_message("X[1]", X[1], str))

    def _init_tables_from_sequence(self, X, key=None):
        """Initializes the spec from a list-like 'X'"""
        assert is_list_like(X), "'X' must be a list-like"

        # Check the input sequence
        self._check_input_sequence(X, key=key)

        # Initialize the tables
        if isinstance(X[0], pd.DataFrame):
            self.main_table = PandasTable("main_table", X[0], key=key)
            self.secondary_tables = []
            for index, dataframe in enumerate(X[1:], start=1):
                self.secondary_tables.append(
                    PandasTable(f"secondary_table_{index:02d}", dataframe, key=key)
                )
        else:
            self.main_table = FileTable("main_table", X[0], key=key)
            self.secondary_tables = []
            for index, table_path in enumerate(X[1:], start=1):
                self.secondary_tables.append(
                    FileTable(f"secondary_table_{index:02d}", table_path, key=key)
                )

        # Create a list of relations
        main_table_name = self.main_table.name
        self.relations = [
            (main_table_name, table.name, False) for table in self.secondary_tables
        ]

    def _check_input_sequence(self, X, key=None):
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
                    type_error_message(f"Table at index {i}", X[i], main_table_type)
                    + " as the first table in X"
                )

        # Check the key for the main_table (it is the same for the others)
        _check_table_key("main_table", key)

    def _init_tables_from_mapping(self, X):
        """Initializes the table spec from a dict-like 'X'"""
        assert is_dict_like(X), "'X' must be dict-like"

        # Check the input mapping
        check_dataset_spec(X)

        # Initialize tables objects
        if len(X["tables"]) == 1:
            main_table_name = list(X["tables"])[0]
            main_table_source, main_table_key = list(X["tables"].values())[0]
            if isinstance(main_table_key, str):
                main_table_key = [main_table_key]
        else:
            main_table_name = X["main_table"]
            main_table_source, main_table_key = X["tables"][main_table_name]

        # Initialize a file dataset
        if isinstance(main_table_source, str):
            # Obtain the file format parameters
            if "format" in X:
                self.sep, self.header = X["format"]
            else:
                self.sep = "\t"
                self.header = True

            # Initialize the tables
            self.main_table = FileTable(
                main_table_name,
                main_table_source,
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
        # Initialize a Pandas dataset
        elif isinstance(main_table_source, pd.DataFrame):
            self.main_table = PandasTable(
                main_table_name,
                main_table_source,
                key=main_table_key,
            )
            self.secondary_tables = []
            for table_name, (table_source, table_key) in X["tables"].items():
                if table_name != main_table_name:
                    self.secondary_tables.append(
                        PandasTable(table_name, table_source, key=table_key)
                    )
        # Initialize a sparse dataset (monotable)
        elif isinstance(main_table_source, sp.spmatrix):
            self.main_table = SparseTable(
                main_table_name,
                main_table_source,
                key=main_table_key,
            )
            self.secondary_tables = []
        # Initialize a numpyarray dataset (monotable)
        else:
            self.main_table = NumpyTable(
                main_table_name,
                main_table_source,
            )
            if len(X["tables"]) > 1:
                raise ValueError(
                    "Multi-table schemas are only allowed "
                    "with pandas dataframe source tables"
                )
            self.secondary_tables = []

        # If the relations are not specified initialize to a star schema
        if "relations" not in X:
            self.relations = [
                (self.main_table.name, table.name, False)
                for table in self.secondary_tables
            ]
        # Otherwise initialize the relations in the spec
        else:
            relations = []
            for relation in X["relations"]:
                parent, child = relation[:2]
                relations.append(
                    (parent, child, relation[2] if len(relation) == 3 else False)
                )
            self.relations = relations

    def _init_target_column(self, y):
        assert self.main_table is not None
        assert self.secondary_tables is not None

        # Check y's type
        # For in memory target columns:
        # - column_or_1d checks *and transforms* to a numpy.array if successful
        # - warn=True in column_or_1d is necessary to pass sklearn checks
        if isinstance(y, str):
            y_checked = y
        else:
            y_checked = column_or_1d(y, warn=True)

        # Check the target type coherence with those of X's tables
        if isinstance(
            self.main_table, (PandasTable, SparseTable, NumpyTable)
        ) and isinstance(y_checked, str):
            if isinstance(self.main_table, PandasTable):
                type_message = "pandas.DataFrame"
            elif isinstance(self.main_table, SparseTable):
                type_message = "scipy.sparse.spmatrix"
            else:
                type_message = "numpy.ndarray"
            raise TypeError(
                type_error_message("y", y, "array-like")
                + f" (X's tables are of type {type_message})"
            )
        if isinstance(self.main_table.data_source, str) and not isinstance(
            y_checked, str
        ):
            raise TypeError(
                type_error_message("y", y, str)
                + " (X's tables are of type str [file paths])"
            )

        # Initialize the members related to the target
        # Case when y is a memory array
        if hasattr(y_checked, "__array__"):
            self.target_column = y_checked

            # Initialize the id of the target column
            if isinstance(y, pd.Series) and y.name is not None:
                self.target_column_id = y.name
            elif isinstance(y, pd.DataFrame):
                self.target_column_id = y.columns[0]
            else:
                if pd.api.types.is_integer_dtype(self.main_table.column_ids):
                    self.target_column_id = self.main_table.column_ids[-1] + 1
                else:
                    assert pd.api.types.is_string_dtype(self.main_table.column_ids)
                    self.target_column_id = "UnknownTargetColumn"

            # Fail if there is a column in the main_table with the target column's name
            if self.target_column_id in self.main_table.column_ids:
                raise ValueError(
                    f"Target column name '{self.target_column_id}' "
                    f"is already present in the main table. "
                    f"Column names: {list(self.main_table.column_ids)}"
                )
        # Case when y is column id: Set both the column and the id to it
        else:
            assert isinstance(y, str), type_error_message("y", y, str)
            self.target_column = y
            self.target_column_id = y

            # Check the target column exists in the main table
            if self.target_column_id not in self.main_table.column_ids:
                raise ValueError(
                    f"Target column '{self.target_column}' not present in main table. "
                    f"Column names: {list(self.main_table.column_ids)}'"
                )

            # Force the target column type from the parameters
            if self.categorical_target:
                self.main_table.khiops_types[self.target_column_id] = "Categorical"
            else:
                self.main_table.khiops_types[self.target_column_id] = "Numerical"

    @property
    def is_in_memory(self):
        """bool : ``True`` if the dataset is in-memory

        A dataset is in-memory if it is constituted either of only pandas.DataFrame
        tables, numpy.ndarray, or scipy.sparse.spmatrix tables.
        """

        return isinstance(self.main_table, (PandasTable, NumpyTable, SparseTable))

    @property
    def table_type(self):
        """type : The table type of this dataset's tables

        Possible values:

        - `PandasTable`
        - `NumpyTable`
        - `SparseTable`
        - `FileTable`
        """
        return type(self.main_table)

    @property
    def is_multitable(self):
        """bool : ``True`` if the dataset is multitable"""
        return self.secondary_tables is not None and len(self.secondary_tables) > 0

    def to_spec(self):
        """Returns a dictionary specification of this dataset"""
        ds_spec = {}
        ds_spec["main_table"] = self.main_table.name
        ds_spec["tables"] = {}
        ds_spec["tables"][self.main_table.name] = (
            self.main_table.data_source,
            self.main_table.key,
        )
        for table in self.secondary_tables:
            ds_spec["tables"][table.name] = (table.data_source, table.key)
        if self.relations:
            ds_spec["relations"] = []
            ds_spec["relations"].extend(self.relations)
        if self.table_type == FileTable:
            ds_spec["format"] = (self.sep, self.header)

        return ds_spec

    def copy(self):
        """Creates a copy of the dataset

        Referenced pandas.DataFrame's, numpy.nparray's and scipy.sparse.spmatrix's in
        tables are copied as references.
        """
        return Dataset(self.to_spec())

    def get_table(self, table_name):
        """Returns a table by its name

        Parameters
        ----------
        table_name: str
            The name of the table to be retrieved.

        Returns
        -------
        `DatasetTable`
            The table object for the specified name.

        Raises
        ------
        `KeyError`
            If there is no table with the specified name.
        """
        return self._tables_by_name[table_name]

    def create_khiops_dictionary_domain(self):
        """Creates a Khiops dictionary domain representing this dataset

        Returns
        -------
        `.DictionaryDomain`
            The dictionary domain object representing this dataset
        """
        assert self.main_table is not None, "'main_table' must be initialized"

        # Create root dictionary and add it to the domain
        dictionary_domain = kh.DictionaryDomain()
        main_dictionary = self.main_table.create_khiops_dictionary()
        dictionary_domain.add_dictionary(main_dictionary)

        # For in-memory datasets: Add the target variable if available
        if self.is_in_memory and self.target_column is not None:
            variable = kh.Variable()
            variable.name = get_khiops_variable_name(self.target_column_id)
            if self.categorical_target:
                variable.type = "Categorical"
            else:
                variable.type = "Numerical"
            main_dictionary.add_variable(variable)

        # Create the dictionaries for each secondary table and the table variables in
        # root dictionary that point to each secondary table
        # This is performed using a breadth-first-search over the graph of relations
        # Note: In general 'name' and 'object_type' fields of Variable can be different
        if self.secondary_tables:
            main_dictionary.root = True
            table_names = [table.name for table in self.secondary_tables]
            tables_to_visit = [self.main_table.name]
            while tables_to_visit:
                current_table = tables_to_visit.pop(0)
                for relation in self.relations:
                    parent_table, child_table, is_one_to_one_relation = relation
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
                        table_variable = kh.Variable()
                        if is_one_to_one_relation:
                            table_variable.type = "Entity"
                        else:
                            table_variable.type = "Table"
                        table_variable.name = table.name
                        table_variable.object_type = table.name
                        parent_table_dictionary.add_variable(table_variable)

        return dictionary_domain

    def create_table_files_for_khiops(self, output_dir, sort=True):
        """Prepares the tables of the dataset to be used by Khiops

        If this is a multi-table dataset it will create sorted copies the tables.

        Parameters
        ----------
        output_dir : str
            The directory where the sorted tables will be created.

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
            self.is_multitable or self.main_table.key is not None
        )

        # In-memory dataset: Create the table files and add the target column
        if self.is_in_memory:
            main_table_path = self.main_table.create_table_file_for_khiops(
                output_dir,
                sort=sort_main_table,
                target_column=self.target_column,
                target_column_id=self.target_column_id,
            )
        # File dataset: Create the table files (the target column is in the file)
        else:
            main_table_path = self.main_table.create_table_file_for_khiops(
                output_dir,
                sort=sort_main_table,
            )

        # Create a copy of each secondary table
        secondary_table_paths = {}
        for table in self.secondary_tables:
            secondary_table_paths[table.name] = table.create_table_file_for_khiops(
                output_dir, sort=sort
            )

        return main_table_path, secondary_table_paths

    def __repr__(self):
        return str(self.create_khiops_dictionary_domain())


# pylint: enable=invalid-name


class DatasetTable(ABC):
    """A generic dataset table"""

    def __init__(self, name, key=None):
        # Check input
        if not isinstance(name, str):
            raise TypeError(type_error_message("name", name, str))
        if not name:
            raise ValueError("'name' cannot be empty")
        if key is not None:
            if not is_list_like(key) and not isinstance(key, (str, int)):
                raise TypeError(type_error_message("key", key, str, int, "list-like"))
            if is_list_like(key):
                for column_index, column_id in enumerate(key):
                    if not isinstance(column_id, (str, int)):
                        raise TypeError(
                            type_error_message(
                                f"key[{column_index}]", column_id, str, int
                            )
                            + f" at table '{name}'"
                        )

        # Initialization (must be completed by concrete sub-classes)
        self.name = name
        self.data_source = None
        if is_list_like(key) or key is None:
            self.key = key
        else:
            self.key = [key]
        self.column_ids = None
        self.khiops_types = None
        self.n_samples = None

    def check_key(self):
        """Checks that the key columns exist"""
        if self.key is not None:
            if not is_list_like(self.key):
                raise TypeError(
                    type_error_message("key", self.key, str, int, "list-like")
                )
            for column_name in self.key:
                if column_name not in self.column_ids:
                    raise ValueError(
                        f"Column '{column_name}' not present in table '{self.name}'"
                    )

    @abstractmethod
    def create_table_file_for_khiops(self, output_dir, sort=True):
        """Creates a copy of the table at the specified directory"""

    def n_features(self):
        """Returns the number of features of the table

        The target column does not count.
        """
        return len(self.column_ids)

    def create_khiops_dictionary(self):
        """Creates a Khiops dictionary representing this table

        Returns
        -------
        `.Dictionary`:
            The Khiops Dictionary object describing this table's schema

        """
        assert self.column_ids is not None, "Dataset column list is None"
        assert self.key is None or is_list_like(self.key), "'key' is not list-like"

        # Create dictionary object
        dictionary = kh.Dictionary()
        dictionary.name = self.name
        if self.key is not None:
            dictionary.key = self.key

        # For each column add a Khiops variable to the dictionary
        for column_id in self.column_ids:
            variable = kh.Variable()
            variable.name = get_khiops_variable_name(column_id)

            # Set the type of the column/variable
            # Case of a column in the key : Set to categorical
            if self.key is not None and column_id in self.key:
                variable.type = "Categorical"
            # The rest of columns: Obtain the type from dtypes
            else:
                variable.type = self.khiops_types[column_id]
            dictionary.add_variable(variable)
        return dictionary


class PandasTable(DatasetTable):
    """DatasetTable encapsulating a pandas dataframe

    Parameters
    ----------
    name : str
        Name for the table.
    dataframe : `pandas.DataFrame`
        The data frame to be encapsulated. It must be non-empty.
    key : list of str, optional
        The names of the columns composing the key.
    """

    def __init__(self, name, dataframe, key=None):
        # Call the parent method
        super().__init__(name, key=key)

        # Check inputs specific to this sub-class
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(type_error_message("dataframe", dataframe, pd.DataFrame))
        if dataframe.shape[0] == 0:
            raise ValueError("'dataframe' is empty")

        # Initialize the attributes
        self.data_source = dataframe
        self.n_samples = len(self.data_source)

        # Initialize feature columns and verify their types
        self.column_ids = self.data_source.columns.values
        if not np.issubdtype(self.column_ids.dtype, np.integer):
            if np.issubdtype(self.column_ids.dtype, object):
                for i, column_id in enumerate(self.column_ids):
                    if not isinstance(column_id, str):
                        raise TypeError(
                            f"Dataframe column ids must be either all integers or "
                            f"all strings. Column id at index {i} ('{column_id}') is"
                            f" of type '{type(column_id).__name__}'"
                        )
            else:
                raise TypeError(
                    f"Dataframe column ids must be either all integers or "
                    f"all strings. The column index has dtype "
                    f"'{self.column_ids.dtype}'"
                )

        # Initialize Khiops types
        self.khiops_types = {
            column_id: get_khiops_type(self.data_source.dtypes[column_id])
            for column_id in self.column_ids
        }

        # Check key integrity
        self.check_key()

    def __repr__(self):
        dtypes_str = (
            str(self.data_source.dtypes).replace("\n", ", ")[:-16].replace("    ", ":")
        )
        return (
            f"<{self.__class__.__name__}; cols={list(self.column_ids)}; "
            f"dtypes={dtypes_str}>"
        )

    def get_column_dtype(self, column_id):
        if column_id not in self.data_source.dtypes:
            raise KeyError(f"Column '{column_id}' not found in the dtypes field")
        return self.data_source.dtypes[column_id]

    def create_table_file_for_khiops(
        self, output_dir, sort=True, target_column=None, target_column_id=None
    ):
        assert not sort or self.key is not None, "Cannot sort table without a key"
        assert not sort or is_list_like(
            self.key
        ), "Cannot sort table with a key is that is not list-like"
        assert not sort or len(self.key) > 0, "Cannot sort table with an empty key"
        assert target_column is not None or target_column_id is None
        assert target_column_id is not None or target_column is None

        # Create the output table resource object
        output_table_path = fs.get_child_path(output_dir, f"{self.name}.txt")

        # Write the output dataframe
        output_dataframe = self.data_source.copy()
        output_names = {
            column_id: get_khiops_variable_name(column_id)
            for column_id in self.column_ids
        }
        output_dataframe.rename(columns=output_names, inplace=True)
        if target_column is not None:
            output_dataframe[get_khiops_variable_name(target_column_id)] = (
                target_column.copy()
            )

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
            fs.write(
                output_table_path, output_dataframe_stream.getvalue().encode("utf-8")
            )

        return output_table_path


class NumpyTable(DatasetTable):
    """DatasetTable encapsulating a NumPy array

    Parameters
    ----------
    name : str
        Name for the table.
    array : `numpy.ndarray` of shape (n_samples, n_features_in)
        The data frame to be encapsulated.
    key : :external:term`array-like` of int, optional
        The names of the columns composing the key.
    """

    def __init__(self, name, array, key=None):
        # Call the parent method
        super().__init__(name, key=key)

        # Check the array's types and shape
        if not hasattr(array, "__array__"):
            raise TypeError(type_error_message("array", array, np.ndarray))

        # Initialize the members
        self.data_source = check_array(array, ensure_2d=True, force_all_finite=False)
        self.column_ids = column_or_1d(range(self.data_source.shape[1]))
        self.khiops_types = {
            column_id: get_khiops_type(self.data_source.dtype)
            for column_id in self.column_ids
        }
        self.n_samples = len(self.data_source)

    def __repr__(self):
        dtype_str = str(self.data_source.dtype)
        return (
            f"<{self.__class__.__name__}; cols={list(self.column_ids)}; "
            f"dtype={dtype_str}; target={self.target_column_id}>"
        )

    def get_column_dtype(self, _):
        return self.data_source.dtype

    def create_table_file_for_khiops(
        self, output_dir, sort=True, target_column=None, target_column_id=None
    ):
        assert not sort or self.key is not None, "Cannot sort table without a key"
        assert not sort or is_list_like(
            self.key
        ), "Cannot sort table with a key is that is not list-like"
        assert not sort or len(self.key) > 0, "Cannot sort table with an empty key"

        # Create the output table resource object
        output_table_path = fs.get_child_path(output_dir, f"{self.name}.txt")

        # Write the output dataframe
        # Note: This is not optimized for memory.
        output_dataframe = pd.DataFrame(self.data_source.copy())
        output_dataframe.columns = [
            get_khiops_variable_name(column_id) for column_id in self.column_ids
        ]
        if target_column is not None:
            output_dataframe[get_khiops_variable_name(target_column_id)] = (
                target_column.copy()
            )

        # Sort by key if requested (as string)
        if sort:
            np.sort(
                output_dataframe,
                by=self.key,
                key=lambda array: array.astype("str"),
                inplace=True,
                kind="mergesort",
            )

        # Write the dataframe to an internal table file
        with io.StringIO() as output_dataframe_stream:
            write_internal_data_table(output_dataframe, output_dataframe_stream)
            fs.write(
                output_table_path, output_dataframe_stream.getvalue().encode("utf-8")
            )

        return output_table_path


class SparseTable(DatasetTable):
    """DatasetTable encapsulating a SciPy sparse matrix

    Parameters
    ----------
    name : str
        Name for the table.
    matrix : `scipy.sparse.spmatrix`
        The sparse matrix to be encapsulated.
    key : list of str, optional
        The names of the columns composing the key.
    """

    def __init__(self, name, matrix, key=None):
        assert key is None, "'key' must be unset for sparse matrix tables"
        # Call the parent method
        super().__init__(name, key=key)

        # Check the sparse matrix types
        if not isinstance(matrix, sp.spmatrix):
            raise TypeError(
                type_error_message("matrix", matrix, "scipy.sparse.spmatrix")
            )
        if not np.issubdtype(matrix.dtype, np.number):
            raise TypeError(
                type_error_message("'matrix' dtype", matrix.dtype, "numeric")
            )

        # Initialize the members
        self.data_source = matrix
        self.column_ids = column_or_1d(range(matrix.shape[1]))
        self.khiops_types = {
            column_id: get_khiops_type(matrix.dtype) for column_id in self.column_ids
        }
        self.n_samples = self.data_source.shape[0]

    def __repr__(self):
        dtype_str = str(self.data_source.dtype)
        return (
            f"<{self.__class__.__name__}; cols={list(self.column_ids)}; "
            f"dtype={dtype_str}>"
        )

    def get_column_dtype(self, _):
        return self.data_source.dtype

    def create_khiops_dictionary(self):
        """Creates a Khiops dictionary representing this sparse table

        Adds metadata to each sparse variable

        Returns
        -------
        `.Dictionary`:
            The Khiops Dictionary object describing this table's schema

        """

        # create dictionary as usual
        dictionary = super().create_khiops_dictionary()

        # create variable block for containing the sparse variables
        variable_block = VariableBlock()
        variable_block.name = "SparseVariables"

        # For each variable, add metadata, named `VarKey`
        variable_names = [variable.name for variable in dictionary.variables]
        for i, variable_name in enumerate(variable_names, 1):
            variable = dictionary.remove_variable(variable_name)
            variable.meta_data.add_value("VarKey", i)
            variable_block.add_variable(variable)
        dictionary.add_variable_block(variable_block)

        return dictionary

    def _flatten(self, iterable):
        if isinstance(iterable, Iterable):
            for iterand in iterable:
                if isinstance(iterand, Iterable):
                    yield from self._flatten(iterand)
                else:
                    yield iterand

    def _write_sparse_block(self, row_index, stream, target_value=None):

        # Access the sparse row
        row = self.data_source.getrow(row_index)
        # Variable indices are not always sorted in `row.indices`
        # Khiops needs variable indices to be sorted
        sorted_indices = np.sort(row.nonzero()[1], axis=-1, kind="mergesort")

        # Flatten row for Python < 3.9 scipy.sparse.lil_matrix whose API
        # is not homogeneous with other sparse matrices: it stores
        # opaque Python lists as elements
        # Thus:
        # - if isinstance(self.data_source, sp.lil_matrix) and Python 3.8, then
        # row.data is np.array([list([...])])
        # - else, row.data is np.array([...])
        # TODO: remove this flattening once Python 3.8 support is dropped
        sorted_data = np.fromiter(self._flatten(row.data), row.data.dtype)[
            sorted_indices.argsort()
        ]
        for variable_index, variable_value in zip(sorted_indices, sorted_data):
            stream.write(f"{variable_index + 1}:{variable_value} ")

        # Write the target value at the end of the record if available
        if target_value is not None:
            stream.write(f"\t{target_value}\n")
        else:
            stream.write("\n")

    def create_table_file_for_khiops(
        self, output_dir, sort=True, target_column=None, target_column_id=None
    ):
        assert target_column is not None or target_column_id is None
        assert target_column_id is not None or target_column is None

        # Create the output table resource object
        output_table_path = fs.get_child_path(output_dir, f"{self.name}.txt")

        # Write the sparse matrix to an internal table file
        with io.StringIO() as output_sparse_matrix_stream:
            if target_column is not None:
                output_sparse_matrix_stream.write(
                    f"SparseVariables\t{get_khiops_variable_name(target_column_id)}\n"
                )
                for target_value, row_index in zip(
                    target_column, range(self.data_source.shape[0])
                ):
                    self._write_sparse_block(
                        row_index,
                        output_sparse_matrix_stream,
                        target_value=target_value,
                    )
            else:
                output_sparse_matrix_stream.write("SparseVariables\n")
                for row_index in range(self.data_source.shape[0]):
                    self._write_sparse_block(row_index, output_sparse_matrix_stream)
            fs.write(
                output_table_path,
                output_sparse_matrix_stream.getvalue().encode("utf-8"),
            )

        return output_table_path


class FileTable(DatasetTable):
    """DatasetTable encapsulating a delimited text data file

    Parameters
    ----------
    name : str
        Name for the table.
    path : str
        Path of the file containing the table.
    key : list-like of str, optional
        The names of the columns composing the key.
    sep : str, optional
        Field separator character. If not specified it will be inferred from the file.
    header : bool, optional
        Indicates if the table.
    """

    def __init__(
        self,
        name,
        path,
        key=None,
        sep="\t",
        header=True,
    ):
        # Initialize parameters
        super().__init__(name=name, key=key)

        # Check the parameters specific to this sub-class
        if not isinstance(path, str):
            raise TypeError(type_error_message("path", path, str))
        if not fs.exists(path):
            raise ValueError(f"Non-existent data table file: {path}")

        # Initialize members specific to this sub-class
        self.data_source = path
        self.sep = sep
        self.header = header

        # Build a dictionary file from the input data table
        # Note: We use export_dictionary_as_json instead of read_dictionary_file
        #       because it makes fail the sklearn mocked tests (this is technical debt)
        try:
            tmp_kdic_path = kh.get_runner().create_temp_file("file_table_", ".kdic")
            tmp_kdicj_path = kh.get_runner().create_temp_file("file_table_", ".kdicj")
            kh.build_dictionary_from_data_table(
                self.data_source,
                self.name,
                tmp_kdic_path,
                field_separator=self.sep,
                header_line=header,
            )
            kh.export_dictionary_as_json(tmp_kdic_path, tmp_kdicj_path)
            json_domain = json.loads(fs.read(tmp_kdicj_path))
        finally:
            fs.remove(tmp_kdic_path)
            fs.remove(tmp_kdicj_path)

        # Alert the user if the parsing failed
        if len(json_domain["dictionaries"]) == 0:
            raise KhiopsRuntimeError(
                f"Failed to build a dictionary "
                f"from data table file: {self.data_source}"
            )

        # Set the column names and types
        variables = json_domain["dictionaries"][0]["variables"]
        self.column_ids = [var["name"] for var in variables]
        self.khiops_types = {var["name"]: var["type"] for var in variables}

        # Check key integrity
        self.check_key()

    def create_table_file_for_khiops(self, output_dir, sort=True):
        assert not sort or self.key is not None, "key is 'None'"

        # Create the input and output file resources
        if sort:
            output_table_file_path = fs.get_child_path(
                output_dir, f"sorted_{self.name}.txt"
            )
        else:
            output_table_file_path = fs.get_child_path(
                output_dir, f"copy_{self.name}.txt"
            )

        # Fail if they have the same path
        if output_table_file_path == self.data_source:
            raise ValueError(f"Cannot overwrite this table's path: {self.data_source}")

        # Create a sorted copy if requested
        if sort:
            # Create the sorting dictionary domain
            sort_dictionary_domain = kh.DictionaryDomain()
            sort_dictionary_domain.add_dictionary(self.create_khiops_dictionary())

            # Delegate the sorting and copy to khiops.core.sort_data_table
            # We use the same input format of the original table
            kh.sort_data_table(
                sort_dictionary_domain,
                self.name,
                self.data_source,
                output_table_file_path,
                self.key,
                field_separator=self.sep,
                header_line=self.header,
                output_field_separator=self.sep,
                output_header_line=self.header,
            )

        # Otherwise copy the contents to the output file
        else:
            fs.write(output_table_file_path, fs.read(self.data_source))

        return output_table_file_path
