######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Classes for handling diverse data tables"""
import csv
import io
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import sklearn
from scipy import sparse as sp
from sklearn.utils import check_array
from sklearn.utils.validation import column_or_1d

import khiops.core as kh
import khiops.core.internals.filesystems as fs
from khiops.core.dictionary import VariableBlock
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

    # Check the "main_table" field
    if "main_table" not in ds_spec:
        raise ValueError("'main_table' entry missing from dataset dict spec")
    if not isinstance(ds_spec["main_table"], tuple):
        raise TypeError(
            type_error_message("'main_table' entry", ds_spec["main_table"], tuple)
        )
    if len(ds_spec["main_table"]) != 2:
        raise ValueError("'main_table' must be a 2-element tuple")

    # Multi-table specific table checks
    if "additional_data_tables" in ds_spec:
        _check_multitable_spec(ds_spec)


def _check_table_entry(table_path, table_spec):
    if not isinstance(table_path, str):
        raise TypeError(type_error_message("Table path", table_path, str))
    if not isinstance(table_spec, tuple):
        raise TypeError(
            type_error_message(f"'{table_path}' table entry", table_spec, tuple)
        )
    if len(table_spec) not in (2, 3):
        raise ValueError(
            f"'{table_path}' table entry must have size 2 or 3, not {len(table_spec)}"
        )
    if len(table_spec) == 3 and not isinstance(table_spec[2], bool):
        raise TypeError(
            type_error_message(
                f"Table at data path {table_path} 1-1 flag",
                table_spec[2],
                bool,
            )
        )
    source, key = table_spec[:2]
    if not isinstance(source, (pd.DataFrame, sp.spmatrix, str)) and not hasattr(
        source, "__array__"
    ):
        raise TypeError(
            type_error_message(
                f"Source of table at data path '{table_path}'",
                source,
                "array-like",
                "scipy.sparse.spmatrix",
                str,
            )
        )
    _check_table_key(table_path, key)


def _check_table_key(table_path, key):
    if key is not None:
        if not is_list_like(key):
            raise TypeError(
                type_error_message(f"'{table_path}' table's key", key, Sequence)
            )
        if len(key) == 0:
            raise ValueError(f"'{table_path}' table's key is empty")
        for column_name in key:
            if not isinstance(column_name, str):
                raise TypeError(
                    type_error_message(
                        f"'{table_path}' table's key column name",
                        column_name,
                        str,
                    )
                )


def _check_multitable_spec(ds_spec):
    # Check that "additional_data_tables" is present
    assert "additional_data_tables" in ds_spec

    # Check the "additional_data_tables" field
    if not is_dict_like(ds_spec["additional_data_tables"]):
        raise TypeError(
            type_error_message(
                "'additional_data_tables' entry",
                ds_spec["additional_data_tables"],
                Mapping,
            )
        )
    for table_path, table_entry in ds_spec["additional_data_tables"].items():
        _check_table_entry(table_path, table_entry)

    # Check that all the tables have the same type as the main
    # Check that the main table's key is contained in subtable keys
    main_table_type = type(ds_spec["main_table"][0])
    main_table_key = ds_spec["main_table"][1]
    if main_table_key is None:
        raise ValueError(
            "The key of the main table is 'None': "
            "table keys must be specified in multi-table datasets"
        )
    if not main_table_key:
        raise ValueError(
            "The key of the main table is empty: "
            "table keys must be specified in multi-table datasets"
        )
    for table_path, table_spec in ds_spec["additional_data_tables"].items():
        table_source = table_spec[0]
        if not isinstance(table_source, main_table_type):
            raise ValueError(
                f"Additional data table at data path '{table_path}' has type "
                f"'{type(table_source).__name__}' which is different from the "
                f"main table's type '{main_table_type.__name__}'."
            )
        table_key = table_spec[1]
        if table_key is None:
            raise ValueError(
                f"Key of secondary table at path '{table_path}' is 'None': "
                "table keys must be specified in multi-table datasets"
            )
        if not set(main_table_key).issubset(table_key):
            table_key_msg = f"[{', '.join(table_key)}]"
            main_table_key_msg = f"[{', '.join(main_table_key)}]"
            raise ValueError(
                f"Table at data path '{table_path}' "
                f"key ({table_key_msg}) does not contain that of the main table "
                f"({main_table_key_msg})."
            )


def table_name_of_path(table_path):
    """Returns the table name as the last fragment of the table data path

    Parameters
    ----------
    table_path: str
        Data path of the table, in the format "path/to/table".

    Returns
    -------
    str
        The name of the table.
    """
    return table_path.split("/")[-1]


def _upgrade_mapping_spec(ds_spec):
    assert is_dict_like(ds_spec)
    new_ds_spec = {}
    new_ds_spec["additional_data_tables"] = {}
    for table_name, table_data in ds_spec["tables"].items():
        table_df, table_key = table_data
        if not is_list_like(table_key):
            table_key = [table_key]
        if table_name == ds_spec["main_table"]:
            new_ds_spec["main_table"] = (table_df, table_key)
        else:
            table_path = [table_name]
            is_entity = False

            # Cycle 4 times on the relations to get all transitive relation, like:
            # - current table name N
            # - main table name N1
            # - and relations: (N1, N2), (N2, N3), (N3, N)
            # the data-path must be N2/N3/N
            # Note: this is a heuristic that should be replaced with a graph
            # traversal procedure
            # If no "relations" key exists, then one has a star schema and
            # the data-paths are the names of the secondary tables themselves
            # (with respect to the main table)
            if "relations" in ds_spec:
                for relation in list(ds_spec["relations"]) * 4:
                    left, right = relation[:2]
                    if len(relation) == 3 and right == table_name:
                        is_entity = relation[2]
                    if (
                        left != ds_spec["main_table"]
                        and left not in table_path
                        and right in table_path
                    ):
                        table_path.insert(0, left)
            table_path = "/".join(table_path)
            if is_entity:
                table_data = (table_df, table_key, is_entity)
            else:
                table_data = (table_df, table_key)
            new_ds_spec["additional_data_tables"][table_path] = table_data
    return new_ds_spec


def get_khiops_type(numpy_type, categorical_str_max_size=None):
    """Translates a numpy dtype to a Khiops dictionary type

    Parameters
    ----------
    numpy_type : `numpy.dtype`
        Numpy type of the column
    categorical_str_max_size : `int`, optional
        Maximum length of the entries of the column whose type is ``numpy_type``.

    Returns
    -------
    str
        Khiops type name. Either "Categorical", "Text", "Numerical" or "Timestamp".

    .. note::
        The "Text" Khiops type is inferred if the Numpy type is "string"
        and the maximum length of the entries of that type is greater than 100.

    """
    # Check categorical_str_max_size type
    if categorical_str_max_size is not None and not isinstance(
        categorical_str_max_size, (int, np.int64)
    ):
        raise TypeError(
            type_error_message(
                "categorical_str_max_size",
                categorical_str_max_size,
                int,
                np.int64,
            )
        )

    # Get the Numpy dtype in lowercase
    lower_numpy_type = str(numpy_type).lower()

    # timedelta64 and datetime64 types
    if "datetime64" in lower_numpy_type or "timedelta64" in lower_numpy_type:
        khiops_type = "Timestamp"
    # float<x>, int<x>, uint<x> types
    elif "int" in lower_numpy_type or "float" in lower_numpy_type:
        khiops_type = "Numerical"
    elif lower_numpy_type == "string":
        if categorical_str_max_size is not None and categorical_str_max_size > 100:
            khiops_type = "Text"
        else:
            khiops_type = "Categorical"
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

    Khiops cannot handle multi-line records.
    Hence, the carriage returns / line feeds need to be removed from the records
    before data is handed over to Khiops.

    Parameters
    ----------
    dataframe : `pandas.DataFrame`
        The dataframe to write.
    file_path_or_stream : str or file object
        The path of the internal data table file to be written or a writable file
        object.
    """
    # Replace carriage returns / line feeds by blanks spaces
    # in order to always keep mono-lines text fields
    dataframe = dataframe.replace(["\r", "\n"], " ", regex=True)

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


def _column_or_1d_with_dtype(y, dtype=None):
    # 'dtype' has been introduced on `column_or_1d' since Scikit-learn 1.2;
    if sklearn.__version__ < "1.2":
        if pd.api.types.is_string_dtype(dtype) and y.isin(["True", "False"]).all():
            warnings.warn(
                "'y' stores strings restricted to 'True'/'False' values: "
                "The predict method may return a bool vector."
            )
        return column_or_1d(y, warn=True)
    else:
        return column_or_1d(y, warn=True, dtype=dtype)


class Dataset:
    """A representation of a dataset

    Parameters
    ----------
    X : `pandas.DataFrame` or dict
        Either:
          - A single dataframe
          - A ``dict`` dataset specification
    y : `pandas.Series`, `pandas.DataFrame` or `numpy.ndarray`, optional
        The target column.
    categorical_target : bool, default True
        ``True`` if the vector ``y`` should be considered as a categorical variable. If
        ``False`` it is considered as numeric. Ignored if ``y`` is ``None``.
    """

    def __init__(self, X, y=None, categorical_target=True):
        # Initialize members
        self.main_table = None
        self.additional_data_tables = None
        self.categorical_target = categorical_target
        self.target_column = None
        self.target_column_id = None
        self.sep = None
        self.header = None

        # Initialization from different types of input "X"
        # A single pandas dataframe
        if isinstance(X, pd.DataFrame):
            self.main_table = PandasTable("main_table", X)
            self.additional_data_tables = []
        # A single numpy array (or compatible object)
        elif hasattr(X, "__array__") or is_list_like(X):
            self.main_table = NumpyTable("main_table", X)
            self.additional_data_tables = []
        # A scipy.sparse.spmatrix
        elif isinstance(X, sp.spmatrix):
            self.main_table = SparseTable("main_table", X)
            self.additional_data_tables = []
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
        # A a dataset dict spec
        elif is_dict_like(X):
            self._init_tables_from_mapping(X)
        # Fail if X is not recognized
        else:
            raise TypeError(type_error_message("X", X, "array-like", Mapping, Sequence))

        # Initialization of the target column if any
        if y is not None:
            self._init_target_column(y)

        # Index the tables by name
        self._tables_by_name = {
            table.name: table
            for table in [self.main_table]
            + [table for _, table, _ in self.additional_data_tables]
        }

        # Post-conditions
        assert self.main_table is not None, "'main_table' is 'None' after init"
        assert isinstance(
            self.additional_data_tables, list
        ), "'secondary_tables' is not a list after init"
        assert not self.is_multitable or len(
            self.additional_data_tables
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

        # Detect if deprecated mapping specification syntax is used;
        # if so, issue deprecation warning and transform it to the new syntax
        if "tables" in X.keys() and isinstance(X.get("main_table"), str):
            warnings.warn(
                deprecation_message(
                    "This multi-table dataset specification format",
                    "11.0.1",
                    replacement=(
                        "the new data-path-based format, as documented in "
                        ":doc:`multi_table_primer`."
                    ),
                    quote=False,
                )
            )
            X = _upgrade_mapping_spec(X)

        # Check the input mapping
        check_dataset_spec(X)

        main_table_name = "main_table"
        main_table_source, main_table_key = X["main_table"]

        # Initialize a Pandas dataset
        if isinstance(main_table_source, pd.DataFrame):
            self.main_table = PandasTable(
                main_table_name,
                main_table_source,
                key=main_table_key,
            )
            self.additional_data_tables = []
            if "additional_data_tables" in X:
                for table_path, table_spec in X["additional_data_tables"].items():
                    table_source, table_key = table_spec[:2]
                    table_name = table_name_of_path(table_path)
                    table = PandasTable(
                        table_name,
                        table_source,
                        key=table_key,
                    )
                    is_one_to_one_relation = False
                    if len(table_spec) == 3 and table_spec[2] is True:
                        is_one_to_one_relation = True

                    self.additional_data_tables.append(
                        (table_path, table, is_one_to_one_relation)
                    )
        # Initialize a sparse dataset (monotable)
        elif isinstance(main_table_source, sp.spmatrix):
            self.main_table = SparseTable(
                name=main_table_name,
                matrix=main_table_source,
                key=main_table_key,
            )
            self.additional_data_tables = []
        # Initialize a numpyarray dataset (monotable)
        elif hasattr(main_table_source, "__array__"):
            self.main_table = NumpyTable(
                name=main_table_name,
                array=main_table_source,
            )
            if "additional_data_tables" in X and len(X["additional_data_tables"]) > 0:
                raise ValueError(
                    "Multi-table schemas are only allowed "
                    "with pandas dataframe source tables"
                )
            self.additional_data_tables = []
        else:
            raise TypeError(
                type_error_message(
                    "X's main table", main_table_source, "array-like", Mapping
                )
            )

    def _init_target_column(self, y):
        assert self.main_table is not None
        assert self.additional_data_tables is not None

        # Check y's type
        # For in memory target columns:
        # - column_or_1d checks *and transforms* to a numpy.array if successful
        # - warn=True in column_or_1d is necessary to pass sklearn checks
        if isinstance(y, str):
            y_checked = y
        # pandas.Series, pandas.DataFrame or numpy.ndarray
        else:
            if hasattr(y, "dtype"):
                if isinstance(y.dtype, pd.CategoricalDtype):
                    y_checked = _column_or_1d_with_dtype(
                        y, dtype=y.dtype.categories.dtype
                    )
                else:
                    y_checked = _column_or_1d_with_dtype(y, dtype=y.dtype)
            elif hasattr(y, "dtypes"):
                if isinstance(y.dtypes.iloc[0], pd.CategoricalDtype):
                    y_checked = _column_or_1d_with_dtype(
                        y, dtype=y.dtypes.iloc[0].categories.dtype
                    )
                else:
                    y_checked = _column_or_1d_with_dtype(y)
            else:
                y_checked = _column_or_1d_with_dtype(y)
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
    def table_type(self):
        """type : The table type of this dataset's tables

        Possible values:

        - `PandasTable`
        - `NumpyTable`
        - `SparseTable`
        """
        return type(self.main_table)

    @property
    def is_multitable(self):
        """bool : ``True`` if the dataset is multitable"""
        return (
            self.additional_data_tables is not None
            and len(self.additional_data_tables) > 0
        )

    def to_spec(self):
        """Returns a dictionary specification of this dataset"""
        ds_spec = {}
        ds_spec["main_table"] = (self.main_table.data_source, self.main_table.key)
        ds_spec["additional_data_tables"] = {}
        for table_path, table, is_one_to_one_relation in self.additional_data_tables:
            assert table_path is not None
            ds_spec["additional_data_tables"][table_path] = (
                table.data_source,
                table.key,
                is_one_to_one_relation,
            )

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

        # Add the target variable if available
        if self.target_column is not None:
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
        if self.additional_data_tables:
            main_dictionary.root = True
            for (
                table_path,
                table,
                is_one_to_one_relation,
            ) in self.additional_data_tables:
                if not "/" in table_path:
                    parent_table_name = self.main_table.name
                else:
                    table_path_fragments = table_path.split("/")
                    parent_table_name = table_name_of_path(
                        "/".join(table_path_fragments[:-1])
                    )
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
            - A dictionary containing the relation [data-path -> file-path] for the
              secondary tables. The dictionary is empty for monotable datasets.
        """
        # Sort the main table unless:
        # - The caller specifies not to do it (sort = False)
        # - The dataset is mono-table and the main table has no key
        sort_main_table = sort and (
            self.is_multitable or self.main_table.key is not None
        )

        # Create the table files and add the target column
        main_table_path = self.main_table.create_table_file_for_khiops(
            output_dir,
            sort=sort_main_table,
            target_column=self.target_column,
            target_column_id=self.target_column_id,
        )

        # Create a copy of each secondary table
        secondary_table_paths = {}
        for table_path, table, _ in self.additional_data_tables:
            assert table_path is not None
            secondary_table_paths[table_path] = table.create_table_file_for_khiops(
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
            if not is_list_like(key):
                raise TypeError(type_error_message("key", key, "list-like"))
            for column_index, column_id in enumerate(key):
                if not isinstance(column_id, (str, int)):
                    raise TypeError(
                        type_error_message(f"key[{column_index}]", column_id, str, int)
                        + f" at table '{name}'"
                    )

        # Initialization (must be completed by concrete sub-classes)
        self.name = name
        self.data_source = None
        self.key = key
        self.column_ids = None
        self.khiops_types = None
        self.n_samples = None

    def check_key(self):
        """Checks that the key columns exist"""
        if self.key is not None:
            if not is_list_like(self.key):
                raise TypeError(type_error_message("key", self.key, "list-like"))
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
        super().__init__(name=name, key=key)

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
        self.khiops_types = {}
        for column_id in self.column_ids:
            column = self.data_source[column_id]
            column_numpy_type = column.dtype
            column_max_size = None
            if isinstance(column_numpy_type, pd.StringDtype):
                column_max_size = column.str.len().max()
            self.khiops_types[column_id] = get_khiops_type(
                column_numpy_type, column_max_size
            )

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
    array : `numpy.ndarray` of shape (n_samples, n_features_in) or Sequence
        The data frame to be encapsulated.
    key : :external:term`array-like` of int, optional
        The names of the columns composing the key.
    """

    def __init__(self, name, array, key=None):
        # Call the parent method
        super().__init__(name=name, key=key)

        # Check the array's types and shape
        if not hasattr(array, "__array__") and not is_list_like(array):
            raise TypeError(type_error_message("array", array, np.ndarray, Sequence))

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
