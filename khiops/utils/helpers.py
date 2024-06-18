"""General helper functions"""

import itertools
import os

from sklearn.model_selection import train_test_split

from khiops import core as kh
from khiops.core.internals.common import is_dict_like, type_error_message
from khiops.utils.dataset import Dataset, FileTable, PandasTable


def sort_dataset(ds_spec, output_dir=None):
    """Sorts a dataset by its table key columns


    The dataset may be multi-table or not. If it is monotable the key of the only table
    must be specified.

    Parameters
    ----------
    ds_spec: dict
        A dataset spec. The tables must be either `pandas.DataFrame` or file path
        references.
    output_dir: str, optional
        *Only for file datasets:* The output directory for the sorted files.


    Notes
    -----

    The sorting algorithm is mergesort, which ensures sort stability. The sorting engine
    for dataframes is Pandas and for file-based datasets is Khiops.

    Examples
    --------
    See the following functions of the ``samples.py`` documentation script:
        - `samples.sort_data_tables_mt()`
    """
    # Check the types
    if not is_dict_like(ds_spec):
        raise TypeError(type_error_message("ds_spec", ds_spec, "dict-like"))

    # Build the dataset
    ds = Dataset(ds_spec)

    # Check special arguments in function of the dataset
    if ds.table_type() == FileTable and output_dir is None:
        raise ValueError("'output_dir' must be specified for file based datasets")

    # Make a copy of the dataset (note: data sources are just reference)
    out_ds = ds.copy()

    # Replace each datasource with the sorted table
    for table in [out_ds.main_table] + out_ds.secondary_tables:
        if isinstance(table, PandasTable):
            table.data_source = _sort_df_table(table)
        else:
            assert isinstance(table, FileTable)
            table.data_source = _sort_file_table(table, ds.sep, ds.header, output_dir)

    return out_ds.to_spec()


def _sort_df_table(table):
    assert isinstance(table, PandasTable), type_error_message(
        "table", table, PandasTable
    )
    out_data_source = table.data_source.sort_values(
        by=table.key,
        key=lambda array: array.astype("str"),
        inplace=False,
        kind="mergesort",
    )

    return out_data_source


def _sort_file_table(table, sep, header, output_dir):
    assert isinstance(table, FileTable), type_error_message("table", table, FileTable)
    domain = kh.DictionaryDomain()
    dictionary = table.create_khiops_dictionary()
    domain.add_dictionary(dictionary)
    out_data_source = os.path.join(output_dir, f"{dictionary.name}.txt")
    kh.sort_data_table(
        domain,
        dictionary.name,
        table.data_source,
        out_data_source,
        field_separator=sep,
        header_line=header,
        output_field_separator=sep,
        output_header_line=header,
    )

    return out_data_source


# Note: We build the splits with lists and itertools.chain avoid pylint warning about
# unbalanced-tuple-unpacking. See issue https://github.com/pylint-dev/pylint/issues/5671


def train_test_split_dataset(
    ds_spec, target_column=None, test_size=0.25, output_dir=None, **kwargs
):
    """Splits a dataset spec into train and test

    Parameters
    ----------
    ds_spec : ``dict``
        A dataset spec. The tables must be either `pandas.DataFrame` or file path
        references.
    target_column : :external:term:`array-like`, optional
        The target values.
    test_size : float, default 0.25
        The proportion of the dataset (between 0.0 and 1.0) to be included in the test
        split.
    output_dir : str, optional
        *Only for file datasets:* The output directory for the split data files.
    ... :
        Other optional parameters for `sklearn.model_selection.train_test_split`


    Examples
    --------
    See the following functions of the ``samples_sklearn.py`` documentation script:
        - `samples_sklearn.khiops_classifier_multitable_star`
        - `samples_sklearn.khiops_classifier_multitable_star_file`
        - `samples_sklearn.khiops_classifier_multitable_snowflake`
    """
    # Check the types
    if not is_dict_like(ds_spec):
        raise TypeError(type_error_message("ds_spec", ds_spec, "dict-like"))

    # Build the dataset for the feature table
    ds = Dataset(ds_spec)

    # Check the parameter coherence
    if not ds.is_in_memory():
        if target_column is not None:
            raise ValueError("'target_column' cannot be used with file path datasets")
        if output_dir is None:
            raise ValueError("'output_dir' must be specified for file path datasets")
        if not isinstance(output_dir, str):
            raise TypeError(type_error_message("output_dir", output_dir, str))

    # Perform the split for each type of dataset
    if ds.is_in_memory():
        # Obtain the keys for the other test_train_split function
        sklearn_split_params = {}
        for param in ("train_size", "random_state", "shuffle", "stratify"):
            if param in kwargs:
                sklearn_split_params[param] = kwargs[param]

        if target_column is None:
            train_ds, test_ds = _train_test_split_in_memory_dataset(
                ds,
                target_column,
                test_size=test_size,
                split_params=sklearn_split_params,
            )
            train_target_column = None
            test_target_column = None
        else:
            train_ds, test_ds, train_target_column, test_target_column = (
                _train_test_split_in_memory_dataset(
                    ds,
                    target_column,
                    test_size=test_size,
                    split_params=sklearn_split_params,
                )
            )
    else:
        train_ds, test_ds = _train_test_split_file_dataset(ds, test_size, output_dir)
        train_target_column = None
        test_target_column = None

    # Create the return tuple
    # Note: We use `itertools.chain` to avoid pylint false positive about
    #     unbalanced-tuple-unpacking. This warning appears when calling the function so
    #     users would be warned. To remove when the following issue is fixed:
    #     https://github.com/pylint-dev/pylint/issues/5671
    if target_column is None:
        split = itertools.chain((train_ds.to_spec(), test_ds.to_spec()))
    else:
        split = itertools.chain(
            (
                train_ds.to_spec(),
                test_ds.to_spec(),
                train_target_column,
                test_target_column,
            )
        )

    return split


def _train_test_split_in_memory_dataset(
    ds, target_column, test_size, split_params=None
):
    # Create shallow copies of the feature dataset
    train_ds = ds.copy()
    test_ds = ds.copy()

    # Split the main table and the target (if any)
    if target_column is None:
        train_ds.main_table.data_source, test_ds.main_table.data_source = (
            train_test_split(
                ds.main_table.data_source, test_size=test_size, **split_params
            )
        )
        train_target_column = None
        test_target_column = None
    else:
        (
            train_ds.main_table.data_source,
            test_ds.main_table.data_source,
            train_target_column,
            test_target_column,
        ) = train_test_split(
            ds.main_table.data_source,
            target_column,
            test_size=test_size,
            **split_params,
        )

    # Split the secondary tables tables
    # Note: The tables are traversed in BFS
    todo_relations = [
        relation for relation in ds.relations if relation[0] == ds.main_table.name
    ]
    while todo_relations:
        current_parent_table_name, current_child_table_name, _ = todo_relations.pop(0)
        for relation in ds.relations:
            parent_table_name, _, _ = relation
            if parent_table_name == current_child_table_name:
                todo_relations.append(relation)

        for new_ds in (train_ds, test_ds):
            origin_child_table = ds.get_table(current_child_table_name)
            new_child_table = new_ds.get_table(current_child_table_name)
            new_parent_table = new_ds.get_table(current_parent_table_name)
            new_parent_key_cols_df = new_parent_table.data_source[new_parent_table.key]
            new_child_table.data_source = new_parent_key_cols_df.merge(
                origin_child_table.data_source, on=new_parent_table.key
            )

    # Build the return value
    # Note: We use `itertools.chain` to avoid pylint false positive about
    #     unbalanced-tuple-unpacking. This warning appears when calling the function so
    #     users would be warned. To remove when the following issue is fixed:
    #     https://github.com/pylint-dev/pylint/issues/5671
    if target_column is None:
        split = itertools.chain((train_ds, test_ds))
    else:
        split = itertools.chain(
            (train_ds, test_ds, train_target_column, test_target_column)
        )

    return split


def _train_test_split_file_dataset(ds, test_size, output_dir):
    domain = ds.create_khiops_dictionary_domain()
    secondary_data_paths = domain.extract_data_paths(ds.main_table.name)
    additional_data_tables = {}
    output_additional_data_tables = {
        "train": {},
        "test": {},
    }
    # Initialize the split datasets as copies of the original one
    split_dss = {
        "train": ds.copy(),
        "test": ds.copy(),
    }
    for split, split_ds in split_dss.items():
        split_ds.main_table.data_source = os.path.join(
            output_dir, split, f"{split_ds.main_table.name}.txt"
        )

    for data_path in secondary_data_paths:
        dictionary = domain.get_dictionary_at_data_path(data_path)
        table = ds.get_table(dictionary.name)
        additional_data_tables[data_path] = table.data_source
        for (
            split,
            split_output_additional_data_tables,
        ) in output_additional_data_tables.items():
            data_table_path = os.path.join(output_dir, split, f"{table.name}.txt")
            split_output_additional_data_tables[data_path] = data_table_path
            split_dss[split].get_table(table.name).data_source = data_table_path

    # Construct the split with Khiops by deploying a idempotent model with selection
    kh.deploy_model(
        domain,
        ds.main_table.name,
        ds.main_table.data_source,
        split_dss["train"].main_table.data_source,
        additional_data_tables=additional_data_tables,
        output_additional_data_tables=output_additional_data_tables["train"],
        header_line=ds.header,
        field_separator=ds.sep,
        output_header_line=ds.header,
        output_field_separator=ds.sep,
        sample_percentage=100.0 * (1 - test_size),
        sampling_mode="Include sample",
    )
    kh.deploy_model(
        domain,
        ds.main_table.name,
        ds.main_table.data_source,
        split_dss["test"].main_table.data_source,
        additional_data_tables=additional_data_tables,
        output_additional_data_tables=output_additional_data_tables["test"],
        header_line=ds.header,
        field_separator=ds.sep,
        output_header_line=ds.header,
        output_field_separator=ds.sep,
        sample_percentage=100.0 * (1 - test_size),
        sampling_mode="Exclude sample",
    )

    # Note: We use `itertools.chain` to avoid pylint false positive about
    #     unbalanced-tuple-unpacking. This warning appears when calling the function so
    #     users would be warned. To remove when the following issue is fixed:
    #     https://github.com/pylint-dev/pylint/issues/5671
    return itertools.chain(split_dss.values())
