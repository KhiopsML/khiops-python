"""General helper functions"""

import itertools

from sklearn.model_selection import train_test_split

from khiops.core.internals.common import is_dict_like, type_error_message
from khiops.utils.dataset import Dataset, FileTable

# Note: We build the splits with lists and itertools.chain avoid pylint warning about
# unbalanced-tuple-unpacking. See issue https://github.com/pylint-dev/pylint/issues/5671


def train_test_split_dataset(ds_spec, y=None, test_size=0.25, **kwargs):
    """Splits a multi-table dataset spec into train and test

    Parameters
    ----------
    ds_spec : ``dict``
         A ``dict`` multi-table dataset specification (see :doc:`/multi_table_primer`).
         Only Pandas, NumPy, SciPy tables are accepted in the spec.
    y : :external:term:`array-like` of size (n_samples,) , optional
        The target values. ``n_samples`` is the number of rows of the main table in
        ``ds_spec``.
    test_size : float, default 0.25
        The proportion of the dataset (between 0.0 and 1.0) to be included in the test
        split.
    ... :
        Other optional parameters for `sklearn.model_selection.train_test_split`

    Raises
    ------
    `TypeError`
        If ``ds_spec`` is not dict-like.

    `ValueError`
        If the tables in ``ds_spec`` are file-paths.

    Examples
    --------
    See the following functions of the ``samples_sklearn.py`` documentation script:
        - `samples_sklearn.khiops_classifier_multitable_star`
        - `samples_sklearn.khiops_classifier_multitable_snowflake`

    """
    # Check the types
    if not is_dict_like(ds_spec):
        raise TypeError(type_error_message("ds_spec", ds_spec, "dict-like"))

    # Build the dataset for the feature table
    ds = Dataset(ds_spec)

    # Check the table type of the dataset
    if isinstance(ds.main_table, FileTable):
        raise ValueError(
            "Only Pandas, NumPy, SciPy sparse datasets may be used in this method."
        )

    # Obtain the keys for the other test_train_split function
    sklearn_split_params = {}
    for param in ("train_size", "random_state", "shuffle", "stratify"):
        if param in kwargs:
            sklearn_split_params[param] = kwargs[param]

    # Perform the split with and without target
    train_ds, test_ds, train_target_column, test_target_column = (
        _train_test_split_in_memory_dataset(
            ds,
            y,
            test_size=test_size,
            sklearn_split_params=sklearn_split_params,
        )
    )

    # Create the return tuple
    # Note:
    #     We use `itertools.chain` to avoid pylint false positive about
    #     unbalanced-tuple-unpacking. This warning appears when calling the function so
    #     users would be warned. To remove when the following issue is fixed:
    #     https://github.com/pylint-dev/pylint/issues/5671
    if y is None:
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


def _train_test_split_in_memory_dataset(ds, y, test_size, sklearn_split_params=None):
    # Create shallow copies of the feature dataset
    train_ds = ds.copy()
    test_ds = ds.copy()

    # Split the main table and the target (if any)
    if y is None:
        train_ds.main_table.data_source, test_ds.main_table.data_source = (
            train_test_split(
                ds.main_table.data_source, test_size=test_size, **sklearn_split_params
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
            y,
            test_size=test_size,
            **sklearn_split_params,
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

    return train_ds, test_ds, train_target_column, test_target_column
