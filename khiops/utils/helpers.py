"""General helper functions"""

import os

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
        The dataset dictionary specification. The tables must be either
        `pandas.DataFrame` or file path references.
    output_dir: str, optional
        _Only for file datasets:_ The output directory for the sorted files.


    Notes
    -----

    The sorting algorithm is mergesort, which ensures sort stability. The sorting engine
    for dataframes is Pandas and for file-based datasets is Khiops.

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
