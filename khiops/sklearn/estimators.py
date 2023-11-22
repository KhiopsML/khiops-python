######################################################################################
# Copyright (c) 2023 Orange. All rights reserved.                                    #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Scikit-Learn Estimator Classes for the Khiops AutoML Suite

Class Overview
--------------
The diagram below describes the relationships in this module::

    KhiopsEstimator(ABC, BaseEstimator)
        |
        +- KhiopsCoclustering(ClusterMixin)
        |
        +- KhiopsSupervisedEstimator
           |
           +- KhiopsPredictor
           |  |
           |  +- KhiopsClassifier(ClassifierMixin)
           |  |
           |  +- KhiopsRegressor(RegressorMixin)
           |
           +- KhiopsEncoder(TransformerMixin)
"""
import io
import warnings
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    ClusterMixin,
    RegressorMixin,
    TransformerMixin,
)
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import assert_all_finite, check_is_fitted, column_or_1d

import khiops.core as kh
import khiops.core.internals.filesystems as fs
from khiops.core.dictionary import DictionaryDomain
from khiops.core.helpers import build_multi_table_dictionary_domain
from khiops.core.internals.common import (
    deprecation_message,
    is_dict_like,
    is_list_like,
    type_error_message,
)
from khiops.sklearn.tables import Dataset, read_internal_data_table

# Disable PEP8 variable names because of scikit-learn X,y conventions
# To capture invalid-names other than X,y run:
#   pylint --disable=all --enable=invalid-names estimators.py
# pylint: disable=invalid-name


def _extract_basic_dictionary(dictionary):
    """Extracts a dictionary containing only basic variables"""
    basic_dictionary = dictionary.copy()
    for variable in dictionary.variables:
        variable_is_model_output = (
            "Prediction" in variable.meta_data
            or "Score" in variable.meta_data
            or "Mean" in variable.meta_data
            or "StandardDeviation" in variable.meta_data
            or "Density" in variable.meta_data
            or any(key.startswith("TargetProb") for key in variable.meta_data.keys)
        )
        variable_is_not_basic = variable.type in ["Structure", "Table", "Entity"]
        variable_is_in_rule_block = (
            variable.variable_block is not None and variable.variable_block.rule != ""
        )
        if (
            variable_is_model_output
            or variable_is_not_basic
            or variable.rule
            or variable_is_in_rule_block
        ):
            basic_dictionary.remove_variable(variable.name)
    return basic_dictionary


def _check_dictionary_compatibility(
    model_dictionary,
    dataset_dictionary,
    estimator_class_name,
):
    # Prefix for all error messages
    error_msg_prefix = f"X contains incompatible table '{dataset_dictionary.name}'"

    # Save variable arrays and their size
    model_variables = model_dictionary.variables
    dataset_variables = dataset_dictionary.variables

    # Error if different number of variables
    if len(model_variables) != len(dataset_variables):
        raise ValueError(
            f"{error_msg_prefix}: It has "
            f"{len(dataset_variables)} feature(s) but {estimator_class_name} "
            f"is expecting {len(model_variables)}. Reshape your data."
        )

    # Check variables: Must have same name and type
    for var_index, (model_variable, dataset_variable) in enumerate(
        zip(model_variables, dataset_variables)
    ):
        if model_variable.name != dataset_variable.name:
            raise ValueError(
                f"{error_msg_prefix}: Feature #{var_index} should be named "
                f"'{model_variable.name}' "
                f"instead of '{dataset_variable.name}'"
            )
        if model_variable.type != dataset_variable.type:
            raise ValueError(
                f"{error_msg_prefix}: Feature #{var_index} should convertible to "
                f"'{model_variable.type}' "
                f"instead of '{dataset_variable.type}'"
            )


def _check_categorical_target_type(dataset):
    assert (
        dataset.main_table.target_column_id is not None
    ), "Target column not specified in dataset."
    if not (
        isinstance(dataset.target_column_type, pd.CategoricalDtype)
        or pd.api.types.is_string_dtype(dataset.target_column_type)
        or pd.api.types.is_integer_dtype(dataset.target_column_type)
        or pd.api.types.is_float_dtype(dataset.target_column_type)
    ):
        raise ValueError(
            f"'y' has invalid type '{dataset.target_column_type}'. "
            "Only string, integer, float and categorical types "
            "are accepted for the target."
        )


def _check_numerical_target_type(dataset):
    assert (
        dataset.main_table.target_column_id is not None
    ), "Target column not specified in dataset."
    if not pd.api.types.is_numeric_dtype(dataset.target_column_type):
        raise ValueError(
            f"Unknown label type '{dataset.target_column_type}'. "
            "Expected a numerical type."
        )
    if dataset.is_in_memory() and dataset.main_table.target_column is not None:
        assert_all_finite(dataset.main_table.target_column)


def _cleanup_dir(target_dir):
    """Cleanups a directory with only files in it

    Parameters
    ----------
    target_dir : str
        path or URI of the directory to clean.
    """
    if not isinstance(target_dir, str):
        raise TypeError(type_error_message("target_dir", target_dir, str))
    for file_name in fs.list_dir(target_dir):
        fs.remove(fs.get_child_path(target_dir, file_name))
    if fs.is_local_resource(target_dir):
        fs.remove(target_dir)


class KhiopsEstimator(ABC, BaseEstimator):
    """Base class for Khiops Scikit-learn estimators

    Parameters
    ----------
    verbose : bool, default ``False``
        If ``True`` it prints debug information and it does not erase temporary files
        when fitting, predicting or transforming.
    output_dir : str, optional
        Path of the output directory for the resulting artifacts of Khiops learning
        tasks. See concrete estimator classes for more information about this parameter.
    auto_sort : bool, default ``True``
        *Advanced.*: See concrete estimator classes for information about this
        parameter.
    key : str, optional
        The name of the column to be used as key.
        **Deprecated** will be removed in khiops-python 11.
    internal_sort : bool, optional
        *Advanced.*: See concrete estimator classes for information about this
        parameter.
        **Deprecated** will be removed in khiops-python 11. Use the ``auto_sort``
        estimator parameter instead.
    """

    def __init__(
        self,
        key=None,
        verbose=False,
        output_dir=None,
        auto_sort=True,
        internal_sort=None,
    ):
        # Set the estimator parameters and internal variables
        self._khiops_model_prefix = None
        self.key = key
        self.output_dir = output_dir
        self.verbose = verbose

        # Set auto_sort and show a deprecation message for internal_sort
        if internal_sort is not None:
            self.auto_sort = internal_sort
            warnings.warn(
                deprecation_message(
                    "the 'internal_sort' estimator parameter",
                    "11.0.0",
                    replacement="the 'auto_sort' estimator parameter",
                    quote=False,
                )
            )
        else:
            self.auto_sort = auto_sort

        # Make sklearn get_params happy
        self.internal_sort = internal_sort

    def _more_tags(self):
        return {"allow_nan": True, "accept_large_sparse": False}

    def _undefine_estimator_attributes(self):
        """Undefines all *estimator* attributes (those that end with _)"""
        for attribute_name in dir(self):
            if not attribute_name.startswith("_") and attribute_name.endswith("_"):
                delattr(self, attribute_name)

    def _get_main_dictionary(self):
        """Returns the model's main Khiops dictionary"""
        assert self.model_ is not None, "Model dictionary domain not available."
        return self.model_.get_dictionary(self.model_main_dictionary_name_)

    def export_report_file(self, report_file_path):
        """Exports the model report to a JSON file

        Parameters
        ----------
        report_file_path : str
            The location of the exported report file.

        Raises
        ------
        `ValueError`
            When the instance is not fitted.
        """
        if not self.is_fitted_:
            raise ValueError(f"{self.__class__.__name__} not fitted yet.")
        if self.model_report_ is None:
            raise ValueError("Report not available (imported model?).")
        self.model_report_.write_khiops_json_file(report_file_path)

    def export_dictionary_file(self, dictionary_file_path):
        """Export the model's Khiops dictionary file (.kdic)"""
        if not self.is_fitted_:
            raise ValueError(f"{self.__class__.__name__} not fitted yet.")
        self.model_.export_khiops_dictionary_file(dictionary_file_path)

    def _import_model(self, kdic_path):
        """Sets model instance attribute by importing model from ``.kdic``"""
        self.model_ = kh.read_dictionary_file(kdic_path)

    def _get_output_dir(self, fallback_dir):
        if self.output_dir:
            return self.output_dir
        else:
            return fallback_dir

    def _cleanup_computation_dir(self, computation_dir):
        """Cleans up the computation dir according to the verbose mode"""
        if not self.verbose:
            _cleanup_dir(computation_dir)
        else:
            print(
                "khiops-python sklearn temporary files located at: " + computation_dir
            )

    def fit(self, X, y=None, **kwargs):
        """Fit the estimator

        Returns
        -------
        self : `KhiopsEstimator`
            The fitted estimator instance.
        """
        # Check for common sklearn parameters to comply with sklearn's check_estimator
        if "sample_weight" in kwargs:
            raise ValueError(
                f"{self.__class__.__name__} does not accept "
                "the 'sample_weight' parameter"
            )

        # Create temporary directory and tables
        computation_dir = self._create_computation_dir("fit")
        initial_runner_temp_dir = kh.get_runner().root_temp_dir
        kh.get_runner().root_temp_dir = computation_dir

        # Create the dataset, fit the model and reset in case of any failure
        try:
            categorical_target = kwargs.get("categorical_target", True)
            dataset = Dataset(X, y, categorical_target=categorical_target, key=self.key)
            self._fit(dataset, computation_dir, **kwargs)
        # Undefine any attributes to pass to "not fitted"
        except:
            self._undefine_estimator_attributes()
            raise
        # Cleanup and restore the runner's temporary dir
        finally:
            self._cleanup_computation_dir(computation_dir)
            kh.get_runner().root_temp_dir = initial_runner_temp_dir

        # If on "fitted" state then:
        # - self.model_ must be a DictionaryDomain
        # - self.model_report_ must be a KhiopsJSONObject
        assert not self.is_fitted_ or isinstance(self.model_, kh.DictionaryDomain)
        assert not self.is_fitted_ or isinstance(
            self.model_report_, kh.KhiopsJSONObject
        )

        return self

    def _fit(self, dataset, computation_dir, **kwargs):
        """Template pattern of a fit method

        Parameters
        ----------
        dataset : `Dataset`
            The learning dataset.
        computation_dir : str
            Path or URI where the Khiops computation results will be stored.

        The called methods are reimplemented in concrete sub-classes
        """
        # Check model parameters
        self._fit_check_params(dataset, **kwargs)

        # Check the dataset
        self._fit_check_dataset(dataset)

        # Train the model
        self._fit_train_model(dataset, computation_dir, **kwargs)
        self.n_features_in_ = dataset.main_table.n_features()

        # If the main attributes are of the proper type finish the fitting
        # Otherwise it means there was an abort (early return) of the previous steps
        if isinstance(self.model_, kh.DictionaryDomain) and isinstance(
            self.model_report_, kh.KhiopsJSONObject
        ):
            self._fit_training_post_process(dataset)
            self.is_fitted_ = True
            self.is_multitable_model_ = dataset.is_multitable()

    def _fit_check_params(self, dataset, **_):
        """Check the model parameters including those data dependent (in kwargs)"""
        if (
            self.key is not None
            and not is_list_like(self.key)
            and not isinstance(self.key, str)
        ):
            raise TypeError(type_error_message("key", self.key, str, "list-like"))

        if not dataset.is_in_memory() and self.output_dir is None:
            raise ValueError("'output_dir' is not set but dataset is file-based")

    def _fit_check_dataset(self, dataset):
        """Checks the pre-conditions of the tables to build the model"""
        if (
            dataset.main_table.n_samples is not None
            and dataset.main_table.n_samples <= 1
        ):
            raise ValueError(
                "Table contains one sample or less. It must contain at least 2."
            )

    @abstractmethod
    def _fit_train_model(self, dataset, computation_dir, **kwargs):
        """Builds the model with one or more calls to khiops.core.api

        It must return the path of the ``.kdic`` Khiops model file and the JSON report.
        """

    @abstractmethod
    def _fit_training_post_process(self, dataset):
        """Loads the model's data from Khiops files into the object"""

    def _transform(
        self,
        dataset,
        computation_dir,
        _transform_create_deployment_model_fun,
        drop_key,
    ):
        """Generic template method to implement transform, predict and predict_proba"""
        # Check if the model is fitted
        check_is_fitted(self)

        # Check if the dataset is consistent with the model
        self._transform_check_dataset(dataset)

        # Create a deployment dataset
        # Note: The input dataset is not necessarily ready to be deployed
        deployment_dataset = self._transform_create_deployment_dataset(
            dataset, computation_dir
        )

        # Create a deployment dictionary
        deployment_dictionary_domain = _transform_create_deployment_model_fun()

        # Deploy the model
        output_table_path = self._transform_deploy_model(
            deployment_dataset,
            deployment_dictionary_domain,
            self.model_main_dictionary_name_,
            computation_dir,
        )

        # Post-process to return the correct output type
        return self._transform_deployment_post_process(
            deployment_dataset, output_table_path, drop_key
        )

    def _transform_create_deployment_dataset(self, dataset, _):
        """Creates if necessary a new dataset to execute the model deployment

        The default behavior is to return the same dataset.
        """
        return dataset

    def _transform_deploy_model(
        self,
        deployment_dataset,
        model_dictionary_domain,
        model_dictionary_name,
        computation_dir,
    ):
        """Deploys a generic Khiops transformation model

        It allows to implement `predict`, `predict_proba` and `transform` methods in the
        sub-classes `KhiopsEncoder`, `KhiopsClassifier`, `KhiopsRegressor`.

        Parameters
        ----------
        X : :external:term:`array-like` of shape (n_samples, n_features_in) or dict
            Training dataset. Either an :external:term:`array-like` or a ``dict``
            specification for multi-table datasets (see :doc:`/multi_table_primer`).
            *Deprecated input modes* (will be removed in khiops-python 11):

            - tuple: A pair (``path_to_file``, ``separator``).
            - list: A sequence of dataframes or paths, or pairs path-separator. The
              first element of the list is the main table and the following are
              secondary ones joined to the main table using ``key`` estimator parameter

        root_name : str
            Name of root table in trained Khiops model .kdic

        Returns
        -------
        str
            The path of the table file containing the deployed model.

        """
        assert self._khiops_model_prefix is not None, "Model prefix is not set"

        # Create the table files to be used by Khiops
        (
            main_table_path,
            secondary_table_paths,
        ) = deployment_dataset.create_table_files_for_khiops(
            computation_dir, sort=self.auto_sort
        )

        # Build the 'additional_data_tables' argument
        secondary_data_paths = model_dictionary_domain.extract_data_paths(
            model_dictionary_name
        )
        additional_data_tables = {}
        for data_path in secondary_data_paths:
            dictionary = model_dictionary_domain.get_dictionary_at_data_path(data_path)
            assert dictionary.name.startswith(self._khiops_model_prefix), (
                f"Dictionary '{dictionary.name}' "
                f"does not have prefix '{self._khiops_model_prefix}'"
            )
            initial_dictionary_name = dictionary.name.replace(
                self._khiops_model_prefix, "", 1
            )

            additional_data_tables[data_path] = secondary_table_paths[
                initial_dictionary_name
            ]

        # Set output path files
        output_dir = self._get_output_dir(computation_dir)
        log_file_path = fs.get_child_path(output_dir, "khiops.log")
        output_data_table_path = fs.get_child_path(output_dir, "transformed.txt")

        # Set the format parameters depending on the type of dataset
        if deployment_dataset.is_in_memory():
            field_separator = "\t"
            header_line = True
        else:
            field_separator = deployment_dataset.main_table.sep
            header_line = deployment_dataset.main_table.header

        # Call to core function deploy_model
        kh.deploy_model(
            model_dictionary_domain,
            model_dictionary_name,
            main_table_path,
            output_data_table_path,
            additional_data_tables=additional_data_tables,
            detect_format=False,
            field_separator=field_separator,
            header_line=header_line,
            output_field_separator=field_separator,
            output_header_line=header_line,
            log_file_path=log_file_path,
            trace=self.verbose,
        )

        return output_data_table_path

    def _transform_check_dataset(self, dataset):
        """Checks the dataset before deploying a model on them"""
        if not dataset.is_in_memory() and self.output_dir is None:
            raise ValueError("'output_dir' is not set but dataset is file-based")

    def _transform_deployment_post_process(
        self, deployment_dataset, output_table_path, drop_key
    ):
        # Return a dataframe for dataframe based datasets
        if deployment_dataset.is_in_memory():
            # Read the transformed table with the internal table settings
            with io.BytesIO(fs.read(output_table_path)) as output_table_stream:
                output_table_df = read_internal_data_table(output_table_stream)

            # On multi-table:
            # - Reorder the table to the original table order
            #     - Because transformed data table file is sorted by key
            # - Drop the key columns if specified
            if deployment_dataset.is_multitable():
                key_df = deployment_dataset.main_table.dataframe[
                    deployment_dataset.main_table.key
                ]
                output_table_df_or_path = key_df.merge(
                    output_table_df, on=deployment_dataset.main_table.key
                )
                if drop_key:
                    output_table_df_or_path.drop(
                        deployment_dataset.main_table.key, axis=1, inplace=True
                    )
            # On mono-table: Return the read dataframe as-is
            else:
                output_table_df_or_path = output_table_df
        # Return a file path for file based datasets
        else:
            output_table_df_or_path = output_table_path

        assert isinstance(
            output_table_df_or_path, (str, pd.DataFrame)
        ), type_error_message(
            "output_table_df_or_path", output_table_df_or_path, str, pd.DataFrame
        )
        return output_table_df_or_path

    def _create_computation_dir(self, method_name):
        """Creates a temporary computation directory"""
        return kh.get_runner().create_temp_dir(
            prefix=f"{self.__class__.__name__}_{method_name}_"
        )


class KhiopsCoclustering(KhiopsEstimator, ClusterMixin):
    """A Khiops Coclustering model

    A coclustering is a non-supervised piecewise constant density estimator.

    Parameters
    ----------
    build_distance_vars : bool, default ``False``
        If ``True`` includes a cluster distance variable in the deployment
    build_frequency_vars : bool, default ``False``
        If ``True`` includes the frequency variables in the deployment.
    build_name_var : bool, default ``False``
        If ``True`` includes a cluster id variable in the deployment.
    verbose : bool, default ``False``
        If ``True`` it prints debug information and it does not erase temporary files
        when fitting, predicting or transforming.
    output_dir : str, optional
        Path of the output directory for the ``Coclustering.khcj`` report file and the
        ``Coclustering.kdic`` modeling dictionary file.
    auto_sort : bool, default ``True``
        *Advanced.* Only for multi-table inputs: If ``True`` input tables are
        automatically sorted by their key before executing Khiops. If the input
        tables are already sorted by their keys set this parameter to ``False``
        to speed up the processing. This affects the `predict` method.
        *Note* The sort by key is performed in a left-to-right, hierarchical,
        lexicographic manner.
    max_part_numbers : dict, optional
        Maximum number of clusters for each of the co-clustered column. Specifically, a
        key-value pair of this dictionary represents the column name and its respective
        maximum number of clusters. If not specified there is no maximun number of
        clusters is imposed on any column.
        **Deprecated** will be removed in khiops-python 11. Use the ``max_part_number``
        parameter of the `fit` method.
    variables : list of str, optional
        A list of column names/indexes to use in the coclustering.
        **Deprecated** will be removed in Khiops 11. Use the ``columns`` parameter of
        the `fit` method.
    key : str, optional
        *Multi-table only* : The name of the column to be used as key.
        **Deprecated** will be removed in Khiops 11. Use ``id_column`` parameter of
        the `fit` method.
    internal_sort : bool, optional
        *Advanced.* Only for multi-table inputs: If ``True`` input tables are
        automatically sorted by their key before executing Khiops. If the input
        tables are already sorted by their keys set this parameter to ``False``
        to speed up the processing. This affects the `predict` method.
        *Note* The sort by key is performed in a left-to-right, hierarchical,
        lexicographic manner.
        **Deprecated** will be removed in khiops-python 11. Use the ``auto_sort``
        parameter of the estimator instead.

    Attributes
    ----------
    is_fitted_ : bool
        ``True`` if the estimator is fitted.
    is_multitable_model_ : bool
        ``True`` if the model was fitted on a multi-table dataset.
    model_ : `.DictionaryDomain`
        The Khiops dictionary domain for the trained coclustering. For coclustering it
        is a multi-table dictionary even though the model is single-table.
    model_main_dictionary_name_ : str
        The name of the main Khiops dictionary within the ``model_`` domain.
    model_report_ : `.CoclusteringResults`
        The Khiops report object.
    model_report_raw_ : dict
        JSON object of the Khiops report.
        **Deprecated** will be removed in khiops-python 11. Use the ``json_data``
        attribute of the ``model_report_`` estimator attribute instead.

    Examples
    --------
    See the following functions of the ``samples_sklearn.py`` documentation script:
        - `samples_sklearn.khiops_coclustering()`
    """

    def __init__(
        self,
        verbose=False,
        output_dir=None,
        auto_sort=True,
        build_name_var=True,
        build_distance_vars=False,
        build_frequency_vars=False,
        max_part_numbers=None,
        key=None,
        variables=None,
        internal_sort=None,
    ):
        super().__init__(
            key=key,
            verbose=verbose,
            output_dir=output_dir,
            auto_sort=auto_sort,
            internal_sort=internal_sort,
        )
        self._khiops_model_prefix = "CC_"
        self.build_name_var = build_name_var
        self.build_distance_vars = build_distance_vars
        self.build_frequency_vars = build_frequency_vars
        self.variables = variables
        self.max_part_numbers = max_part_numbers
        self.model_id_column = None

        # Deprecation message for 'key' and 'variables' constructor parameter
        if key is not None:
            warnings.warn(
                deprecation_message(
                    "'key' estimator parameter",
                    "11.0.0",
                    replacement="'id_column' parameter of the 'fit' method",
                    quote=False,
                )
            )
        if variables is not None:
            warnings.warn(
                deprecation_message(
                    "'variables' estimator parameter",
                    "11.0.0",
                    replacement="'columns' parameter of the 'fit' method",
                    quote=False,
                )
            )
        if max_part_numbers is not None:
            warnings.warn(
                deprecation_message(
                    "'max_part_numbers' estimator parameter",
                    "11.0.0",
                    replacement="'max_part_numbers' parameter of the 'simplify' method",
                    quote=False,
                )
            )

    def fit(self, X, y=None, **kwargs):
        """Trains a Khiops Coclustering model

        Parameters
        ----------
        X : :external:term:`array-like` of shape (n_samples, n_features_in) or dict
            Training dataset. Either an :external:term:`array-like` or a ``dict``
            specification for multi-table datasets (see :doc:`/multi_table_primer`).
            *Deprecated input modes* (will be removed in khiops-python 11):

            - tuple: A pair (``path_to_file``, ``separator``).
            - list: A sequence of dataframes or paths, or pairs path-separator. The
              first element of the list is the main table and the following are
              secondary ones joined to the main table using ``key`` estimator parameter.

        id_column : str
            The column that contains the id of the instance.
        columns : list, optional
            The columns to be co-clustered. If not specified it uses all columns.
        max_part_numbers : dict, optional
            Maximum number of clusters for each of the co-clustered column.
            Specifically, a key-value pair of this dictionary represents the column name
            and its respective maximum number of clusters. If not specified, then no
            maximum number of clusters is imposed on any column.
            **Deprecated** (will be removed in khiops-python 11). Use the ``simplify``
            method instead.

        Returns
        -------
        self : `KhiopsCoclustering`
            The calling estimator instance.
        """
        return super().fit(X, y=y, **kwargs)

    def _fit_check_params(self, dataset, **kwargs):
        # Check that at least one of the build methods parameters is set
        if not (
            self.build_name_var or self.build_distance_vars or self.build_frequency_vars
        ):
            raise ValueError(
                "One of the parameters 'build_name_var', "
                "'build_distance_vars' or 'build_frequency_vars' must be ``True``"
            )

        # If 'columns' specified check that:
        # - Is a sequence of string
        # - Is contained in the columns names of the main table
        columns = kwargs.get("columns", self.variables)
        if columns is not None:
            if not is_list_like(columns):
                raise TypeError(type_error_message("columns", columns, "list-like"))
            else:
                for i, column_id in enumerate(columns):
                    if not isinstance(column_id, (str, int)):
                        raise TypeError(
                            type_error_message(f"columns[{i}]", column_id, str)
                        )
                    if column_id not in dataset.main_table.column_ids:
                        raise ValueError(f"columns[{i}] ('{column_id}') not found in X")

        # Check that 'id_column':
        # - Is specified
        # - Is a string
        # - Is contained in the columns names of the main table
        id_column = kwargs.get("id_column", self.key)
        if id_column is None:
            raise ValueError("'id_column' is a mandatory parameter")
        if not isinstance(id_column, str):
            raise TypeError(type_error_message("key_columns", id_column, str))
        if id_column not in dataset.main_table.column_ids:
            raise ValueError(f"id column '{id_column}' not found in X")

        # Deprecate the 'max_part_numbers' parameter
        max_part_numbers = kwargs.get("max_part_numbers", self.max_part_numbers)
        if max_part_numbers is not None:
            warnings.warn(
                deprecation_message(
                    "'max_part_numbers' 'fit' parameter",
                    "11.0.0",
                    replacement="'max_part_numbers' parameter of the 'simplify' method",
                    quote=False,
                )
            )

    def _fit_train_model(self, dataset, computation_dir, **kwargs):
        assert not dataset.is_multitable(), "Coclustering not available in multitable"

        # Prepare the table files and dictionary for Khiops
        main_table_path, _ = dataset.create_table_files_for_khiops(
            computation_dir, sort=self.auto_sort
        )

        # Set the output paths
        output_dir = self._get_output_dir(computation_dir)
        train_log_file_path = fs.get_child_path(output_dir, "khiops_train_cc.log")

        # Set the 'variables' parameter
        if "columns" in kwargs:
            variables = kwargs["columns"]
        elif self.variables is not None:
            variables = self.variables
        else:
            variables = list(dataset.main_table.column_ids)

        # Train the coclustering model
        coclustering_file_path = kh.train_coclustering(
            dataset.create_khiops_dictionary_domain(),
            dataset.main_table.name,
            main_table_path,
            variables,
            output_dir,
            log_file_path=train_log_file_path,
            trace=self.verbose,
        )

        # Search "No coclustering found" message in the log, warn the user and return
        no_cc_message = "No informative coclustering found in data"
        with io.TextIOWrapper(
            io.BytesIO(fs.read(train_log_file_path)), encoding="utf8", errors="replace"
        ) as train_log_file:
            for line in train_log_file:
                if line.startswith(no_cc_message):
                    warnings.warn(
                        f"{no_cc_message}. Estimator not fitted.", stacklevel=5
                    )
                    return

        # Save the report file
        self.model_report_ = kh.read_coclustering_results_file(coclustering_file_path)
        self.model_report_raw_ = self.model_report_.json_data

        # Save the id column
        if "id_column" in kwargs:
            self.model_id_column = kwargs["id_column"]
        else:
            self.model_id_column = self.key

        # Check that the id column was clustered
        try:
            self.model_report_.coclustering_report.get_dimension(self.model_id_column)
        except KeyError:
            warnings.warn(
                f"Coclustering did not cluster the id column '{self.model_id_column}'. "
                "The report is available but the estimator is not fitted"
            )
            return

        # Create a multi-table dictionary from the schema of the table
        # The root table contains the key of the table and points to the main table
        tmp_domain = dataset.create_khiops_dictionary_domain()
        main_table_dictionary = tmp_domain.get_dictionary(dataset.main_table.name)
        if not main_table_dictionary.key:
            main_table_dictionary.key = [self.model_id_column]
        main_table_dictionary.name = (
            f"{self._khiops_model_prefix}{dataset.main_table.name}"
        )
        self.model_main_dictionary_name_ = (
            f"{self._khiops_model_prefix}Keys_{dataset.main_table.name}"
        )
        self.model_secondary_table_variable_name = (
            f"{self._khiops_model_prefix}{dataset.main_table.name}"
        )
        self._create_coclustering_model_domain(
            tmp_domain, coclustering_file_path, output_dir
        )

        # Update the `model_` attribute of the coclustering estimator to the
        # new coclustering model
        self.model_ = kh.read_dictionary_file(
            fs.get_child_path(output_dir, "Coclustering.kdic")
        )

        # If the deprecated `max_part_numbers` is not None, then call `simplify`
        max_part_numbers = kwargs.get("max_part_numbers", self.max_part_numbers)
        if max_part_numbers is not None:
            # Get simplified estimator
            simplified_cc = self._simplify(max_part_numbers=max_part_numbers)

            # Update main estimator model and report according to the simplified model
            self.model_ = simplified_cc.model_
            self.model_report_ = simplified_cc.model_report_
            self.model_report_raw_ = self.model_report_.json_data

    def _fit_training_post_process(self, dataset):
        assert (
            len(self.model_.dictionaries) == 2
        ), "'model_' does not have exactly 2 dictionaries"

        # Set the main dictionary as Root
        self._get_main_dictionary().root = True

    def _create_coclustering_model_domain(
        self, domain, coclustering_file_path, output_dir
    ):
        """Postprocess the coclustering model

        To this end:
        - build multi-table dictionary domain from input (dataset-specific)
          domain.
        - prepare the coclustering model for deployment, by adding the
          coclustering variables to the the root (multi-table) dictionary of the
          multi-table dictionary domain.

        Parameters
        ----------
        domain : `.DictionaryDomain`
            Input dictionary domain reflecting the structure of the input dataset.
        coclustering_file_path : str
            Path to the coclustering report file.
        output_dir : str
            Path to the output directory, where the deployed model will be
            written on disk.
        """
        # Check for potential programming errors
        assert isinstance(domain, DictionaryDomain)
        assert isinstance(coclustering_file_path, str)
        assert isinstance(output_dir, str)

        # Build multi-table dictionary domain out of the input domain
        mt_domain = build_multi_table_dictionary_domain(
            domain,
            self.model_main_dictionary_name_,
            self.model_secondary_table_variable_name,
        )

        # Create the model by adding the coclustering variables
        # to the multi-table dictionary created before
        prepare_log_file_path = fs.get_child_path(output_dir, "khiops_prepare_cc.log")
        kh.prepare_coclustering_deployment(
            mt_domain,
            self.model_main_dictionary_name_,
            coclustering_file_path,
            self.model_secondary_table_variable_name,
            self.model_id_column,
            output_dir,
            build_cluster_variable=self.build_name_var,
            build_distance_variables=self.build_distance_vars,
            build_frequency_variables=self.build_frequency_vars,
            log_file_path=prepare_log_file_path,
            trace=self.verbose,
        )

    def _simplify(
        self,
        max_preserved_information=0,
        max_cells=0,
        max_total_parts=0,
        max_part_numbers=None,
    ):
        """Simplifies a Khiops coclustering model

        Does *not* check that the estimator is fitted

        Parameters
        ----------
        max_preserved_information : int, default 0
            Maximum information preserve in the simplified coclustering. If equal to 0
            there is no limit.
        max_cells : int, default 0
            Maximum number of cells in the simplified coclustering. If equal to 0 there
            is no limit.
        max_total_parts : int, default 0
            Maximum number of parts totaled over all variables. If equal to 0 there
            is no limit.
        max_part_numbers : dict, optional
            Maximum number of clusters for each of the co-clustered column.
            Specifically, a key-value pair of this dictionary represents the column name
            and its respective maximum number of clusters. If not specified, then no
            maximum number of clusters is imposed on any column.

        Returns
        -------
        self : `KhiopsCoclustering`
            A *new*, simplified `.KhiopsCoclustering` estimator instance.
        """
        # Check parameters: types and authorized value ranges
        assert hasattr(self, "model_report_")
        assert hasattr(self, "model_")
        if not isinstance(max_cells, int):
            raise TypeError(type_error_message("max_cells", max_cells, int))
        elif max_cells < 0:
            raise ValueError("'max_cells' must be greater than 0")

        if not isinstance(max_preserved_information, int):
            raise TypeError(
                type_error_message(
                    "max_preserved_information", max_preserved_information, int
                )
            )
        elif max_preserved_information < 0:
            raise ValueError("'max_preserved_information' must be greater than 0")
        if not isinstance(max_total_parts, int):
            raise TypeError(type_error_message("max_total_parts", max_total_parts, int))
        elif max_total_parts < 0:
            raise ValueError("'max_total_parts' must be greater than 0")
        if max_part_numbers is not None and not is_dict_like(max_part_numbers):
            raise TypeError(
                type_error_message("max_part_numbers", max_part_numbers, "dict-like")
            )
        if max_part_numbers is not None:
            for key in max_part_numbers.keys():
                if not isinstance(key, str):
                    raise TypeError(
                        type_error_message("'max_part_numbers' keys", key, str)
                    )

            for value in max_part_numbers.values():
                if not isinstance(value, int):
                    raise TypeError(
                        type_error_message("'max_part_numbers' values", value, int)
                    )
                elif value < 0:
                    raise ValueError("'max_part_numbers' values must be positive")
        # Create temporary directory and tables
        computation_dir = self._create_computation_dir("simplify")
        output_dir = self._get_output_dir(computation_dir)
        simplify_log_file_path = fs.get_child_path(output_dir, "khiops_simplify_cc.log")
        initial_runner_temp_dir = kh.get_runner().root_temp_dir
        full_coclustering_file_path = fs.get_child_path(
            output_dir, "FullCoclustering.khcj"
        )
        self.model_report_.write_khiops_json_file(full_coclustering_file_path)
        kh.get_runner().root_temp_dir = computation_dir
        try:
            # - simplify_coclustering, then
            # - prepare_coclustering_deployment
            # - prepare coclustering deployment and re-initialise the `model_`
            #   attribute accordingly
            kh.simplify_coclustering(
                full_coclustering_file_path,
                "Coclustering.khc",
                output_dir,
                max_preserved_information=max_preserved_information,
                max_cells=max_cells,
                max_total_parts=max_total_parts,
                max_part_numbers=max_part_numbers,
                log_file_path=simplify_log_file_path,
                trace=self.verbose,
            )

            # Get dataset dictionary from model; it should not be root
            dataset_dictionary = self.model_.get_dictionary(
                self.model_secondary_table_variable_name
            )
            assert (
                not dataset_dictionary.root
            ), "Dataset dictionary in the coclustering model should not be root"
            if not dataset_dictionary.key:
                dataset_dictionary.key = self.model_id_column
            domain = DictionaryDomain()
            domain.add_dictionary(dataset_dictionary)
            simplified_coclustering_file_path = fs.get_child_path(
                output_dir, "Coclustering.khcj"
            )

            # Create new (simplified) coclustering estimator
            simplified_cc = KhiopsCoclustering()

            # Set its parameters according to the original estimator
            simplified_cc.set_params(**self.get_params())

            # Copy relevant attributes
            # Note: do not copy `model_*` attributes, that get rebuilt anyway
            for attribute_name in (
                "is_fitted_",
                "is_multitable_model_",
                "model_main_dictionary_name_",
                "model_id_column",
            ):
                if hasattr(self, attribute_name):
                    setattr(
                        simplified_cc, attribute_name, getattr(self, attribute_name)
                    )

            # Set the coclustering report according to the simplification
            simplified_cc.model_report_ = kh.read_coclustering_results_file(
                simplified_coclustering_file_path
            )
            simplified_cc.model_report_raw_ = simplified_cc.model_report_.json_data

            # Build the individual-variable coclustering model
            self._create_coclustering_model_domain(
                domain, simplified_coclustering_file_path, output_dir
            )

            # Set the `model_` attribute of the new coclustering estimator to
            # the new coclustering model
            simplified_cc.model_ = kh.read_dictionary_file(
                fs.get_child_path(output_dir, "Coclustering.kdic")
            )
        finally:
            self._cleanup_computation_dir(computation_dir)
            kh.get_runner().root_temp_dir = initial_runner_temp_dir
        return simplified_cc

    def simplify(
        self,
        max_preserved_information=0,
        max_cells=0,
        max_total_parts=0,
        max_part_numbers=None,
    ):
        """Creates a simplified coclustering model from the current instance

        Parameters
        ----------
        max_preserved_information : int, default 0
            Maximum information preserve in the simplified coclustering. If equal to 0
            there is no limit.
        max_cells : int, default 0
            Maximum number of cells in the simplified coclustering. If equal to 0 there
            is no limit.
        max_total_parts : int, default 0
            Maximum number of parts totaled over all variables. If equal to 0 there
            is no limit.
        max_part_numbers : dict, optional
            Maximum number of clusters for each of the co-clustered column.
            Specifically, a key-value pair of this dictionary represents the column name
            and its respective maximum number of clusters. If not specified, then no
            maximum number of clusters is imposed on any column.

        Returns
        -------
        self : `KhiopsCoclustering`
            A *new*, simplified `.KhiopsCoclustering` estimator instance.
        """
        # Check that the estimator is fitted:
        if not self.is_fitted_:
            raise ValueError("Only fitted coclustering estimators can be simplified")

        return self._simplify(
            max_preserved_information=max_preserved_information,
            max_cells=max_cells,
            max_total_parts=max_total_parts,
            max_part_numbers=max_part_numbers,
        )

    def predict(self, X):
        """Predicts the most probable cluster for the test dataset X

        Parameters
        ----------
        X : :external:term:`array-like` of shape (n_samples, n_features_in) or dict
            Training dataset. Either an :external:term:`array-like` or a ``dict``
            specification for multi-table datasets (see :doc:`/multi_table_primer`).
            *Deprecated input modes* (will be removed in khiops-python 11):

            - tuple: A pair (``path_to_file``, ``separator``).
            - list: A sequence of dataframes or paths, or pairs path-separator. The
              first element of the list is the main table and the following are
              secondary ones joined to the main table using ``key`` estimator parameter.

        Returns
        -------
        `numpy.ndarray`
            An array containing the encoded columns. A first column containing key
            column ids is added in multi-table mode.

            *Deprecated return values* (will be removed in khiops-python 11): str for
            file based dataset specification.
        """
        # Create temporary directory
        computation_dir = self._create_computation_dir("predict")
        initial_runner_temp_dir = kh.get_runner().root_temp_dir
        kh.get_runner().root_temp_dir = computation_dir

        # Create the input dataset
        dataset = Dataset(X)

        # Call the template transform method
        try:
            y_pred = super()._transform(
                dataset,
                computation_dir,
                self._transform_prepare_deployment_model_for_predict,
                False,
            )
        # Cleanup and restore the runner's temporary dir
        finally:
            self._cleanup_computation_dir(computation_dir)
            kh.get_runner().root_temp_dir = initial_runner_temp_dir

        # Transform to numpy.array for in-memory inputs
        if dataset.is_in_memory():
            y_pred = y_pred.to_numpy()

        return y_pred

    def _transform_check_dataset(self, dataset):
        """Checks the tables before deploying a model on them"""
        assert (
            len(self.model_.dictionaries) == 2
        ), "'model' does not have exactly 2 dictionaries"

        # Call the parent method
        super()._transform_check_dataset(dataset)

        # Coclustering models are special:
        # - They are mono-table only
        # - They are deployed with a multitable model whose main table contain
        #   the keys of the input table and the secondary table is the input table
        if dataset.is_multitable():
            raise ValueError("Coclustering models not available in multi-table mode")

        # The "model dictionary domain" in the coclustering case it is just composed
        # of the secondary table. The main "keys" table is a technical object.
        # So we check the compatibility against only this dictionary and override
        # the parents implementaion
        for dictionary in self.model_.dictionaries:
            if dictionary.name != self.model_main_dictionary_name_:
                _check_dictionary_compatibility(
                    dictionary,
                    dataset.main_table.create_khiops_dictionary(),
                    self.__class__.__name__,
                )

    def _transform_create_deployment_dataset(self, dataset, computation_dir):
        assert not dataset.is_multitable(), "'dataset' is multitable"

        # Build the multitable deployment dataset
        keys_table_name = f"keys_{dataset.main_table.name}"
        deploy_dataset_spec = {}
        deploy_dataset_spec["main_table"] = keys_table_name
        deploy_dataset_spec["tables"] = {}
        if dataset.is_in_memory():
            # Extract the keys from the main table
            keys_table_dataframe = pd.DataFrame(
                {
                    self.model_id_column: dataset.main_table.dataframe[
                        self.model_id_column
                    ].unique()
                }
            )

            # Create the dataset with the keys table as the main one
            deploy_dataset_spec["tables"][keys_table_name] = (
                keys_table_dataframe,
                self.model_id_column,
            )
            deploy_dataset_spec["tables"][dataset.main_table.name] = (
                dataset.main_table.dataframe,
                self.model_id_column,
            )
        else:
            # Create the table to extract the keys (sorted)
            keyed_dataset = dataset.copy()
            keyed_dataset.main_table.key = [self.model_id_column]
            main_table_path = keyed_dataset.main_table.create_table_file_for_khiops(
                computation_dir, sort=self.auto_sort
            )

            # Create a table storing the main table keys
            keys_table_name = f"keys_{dataset.main_table.name}"
            keys_table_file_path = fs.get_child_path(
                computation_dir, f"raw_{keys_table_name}.txt"
            )
            kh.extract_keys_from_data_table(
                keyed_dataset.create_khiops_dictionary_domain(),
                keyed_dataset.main_table.name,
                main_table_path,
                keys_table_file_path,
                header_line=dataset.header,
                field_separator=dataset.sep,
                output_header_line=dataset.header,
                output_field_separator=dataset.sep,
                trace=self.verbose,
            )
            deploy_dataset_spec["tables"][keys_table_name] = (
                keys_table_file_path,
                self.model_id_column,
            )
            deploy_dataset_spec["tables"][dataset.main_table.name] = (
                dataset.main_table.path,
                self.model_id_column,
            )
            deploy_dataset_spec["format"] = (dataset.sep, dataset.header)

        return Dataset(deploy_dataset_spec)

    def _transform_prepare_deployment_model_for_predict(self):
        return self.model_

    def _transform_deployment_post_process(
        self, deployment_dataset, output_table_path, drop_key
    ):
        assert deployment_dataset.is_multitable()
        return super()._transform_deployment_post_process(
            deployment_dataset, output_table_path, drop_key
        )

    def fit_predict(self, X, y=None, **kwargs):
        """Performs clustering on X and returns result (instead of labels)"""
        return self.fit(X, y, **kwargs).predict(X)


class KhiopsSupervisedEstimator(KhiopsEstimator):
    """Abstract Khiops Supervised Estimator"""

    def __init__(
        self,
        n_features=100,
        n_pairs=0,
        n_trees=10,
        verbose=False,
        output_dir=None,
        auto_sort=True,
        key=None,
        internal_sort=None,
    ):
        super().__init__(
            key=key,
            verbose=verbose,
            output_dir=output_dir,
            auto_sort=auto_sort,
            internal_sort=internal_sort,
        )
        self.n_features = n_features
        self.n_pairs = n_pairs
        self.n_trees = n_trees
        self._predicted_target_meta_data_tag = None

        # Deprecation message for 'key' constructor parameter
        if key is not None:
            warnings.warn(
                deprecation_message(
                    "'key' estimator parameter",
                    "11.0.0",
                    replacement="dict dataset input",
                    quote=False,
                )
            )

    def _more_tags(self):
        return {"require_y": True}

    def _fit_check_dataset(self, dataset):
        super()._fit_check_dataset(dataset)
        self._check_target_type(dataset)

    @abstractmethod
    def _check_target_type(self, dataset):
        """Checks that the target type has the correct type for the estimator"""

    def fit(self, X, y=None, **kwargs):
        """Fits a supervised estimator according to X,y

        Called by the concrete sub-classes `KhiopsEncoder`, `KhiopsClassifier`,
        `KhiopsRegressor`.

        Parameters
        ----------
        X : :external:term:`array-like` of shape (n_samples, n_features_in) or dict
            Training dataset. Either an :external:term:`array-like` or a ``dict``
            specification for multi-table datasets (see :doc:`/multi_table_primer`).
            *Deprecated input modes* (will be removed in khiops-python 11):

            - tuple: A pair (``path_to_file``, ``separator``).
            - list: A sequence of dataframes or paths, or pairs path-separator. The
              first element of the list is the main table and the following are
              secondary ones joined to the main table using ``key`` estimator parameter.

        y : :external:term:`array-like` of shape (n_samples,)
            :external:term:`array-like` object containing the target values.

            **Deprecated input modes** (will be removed in khiops-python 11):
                - str: A path to a data table file for file-based ``dict`` dataset
                  specifications.

        Returns
        -------
        self : `KhiopsSupervisedEstimator`
            The calling estimator instance.
        """
        if y is None:
            raise ValueError(
                "'y' must be specified for fitting "
                f"{self.__class__.__name__} estimator."
            )
        super().fit(X, y=y, **kwargs)
        return self

    def _fit_check_params(self, dataset, **kwargs):
        # Call parent method
        super()._fit_check_params(dataset, **kwargs)

        # Check supervised estimator parameters
        if not isinstance(self.n_features, int):
            raise TypeError(type_error_message("n_features", self.n_features, int))
        if self.n_features < 0:
            raise ValueError("'n_features' must be positive")
        if not isinstance(self.n_trees, int):
            raise TypeError(type_error_message("n_trees", self.n_trees, int))
        if self.n_trees < 0:
            raise ValueError("'n_trees' must be positive")
        if not isinstance(self.n_pairs, int):
            raise TypeError(type_error_message("n_pairs", self.n_pairs, int))
        if self.n_pairs < 0:
            raise ValueError("'n_pairs' must be positive")

    def _fit_train_model(self, dataset, computation_dir, **kwargs):
        # Train the model with Khiops
        train_args, train_kwargs = self._fit_prepare_training_function_inputs(
            dataset, computation_dir
        )
        report_file_path, model_kdic_file_path = self._fit_core_training_function(
            *train_args, **train_kwargs
        )

        # Abort if the model dictionary file does not exist
        if not fs.exists(model_kdic_file_path):
            warnings.warn(
                "Khiops dictionary model not found. Model not fitted", stacklevel=6
            )
            return

        # Save the model domain object and report
        self.model_ = kh.read_dictionary_file(model_kdic_file_path)
        self.model_report_ = kh.read_analysis_results_file(report_file_path)
        self.model_report_raw_ = self.model_report_.json_data

    @abstractmethod
    def _fit_core_training_function(self, *args, **kwargs):
        """A wrapper to the khiops.core training function for the estimator"""

    def _fit_prepare_training_function_inputs(self, dataset, computation_dir):
        # Set output path files
        output_dir = self._get_output_dir(computation_dir)
        log_file_path = fs.get_child_path(output_dir, "khiops.log")

        main_table_path, secondary_table_paths = dataset.create_table_files_for_khiops(
            computation_dir, sort=self.auto_sort
        )

        # Build the 'additional_data_tables' argument
        dataset_domain = dataset.create_khiops_dictionary_domain()
        secondary_data_paths = dataset_domain.extract_data_paths(
            dataset.main_table.name
        )
        additional_data_tables = {}
        for data_path in secondary_data_paths:
            dictionary = dataset_domain.get_dictionary_at_data_path(data_path)
            additional_data_tables[data_path] = secondary_table_paths[dictionary.name]

        # Build the mandatory arguments
        args = [
            dataset.create_khiops_dictionary_domain(),
            dataset.main_table.name,
            main_table_path,
            dataset.main_table.get_khiops_variable_name(
                dataset.main_table.target_column_id
            ),
            output_dir,
        ]

        # Build the optional parameters from a copy of the estimator parameters
        kwargs = self.get_params()

        # Remove 'key' and 'output_dir'
        del kwargs["key"]
        del kwargs["output_dir"]

        # Set the sampling percentage to a 100%
        kwargs["sample_percentage"] = 100

        # Set the format parameters depending on the type of dataset
        kwargs["detect_format"] = False
        if dataset.is_in_memory():
            kwargs["field_separator"] = "\t"
            kwargs["header_line"] = True
        else:
            kwargs["field_separator"] = dataset.main_table.sep
            kwargs["header_line"] = dataset.main_table.header

        # Rename parameters to be compatible with khiops.core
        kwargs["max_constructed_variables"] = kwargs.pop("n_features")
        kwargs["max_pairs"] = kwargs.pop("n_pairs")
        kwargs["max_trees"] = kwargs.pop("n_trees")

        # Add the additional_data_tables parameter
        kwargs["additional_data_tables"] = additional_data_tables

        # Set the log file and trace parameters
        kwargs["log_file_path"] = log_file_path
        kwargs["trace"] = kwargs["verbose"]
        del kwargs["verbose"]

        return args, kwargs

    def _fit_training_post_process(self, dataset):
        # Call parent method
        super()._fit_training_post_process(dataset)

        # Set the target variable name
        self.model_target_variable_name_ = dataset.main_table.get_khiops_variable_name(
            dataset.main_table.target_column_id
        )

        # Verify it has at least one dictionary and a root dictionary in multi-table
        if len(self.model_.dictionaries) == 1:
            self.model_main_dictionary_name_ = self.model_.dictionaries[0].name
        else:
            for dictionary in self.model_.dictionaries:
                assert dictionary.name.startswith(self._khiops_model_prefix), (
                    f"Dictionary '{dictionary.name}' "
                    f"does not have prefix '{self._khiops_model_prefix}'"
                )
                initial_dictionary_name = dictionary.name.replace(
                    self._khiops_model_prefix, "", 1
                )
                if initial_dictionary_name == dataset.main_table.name:
                    self.model_main_dictionary_name_ = dictionary.name
        if self.model_main_dictionary_name_ is None:
            raise ValueError("No model dictionary after Khiops call")

        # Remove the target variable in the model dictionary
        model_main_dictionary = self.model_.get_dictionary(
            self.model_main_dictionary_name_
        )
        model_main_dictionary.remove_variable(self.model_target_variable_name_)

    def _transform_check_dataset(self, dataset):
        assert isinstance(dataset, Dataset), "'dataset' is not 'Dataset'"

        # Call the parent method
        super()._transform_check_dataset(dataset)

        # Check the coherence between thi input table and the model
        if self.is_multitable_model_ and not dataset.is_multitable():
            raise ValueError(
                "You are trying to apply on single-table inputs a model which has "
                "been trained on multi-table data."
            )
        if not self.is_multitable_model_ and dataset.is_multitable():
            raise ValueError(
                "You are trying to apply on multi-table inputs a model which has "
                "been trained on single-table data."
            )

        # Error if different number of dictionaries
        dataset_domain = dataset.create_khiops_dictionary_domain()
        if len(self.model_.dictionaries) != len(dataset_domain.dictionaries):
            raise ValueError(
                f"X has {len(dataset_domain.dictionaries)} table(s), "
                f"but {self.__class__.__name__} is expecting "
                f"{len(self.model_.dictionaries)}"
            )

        # Check the main table compatibility
        # Note: Name checking is omitted for the main table
        _check_dictionary_compatibility(
            _extract_basic_dictionary(self._get_main_dictionary()),
            dataset.main_table.create_khiops_dictionary(),
            self.__class__.__name__,
        )

        # Multi-table model: Check name and dictionary coherence of secondary tables
        dataset_secondary_tables_by_name = {
            table.name: table for table in dataset.secondary_tables
        }
        for dictionary in self.model_.dictionaries:
            assert dictionary.name.startswith(self._khiops_model_prefix), (
                f"Dictionary '{dictionary.name}' "
                f"does not have prefix '{self._khiops_model_prefix}'"
            )

            if dictionary.name != self.model_main_dictionary_name_:
                initial_dictionary_name = dictionary.name.replace(
                    self._khiops_model_prefix, "", 1
                )
                if initial_dictionary_name not in dataset_secondary_tables_by_name:
                    raise ValueError(
                        f"X does not contain table {initial_dictionary_name} "
                        f"but {self.__class__.__name__} is expecting it"
                    )
                _check_dictionary_compatibility(
                    _extract_basic_dictionary(dictionary),
                    dataset_secondary_tables_by_name[
                        initial_dictionary_name
                    ].create_khiops_dictionary(),
                    self.__class__.__name__,
                )


class KhiopsPredictor(KhiopsSupervisedEstimator):
    """Abstract Khiops Selective Naive Bayes Predictor"""

    def __init__(
        self,
        n_features=100,
        n_pairs=0,
        n_trees=10,
        verbose=False,
        output_dir=None,
        auto_sort=True,
        key=None,
        internal_sort=None,
    ):
        super().__init__(
            n_features=n_features,
            n_pairs=n_pairs,
            n_trees=n_trees,
            verbose=verbose,
            output_dir=output_dir,
            auto_sort=auto_sort,
            key=key,
            internal_sort=internal_sort,
        )
        # Data to be specified by inherited classes
        self._predicted_target_meta_data_tag = None

    def predict(self, X):
        """Predicts the target variable for the test dataset X

        See the documentation of concrete subclasses for more details.
        """
        # Create temporary directory
        computation_dir = self._create_computation_dir("predict")
        initial_runner_temp_dir = kh.get_runner().root_temp_dir
        kh.get_runner().root_temp_dir = computation_dir

        try:
            # Create the input dataset
            dataset = Dataset(X, key=self.key)

            # Call the template transform method
            y_pred = super()._transform(
                dataset,
                computation_dir,
                self._transform_prepare_deployment_model_for_predict,
                True,
            )
        # Cleanup and restore the runner's temporary dir
        finally:
            self._cleanup_computation_dir(computation_dir)
            kh.get_runner().root_temp_dir = initial_runner_temp_dir

        # Restore the runner's temporary dir
        kh.get_runner().root_temp_dir = initial_runner_temp_dir

        # Return pd.Series in the monotable + pandas case
        assert isinstance(y_pred, (str, pd.DataFrame)), "Expected str or DataFrame"
        return y_pred

    def _transform_prepare_deployment_model_for_predict(self):
        assert (
            self._predicted_target_meta_data_tag is not None
        ), "Predicted target metadata tag is not set"

        # Create a copy of the model dictionary using only the predicted target
        # Also activate the key to reorder the output in the multitable case
        model_copy = self.model_.copy()
        model_dictionary = model_copy.get_dictionary(self.model_main_dictionary_name_)
        for variable in model_dictionary.variables:
            if variable.name in model_dictionary.key:
                variable.used = True
            elif self._predicted_target_meta_data_tag in variable.meta_data:
                variable.used = True
            else:
                variable.used = False
        return model_copy


class KhiopsClassifier(KhiopsPredictor, ClassifierMixin):
    """Khiops Selective Naive Bayes Classifier

    This classifier supports automatic feature engineering on multi-table datasets. See
    :doc:`/multi_table_primer` for more details.

    .. note::

        Visit `the Khiops site <https://khiops.org/learn/understand>`_ to learn
        abouth the automatic feature engineering algorithm.

    Parameters
    ----------
    n_features : int, default 100
        *Multi-table only* : Maximum number of multi-table aggregate features to
        construct. See :doc:`/multi_table_primer` for more details.
    n_pairs : int, default 0
        Maximum number of pair features to construct. These features represent a 2D grid
        partition of the domain of a pair of variables in which is optimized in a way
        that the cells are the purest possible with respect to the target. Only pairs
        which jointly are more informative that its univariate components may be taken
        into account in the classifier.
    n_trees : int, default 10
        Maximum number of decision tree features to construct. The constructed trees
        combine other features, either native or constructed. These features usually
        improve the classifier's performance at the cost of interpretability of the
        model.
    verbose : bool, default ``False``
        If ``True`` it prints debug information and it does not erase temporary files
        when fitting, predicting or transforming.
    output_dir : str, optional
        Path of the output directory for the ``AllReports.khj`` report file and the
        ``Modeling.kdic`` modeling dictionary file. By default these files are deleted.
    auto_sort : bool, default ``True``
        *Advanced.* Only for multi-table inputs: If ``True`` input tables are pre-sorted
        by their key before executing Khiops. If the input tables are already sorted by
        their keys set this parameter to ``False`` to speed up the processing. This
        affects the `fit`, `predict` and `predict_proba` methods.
        *Note* The sort by key is performed in a left-to-right, hierarchical,
        lexicographic manner.
    key : str, optional
        *Multi-table only* : The name of the column to be used as key.
        **Deprecated** will be removed in khiops-python 11. Use ``dict`` dataset
        specifications in ``fit``, ``fit_predict``, ``predict`` and ``predict_proba``.
    internal_sort : bool, optional
        *Advanced.* Only for multi-table inputs: If ``True`` input tables are pre-sorted
        by their key before executing Khiops. If the input tables are already sorted by
        their keys set this parameter to ``False`` to speed up the processing. This
        affects the `fit`, `predict` and `predict_proba` methods.
        *Note* The sort by key is performed in a left-to-right, hierarchical,
        lexicographic manner.
        **Deprecated** will be removed in khiops-python 11. Use the ``auto_sort``
        estimator parameter instead.

    Attributes
    ----------
    classes_ : `numpy.ndarray`
        The list of classes seen in training. Depending on the traning target the
        contents are ``int`` or ``str``.
    is_fitted_ : bool
        ``True`` if the estimator is fitted.
    is_multitable_model_ : bool
        ``True`` if the model was fitted on a multi-table dataset.
    model_ : `.DictionaryDomain`
        The Khiops dictionary domain for the trained classifier.
    model_main_dictionary_name_ : str
        The name of the main Khiops dictionary within the ``model_`` domain.
    model_report_ : `.AnalysisResults`
        The Khiops report object.
    model_report_raw_ : dict
        JSON object of the Khiops report.
        **Deprecated** will be removed in khiops-python 11. Use the ``json_data``
        attribute of the ``model_report_`` estimator attribute instead.

    Examples
    --------
    See the following functions of the ``samples_sklearn.py`` documentation script:
        - `samples_sklearn.khiops_classifier()`
        - `samples_sklearn.khiops_classifier_multiclass()`
        - `samples_sklearn.khiops_classifier_multitable_star()`
        - `samples_sklearn.khiops_classifier_multitable_snowflake()`
        - `samples_sklearn.khiops_classifier_pickle()`
        - `samples_sklearn.khiops_classifier_multitable_star_file()`
    """

    def __init__(
        self,
        n_features=100,
        n_pairs=0,
        n_trees=10,
        verbose=False,
        output_dir=None,
        auto_sort=True,
        key=None,
        internal_sort=None,
    ):
        super().__init__(
            n_features=n_features,
            n_pairs=n_pairs,
            n_trees=n_trees,
            verbose=verbose,
            output_dir=output_dir,
            auto_sort=auto_sort,
            key=key,
            internal_sort=internal_sort,
        )
        self._khiops_model_prefix = "SNB_"
        self._predicted_target_meta_data_tag = "Prediction"

    def _is_real_target_dtype_integer(self):
        assert self._original_target_type is not None, "Original target type not set"
        return pd.api.types.is_integer_dtype(self._original_target_type) or (
            isinstance(self._original_target_type, pd.CategoricalDtype)
            and pd.api.types.is_integer_dtype(self._original_target_type.categories)
        )

    def _sorted_prob_variable_names(self):
        """Returns the model probability variable names in the order of self.classes_"""
        assert self.is_fitted_, "Model not fit yet"

        # Collect the probability variables from the model main dictionary
        prob_variables = []
        for variable in self._get_main_dictionary().variables:
            for key in variable.meta_data.keys:
                if key.startswith("TargetProb"):
                    prob_variables.append((variable, key))

        # Collect the probability variable names in the same order of self.classes_
        sorted_prob_variable_names = []
        for class_name in self.classes_:
            for variable, prob_key in prob_variables:
                if str(class_name) == variable.meta_data.get_value(prob_key):
                    sorted_prob_variable_names.append(variable.name)

        return sorted_prob_variable_names

    def fit(self, X, y, **kwargs):
        """Fits a Selective Naive Bayes classifier according to X, y

        Parameters
        ----------
        X : :external:term:`array-like` of shape (n_samples, n_features_in) or dict
            Training dataset. Either an :external:term:`array-like` or a ``dict``
            specification for multi-table datasets (see :doc:`/multi_table_primer`).
            *Deprecated input modes* (will be removed in khiops-python 11):

            - tuple: A pair (``path_to_file``, ``separator``).
            - list: A sequence of dataframes or paths, or pairs path-separator. The
              first element of the list is the main table and the following are
              secondary ones joined to the main table using ``key`` estimator parameter.

        y : :external:term:`array-like` of shape (n_samples,)
            :external:term:`array-like` object containing the target values.

            **Deprecated input modes** (will be removed in khiops-python 11):
                - str: A path to a data table file for file-based ``dict`` dataset
                  specifications.

        Returns
        -------
        self : `KhiopsClassifier`
            The calling estimator instance.
        """
        kwargs["categorical_target"] = True
        return super().fit(X, y, **kwargs)

    def _check_target_type(self, dataset):
        _check_categorical_target_type(dataset)

    def _fit_check_dataset(self, dataset):
        # Call the parent method
        super()._fit_check_dataset(dataset)

        # Check that the target is for classification in in_memory_tables
        if dataset.is_in_memory():
            current_type_of_target = type_of_target(dataset.main_table.target_column)
            if current_type_of_target not in ["binary", "multiclass"]:
                raise ValueError(
                    f"Unknown label type: '{current_type_of_target}' "
                    "for classification. Maybe you passed a floating point target?"
                )
        # Check if the target has more than 1 class
        if (
            dataset.is_in_memory()
            and len(np.unique(dataset.main_table.target_column)) == 1
        ):
            raise ValueError(
                f"{self.__class__.__name__} can't train when only one class is present."
            )

    def _fit_core_training_function(self, *args, **kwargs):
        return kh.train_predictor(*args, **kwargs)

    def _fit_training_post_process(self, dataset):
        # Call the parent's method
        super()._fit_training_post_process(dataset)

        # Save the target datatype
        self._original_target_type = dataset.target_column_type

        # Save class values in the order of deployment
        self.classes_ = []
        for variable in self._get_main_dictionary().variables:
            for key in variable.meta_data.keys:
                if key.startswith("TargetProb"):
                    self.classes_.append(variable.meta_data.get_value(key))
        if self._is_real_target_dtype_integer():
            self.classes_ = [int(class_value) for class_value in self.classes_]
            self.classes_.sort()
        self.classes_ = column_or_1d(self.classes_)

        # Warn when there are no informative variables
        if self.model_report_.preparation_report.informative_variable_number == 0:
            warnings.warn(
                "There are no informative variables. "
                "The fitted model is the majority class classifier.",
                stacklevel=6,
            )

        # Set the target class probabilites as used
        # (only the predicted classes is obtained without this step prior to Khiops 10)
        for variable in self._get_main_dictionary().variables:
            for key in variable.meta_data.keys:
                if key.startswith("TargetProb"):
                    variable.used = True

    def predict(self, X):
        """Predicts the most probable class for the test dataset X

        The predicted class of an input sample is the arg-max of the conditional
        probabilities P(y|X) for each value of y.

        Parameters
        ----------
        X : :external:term:`array-like` of shape (n_samples, n_features_in) or dict
            Training dataset. Either an :external:term:`array-like` or a ``dict``
            specification for multi-table datasets (see :doc:`/multi_table_primer`).
            *Deprecated input modes* (will be removed in khiops-python 11):

            - tuple: A pair (``path_to_file``, ``separator``).
            - list: A sequence of dataframes or paths, or pairs path-separator. The
              first element of the list is the main table and the following are
              secondary ones joined to the main table using ``key`` estimator parameter.

        Returns
        -------
        `numpy.ndarray`
            An array containing the encoded columns. A first column containing key
            column ids is added in multi-table mode. The `numpy.dtype` of the array is
            integer if the classifier was learned with an integer ``y``. Otherwise it
            will be ``str``.

            The key columns are added for multi-table tasks.
        """
        # Call the parent's method
        y_pred = super().predict(X)

        # Adjust the data type according to the original target type
        # Note: String is coerced explictly because astype does not work as expected
        if isinstance(y_pred, pd.DataFrame):
            # Transform to numpy.ndarray
            y_pred = y_pred.to_numpy(copy=False).ravel()

            # If integer and string just transform
            if pd.api.types.is_integer_dtype(self._original_target_type):
                y_pred = y_pred.astype(self._original_target_type)
            elif pd.api.types.is_string_dtype(self._original_target_type):
                y_pred = y_pred.astype(str, copy=False)
            # If category first coerce the type to the categories' type
            else:
                assert pd.api.types.is_categorical_dtype(self._original_target_type)
                if pd.api.types.is_integer_dtype(
                    self._original_target_type.categories.dtype
                ):
                    y_pred = y_pred.astype(
                        self._original_target_type.categories.dtype, copy=False
                    )
                else:
                    y_pred = y_pred.astype(str, copy=False)

        assert isinstance(y_pred, (str, np.ndarray)), "Expected str or np.array"
        return y_pred

    def predict_proba(self, X):
        """Predicts the class probabilities for the test dataset X

        Parameters
        ----------
        X : :external:term:`array-like` of shape (n_samples, n_features_in) or dict
            Training dataset. Either an :external:term:`array-like` or a ``dict``
            specification for multi-table datasets (see :doc:`/multi_table_primer`).
            *Deprecated input modes* (will be removed in khiops-python 11):

            - tuple: A pair (``path_to_file``, ``separator``).
            - list: A sequence of dataframes or paths, or pairs path-separator. The
              first element of the list is the main table and the following are
              secondary ones joined to the main table using ``key`` estimator parameter.

        Returns
        -------
        `numpy.array` or str
            The probability of the samples for each class in the model.  The columns are
            named with the pattern ``Prob<class>`` for each ``<class>`` found in the
            training dataset. The output data container depends on ``X``:

                - Dataframe or dataframe-based ``dict`` dataset specification:
                  `numpy.array`
                - File-based ``dict`` dataset specification: A CSV file (the method
                  returns its path).

            The key columns are added for multi-table tasks.
        """
        # Create temporary directory and tables
        computation_dir = self._create_computation_dir("predict_proba")
        initial_runner_temp_dir = kh.get_runner().root_temp_dir
        kh.get_runner().root_temp_dir = computation_dir

        # Create the input dataset

        # Call the generic transfrom method
        try:
            dataset = Dataset(X, key=self.key)
            y_probas = self._transform(
                dataset,
                computation_dir,
                self._transform_prepare_deployment_model_for_predict_proba,
                True,
            )
        # Cleanup and restore the runner's temporary dir
        finally:
            self._cleanup_computation_dir(computation_dir)
            kh.get_runner().root_temp_dir = initial_runner_temp_dir

        # For in-memory datasets:
        # - Reorder the columns to that of self.classes_
        # - Transform to np.ndarray
        if dataset.is_in_memory():
            assert isinstance(
                y_probas, (pd.DataFrame, np.ndarray)
            ), "y_probas is not a Pandas DataFrame nor Numpy array"
            y_probas = y_probas.reindex(
                self._sorted_prob_variable_names(), axis=1, copy=False
            ).to_numpy(copy=False)

        assert isinstance(y_probas, (str, np.ndarray)), "Expected str or np.ndarray"
        return y_probas

    def _transform_prepare_deployment_model_for_predict_proba(self):
        # Create a copy of the model dictionary with only the probabilities used
        # We also activate the key to reorder the output in the multitable case
        model_copy = self.model_.copy()
        model_dictionary = model_copy.get_dictionary(self.model_main_dictionary_name_)
        for variable in model_dictionary.variables:
            for key in variable.meta_data.keys:
                if variable.name in model_dictionary.key:
                    variable.used = True
                elif key.startswith("TargetProb"):
                    variable.used = True
                else:
                    variable.used = False
        return model_copy


class KhiopsRegressor(KhiopsPredictor, RegressorMixin):
    """Khiops Selective Naive Bayes Regressor

    This regressor supports automatic feature engineering on multi-table datasets. See
    :doc:`/multi_table_primer` for more details.

    .. note::

        Visit `the Khiops site <https://khiops.org/learn/understand>`_ to learn
        about the automatic feature engineering algorithm.

    Parameters
    ----------
    n_features : int, default 100
        *Multi-table only* : Maximum number of multi-table aggregate features to
        construct. See :doc:`/multi_table_primer` for more details.
    n_pairs : int, default 0
        Maximum number of pair features to construct. These features represent a 2D grid
        partition of the domain of a pair of variables in which is optimized in a way
        that the cells are the purest possible with respect to the target. Only pairs
        which jointly are more informative that its univariate components may be taken
        into account in the regressor.
    verbose : bool, default ``False``
        If ``True`` it prints debug information and it does not erase temporary files
        when fitting, predicting or transforming.
    output_dir : str, optional
        Path of the output directory for the ``AllReports.khj`` report file and the
        ``Modeling.kdic`` modeling dictionary file. By default these files are deleted.
    auto_sort : bool, default ``True``
        *Advanced.* Only for multi-table inputs: If ``True`` input tables are pre-sorted
        by their key before executing Khiops. If the input tables are already sorted by
        their keys set this parameter to ``False`` to speed up the processing. This
        affects the `fit` and `predict` methods.
        *Note* The sort by key is performed in a left-to-right, hierarchical,
        lexicographic manner.
    key : str, optional
        *Multi-table only* : The name of the column to be used as key.
        **Deprecated** will be removed in khiops-python 11. Use ``dict`` dataset
        specifications in ``fit``, ``fit_predict``  and ``predict``.
    internal_sort : bool, optional
        *Advanced.* Only for multi-table inputs: If ``True`` input tables are pre-sorted
        by their key before executing Khiops. If the input tables are already sorted by
        their keys set this parameter to ``False`` to speed up the processing. This
        affects the `fit` and `predict` methods.
        *Note* The sort by key is performed in a left-to-right, hierarchical,
        lexicographic manner.
        **Deprecated** will be removed in khiops-python 11. Use the ``auto_sort``
        estimator parameter instead.

    Attributes
    ----------
    is_fitted_ : bool
        ``True`` if the estimator is fitted.
    is_multitable_model_ : bool
        ``True`` if the model was fitted on a multi-table dataset.
    model_ : `.DictionaryDomain`
        The Khiops dictionary domain for the trained regressor.
    model_main_dictionary_name_ : str
        The name of the main Khiops dictionary within the ``model_`` domain.
    model_report_ : `.AnalysisResults`
        The Khiops report object.
    model_report_raw_ : dict
        JSON object of the Khiops report.
        **Deprecated** will be removed in khiops-python 11. Use the ``json_data``
        attribute of the ``model_report_`` estimator attribute instead.

    Examples
    --------
    See the following functions of the ``samples_sklearn.py`` documentation script:
        - `samples_sklearn.khiops_regressor()`
    """

    def __init__(
        self,
        n_features=100,
        n_pairs=0,
        n_trees=0,
        verbose=False,
        output_dir=None,
        auto_sort=True,
        key=None,
        internal_sort=None,
    ):
        super().__init__(
            n_features=n_features,
            n_pairs=n_pairs,
            n_trees=n_trees,
            verbose=verbose,
            output_dir=output_dir,
            auto_sort=auto_sort,
            key=key,
            internal_sort=internal_sort,
        )
        self._khiops_model_prefix = "SNB_"
        self._predicted_target_meta_data_tag = "Mean"

    def fit(self, X, y=None, **kwargs):
        """Fits a Selective Naive Bayes regressor according to X, y

        .. warning::
            Make sure that the type of ``y`` is float. This is easily done with ``y =
            y.astype(float)``.

        Parameters
        ----------
        X : :external:term:`array-like` of shape (n_samples, n_features_in) or dict
            Training dataset. Either an :external:term:`array-like` or a ``dict``
            specification for multi-table datasets (see :doc:`/multi_table_primer`).
            *Deprecated input modes* (will be removed in khiops-python 11):

            - tuple: A pair (``path_to_file``, ``separator``).
            - list: A sequence of dataframes or paths, or pairs path-separator. The
              first element of the list is the main table and the following are
              secondary ones joined to the main table using ``key`` estimator parameter.

        y : :external:term:`array-like` of shape (n_samples,)
            :external:term:`array-like` object containing the target values.

            **Deprecated input modes** (will be removed in khiops-python 11):
                - str: A path to a data table file for file-based ``dict`` dataset
                  specifications.
        Returns
        -------
        self : `KhiopsRegressor`
            The calling estimator instance.
        """
        if self.n_trees > 0:
            warnings.warn("Khiops does not support n_trees > 0 for regression models.")
        kwargs["categorical_target"] = False
        return super().fit(X, y=y, **kwargs)

    def _fit_core_training_function(self, *args, **kwargs):
        return kh.train_predictor(*args, **kwargs)

    def _fit_train_model(self, dataset, computation_dir, **kwargs):
        # Call the parent method
        super()._fit_train_model(dataset, computation_dir, **kwargs)

        # Warn when there are no informative variables
        if self.model_report_.preparation_report.informative_variable_number == 0:
            warnings.warn(
                "There are no informative variables. "
                "The fitted model is the mean regressor."
            )

    def _fit_training_post_process(self, dataset):
        # Call parent method
        super()._fit_training_post_process(dataset)

        # Remove variables depending on the target
        variables_to_eliminate = []
        for variable in self._get_main_dictionary().variables:
            if (
                "TargetVariableRank" in variable.meta_data
                or "Density" in variable.meta_data
                or "DensityRank" in variable.meta_data
            ):
                variables_to_eliminate.append(variable.name)
        for variable_name in variables_to_eliminate:
            self._get_main_dictionary().remove_variable(variable_name)

    def _check_target_type(self, dataset):
        _check_numerical_target_type(dataset)

    # Deactivate useless super delegation because the method have different docstring
    # pylint: disable=useless-super-delegation

    def predict(self, X):
        """Predicts the regression values for the test dataset X

        The predicted value is estimated by the Selective Naive Bayes Regressor learned
        during fit step.

        Parameters
        ----------
        X : :external:term:`array-like` of shape (n_samples, n_features_in) or dict
            Training dataset. Either an :external:term:`array-like` or a ``dict``
            specification for multi-table datasets (see :doc:`/multi_table_primer`).
            *Deprecated input modes* (will be removed in khiops-python 11):

            - tuple: A pair (``path_to_file``, ``separator``).
            - list: A sequence of dataframes or paths, or pairs path-separator. The
              first element of the list is the main table and the following are
              secondary ones joined to the main table using ``key`` estimator parameter.

        Returns
        -------
        `numpy.ndarray`
            An array containing the encoded columns. A first column containing key
            column ids is added in multi-table mode. The key columns are added for
            multi-table tasks.

            *Deprecated return values* (will be removed in khiops-python 11): str for
            file based dataset specification.

        """
        # Call the parent's method
        y_pred = super().predict(X)

        # Transform to np.ndarray for in-memory datasets
        if isinstance(y_pred, pd.DataFrame):
            y_pred = y_pred.astype("float64", copy=False).to_numpy(copy=False).ravel()

        assert isinstance(y_pred, (str, np.ndarray)), "Expected str or np.array"
        return y_pred

    # pylint: enable=useless-super-delegation


class KhiopsEncoder(KhiopsSupervisedEstimator, TransformerMixin):
    """Khiops supervised discretization/grouping encoder

    Parameters
    ----------
    categorical_target : bool, default ``True``
        ``True`` if the target column is categorical.
    n_features : int, default 100
        *Multi-table only* : Maximum number of multi-table aggregate features to
        construct. See :doc:`/multi_table_primer` for more details.
    n_pairs : int, default 0
        Maximum number of pair features to construct. These features represent a 2D grid
        partition of the domain of a pair of variables in which is optimized in a way
        that the cells are the purest possible with respect to the target.
    n_trees : int, default 10
        Maximum number of decision tree features to construct. The constructed trees
        combine other features, either native or constructed. These features usually
        improve a predictor's performance at the cost of interpretability of the model.
    keep_initial_variables : bool, default ``False``
        If ``True`` the original columns are kept in the transformed data.
    transform_type_categorical : str, default "part_id"
        Type of transformation for categorical variables. Valid values:
            - "part_id"
            - "part_label"
            - "dummies"
            - "conditional_info"

        See the documentation for the ``categorical_recoding_method`` parameter of the
        `~.api.train_recoder` function for more details.
    transform_type_numerical : str, default "part_id"
        One of the following strings are valid:
            - "part_id"
            - "part_label"
            - "dummies"
            - "conditional_info"
            - "center_reduction"
            - "0-1_normalization"
            - "rank_normalization"

        See the documentation for the ``numerical_recoding_method`` parameter of the
        `~.api.train_recoder` function for more details.
    verbose : bool, default ``False``
        If ``True`` it prints debug information and it does not erase temporary files
        when fitting, predicting or transforming.
    output_dir : str, optional
        Path of the output directory for the ``AllReports.khj`` report file and the
        ``Modeling.kdic`` modeling dictionary file. By default these files are deleted.
    auto_sort : bool, default ``True``
        *Advanced.* Only for multi-table inputs: If ``True`` input tables are pre-sorted
        by their key before executing Khiops. If the input tables are already sorted by
        their keys set this parameter to ``False`` to speed up the processing. This
        affects the `fit` and `transform` methods.
        *Note* The sort by key is performed in a left-to-right, hierarchical,
        lexicographic manner.
    key : str, optional
        *Multi-table only* : The name of the column to be used as key.
        **Deprecated** will be removed in khiops-python 11. Use ``dict`` dataset
        specifications in ``fit`` and ``transform``.
    internal_sort : bool, optional
        *Advanced.* Only for multi-table inputs: If ``True`` input tables are pre-sorted
        by their key before executing Khiops. If the input tables are already sorted by
        their keys set this parameter to ``False`` to speed up the processing. This
        affects the `fit` and `transform` methods.
        *Note* The sort by key is performed in a left-to-right, hierarchical,
        lexicographic manner.
        **Deprecated** will be removed in khiops-python 11. Use the ``auto_sort``
        estimator parameter instead.

    Attributes
    ----------
    is_fitted_ : bool
        ``True`` if the estimator is fitted.
    is_multitable_model_ : bool
        ``True`` if the model was fitted on a multi-table dataset.
    model_ : `.DictionaryDomain`
        The Khiops dictionary domain for the trained encoder.
    model_main_dictionary_name_ : str
        The name of the main Khiops dictionary within the ``model_`` domain.
    model_report_ : `.AnalysisResults`
        The Khiops report object.
    model_report_raw_ : dict
        JSON object of the Khiops report.
        **Deprecated** will be removed in khiops-python 11. Use the ``json_data``
        attribute of the ``model_report_`` estimator attribute instead.

    Examples
    --------
    See the following functions of the ``samples_sklearn.py`` documentation script:
        - `samples_sklearn.khiops_encoder()`
        - `samples_sklearn.khiops_encoder_multitable_star()`
        - `samples_sklearn.khiops_encoder_multitable_snowflake()`
    """

    def __init__(
        self,
        categorical_target=True,
        n_features=100,
        n_pairs=0,
        n_trees=0,
        transform_type_categorical="part_id",
        transform_type_numerical="part_id",
        keep_initial_variables=False,
        verbose=False,
        output_dir=None,
        auto_sort=True,
        key=None,
        internal_sort=None,
    ):
        super().__init__(
            n_features=n_features,
            n_pairs=n_pairs,
            n_trees=n_trees,
            verbose=verbose,
            output_dir=output_dir,
            auto_sort=auto_sort,
            key=key,
            internal_sort=internal_sort,
        )
        self.categorical_target = categorical_target
        self.transform_type_categorical = transform_type_categorical
        self.transform_type_numerical = transform_type_numerical
        self.keep_initial_variables = keep_initial_variables
        self._khiops_model_prefix = "R_"

    def more_tags(self):
        return {"preserves_dtype": []}

    def _categorical_transform_method(self):
        _transform_types_categorical = {
            "part_id": "part Id",
            "part_label": "part label",
            "dummies": "0-1 binarization",
            "conditional_info": "conditional info",
            None: "none",
        }
        if self.transform_type_categorical not in _transform_types_categorical:
            raise ValueError(
                "'transform_type_categorical' must be one of the following:"
                ",".join(_transform_types_categorical.keys)
            )
        return _transform_types_categorical[self.transform_type_categorical]

    def _numerical_transform_method(self):
        _transform_types_numerical = {
            "part_id": "part Id",
            "part_label": "part label",
            "dummies": "0-1 binarization",
            "conditional_info": "conditional info",
            "center_reduction": "center-reduction",
            "0-1_normalization": "0-1 normalization",
            "rank_normalization": "rank normalization",
            None: "none",
        }
        if self.transform_type_categorical not in _transform_types_numerical:
            raise ValueError(
                "'transform_type_numerical' must be one of the following:"
                ",".join(_transform_types_numerical.keys)
            )
        return _transform_types_numerical[self.transform_type_numerical]

    def _fit_check_params(self, dataset, **kwargs):
        # Call parent method
        super()._fit_check_params(dataset, **kwargs)

        # Check 'transform_type_categorical' parameter
        if not isinstance(self.transform_type_categorical, str):
            raise TypeError(
                type_error_message(
                    "transform_type_categorical", self.transform_type_categorical, str
                )
            )
        self._categorical_transform_method()  # Raises ValueError if invalid

        # Check 'transform_type_numerical' parameter
        if not isinstance(self.transform_type_numerical, str):
            raise TypeError(
                type_error_message(
                    "transform_type_numerical", self.transform_type_numerical, str
                )
            )

        self._numerical_transform_method()  # Raises ValueError if invalid

        # Check coherence between transformation types and tree number
        if (
            self.transform_type_categorical is None
            and self.transform_type_numerical is None
            and self.n_trees == 0
        ):
            raise ValueError(
                "transform_type_categorical and transform_type_numerical "
                "cannot be both None with n_trees == 0."
            )

    def _check_target_type(self, dataset):
        if self.categorical_target:
            _check_categorical_target_type(dataset)
        else:
            _check_numerical_target_type(dataset)

    def _fit_core_training_function(self, *args, **kwargs):
        return kh.train_recoder(*args, **kwargs)

    # Deactivate useless super delegation because the method have different docstring
    # pylint: disable=useless-super-delegation

    def fit(self, X, y=None, **kwargs):
        """Fits the Khiops Encoder according to X, y

        Parameters
        ----------
        X : :external:term:`array-like` of shape (n_samples, n_features_in) or dict
            Training dataset. Either an :external:term:`array-like` or a ``dict``
            specification for multi-table datasets (see :doc:`/multi_table_primer`).
            *Deprecated input modes* (will be removed in khiops-python 11):

            - tuple: A pair (``path_to_file``, ``separator``).
            - list: A sequence of dataframes or paths, or pairs path-separator. The
              first element of the list is the main table and the following are
              secondary ones joined to the main table using ``key`` estimator parameter.

        y : :external:term:`array-like` of shape (n_samples,)
            :external:term:`array-like` object containing the target values.

            **Deprecated input modes** (will be removed in khiops-python 11):
                - str: A path to a data table file for file-based ``dict`` dataset
                  specifications.

        Returns
        -------
        self : `KhiopsEncoder`
            The calling estimator instance.
        """
        kwargs["categorical_target"] = self.categorical_target
        return super().fit(X, y, **kwargs)

    # pylint: enable=useless-super-delegation

    def _fit_prepare_training_function_inputs(self, dataset, computation_dir):
        # Call the parent method
        args, kwargs = super()._fit_prepare_training_function_inputs(
            dataset, computation_dir
        )

        # Rename encoder parameters, delete unused ones
        kwargs["keep_initial_categorical_variables"] = kwargs["keep_initial_variables"]
        kwargs["keep_initial_numerical_variables"] = kwargs.pop(
            "keep_initial_variables"
        )
        kwargs["categorical_recoding_method"] = self._categorical_transform_method()
        kwargs["numerical_recoding_method"] = self._numerical_transform_method()
        del kwargs["transform_type_categorical"]
        del kwargs["transform_type_numerical"]
        del kwargs["categorical_target"]

        return args, kwargs

    def _fit_training_post_process(self, dataset):
        # Call parent method
        super()._fit_training_post_process(dataset)

        # Eliminate the target variable from the main dictionary
        self._get_main_dictionary()

        # Save the encoded feature names
        self.feature_names_out_ = []
        for variable in self._get_main_dictionary().variables:
            if variable.used:
                self.feature_names_out_.append(variable.name)

        # Activate the key columns in multitable
        if len(self.model_.dictionaries) > 1:
            for key_variable_name in self._get_main_dictionary().key:
                self._get_main_dictionary().get_variable(key_variable_name).used = True

    def transform(self, X):
        """Transforms X with a fitted Khiops supervised encoder

        .. note::
            Numerical variables are encoded to categorical ones. See the
            ``transform_type_numerical`` parameter for details.

        Parameters
        ----------
        X : :external:term:`array-like` of shape (n_samples, n_features_in) or dict
            Training dataset. Either an :external:term:`array-like` or a ``dict``
            specification for multi-table datasets (see :doc:`/multi_table_primer`).
            *Deprecated input modes* (will be removed in khiops-python 11):

            - tuple: A pair (``path_to_file``, ``separator``).
            - list: A sequence of dataframes or paths, or pairs path-separator. The
              first element of the list is the main table and the following are
              secondary ones joined to the main table using ``key`` estimator parameter.

        Returns
        -------
        `numpy.ndarray`
            An array containing the encoded columns. A first column containing key
            column ids is added in multi-table mode.

            *Deprecated return values* (will be removed in khiops-python 11): str for
            file based dataset specification.
        """
        # Create temporary directory
        computation_dir = self._create_computation_dir("transform")
        initial_runner_temp_dir = kh.get_runner().root_temp_dir
        kh.get_runner().root_temp_dir = computation_dir

        # Create and transform the dataset
        try:
            dataset = Dataset(X, key=self.key)
            X_transformed = super()._transform(
                dataset,
                computation_dir,
                self.model_.copy,
                True,
            )
        # Cleanup and restore the runner's temporary dir
        finally:
            self._cleanup_computation_dir(computation_dir)
            kh.get_runner().root_temp_dir = initial_runner_temp_dir
        if dataset.is_in_memory():
            return X_transformed.to_numpy(copy=False)
        return X_transformed

    def fit_transform(self, X, y=None, **kwargs):
        """Fit and transforms its inputs

        Parameters
        ----------
        X : :external:term:`array-like` of shape (n_samples, n_features_in) or dict
            Training dataset. Either an :external:term:`array-like` or a ``dict``
            specification for multi-table datasets (see :doc:`/multi_table_primer`).
            *Deprecated input modes* (will be removed in khiops-python 11):

            - tuple: A pair (``path_to_file``, ``separator``).
            - list: A sequence of dataframes or paths, or pairs path-separator. The
              first element of the list is the main table and the following are
              secondary ones joined to the main table using ``key`` estimator parameter.

        y : :external:term:`array-like` of shape (n_samples,)
            :external:term:`array-like` object containing the target values.

            **Deprecated input modes** (will be removed in khiops-python 11):
                - str: A path to a data table file for file-based ``dict`` dataset
                  specifications.

        Returns
        -------
        self : `KhiopsEncoder`
            The calling estimator instance.
        """
        return self.fit(X, y, **kwargs).transform(X)
