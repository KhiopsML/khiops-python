######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
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
from sklearn.exceptions import NotFittedError
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import assert_all_finite, check_is_fitted, column_or_1d

import khiops.core as kh
import khiops.core.internals.filesystems as fs
from khiops.core.dictionary import DictionaryDomain
from khiops.core.helpers import _build_multi_table_dictionary_domain
from khiops.core.internals.common import is_dict_like, is_list_like, type_error_message
from khiops.sklearn.dataset import (
    Dataset,
    get_khiops_variable_name,
    read_internal_data_table,
)

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
    ds_dictionary,
    estimator_class_name,
    target_variable_name=None,
):
    # Prefix for all error messages
    error_msg_prefix = (
        f"Model {estimator_class_name} incompatible with "
        f"table '{ds_dictionary.name}'"
    )

    # Put the variable names in sets
    model_variable_names = {var.name for var in model_dictionary.variables}
    ds_variable_names = {var.name for var in ds_dictionary.variables}

    # The only feature that may be missing of the dataset is the target
    model_var_names_not_in_ds = model_variable_names - ds_variable_names
    if len(model_var_names_not_in_ds) > 0:
        if target_variable_name is None:
            effective_model_var_names_not_in_ds = model_var_names_not_in_ds
        else:
            effective_model_var_names_not_in_ds = model_var_names_not_in_ds - {
                target_variable_name
            }
        if len(effective_model_var_names_not_in_ds) > 0:
            raise ValueError(
                f"{error_msg_prefix}: Missing features: "
                f"{effective_model_var_names_not_in_ds}."
            )

    # Raise an error if there are extra features in the input
    ds_var_names_not_in_model = ds_variable_names - model_variable_names
    if len(ds_var_names_not_in_model) > 0:
        raise ValueError(
            f"{error_msg_prefix}: Features not in model: {ds_var_names_not_in_model}."
        )

    # Check the type
    for ds_var in ds_dictionary.variables:
        model_var = model_dictionary.get_variable(ds_var.name)
        if ds_var.type != model_var.type:
            if model_var.type == "Categorical":
                warnings.warn(
                    f"X contains variable '{ds_var.name}' which was deemed "
                    "numerical. It will be coerced to categorical."
                )
            else:
                raise ValueError(
                    f"{error_msg_prefix}: Khiops type for variable "
                    f"'{ds_var.name}' should be '{model_var.type}' "
                    f"not '{ds_var.type}'"
                )


def _check_categorical_target_type(ds):
    if ds.target_column is None:
        raise ValueError("Target vector is not specified.")

    if not (
        isinstance(ds.target_column.dtype, pd.CategoricalDtype)
        or pd.api.types.is_string_dtype(ds.target_column.dtype)
        or pd.api.types.is_integer_dtype(ds.target_column.dtype)
        or pd.api.types.is_float_dtype(ds.target_column.dtype)
        or pd.api.types.is_bool_dtype(ds.target_column.dtype)
    ):
        raise ValueError(
            f"'y' has invalid type '{ds.target_column_type}'. "
            "Only string, integer, float and categorical types "
            "are accepted for the target."
        )


def _check_numerical_target_type(ds):
    # Check that the target column is specified
    if ds.target_column is None:
        raise ValueError("Target vector is not specified.")

    # Check that the column is numerical and that the values are finite
    # The latter is required by sklearn
    if not pd.api.types.is_numeric_dtype(ds.target_column.dtype):
        raise ValueError(
            f"Unknown label type '{ds.target_column.dtype}'. "
            "Expected a numerical type."
        )
    if ds.target_column is not None:
        assert_all_finite(ds.target_column)


def _check_pair_parameters(estimator):
    assert isinstance(estimator, (KhiopsClassifier, KhiopsEncoder)), type_error_message(
        "estimator", estimator, KhiopsClassifier, KhiopsEncoder
    )
    if not isinstance(estimator.n_pairs, int):
        raise TypeError(type_error_message("n_pairs", estimator.n_pairs, int))
    if estimator.n_pairs < 0:
        raise ValueError("'n_pairs' must be positive")
    if estimator.specific_pairs is not None:
        if not is_list_like(estimator.specific_pairs):
            raise TypeError(
                type_error_message(
                    "specific_pairs", estimator.specific_pairs, "list-like"
                )
            )
        else:
            for pair in estimator.specific_pairs:
                if not isinstance(pair, tuple):
                    raise TypeError(type_error_message(pair, pair, tuple))
    if not isinstance(estimator.all_possible_pairs, bool):
        raise TypeError(
            type_error_message("all_possible_pairs", estimator.all_possible_pairs, bool)
        )

    # Check 'group_target_value' parameter
    if not isinstance(estimator.group_target_value, bool):
        raise TypeError(
            type_error_message("group_target_value", estimator.group_target_value, bool)
        )


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

    .. note::
         The input features collection X needs to have single-line records
         so that Khiops can handle them.
         Hence, multi-line records are preprocessed:
         carriage returns / line feeds are replaced
         with blank spaces before being handed over to Khiops.

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
    """

    def __init__(
        self,
        verbose=False,
        output_dir=None,
        auto_sort=True,
    ):
        # Set the estimator parameters and internal variables
        self._khiops_model_prefix = None
        self.output_dir = output_dir
        self.verbose = verbose
        self.auto_sort = auto_sort

    def __sklearn_tags__(self):
        # We disable this because this import is only available for scikit-learn>=1.6
        # pylint: disable=import-outside-toplevel
        try:
            from sklearn.utils import TransformerTags
        except ImportError as exc:
            raise NotImplementedError("__sklearn_tags__ API unsupported.") from exc

        # Set the tags from _more_tags
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = self._more_tags()["allow_nan"]
        for tag, tag_value in self._more_tags().items():
            if tag == "requires_y":
                tags.target_tags.required = tag_value
            elif tag == "preserves_dtype":
                tags.transformer_tags = TransformerTags()
                tags.transformer_tags.preserves_dtype = tag_value
        return tags

    def _more_tags(self):
        return {"allow_nan": True}

    def _undefine_estimator_attributes(self):
        """Undefines all sklearn estimator attributes (ie. pass to "not fit" state)

        Sklearn estimator attributes follow the convention that their names end in "_".
        See https://scikit-learn.org/stable/glossary.html#term-attributes
        """
        for attribute_name in dir(self):
            if not attribute_name.startswith("_") and attribute_name.endswith("_"):
                delattr(self, attribute_name)

    def _get_main_dictionary(self):
        """Returns the model's main Khiops dictionary"""
        self._assert_is_fitted()
        return self.model_.get_dictionary(self.model_main_dictionary_name_)

    def _read_model_from_dictionary_file(self, model_dictionary_file_path):
        """Removes dictionaries that do not have the model prefix in their name

        This function is necessary for the regression case because Khiops generates a
        baseline model which has to be removed for sklearn predictor.
        """
        model = kh.read_dictionary_file(model_dictionary_file_path)
        assert self._khiops_model_prefix is not None
        for dictionary_name in [kdic.name for kdic in model.dictionaries]:
            if not dictionary_name.startswith(self._khiops_model_prefix):
                model.remove_dictionary(dictionary_name)
        return model

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
        check_is_fitted(self)
        if self.model_report_ is None:
            raise ValueError("Report not available (imported model?).")
        self.model_report_.write_khiops_json_file(report_file_path)

    def export_dictionary_file(self, dictionary_file_path):
        """Export the model's Khiops dictionary file (.kdic)"""
        check_is_fitted(self)
        self.model_.export_khiops_dictionary_file(dictionary_file_path)

    def _import_model(self, kdic_path):
        """Sets model instance attribute by importing model from ``.kdic``"""
        self.model_ = self._read_model_from_dictionary_file(kdic_path)

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
            print("khiops sklearn temporary files located at: " + computation_dir)

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

        # Create the dataset, fit the model and reset in case of any failure
        try:
            categorical_target = kwargs.get("categorical_target", True)
            dataset = Dataset(X, y, categorical_target=categorical_target)
            self._fit(dataset, computation_dir, **kwargs)
        # Undefine any attributes to pass to "not fitted"
        except:
            self._undefine_estimator_attributes()
            raise
        # Cleanup and restore the runner's temporary dir
        finally:
            self._cleanup_computation_dir(computation_dir)

        # If on "fitted" state then:
        # - self.model_ must be a DictionaryDomain
        # - self.model_report_ must be a KhiopsJSONObject
        try:
            check_is_fitted(self)
            assert isinstance(self.model_, kh.DictionaryDomain)
            assert isinstance(self.model_report_, kh.KhiopsJSONObject)
            assert isinstance(self.model_, kh.DictionaryDomain)
        # Note:
        #   We ignore any raised NotFittedError by check_is_fitted because we are using
        #   the try/catch as an if/else. The code intended is
        #     if check_is_fitted(self):
        #         # asserts
        #   But check_is_fitted has a do-nothing or raise pattern.
        except NotFittedError:
            pass

        return self

    def _fit(self, ds, computation_dir, **kwargs):
        """Template pattern of a fit method

        Parameters
        ----------
        ds : `Dataset`
            The learning dataset.
        computation_dir : str
            Path or URI where the Khiops computation results will be stored.

        The called methods are reimplemented in concrete sub-classes
        """
        # Check model parameters
        self._fit_check_params(ds, **kwargs)

        # Check the dataset
        self._fit_check_dataset(ds)

        # Train the model
        self._fit_train_model(ds, computation_dir, **kwargs)

        # If the main attributes are of the proper type finish the fitting
        # Otherwise it means there was an abort (early return) of the previous steps
        if (
            hasattr(self, "model_")
            and isinstance(self.model_, kh.DictionaryDomain)
            and hasattr(self, "model_report_")
            and isinstance(self.model_report_, kh.KhiopsJSONObject)
        ):
            self._fit_training_post_process(ds)
            self.is_multitable_model_ = ds.is_multitable
            self.n_features_in_ = ds.main_table.n_features()

    def _fit_check_params(self, _ds, **_):
        """Check the model parameters including those data dependent (in kwargs)"""

    def _fit_check_dataset(self, ds):
        """Checks the pre-conditions of the tables to build the model"""
        if ds.main_table.n_samples is not None and ds.main_table.n_samples <= 1:
            raise ValueError(
                "Table contains one sample or less. It must contain at least 2."
            )

    @abstractmethod
    def _fit_train_model(self, ds, computation_dir, **kwargs):
        """Builds the model with one or more calls to khiops.core.api

        At a minimum it sets the following attributes:

        - self.model_
        - self.model_report_
        """

    @abstractmethod
    def _fit_training_post_process(self, ds):
        """Loads the model's data from Khiops files into the object"""

    @abstractmethod
    def _transform_check_dataset(self, ds):
        """Checks that the dataset is consistent with the model"""

    def _transform(
        self,
        ds,
        computation_dir,
        _transform_prepare_deployment_fun,
        drop_key,
        transformed_file_name,
    ):
        """Generic template method to implement transform, predict and predict_proba"""
        # Check if the model is fitted
        check_is_fitted(self)

        # Check if the dataset is consistent with the model
        self._transform_check_dataset(ds)

        # Create a deployment dataset
        # Note: The input dataset isn't ready for deployment in the case of coclustering
        deployment_ds = self._transform_create_deployment_dataset(ds, computation_dir)

        # Create a deployment dictionary and the internal table column dtypes
        deployment_dictionary_domain, internal_table_column_dtypes = (
            _transform_prepare_deployment_fun(ds)
        )

        # Deploy the model
        output_table_path = self._transform_deploy_model(
            deployment_ds,
            deployment_dictionary_domain,
            self.model_main_dictionary_name_,
            computation_dir,
            transformed_file_name,
        )

        # Post-process to return the correct output type and order
        # Load the table as a dataframe
        with io.BytesIO(fs.read(output_table_path)) as output_table_stream:
            raw_output_table_df = read_internal_data_table(
                output_table_stream, column_dtypes=internal_table_column_dtypes
            )

        # On multi-table:
        # - Reorder the table to the original table order
        #     - Because transformed data table file is sorted by key
        # - Drop the key columns if specified
        if deployment_ds.is_multitable:
            key_df = deployment_ds.main_table.data_source[deployment_ds.main_table.key]
            output_table_df = key_df.merge(
                raw_output_table_df, on=deployment_ds.main_table.key
            )
            if drop_key:
                output_table_df.drop(deployment_ds.main_table.key, axis=1, inplace=True)
        # On mono-table: Return the read dataframe as-is
        else:
            output_table_df = raw_output_table_df

        return output_table_df

    def _transform_create_deployment_dataset(self, ds, _):
        """Creates if necessary a new dataset to execute the model deployment

        The default behavior is to return the same dataset.
        """
        return ds

    def _transform_deploy_model(
        self,
        deployment_ds,
        model_dictionary_domain,
        model_dictionary_name,
        computation_dir,
        transformed_file_name,
    ):
        """Deploys a generic Khiops transformation model

        It allows to implement `predict`, `predict_proba` and `transform` methods in the
        sub-classes `KhiopsEncoder`, `KhiopsClassifier`, `KhiopsRegressor`.

        Parameters
        ----------
        X : :external:term:`array-like` of shape (n_samples, n_features_in) or dict
            Training dataset. Either an :external:term:`array-like` or a ``dict``
            specification for multi-table datasets (see :doc:`/multi_table_primer`).

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
            additional_data_table_paths,
        ) = deployment_ds.create_table_files_for_khiops(
            computation_dir, sort=self.auto_sort
        )

        # Build the 'additional_data_tables' argument
        secondary_data_paths = model_dictionary_domain.extract_data_paths(
            model_dictionary_name
        )
        for data_path in secondary_data_paths:
            dictionary = model_dictionary_domain.get_dictionary_at_data_path(data_path)
            assert dictionary.name.startswith(self._khiops_model_prefix), (
                f"Dictionary '{dictionary.name}' "
                f"does not have prefix '{self._khiops_model_prefix}'"
            )

        # Set output path files
        output_dir = self._get_output_dir(computation_dir)
        log_file_path = fs.get_child_path(output_dir, "khiops.log")
        output_data_table_path = fs.get_child_path(output_dir, transformed_file_name)

        # Call to core function deploy_model
        kh.deploy_model(
            model_dictionary_domain,
            model_dictionary_name,
            main_table_path,
            output_data_table_path,
            additional_data_tables=additional_data_table_paths,
            detect_format=False,
            field_separator="\t",
            header_line=True,
            output_field_separator="\t",
            output_header_line=True,
            log_file_path=log_file_path,
            trace=self.verbose,
        )

        return output_data_table_path

    def _create_computation_dir(self, method_name):
        """Creates a temporary computation directory"""
        return kh.get_runner().create_temp_dir(
            prefix=f"{self.__class__.__name__}_{method_name}_"
        )

    def _assert_is_fitted(self):
        try:
            check_is_fitted(self)
        except NotFittedError as exc:
            raise AssertionError("Model not fitted") from exc


# Note: scikit-learn **requires** inherit first the mixins and then other classes
class KhiopsCoclustering(ClusterMixin, KhiopsEstimator):
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

    Attributes
    ----------
    is_multitable_model_ : bool
        ``True`` if the model was fitted on a multi-table dataset.
    model_ : `.DictionaryDomain`
        The Khiops dictionary domain for the trained coclustering. For coclustering it
        is a multi-table dictionary even though the model is single-table.
    model_main_dictionary_name_ : str
        The name of the main Khiops dictionary within the ``model_`` domain.
    model_report_ : `.CoclusteringResults`
        The Khiops report object.

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
    ):
        super().__init__(
            verbose=verbose,
            output_dir=output_dir,
            auto_sort=auto_sort,
        )
        self._khiops_model_prefix = "CC_"
        self.build_name_var = build_name_var
        self.build_distance_vars = build_distance_vars
        self.build_frequency_vars = build_frequency_vars
        self.model_id_column = None

    def __sklearn_tags__(self):
        # If we don't implement this trivial method it's not found by the sklearn. This
        # is likely due to the complex resolution of the multiple inheritance.
        # pylint: disable=useless-parent-delegation
        return super().__sklearn_tags__()

    def fit(self, X, y=None, **kwargs):
        """Trains a Khiops Coclustering model

        Parameters
        ----------
        X : :external:term:`array-like` of shape (n_samples, n_features_in) or dict
            Training dataset. Either an :external:term:`array-like` or a ``dict``
            specification for multi-table datasets (see :doc:`/multi_table_primer`).
        id_column : str
            The column that contains the id of the instance.
        columns : list, optional
            The columns to be co-clustered. If not specified it uses all columns.

        Returns
        -------
        self : `KhiopsCoclustering`
            The calling estimator instance.
        """
        return super().fit(X, y=y, **kwargs)

    def _fit_check_params(self, ds, **kwargs):
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
        columns = kwargs.get("columns")
        if columns is not None:
            if not is_list_like(columns):
                raise TypeError(type_error_message("columns", columns, "list-like"))
            else:
                for i, column_id in enumerate(columns):
                    if not isinstance(column_id, (str, int)):
                        raise TypeError(
                            type_error_message(f"columns[{i}]", column_id, str)
                        )
                    if column_id not in ds.main_table.column_ids:
                        raise ValueError(f"columns[{i}] ('{column_id}') not found in X")

        # Check that 'id_column':
        # - Is specified
        # - Is a string
        # - Is contained in the columns names of the main table
        id_column = kwargs.get("id_column")
        if id_column is None:
            raise ValueError("'id_column' is a mandatory parameter")
        if not isinstance(id_column, str):
            raise TypeError(type_error_message("key_columns", id_column, str))
        if id_column not in ds.main_table.column_ids:
            raise ValueError(f"id column '{id_column}' not found in X")

    def _fit_train_model(self, ds, computation_dir, **kwargs):
        assert not ds.is_multitable, "Coclustering not available in multitable"

        # Prepare the table files and dictionary for Khiops
        main_table_path, _ = ds.create_table_files_for_khiops(
            computation_dir, sort=self.auto_sort
        )

        # Set the output paths
        output_dir = self._get_output_dir(computation_dir)
        train_log_file_path = fs.get_child_path(output_dir, "khiops_train_cc.log")

        # Set the 'variables' parameter
        if "columns" in kwargs:
            variables = kwargs["columns"]
        else:
            variables = list(ds.main_table.column_ids)

        coclustering_file_path = fs.get_child_path(
            output_dir, f"{ds.main_table.name}_Coclustering.khcj"
        )

        # Train the coclustering model
        coclustering_file_path = kh.train_coclustering(
            ds.create_khiops_dictionary_domain(),
            ds.main_table.name,
            main_table_path,
            variables,
            coclustering_file_path,
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

        # Save the id column
        if "id_column" in kwargs:
            self.model_id_column = kwargs["id_column"]

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
        tmp_domain = ds.create_khiops_dictionary_domain()
        main_table_dictionary = tmp_domain.get_dictionary(ds.main_table.name)
        if not main_table_dictionary.key:
            main_table_dictionary.key = [self.model_id_column]
        main_table_dictionary.name = f"{self._khiops_model_prefix}{ds.main_table.name}"
        self.model_main_dictionary_name_ = (
            f"{self._khiops_model_prefix}{ds.main_table.name}"
        )
        self.model_secondary_table_variable_name = (
            f"{self._khiops_model_prefix}original_{ds.main_table.name}"
        )
        self._create_coclustering_model_domain(
            tmp_domain, coclustering_file_path, output_dir
        )

        # Update the `model_` attribute of the coclustering estimator to the
        # new coclustering model
        self.model_ = self._read_model_from_dictionary_file(
            fs.get_child_path(
                output_dir, f"{self.model_main_dictionary_name_}_deployed.kdic"
            )
        )

    def _fit_training_post_process(self, ds):
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
        mt_domain = _build_multi_table_dictionary_domain(
            domain,
            self.model_main_dictionary_name_,
            self.model_secondary_table_variable_name,
            update_secondary_table_name=True,
        )

        # Create the model by adding the coclustering variables
        # to the multi-table dictionary created before
        prepare_log_file_path = fs.get_child_path(output_dir, "khiops_prepare_cc.log")
        deployed_coclustering_dictionary_file_path = fs.get_child_path(
            output_dir, f"{self.model_main_dictionary_name_}_deployed.kdic"
        )
        kh.prepare_coclustering_deployment(
            mt_domain,
            self.model_main_dictionary_name_,
            coclustering_file_path,
            self.model_secondary_table_variable_name,
            self.model_id_column,
            deployed_coclustering_dictionary_file_path,
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
        full_coclustering_file_path = fs.get_child_path(
            output_dir, "FullCoclustering.khcj"
        )
        simplified_coclustering_file_path = fs.get_child_path(
            output_dir, "Coclustering.khcj"
        )
        self.model_report_.write_khiops_json_file(full_coclustering_file_path)
        try:
            # - simplify_coclustering, then
            # - prepare_coclustering_deployment
            # - prepare coclustering deployment and re-initialise the `model_`
            #   attribute accordingly
            kh.simplify_coclustering(
                full_coclustering_file_path,
                simplified_coclustering_file_path,
                max_preserved_information=max_preserved_information,
                max_cells=max_cells,
                max_total_parts=max_total_parts,
                max_part_numbers=max_part_numbers,
                log_file_path=simplify_log_file_path,
                trace=self.verbose,
            )

            # Get dataset dictionary from model; it should not be root
            ds_dictionary = self.model_.get_dictionary(
                self.model_secondary_table_variable_name
            )
            assert (
                not ds_dictionary.root
            ), "Dataset dictionary in the coclustering model should not be root"
            if not ds_dictionary.key:
                ds_dictionary.key = self.model_id_column
            domain = DictionaryDomain()
            domain.add_dictionary(ds_dictionary)
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

            # Build the individual-variable coclustering model
            self._create_coclustering_model_domain(
                domain, simplified_coclustering_file_path, output_dir
            )

            # Set the `model_` attribute of the new coclustering estimator to
            # the new coclustering model
            simplified_cc.model_ = self._read_model_from_dictionary_file(
                fs.get_child_path(
                    output_dir, f"{self.model_main_dictionary_name_}_deployed.kdic"
                )
            )
        finally:
            self._cleanup_computation_dir(computation_dir)
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
        check_is_fitted(self)

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

        Returns
        -------
        `ndarray <numpy.ndarray>`
            An array containing the encoded columns. A first column containing key
            column ids is added in multi-table mode.
        """
        # Create temporary directory
        computation_dir = self._create_computation_dir("predict")

        # Create the input dataset
        ds = Dataset(X)

        # Call the template transform method
        try:
            y_pred = super()._transform(
                ds,
                computation_dir,
                self._transform_prepare_deployment_for_predict,
                False,
                "predict.txt",
            )
        # Cleanup and restore the runner's temporary dir
        finally:
            self._cleanup_computation_dir(computation_dir)

        # Transform to numpy.array
        y_pred = y_pred.to_numpy()

        return y_pred

    def _transform_check_dataset(self, ds):
        """Checks the tables before deploying a model on them"""
        assert (
            len(self.model_.dictionaries) == 2
        ), "'model' does not have exactly 2 dictionaries"

        # Coclustering models are special:
        # - They are mono-table only
        # - They are deployed with a multitable model whose main table contain
        #   the keys of the input table and the secondary table is the input table
        if ds.is_multitable:
            raise ValueError("Coclustering models not available in multi-table mode")

        # The "model dictionary domain" in the coclustering case it is just composed
        # of the secondary table. The main "keys" table is a technical object.
        # So we check the compatibility against only this dictionary and override
        # the parents implementation
        for dictionary in self.model_.dictionaries:
            if dictionary.name != self.model_main_dictionary_name_:
                _check_dictionary_compatibility(
                    dictionary,
                    ds.main_table.create_khiops_dictionary(),
                    self.__class__.__name__,
                )

    def _transform_create_deployment_dataset(self, ds, _):
        assert not ds.is_multitable, "'dataset' is multitable"

        # Build the multitable deployment dataset
        deploy_dataset_spec = {}
        deploy_dataset_spec["additional_data_tables"] = {}

        # Extract the keys from the main table
        keys_table_dataframe = pd.DataFrame(
            {
                self.model_id_column: ds.main_table.data_source[
                    self.model_id_column
                ].unique()
            }
        )

        # Create the dataset with the keys table as the main one
        deploy_dataset_spec["main_table"] = (
            keys_table_dataframe,
            [self.model_id_column],
        )
        deploy_dataset_spec["additional_data_tables"][
            f"{self._khiops_model_prefix}original_main_table"
        ] = (
            ds.main_table.data_source,
            [self.model_id_column],
        )

        return Dataset(deploy_dataset_spec)

    def _transform_prepare_deployment_for_predict(self, _):
        # TODO: Replace the second return value (the output columns' dtypes) with a
        #       proper value instead of `None`. In the current state, it will use pandas
        #       type auto-detection to load the internal table into memory.
        return self.model_.copy(), None

    def fit_predict(self, X, y=None, **kwargs):
        """Performs clustering on X and returns result (instead of labels)"""
        return self.fit(X, y, **kwargs).predict(X)


class KhiopsSupervisedEstimator(KhiopsEstimator):
    """Abstract Khiops Supervised Estimator"""

    def __init__(
        self,
        n_features=100,
        n_trees=10,
        n_text_features=10000,
        type_text_features="words",
        specific_pairs=None,
        all_possible_pairs=True,
        construction_rules=None,
        verbose=False,
        output_dir=None,
        auto_sort=True,
    ):
        super().__init__(
            verbose=verbose,
            output_dir=output_dir,
            auto_sort=auto_sort,
        )
        self.n_features = n_features
        self.n_trees = n_trees
        self.n_text_features = n_text_features
        self.type_text_features = type_text_features
        self.specific_pairs = specific_pairs
        self.all_possible_pairs = all_possible_pairs
        self.construction_rules = construction_rules
        self._original_target_dtype = None
        self._predicted_target_meta_data_tag = None
        self._khiops_baseline_model_prefix = None

    def __sklearn_tags__(self):
        # If we don't implement this trivial method it's not found by the sklearn. This
        # is likely due to the complex resolution of the multiple inheritance.
        # pylint: disable=useless-parent-delegation
        return super().__sklearn_tags__()

    def _more_tags(self):
        more_tags = super()._more_tags()
        more_tags["requires_y"] = True
        return more_tags

    def _fit_check_dataset(self, ds):
        super()._fit_check_dataset(ds)
        self._check_target_type(ds)

    @abstractmethod
    def _check_target_type(self, ds):
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

        y : :external:term:`array-like` of shape (n_samples,)
            The target values.

        Returns
        -------
        self : `KhiopsSupervisedEstimator`
            The calling estimator instance.
        """
        if y is None:
            raise ValueError(
                f"{self.__class__.__name__} requires y to be passed, "
                "but the target y is None"
            )
        super().fit(X, y=y, **kwargs)
        return self

    def _fit_check_params(self, ds, **kwargs):
        # Call parent method
        super()._fit_check_params(ds, **kwargs)

        # Check supervised estimator parameters
        if not isinstance(self.n_features, int):
            raise TypeError(type_error_message("n_features", self.n_features, int))
        if self.n_features < 0:
            raise ValueError("'n_features' must be positive")
        if not isinstance(self.n_trees, int):
            raise TypeError(type_error_message("n_trees", self.n_trees, int))
        if self.n_trees < 0:
            raise ValueError("'n_trees' must be positive")
        if not isinstance(self.n_text_features, int):
            raise TypeError(
                type_error_message("n_text_features", self.n_text_features, int)
            )
        if self.n_text_features < 0:
            raise ValueError("'n_text_features' must be positive")
        if not isinstance(self.type_text_features, str):
            raise TypeError(
                type_error_message("type_text_features", self.type_text_features, str)
            )
        if self.type_text_features not in ("words", "ngrams", "tokens"):
            raise ValueError(
                "'type_text_features' must be among 'words', 'ngrams' or 'tokens'"
            )
        if self.construction_rules is not None:
            if not is_list_like(self.construction_rules):
                raise TypeError(
                    type_error_message(
                        "construction_rules", self.construction_rules, "list-like"
                    )
                )
            else:
                for rule in self.construction_rules:
                    if not isinstance(rule, str):
                        raise TypeError(type_error_message(rule, rule, str))

    def _fit_train_model(self, ds, computation_dir, **kwargs):
        # Train the model with Khiops
        train_args, train_kwargs = self._fit_prepare_training_function_inputs(
            ds, computation_dir
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
        self.model_ = self._read_model_from_dictionary_file(model_kdic_file_path)
        self.model_report_ = kh.read_analysis_results_file(report_file_path)

    @abstractmethod
    def _fit_core_training_function(self, *args, **kwargs):
        """A wrapper to the khiops.core training function for the estimator"""

    def _fit_prepare_training_function_inputs(self, ds, computation_dir):
        # Set output path files
        output_dir = self._get_output_dir(computation_dir)
        report_file_path = fs.get_child_path(
            output_dir, f"{ds.main_table.name}_AnalysisResults.khj"
        )
        log_file_path = fs.get_child_path(output_dir, "khiops.log")

        main_table_path, secondary_table_paths = ds.create_table_files_for_khiops(
            computation_dir, sort=self.auto_sort
        )

        # Build the 'additional_data_tables' argument
        ds_domain = ds.create_khiops_dictionary_domain()
        secondary_data_paths = ds_domain.extract_data_paths(ds.main_table.name)
        additional_data_tables = {}
        for data_path in secondary_data_paths:
            path_bits = []
            data_path_fragments = data_path.split("/")
            for path_fragment in data_path_fragments:
                path_subfragments = path_fragment.split("§")
                for path_subfragment in path_subfragments:
                    if path_subfragment not in path_bits:
                        path_bits.append(path_subfragment)
            simplified_data_path = "/".join(path_bits)

            additional_data_tables[data_path] = secondary_table_paths[
                simplified_data_path
            ]

        # Build the mandatory arguments
        args = [
            ds_domain,
            ds.main_table.name,
            main_table_path,
            get_khiops_variable_name(ds.target_column_id),
            report_file_path,
        ]

        # Build the optional parameters from a copy of the estimator parameters
        kwargs = self.get_params()

        # Remove non core.api params
        del kwargs["output_dir"]
        del kwargs["auto_sort"]

        # Set the sampling percentage to a 100%
        kwargs["sample_percentage"] = 100

        # Set the format parameters depending on the type of dataset
        kwargs["detect_format"] = False
        kwargs["field_separator"] = "\t"
        kwargs["header_line"] = True

        # Rename parameters to be compatible with khiops.core
        kwargs["max_constructed_variables"] = kwargs.pop("n_features")
        kwargs["max_trees"] = kwargs.pop("n_trees")
        kwargs["max_text_features"] = kwargs.pop("n_text_features")
        kwargs["text_features"] = kwargs.pop("type_text_features")

        # Add the additional_data_tables parameter
        kwargs["additional_data_tables"] = additional_data_tables

        # Set the log file and trace parameters
        kwargs["log_file_path"] = log_file_path
        kwargs["trace"] = kwargs["verbose"]
        del kwargs["verbose"]

        return args, kwargs

    def _fit_training_post_process(self, ds):
        # Call parent method
        super()._fit_training_post_process(ds)

        # Save the target and key column dtype's
        if self._original_target_dtype is None:
            self._original_target_dtype = ds.target_column.dtype
        if ds.main_table.key is not None:
            self._original_key_dtypes = {}
            for column_id in ds.main_table.key:
                self._original_key_dtypes[column_id] = ds.main_table.get_column_dtype(
                    column_id
                )
        else:
            self._original_key_dtypes = None

        # Set the target variable name
        self.model_target_variable_name_ = get_khiops_variable_name(ds.target_column_id)

        # Verify it has at least one dictionary and a root dictionary in multi-table
        if len(self.model_.dictionaries) == 1:
            self.model_main_dictionary_name_ = self.model_.dictionaries[0].name
        else:
            for dictionary in self.model_.dictionaries:

                # The baseline model is mandatory for regression;
                # absent for classification and encoding
                assert dictionary.name.startswith(
                    self._khiops_model_prefix
                ) or dictionary.name.startswith(self._khiops_baseline_model_prefix), (
                    f"Dictionary '{dictionary.name}' "
                    f"does not have prefix '{self._khiops_model_prefix}' "
                    f"or '{self._khiops_baseline_model_prefix}'."
                )
                # Skip baseline model
                if dictionary.name.startswith(self._khiops_model_prefix):
                    initial_dictionary_name = dictionary.name.replace(
                        self._khiops_model_prefix, "", 1
                    )
                    if initial_dictionary_name == ds.main_table.name:
                        self.model_main_dictionary_name_ = dictionary.name
        if self.model_main_dictionary_name_ is None:
            raise ValueError("No model dictionary after Khiops call")

    def _transform_check_dataset(self, ds):
        assert isinstance(ds, Dataset), "'ds' is not 'Dataset'"

        # Check the coherence between the input table and the model
        if self.is_multitable_model_ and not ds.is_multitable:
            raise ValueError(
                "You are trying to apply on single-table inputs a model which has "
                "been trained on multi-table data."
            )
        if not self.is_multitable_model_ and ds.is_multitable:
            raise ValueError(
                "You are trying to apply on multi-table inputs a model which has "
                "been trained on single-table data."
            )

        # Error if different number of dictionaries
        ds_domain = ds.create_khiops_dictionary_domain()
        if len(self.model_.dictionaries) != len(ds_domain.dictionaries):
            raise ValueError(
                f"X has {len(ds_domain.dictionaries)} table(s), "
                f"but {self.__class__.__name__} is expecting "
                f"{len(self.model_.dictionaries)}"
            )

        # Check the main table compatibility
        # Note: Name checking is omitted for the main table
        _check_dictionary_compatibility(
            _extract_basic_dictionary(self._get_main_dictionary()),
            ds.main_table.create_khiops_dictionary(),
            self.__class__.__name__,
            target_variable_name=self.model_target_variable_name_,
        )

        # Multi-table model: Check name and dictionary coherence of secondary tables
        dataset_secondary_tables_by_name = {
            table.name: table for _, table, _ in ds.additional_data_tables
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
        n_trees=10,
        n_text_features=10000,
        type_text_features="words",
        n_selected_features=0,
        n_evaluated_features=0,
        specific_pairs=None,
        all_possible_pairs=True,
        construction_rules=None,
        verbose=False,
        output_dir=None,
        auto_sort=True,
    ):
        super().__init__(
            n_features=n_features,
            n_trees=n_trees,
            n_text_features=n_text_features,
            type_text_features=type_text_features,
            specific_pairs=specific_pairs,
            all_possible_pairs=all_possible_pairs,
            construction_rules=construction_rules,
            verbose=verbose,
            output_dir=output_dir,
            auto_sort=auto_sort,
        )
        # Data to be specified by inherited classes
        self._predicted_target_meta_data_tag = None
        self._predicted_target_name_prefix = None
        self.n_evaluated_features = n_evaluated_features
        self.n_selected_features = n_selected_features

    def __sklearn_tags__(self):
        # If we don't implement this trivial method it's not found by the sklearn. This
        # is likely due to the complex resolution of the multiple inheritance.
        # pylint: disable=useless-parent-delegation
        return super().__sklearn_tags__()

    def predict(self, X):
        """Predicts the target variable for the test dataset X

        See the documentation of concrete subclasses for more details.
        """
        # Create temporary directory
        computation_dir = self._create_computation_dir("predict")

        try:
            # Create the input dataset
            ds = Dataset(X)

            # Call the template transform method
            y_pred = super()._transform(
                ds,
                computation_dir,
                self._transform_prepare_deployment_for_predict,
                True,
                "predict.txt",
            )
        # Cleanup and restore the runner's temporary dir
        finally:
            self._cleanup_computation_dir(computation_dir)

        # Return pd.Series in the monotable + pandas case
        assert isinstance(y_pred, (str, pd.DataFrame)), "Expected str or DataFrame"
        return y_pred

    def _fit_prepare_training_function_inputs(self, ds, computation_dir):
        # Call the parent method
        args, kwargs = super()._fit_prepare_training_function_inputs(
            ds, computation_dir
        )

        # Rename parameters to be compatible with khiops.core
        kwargs["max_evaluated_variables"] = kwargs.pop("n_evaluated_features")
        kwargs["max_selected_variables"] = kwargs.pop("n_selected_features")

        return args, kwargs

    def _transform_prepare_deployment_for_predict(self, ds):
        assert (
            self._predicted_target_meta_data_tag is not None
        ), "Predicted target metadata tag is not set"
        assert hasattr(
            self, "model_main_dictionary_name_"
        ), "Model main dictionary name has not been set"

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

        # Remove the target variable if it is not present in the input dataset
        # Note: We use `list` to avoid a warning of numpy about the `in` operator
        if self.model_target_variable_name_ not in list(ds.main_table.column_ids):
            model_dictionary.remove_variable(self.model_target_variable_name_)

        # Create the output column dtype dict
        predicted_target_column_name = (
            self._predicted_target_name_prefix + self.model_target_variable_name_
        )
        output_columns_dtype = {
            predicted_target_column_name: self._original_target_dtype
        }
        if self.is_multitable_model_:
            output_columns_dtype.update(self._original_key_dtypes)

        return model_copy, output_columns_dtype

    def _fit_check_params(self, ds, **kwargs):
        # Call parent method
        super()._fit_check_params(ds, **kwargs)

        # Check estimator parameters
        if self.n_evaluated_features < 0:
            raise ValueError("'n_evaluated_features' must be positive")
        if self.n_selected_features < 0:
            raise ValueError("'n_selected_features' must be positive")


# Note: scikit-learn **requires** inherit first the mixins and then other classes
class KhiopsClassifier(ClassifierMixin, KhiopsPredictor):
    # Disable line too long as this docstring *needs* to have lines longer than 88c
    # pylint: disable=line-too-long
    r"""Khiops Selective Naive Bayes Classifier

    This classifier supports automatic feature engineering on multi-table datasets. See
    :doc:`/multi_table_primer` for more details.

    .. note::

        Visit `the Khiops site <https://khiops.org/learn/understand>`_ to learn
        about the automatic feature engineering algorithm.

    Parameters
    ----------
    n_features : int, default 100
        Maximum number of features to construct automatically. See
        :doc:`/multi_table_primer` for more details on the multi-table-specific
        features.
    n_pairs : int, default 0
        Maximum number of pair features to construct. These features are 2D grid
        partitions of univariate feature pairs. The grid is optimized such that in each
        cell the target distribution is well approximated by a constant histogram. Only
        pairs that are jointly more informative than their marginals may be taken into
        account in the classifier.
    n_trees : int, default 10
        Maximum number of decision tree features to construct. The constructed trees
        combine other features, either native or constructed. These features usually
        improve the classifier's performance at the cost of interpretability of the
        model.
    n_text_features : int, default 10000
        Maximum number of text features to construct.
    type_text_features : str, default "words"
        Type of the text features to construct. Can be either one of:
            - "words": sequences of non-space characters
            - "ngrams": sequences of bytes
            - "tokens": user-defined
    n_selected_features : int, default 0
        Maximum number of features to be selected in the SNB predictor. If equal to
        0 it selects all the features kept in the training.
    n_evaluated_features : int, default 0
        Maximum number of features to be evaluated in the SNB predictor training. If
        equal to 0 it evaluates all informative features.
    specific_pairs : list of tuple, optional
        User-specified pairs as a list of 2-tuples of feature names. If a given tuple
        contains only one non-empty feature name, then it generates all the pairs
        containing it (within the maximum limit ``n_pairs``). These pairs have top
        priority: they are constructed first.
    all_possible_pairs : bool, default ``True``
        If ``True`` tries to create all possible pairs within the limit ``n_pairs``.
        Pairs specified with ``specific_pairs`` have top priority: they are constructed
        first.
    construction_rules : list of str, optional
        Allowed rules for the automatic feature construction. If not set, Khiops
        uses the multi-table construction rules listed in
        `kh.DEFAULT_CONSTRUCTION_RULES <khiops.core.api.DEFAULT_CONSTRUCTION_RULES>`
    group_target_value : bool, default ``False``
        Allows grouping of the target values in classification. It can substantially
        increase the training time.
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

    Attributes
    ----------
    n_classes_ : int
        The number of classes seen in training.
    classes_ : `ndarray <numpy.ndarray>` of shape (n_classes\_,)
        The list of classes seen in training. Depending on the training target, the
        contents are ``int`` or ``str``.
    n_features_in_ : int
        The number of features in the main table of the training dataset.
    is_multitable_model_ : bool
        ``True`` if the model was fitted on a multi-table dataset.
    model_ : `.DictionaryDomain`
        The Khiops dictionary domain for the trained classifier.
    model_main_dictionary_name_ : str
        The name of the main Khiops dictionary within the ``model_`` domain.
    model_report_ : `.AnalysisResults`
        The Khiops report object.

    Examples
    --------
    See the following functions of the ``samples_sklearn.py`` documentation script:
        - `samples_sklearn.khiops_classifier()`
        - `samples_sklearn.khiops_classifier_multiclass()`
        - `samples_sklearn.khiops_classifier_multitable_star()`
        - `samples_sklearn.khiops_classifier_multitable_snowflake()`
        - `samples_sklearn.khiops_classifier_pickle()`
    """
    # pylint: enable=line-too-long

    def __init__(
        self,
        n_features=100,
        n_pairs=0,
        n_trees=10,
        n_text_features=10000,
        type_text_features="words",
        n_selected_features=0,
        n_evaluated_features=0,
        specific_pairs=None,
        all_possible_pairs=True,
        construction_rules=None,
        group_target_value=False,
        verbose=False,
        output_dir=None,
        auto_sort=True,
    ):
        super().__init__(
            n_features=n_features,
            n_trees=n_trees,
            n_text_features=n_text_features,
            type_text_features=type_text_features,
            n_selected_features=n_selected_features,
            n_evaluated_features=n_evaluated_features,
            construction_rules=construction_rules,
            verbose=verbose,
            output_dir=output_dir,
            auto_sort=auto_sort,
        )
        self.n_pairs = n_pairs
        self.specific_pairs = specific_pairs
        self.all_possible_pairs = all_possible_pairs
        self.group_target_value = group_target_value
        self._khiops_model_prefix = "SNB_"
        self._predicted_target_meta_data_tag = "Prediction"
        self._predicted_target_name_prefix = "Predicted"

    def __sklearn_tags__(self):
        # If we don't implement this trivial method it's not found by the sklearn. This
        # is likely due to the complex resolution of the multiple inheritance.
        # pylint: disable=useless-parent-delegation
        return super().__sklearn_tags__()

    def _is_real_target_dtype_integer(self):
        return self._original_target_dtype is not None and (
            pd.api.types.is_integer_dtype(self._original_target_dtype)
            or (
                isinstance(self._original_target_dtype, pd.CategoricalDtype)
                and pd.api.types.is_integer_dtype(
                    self._original_target_dtype.categories
                )
            )
        )

    def _is_real_target_dtype_float(self):
        return self._original_target_dtype is not None and (
            pd.api.types.is_float_dtype(self._original_target_dtype)
            or (
                isinstance(self._original_target_dtype, pd.CategoricalDtype)
                and pd.api.types.is_float_dtype(self._original_target_dtype.categories)
            )
        )

    def _is_real_target_dtype_bool(self):
        return self._original_target_dtype is not None and (
            pd.api.types.is_bool_dtype(self._original_target_dtype)
            or (
                isinstance(self._original_target_dtype, pd.CategoricalDtype)
                and pd.api.types.is_bool_dtype(self._original_target_dtype.categories)
            )
        )

    def _sorted_prob_variable_names(self):
        """Returns the model probability variable names in the order of self.classes_"""
        self._assert_is_fitted()

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

    def _fit_check_params(self, ds, **kwargs):
        # Call parent method
        super()._fit_check_params(ds, **kwargs)

        # Check the pair related parameters
        _check_pair_parameters(self)

    def _fit_prepare_training_function_inputs(self, ds, computation_dir):
        # Call the parent method
        args, kwargs = super()._fit_prepare_training_function_inputs(
            ds, computation_dir
        )

        # Rename parameters to be compatible with khiops.core
        kwargs["max_pairs"] = kwargs.pop("n_pairs")

        return args, kwargs

    def fit(self, X, y, **kwargs):
        """Fits a Selective Naive Bayes classifier according to X, y

        Parameters
        ----------
        X : :external:term:`array-like` of shape (n_samples, n_features_in) or dict
            Training dataset. Either an :external:term:`array-like` or a ``dict``
            specification for multi-table datasets (see :doc:`/multi_table_primer`).

        y : :external:term:`array-like` of shape (n_samples,)
            The target values.

        Returns
        -------
        self : `KhiopsClassifier`
            The calling estimator instance.
        """
        kwargs["categorical_target"] = True
        return super().fit(X, y, **kwargs)

    def _check_target_type(self, ds):
        _check_categorical_target_type(ds)

    def _fit_check_dataset(self, ds):
        # Call the parent method
        super()._fit_check_dataset(ds)

        # Check that the target is for classification
        current_type_of_target = type_of_target(ds.target_column)
        if current_type_of_target not in ["binary", "multiclass"]:
            raise ValueError(
                f"Unknown label type: '{current_type_of_target}' "
                "for classification. Maybe you passed a floating point target?"
            )
        # Check if the target has more than 1 class
        if len(np.unique(ds.target_column)) == 1:
            raise ValueError(
                f"{self.__class__.__name__} can't train when only one class is present."
            )

    def _fit_core_training_function(self, *args, **kwargs):
        return kh.train_predictor(*args, **kwargs)

    def _fit_training_post_process(self, ds):
        # Call the parent's method
        super()._fit_training_post_process(ds)

        # Save class values in the order of deployment
        self.classes_ = []
        for variable in self._get_main_dictionary().variables:
            for key in variable.meta_data.keys:
                if key.startswith("TargetProb"):
                    self.classes_.append(variable.meta_data.get_value(key))
        if self._is_real_target_dtype_integer():
            self.classes_ = [int(class_value) for class_value in self.classes_]
        elif self._is_real_target_dtype_float():
            self.classes_ = [float(class_value) for class_value in self.classes_]
        elif self._is_real_target_dtype_bool():
            self.classes_ = [class_value == "True" for class_value in self.classes_]
        self.classes_.sort()
        self.classes_ = column_or_1d(self.classes_)

        # Count number of classes
        self.n_classes_ = len(self.classes_)

        # Warn when there are no informative variables
        if self.model_report_.preparation_report.informative_variable_number == 0:
            warnings.warn(
                "There are no informative variables. "
                "The fitted model is the majority class classifier.",
                stacklevel=6,
            )

        # Set the target class probabilities as used
        # (only the predicted classes are obtained without this step prior to Khiops 10)
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

        Returns
        -------
        `ndarray <numpy.ndarray>`
            An array containing the encoded columns. A first column containing key
            column ids is added in multi-table mode. The `numpy.dtype` of the array
            matches the type of ``y`` used during training. It will be integer, float,
            or boolean if the classifier was trained with a ``y`` of the corresponding
            type. Otherwise it will be ``str``.

            The key columns are added for multi-table tasks.
        """
        # Call the parent's method
        y_pred = super().predict(X)

        # Convert to numpy if it is in memory
        if isinstance(y_pred, pd.DataFrame):
            y_pred = y_pred[
                self._predicted_target_name_prefix + self.model_target_variable_name_
            ].to_numpy(copy=False)

        assert isinstance(y_pred, (np.ndarray, str)), type_error_message(
            "y_pred", y_pred, np.ndarray, str
        )
        return y_pred

    def predict_proba(self, X):
        """Predicts the class probabilities for the test dataset X

        Parameters
        ----------
        X : :external:term:`array-like` of shape (n_samples, n_features_in) or dict
            Training dataset. Either an :external:term:`array-like` or a ``dict``
            specification for multi-table datasets (see :doc:`/multi_table_primer`).

        Returns
        -------
        `numpy.array` or str
            The probability of the samples for each class in the model.  The columns are
            named with the pattern ``Prob<class>`` for each ``<class>`` found in the
            training dataset. The output data container depends on ``X``:

                - Dataframe or dataframe-based ``dict`` dataset specification:
                  `numpy.array`

            The key columns are added for multi-table tasks.
        """
        # Create temporary directory and tables
        computation_dir = self._create_computation_dir("predict_proba")

        # Create the input dataset

        # Call the generic transform method
        try:
            ds = Dataset(X)
            y_probas = self._transform(
                ds,
                computation_dir,
                self._transform_prepare_deployment_for_predict_proba,
                True,
                "predict_proba.txt",
            )
        # Cleanup and restore the runner's temporary dir
        finally:
            self._cleanup_computation_dir(computation_dir)

        # - Reorder the columns to that of self.classes_
        # - Transform to np.ndarray
        assert isinstance(
            y_probas, (pd.DataFrame, np.ndarray)
        ), "y_probas is not a Pandas DataFrame nor Numpy array"
        y_probas = y_probas.reindex(
            self._sorted_prob_variable_names(), axis=1, copy=False
        ).to_numpy(copy=False)

        assert isinstance(y_probas, (str, np.ndarray)), "Expected str or np.ndarray"
        return y_probas

    def _transform_prepare_deployment_for_predict_proba(self, ds):
        assert hasattr(
            self, "model_target_variable_name_"
        ), "Target variable name has not been set"

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

        # Remove the target variable if it is not present in the input dataset
        # Note: We use `list` to avoid a warning of numpy about the `in` operator
        if self.model_target_variable_name_ not in list(ds.main_table.column_ids):
            model_dictionary.remove_variable(self.model_target_variable_name_)

        output_columns_dtype = {}
        if self.is_multitable_model_:
            output_columns_dtype.update(self._original_key_dtypes)
        for variable in model_dictionary.variables:
            if variable.used and variable.name not in model_dictionary.key:
                output_columns_dtype[variable.name] = np.float64

        return model_copy, output_columns_dtype


# Note: scikit-learn **requires** inherit first the mixins and then other classes
class KhiopsRegressor(RegressorMixin, KhiopsPredictor):
    # Disable line too long as this docstring *needs* to have lines longer than 88c
    # pylint: disable=line-too-long
    r"""Khiops Selective Naive Bayes Regressor

    This regressor supports automatic feature engineering on multi-table datasets. See
    :doc:`/multi_table_primer` for more details.

    .. note::

        Visit `the Khiops site <https://khiops.org/learn/understand>`_ to learn
        about the automatic feature engineering algorithm.

    Parameters
    ----------
    n_features : int, default 100
        Maximum number of features to construct automatically. See
        :doc:`/multi_table_primer` for more details on the multi-table-specific
        features.
    n_selected_features : int, default 0
        Maximum number of features to be selected in the SNB predictor. If equal to
        0 it selects all the features kept in the training.
    n_evaluated_features : int, default 0
        Maximum number of features to be evaluated in the SNB predictor training. If
        equal to 0 it evaluates all informative features.
    construction_rules : list of str, optional
        Allowed rules for the automatic feature construction. If not set, Khiops
        uses the multi-table construction rules listed in
        `kh.DEFAULT_CONSTRUCTION_RULES <khiops.core.api.DEFAULT_CONSTRUCTION_RULES>`.
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

    Attributes
    ----------
    n_features_in_ : int
        The number of features in the main table of the training dataset.
    is_multitable_model_ : bool
        ``True`` if the model was fitted on a multi-table dataset.
    model_ : `.DictionaryDomain`
        The Khiops dictionary domain for the trained regressor.
    model_main_dictionary_name_ : str
        The name of the main Khiops dictionary within the ``model_`` domain.
    model_report_ : `.AnalysisResults`
        The Khiops report object.

    Examples
    --------
    See the following functions of the ``samples_sklearn.py`` documentation script:
        - `samples_sklearn.khiops_regressor()`
    """
    # pylint: enable=line-too-long

    def __init__(
        self,
        n_features=100,
        n_trees=0,
        n_text_features=10000,
        type_text_features="words",
        n_selected_features=0,
        n_evaluated_features=0,
        construction_rules=None,
        verbose=False,
        output_dir=None,
        auto_sort=True,
    ):
        super().__init__(
            n_features=n_features,
            n_trees=n_trees,
            n_text_features=n_text_features,
            type_text_features=type_text_features,
            n_selected_features=n_selected_features,
            n_evaluated_features=n_evaluated_features,
            construction_rules=construction_rules,
            verbose=verbose,
            output_dir=output_dir,
            auto_sort=auto_sort,
        )
        self._khiops_model_prefix = "SNB_"
        self._khiops_baseline_model_prefix = "B_"
        self._predicted_target_meta_data_tag = "Mean"
        self._predicted_target_name_prefix = "M"
        self._original_target_dtype = np.float64

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

        y : :external:term:`array-like` of shape (n_samples,)
            The target values.

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

    def _fit_train_model(self, ds, computation_dir, **kwargs):
        # Call the parent method
        super()._fit_train_model(ds, computation_dir, **kwargs)

        # Warn when there are no informative variables
        if self.model_report_.preparation_report.informative_variable_number == 0:
            warnings.warn(
                "There are no informative variables. "
                "The fitted model is the mean regressor."
            )

    def _fit_training_post_process(self, ds):
        # Call parent method
        super()._fit_training_post_process(ds)

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

    def _check_target_type(self, ds):
        _check_numerical_target_type(ds)

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

        Returns
        -------
        `numpy.ndarray` or str

            An array containing the encoded columns. A first column containing key
            column ids is added in multi-table mode. The key columns are added for
            multi-table tasks. The array is in the form of:

            - `numpy.ndarray` if X is :external:term:`array-like`, or dataset spec
              containing `pandas.DataFrame` table.
            - str (a path for the file containing the array) if X is a dataset spec
              containing file-path tables.
        """
        assert (
            self._khiops_baseline_model_prefix is not None
        ), "Baseline model prefix is not set (mandatory for regression)"
        # Call the parent's method
        y_pred = super().predict(X)

        # Transform to np.ndarray
        if isinstance(y_pred, pd.DataFrame):
            y_pred = y_pred.astype("float64", copy=False).to_numpy(copy=False).ravel()

        assert isinstance(y_pred, (str, np.ndarray)), "Expected str or np.array"
        return y_pred

    # pylint: enable=useless-super-delegation


# Note: scikit-learn **requires** inherit first the mixins and then other classes
class KhiopsEncoder(TransformerMixin, KhiopsSupervisedEstimator):
    # Disable line too long as this docstring *needs* to have lines longer than 88c
    # pylint: disable=line-too-long
    r"""Khiops supervised discretization/grouping encoder

    Parameters
    ----------
    categorical_target : bool, default ``True``
        ``True`` if the target column is categorical.
    n_features : int, default 100
        Maximum number of features to construct automatically. See
        :doc:`/multi_table_primer` for more details on the multi-table-specific
        features.
    n_pairs : int, default 0
        Maximum number of pair features to construct. These features are 2D grid
        partitions of univariate feature pairs. The grid is optimized such that in each
        cell the target distribution is well approximated by a constant histogram. Only
        pairs that are jointly more informative than their marginals may be taken into
        account in the encoder.
    n_trees : int, default 10
        Maximum number of decision tree features to construct. The constructed trees
        combine other features, either native or constructed. These features usually
        improve a predictor's performance at the cost of interpretability of the model.
    n_text_features : int, default 10000
        Maximum number of text features to construct.
    type_text_features : str, default "words"
        Type of the text features to construct. Can be either one of:
            - "words": sequences of non-space characters
            - "ngrams": sequences of bytes
            - "tokens": user-defined
    specific_pairs : list of tuple, optional
        User-specified pairs as a list of 2-tuples of feature names. If a given tuple
        contains only one non-empty feature name, then it generates all the pairs
        containing it (within the maximum limit ``n_pairs``). These pairs have top
        priority: they are constructed first.
    all_possible_pairs : bool, default ``True``
        If ``True`` tries to create all possible pairs within the limit ``n_pairs``.
        Pairs specified with ``specific_pairs`` have top priority: they are constructed
        first.
    construction_rules : list of str, optional
        Allowed rules for the automatic feature construction. If not set, Khiops
        uses the multi-table construction rules listed in
        `kh.DEFAULT_CONSTRUCTION_RULES <khiops.core.api.DEFAULT_CONSTRUCTION_RULES>`.
    informative_features_only : bool, default ``True``
        If ``True`` keeps only informative features.
    group_target_value : bool, default ``False``
        Allows grouping of the target values in classification. It can substantially
        increase the training time.
    keep_initial_variables : bool, default ``False``
        If ``True`` the original columns are kept in the transformed data.
    transform_type_categorical : str, default "part_id"
        Type of transformation for categorical features. Valid values:
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
    transform_type_pairs : str, default "part_id"
        Type of transformation for bivariate features. Valid values:
            - "part_id"
            - "part_label"
            - "dummies"
            - "conditional_info"
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

    Attributes
    ----------
    is_multitable_model_ : bool
        ``True`` if the model was fitted on a multi-table dataset.
    model_ : `.DictionaryDomain`
        The Khiops dictionary domain for the trained encoder.
    model_main_dictionary_name_ : str
        The name of the main Khiops dictionary within the ``model_`` domain.
    model_report_ : `.AnalysisResults`
        The Khiops report object.

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
        n_text_features=10000,
        type_text_features="words",
        specific_pairs=None,
        all_possible_pairs=True,
        construction_rules=None,
        informative_features_only=True,
        group_target_value=False,
        keep_initial_variables=False,
        transform_type_categorical="part_id",
        transform_type_numerical="part_id",
        transform_type_pairs="part_id",
        verbose=False,
        output_dir=None,
        auto_sort=True,
    ):
        super().__init__(
            n_features=n_features,
            n_trees=n_trees,
            n_text_features=n_text_features,
            type_text_features=type_text_features,
            construction_rules=construction_rules,
            verbose=verbose,
            output_dir=output_dir,
            auto_sort=auto_sort,
        )
        self.n_pairs = n_pairs
        self.specific_pairs = specific_pairs
        self.all_possible_pairs = all_possible_pairs
        self.categorical_target = categorical_target
        self.group_target_value = group_target_value
        self.transform_type_categorical = transform_type_categorical
        self.transform_type_numerical = transform_type_numerical
        self.transform_type_pairs = transform_type_pairs
        self.informative_features_only = informative_features_only
        self.keep_initial_variables = keep_initial_variables
        self._khiops_model_prefix = "R_"

    def __sklearn_tags__(self):
        # If we don't implement this trivial method it's not found by the sklearn. This
        # is likely due to the complex resolution of the multiple inheritance.
        # pylint: disable=useless-parent-delegation
        return super().__sklearn_tags__()

    def _more_tags(self):
        more_tags = super()._more_tags()
        more_tags["preserves_dtype"] = []
        return more_tags

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

    def _pairs_transform_method(self):
        _transform_types = {
            "part_id": "part Id",
            "part_label": "part label",
            "dummies": "0-1 binarization",
            "conditional_info": "conditional info",
            None: "none",
        }
        if self.transform_type_pairs not in _transform_types:
            raise ValueError(
                "'transform_type_pairs' must be one of the following:"
                ",".join(_transform_types.keys)
            )
        return _transform_types[self.transform_type_pairs]

    def _fit_check_params(self, ds, **kwargs):
        # Call parent method
        super()._fit_check_params(ds, **kwargs)

        # Check the pair related parameters
        _check_pair_parameters(self)

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

        # Check 'transform_type_pairs' parameter
        if not isinstance(self.transform_type_pairs, str):
            raise TypeError(
                type_error_message(
                    "transform_type_pairs", self.transform_type_pairs, str
                )
            )
        self._pairs_transform_method()  # Raises ValueError if invalid

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

        # Check 'informative_features_only' parameter
        if not isinstance(self.informative_features_only, bool):
            raise TypeError(
                type_error_message(
                    "informative_features_only", self.informative_features_only, bool
                )
            )

        # Check 'group_target_value' parameter
        if not isinstance(self.group_target_value, bool):
            raise TypeError(
                type_error_message("group_target_value", self.group_target_value, bool)
            )

    def _fit_train_model(self, ds, computation_dir, **kwargs):
        # Call the parent method
        super()._fit_train_model(ds, computation_dir, **kwargs)

        # Check whether there are any used variables other than the target
        model_doesnt_have_output_vars = True
        for var in self.model_.get_dictionary(f"R_{ds.main_table.name}").variables:
            if var.name != ds.target_column_id and var.used:
                model_doesnt_have_output_vars = False
                break

        # If no informative vars undo the "fit" state (undefine attributes) and warn
        if model_doesnt_have_output_vars:
            self._undefine_estimator_attributes()
            warnings.warn(
                "Encoder is not fit because Khiops didn't create any output "
                "variables. If you want to use non-informative features in the encoder "
                "set 'informative_features_only' to False."
            )

    def _check_target_type(self, ds):
        if self.categorical_target:
            _check_categorical_target_type(ds)
        else:
            _check_numerical_target_type(ds)

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

        y : :external:term:`array-like` of shape (n_samples,)
            The target values.

        Returns
        -------
        self : `KhiopsEncoder`
            The calling estimator instance.
        """
        kwargs["categorical_target"] = self.categorical_target
        return super().fit(X, y, **kwargs)

    # pylint: enable=useless-super-delegation

    def _fit_prepare_training_function_inputs(self, ds, computation_dir):
        # Call the parent method
        args, kwargs = super()._fit_prepare_training_function_inputs(
            ds, computation_dir
        )
        # Rename encoder parameters, delete unused ones
        # to be compatible with khiops.core
        kwargs["max_pairs"] = kwargs.pop("n_pairs")
        kwargs["keep_initial_categorical_variables"] = kwargs["keep_initial_variables"]
        kwargs["keep_initial_numerical_variables"] = kwargs.pop(
            "keep_initial_variables"
        )
        kwargs["categorical_recoding_method"] = self._categorical_transform_method()
        kwargs["numerical_recoding_method"] = self._numerical_transform_method()
        kwargs["pairs_recoding_method"] = self._pairs_transform_method()
        kwargs["informative_variables_only"] = kwargs.pop("informative_features_only")

        del kwargs["transform_type_categorical"]
        del kwargs["transform_type_numerical"]
        del kwargs["transform_type_pairs"]
        del kwargs["categorical_target"]

        return args, kwargs

    def _fit_training_post_process(self, ds):
        # Call parent method
        super()._fit_training_post_process(ds)

        # Eliminate the target variable from the main dictionary
        self._get_main_dictionary()

        # Save the encoded feature names
        self.feature_names_out_ = []
        for variable in self._get_main_dictionary().variables:
            if variable.used and variable.name != ds.target_column_id:
                self.feature_names_out_.append(variable.name)

        # Activate the key columns in multitable
        if len(self.model_.dictionaries) > 1:
            for key_variable_name in self._get_main_dictionary().key:
                self._get_main_dictionary().get_variable(key_variable_name).used = True

    def transform(self, X):
        """Transforms X with a fitted Khiops supervised encoder

        .. note::
            Numerical features are encoded to categorical ones. See the
            ``transform_type_numerical`` parameter for details.

        Parameters
        ----------
        X : :external:term:`array-like` of shape (n_samples, n_features_in) or dict
            Training dataset. Either an :external:term:`array-like` or a ``dict``
            specification for multi-table datasets (see :doc:`/multi_table_primer`).

        Returns
        -------
        `ndarray <numpy.ndarray>`
            An array containing the encoded columns. A first column containing key
            column ids is added in multi-table mode.
        """
        # Create temporary directory
        computation_dir = self._create_computation_dir("transform")

        # Create and transform the dataset
        try:
            ds = Dataset(X)
            X_transformed = super()._transform(
                ds,
                computation_dir,
                self._transform_prepare_deployment_for_transform,
                True,
                "transform.txt",
            )
        # Cleanup and restore the runner's temporary dir
        finally:
            self._cleanup_computation_dir(computation_dir)
        return X_transformed.to_numpy(copy=False)

    def _transform_prepare_deployment_for_transform(self, ds):
        assert hasattr(
            self, "model_target_variable_name_"
        ), "Target variable name has not been set"

        # Create a copy of the model dictionary domain with the target variable
        # if it is not present in the input dataset
        # Note: We use `list` to avoid a warning of numpy about the `in` operator
        model_copy = self.model_.copy()
        model_dictionary = model_copy.get_dictionary(self.model_main_dictionary_name_)
        if self.model_target_variable_name_ not in list(ds.main_table.column_ids):
            model_dictionary.remove_variable(self.model_target_variable_name_)

        # TODO: Replace the second return value (the output columns' dtypes) with a
        #       proper value instead of `None`. In the current state, it will use pandas
        #       type auto-detection to load the internal table into memory.
        return model_copy, None

    def fit_transform(self, X, y=None, **kwargs):
        """Fit and transforms its inputs

        Parameters
        ----------
        X : :external:term:`array-like` of shape (n_samples, n_features_in) or dict
            Training dataset. Either an :external:term:`array-like` or a ``dict``
            specification for multi-table datasets (see :doc:`/multi_table_primer`).

        y : :external:term:`array-like` of shape (n_samples,)
            The target values.

        Returns
        -------
        self : `KhiopsEncoder`
            The calling estimator instance.
        """
        return self.fit(X, y, **kwargs).transform(X)
