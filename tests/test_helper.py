###############################################################################
# Copyright (c) 2022 Orange - All Rights Reserved
# * This software is the confidential and proprietary information of Orange.
# * You shall not disclose such Restricted Information and shall use it only in
#   accordance with the terms of the license agreement you entered into with
#   Orange named the "Khiops - Python Library Evaluation License".
# * Unauthorized copying of this file, via any medium is strictly prohibited.
# * See the "LICENSE.md" file for more details.
###############################################################################
"""Helper test class, for running fit and predict on Khiops estimators"""

import importlib
import os
import pickle
from collections import defaultdict

import pandas as pd
import wrapt
from sklearn.model_selection import train_test_split

import pykhiops.core as pk
import pykhiops.core.filesystems as fs
from pykhiops.core.common import is_iterable
from pykhiops.sklearn.estimators import KhiopsEncoder, KhiopsEstimator


class Mock:
    """Function mock context manager

    Replaces a specified function with a fixture copying-based mock.
    In the end, the mock is again replaced with the original function.
    """

    def __init__(
        self,
        module,
        function,
        resources,
        output_dir_grabber,
        output_file_names_map=None,
    ):
        self.module = module
        self.function = function
        self.resources = resources
        self.output_dir_grabber = output_dir_grabber
        self.output_file_names_map = output_file_names_map
        self.original_function = None

    def __enter__(self):
        """Mock `self.function` with fixture copying

        The fixtures are copied from `self.resources` to output directory
        given by `self.output_dir_grabber`
        """
        # `resources` is a list of file paths that should be output by
        # `function`
        # the treatment of this list is done according to the args and kwargs
        # passed to the wrapper
        # `output_dir_grabber` callback which detects the output directory from
        # the wrapped `function` 's `args`
        # `output_file_names_map` is a dictionary mapping resource file names to
        # output file names which are obtained via callbacks which detect the
        # target file names from the wrapped `function` 's `args`

        module = importlib.import_module(self.module)
        self.original_function = getattr(module, self.function)

        @wrapt.patch_function_wrapper(self.module, self.function)
        def mock(_mocked, _instance, args, _kwargs):
            assert self.output_file_names_map is None or len(
                self.output_file_names_map.keys()
            ) == len(self.resources)
            # copy resources from resource to target according to the args and
            # kwargs
            output_dir = self.output_dir_grabber(*args)
            output_dir_res = fs.create_resource(output_dir)
            if not output_dir_res.exists():
                output_dir_res.make_dir()

            file_paths_to_return = []

            for resource_file_path in self.resources:
                resource = fs.create_resource(resource_file_path)
                resource_file_name = os.path.basename(resource_file_path)
                if self.output_file_names_map is not None:
                    output_file_name = self.output_file_names_map.get(
                        os.path.basename(resource_file_path)
                    )(*args)
                    if output_file_name is not None:
                        output_file_path = os.path.join(output_dir, output_file_name)
                    else:
                        output_file_path = os.path.join(output_dir, resource_file_name)
                else:
                    output_file_path = os.path.join(output_dir, resource_file_name)

                resource.copy_to_local(output_file_path)
                file_paths_to_return.append(output_file_path)
            # now return what has been copied

            return (
                tuple(file_paths_to_return)
                if len(file_paths_to_return) > 1 or len(file_paths_to_return) == 0
                else file_paths_to_return[0]
            )

        return self

    def __exit__(self, *exc):
        """Mock `self.function` to `self.original_function`"""

        @wrapt.patch_function_wrapper(self.module, self.function)
        def original(_mocked, _instance, args, kwargs):
            if self.original_function is not None:
                return self.original_function(*args, **kwargs)

        # return False, so that any exceptions *exc are propagated to the caller
        return False


class PyKhiopsSklearnTestsHelper:
    """Helper functions for the actual tests

    Some of them need to be static so that they can be serialized for
    multiprocessing.
    """

    def create_parameter_trace(self):
        """Create empty, updatable, three-level look-up dictionary"""
        return defaultdict(lambda: defaultdict(list))

    def wrap_with_parameter_trace(self, module, function, function_parameters):
        """Wrap function with parameter trace"""

        @wrapt.patch_function_wrapper(module, function)
        def wrapper(wrapped, _instance, args, kwargs):

            # mutate function_parameters as previously bound / initialized in the
            # outer scope of the `wrap_with_parameter_trace` method by its
            # caller:
            nonlocal function_parameters

            function_parameters[module][function].append(
                {"args": args, "kwargs": kwargs}
            )

            return wrapped(*args, **kwargs)

        return wrapper

    def get_resources_dir(self):
        """Helper to get the directory containing the fixtures"""
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), "resources")

    @classmethod
    def get_two_table_data(cls, dataset_path, root_table_name, secondary_table_name):
        """Read two-table data from two CSV files from sample dataset"""

        samples_dir = pk.get_runner().samples_dir
        data_path = os.path.join(samples_dir, dataset_path)
        root_table = pd.read_csv(
            os.path.join(data_path, f"{root_table_name}.txt"), sep="\t"
        )
        secondary_table = pd.read_csv(
            os.path.join(data_path, f"{secondary_table_name}.txt"), sep="\t"
        )
        return (root_table, secondary_table)

    @classmethod
    def get_monotable_data(cls, dataset_name):
        """Read monotable data from CSV sample dataset"""
        samples_dir = pk.get_runner().samples_dir
        return pd.read_csv(
            os.path.join(samples_dir, dataset_name, f"{dataset_name}.txt"), sep="\t"
        )

    @classmethod
    def prepare_data(cls, data, target_variable, primary_table=None):
        """Prepare training and testing data for automated tests"""
        if primary_table is None:
            data_train, data_test = train_test_split(
                data, test_size=0.3, random_state=1
            )

            y_test = data_test[target_variable]
            y_train = data_train[target_variable]

            x_test = data_test.drop([target_variable], axis=1)
            x_train = data_train.drop([target_variable], axis=1)
        elif isinstance(primary_table, pd.DataFrame):
            root_table_train, root_table_test = train_test_split(
                primary_table, test_size=0.3, random_state=1
            )
            data.index = data[target_variable].values
            x_train = data.loc[root_table_train[target_variable]]
            y_train = None
            x_test = data.loc[root_table_test[target_variable]]
            y_test = None
        else:
            raise TypeError("primary_table must be None or a data frame")
        return ((x_train, y_train), (x_test, y_test))

    @staticmethod
    def _create_and_fit_estimator_to_data(
        estimator_class, observations, labels, fit_kwargs=None, **estimator_kwargs
    ):
        estimator = estimator_class.__call__(**estimator_kwargs)
        fit_kwargs = fit_kwargs if fit_kwargs is not None else {}
        if labels is not None:
            fitted_estimator = estimator.fit(X=observations, y=labels, **fit_kwargs)
        else:
            fitted_estimator = estimator.fit(X=observations, **fit_kwargs)
        return fitted_estimator

    @staticmethod
    def fit_helper(
        estimator_class, data, pickled=True, fit_kwargs=None, **estimator_kwargs
    ):
        """Creates a fitted estimator from a class and a dataset"""
        assert isinstance(
            estimator_class, KhiopsEstimator.__class__
        ), "'estimator_class' should be a class derived from KhiopsEstimator"

        # Unpickle training data if necessary
        training_data = pickle.loads(data) if pickled else data
        assert (
            is_iterable(training_data) and len(training_data) == 2
        ), "'training_data' should be an iterable with 2 elements"

        # Build a fitted estimator
        fitted_estimator = PyKhiopsSklearnTestsHelper._create_and_fit_estimator_to_data(
            estimator_class,
            training_data[0],  # observations
            training_data[1],  # labels
            fit_kwargs,
            **estimator_kwargs,
        )
        return fitted_estimator

    @staticmethod
    def _transform_data_with_encoder(encoder, observations):
        return encoder.transform(X=observations)

    @staticmethod
    def _predict_data_with_estimator(estimator, observations):
        return estimator.predict(X=observations)

    @staticmethod
    def _predict_proba_with_estimator(estimator, observations):
        return estimator.predict_proba(X=observations)

    @staticmethod
    def predict_helper(data, pickled=True, kind="simple"):  # 'proba' accepted
        """Prediction driver helper for tests"""

        estimator_and_data = pickle.loads(data) if pickled else data
        estimator, data = estimator_and_data
        if (
            isinstance(estimator, KhiopsEstimator)
            and is_iterable(data)
            and len(data) == 2
        ):
            if isinstance(estimator, KhiopsEncoder):
                predictions = PyKhiopsSklearnTestsHelper._transform_data_with_encoder(
                    estimator, data[0]
                )
            else:
                if kind == "simple":
                    predictions = (
                        PyKhiopsSklearnTestsHelper._predict_data_with_estimator(
                            estimator, data[0]
                        )
                    )
                elif kind == "proba":
                    predictions = (
                        PyKhiopsSklearnTestsHelper._predict_proba_with_estimator(
                            estimator, data[0]
                        )
                    )
                else:
                    raise ValueError(f"Kind of prediction unknown: '{kind}'")
        else:
            raise TypeError(
                "estimator should be a subclass of KhiopsEstimator "
                "and data should be an iterable of two elements"
            )
        return predictions
