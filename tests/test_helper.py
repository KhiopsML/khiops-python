######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Helper test class, for mocking Khiops core API functions and preparing data"""

import importlib
import os
import shutil
from collections import defaultdict

import pandas as pd
import wrapt
from sklearn.model_selection import train_test_split

import khiops.core as kh
from khiops.core.internals.common import type_error_message


class CoreApiFunctionMock:
    """Function mock context manager

    Replaces a specified function with a fixture copying-based mock.
    In the end, the mock is again replaced with the original function.

    Parameters
    ----------
    module : str
        Name of the module that contains the mocked function.
    function : str
        Name of the mocked function.
    fixture : dict
        Fixture with the output files and return values to mock. See above for more
        details.
    """

    api_function_specs = {
        ("khiops.core.api", "export_dictionary_as_json"): {
            "output_path_arg_index": 1,
            "output_file_key": "kdicj_path",
            "return_value_number": 1,
        },
        ("khiops.core", "train_predictor"): {
            "output_path_arg_index": 4,
            "output_file_key": "report_path",
            "return_value_number": 2,
        },
        ("khiops.core", "train_recoder"): {
            "output_path_arg_index": 4,
            "output_file_key": "report_path",
            "return_value_number": 2,
        },
        ("khiops.core", "deploy_model"): {
            "output_path_arg_index": 3,
            "output_file_key": "output_data_table",
            "return_value_number": 0,
        },
        ("khiops.core", "train_coclustering"): {
            "output_path_arg_index": 4,
            "output_file_key": "report_path",
            "return_value_number": 1,
        },
        ("khiops.core", "simplify_coclustering"): {
            "output_path_arg_index": 1,
            "output_file_key": "report_path",
            "return_value_number": 0,
        },
        ("khiops.core", "prepare_coclustering_deployment"): {
            "output_path_arg_index": 5,
            "output_file_key": "deploy_kdic_path",
            "return_value_number": 0,
        },
        ("khiops.core", "extract_keys_from_data_table"): {
            "output_path_arg_index": 3,
            "output_file_key": "keys_table_path",
            "return_value_number": 0,
        },
    }

    def __init__(
        self,
        module_name,
        function_name,
        fixture,
    ):
        if (module_name, function_name) not in self.api_function_specs:
            raise ValueError(
                f"Unsupported API function '{function_name}' on module '{module_name}'"
            )
        self.module_name = module_name
        self.function_name = function_name
        self.fixture = fixture
        self._original_function = None

        self._check_fixture()

    def _check_fixture(self):
        # Check container types
        if not isinstance(self.fixture, dict):
            raise TypeError(f"fixture must be dict not {type(self.fixture).__name__}.")
        if not isinstance(self.fixture["output_file_paths"], dict):
            raise TypeError(
                "fixture['output_file_paths'] must be dict "
                f"not {type(self.fixture['output_file_paths']).__name__}."
            )
        if not isinstance(self.fixture["return_values"], list):
            raise TypeError(
                "fixture['return_values'] must be list not "
                f"{type(self.fixture['return_values']).__name__}."
            )
        if not "output_file_paths" in self.fixture:
            raise ValueError("Missing 'output_file_paths' key in fixture")

        # Check contents of containers
        if self.output_file_key not in self.fixture["output_file_paths"]:
            raise ValueError(
                f"Missing output file key '{self.output_file_key}' "
                f"for function '{self.function_name}' "
                f"of module '{self.module_name}'."
            )
        if len(self.fixture["return_values"]) != self.return_value_number:
            raise ValueError(
                f"Found {len(self.fixture['return_values'])} "
                f"return values in fixture but expected {self.return_value_number}."
            )
        for index, return_value_tuple in enumerate(self.fixture["return_values"]):
            if not isinstance(return_value_tuple, tuple):
                raise TypeError(
                    f"Return value specification at index {index} "
                    f"must be tuple not {type(return_value_tuple).__name__}."
                )

    @property
    def output_path_arg_index(self):
        return self.api_function_specs[(self.module_name, self.function_name)][
            "output_path_arg_index"
        ]

    @property
    def output_file_key(self):
        return self.api_function_specs[(self.module_name, self.function_name)][
            "output_file_key"
        ]

    @property
    def return_value_number(self):
        return self.api_function_specs[(self.module_name, self.function_name)][
            "return_value_number"
        ]

    def __enter__(self):
        """Mock the function with fixture resources copying"""
        # Save the original function
        module = importlib.import_module(self.module_name)
        assert hasattr(
            module, self.function_name
        ), f"Core API function '{self.function_name}' not found"
        self._original_function = getattr(module, self.function_name)

        # Replace the function with the mock
        @wrapt.patch_function_wrapper(self.module_name, self.function_name)
        def function_mock(_mocked, _instance, args, kwargs):
            # Function with output dir: Copy the output_files to the specified directory
            copied_output_file_paths = {}
            output_file_path = args[self.output_path_arg_index]
            resource_file_path = self.fixture["output_file_paths"][self.output_file_key]
            shutil.copyfile(resource_file_path, output_file_path)
            copied_output_file_paths[self.output_file_key] = output_file_path

            # Attempt to copy resource predictor file
            if "predictor_kdic_path" in self.fixture["output_file_paths"]:
                resource_predictor_file_path = self.fixture["output_file_paths"][
                    "predictor_kdic_path"
                ]
                if os.path.exists(resource_predictor_file_path):
                    output_predictor_file_path = (
                        KhiopsTestHelper.model_path_of_report_path(output_file_path)
                    )
                    shutil.copyfile(
                        resource_predictor_file_path, output_predictor_file_path
                    )
                    copied_output_file_paths["predictor_kdic_path"] = (
                        output_predictor_file_path
                    )

            # Copy the log file if specified in the fixture
            if "log_file_path" in self.fixture["extra_file_paths"]:
                assert "log_file_path" in kwargs and kwargs["log_file_path"] is not None
                log_file_path = self.fixture["extra_file_paths"]["log_file_path"]
                shutil.copyfile(log_file_path, kwargs["log_file_path"])

            # Build mocked output
            mocked_return_value = None
            if self.return_value_number >= 1:
                # Build the return values list
                mocked_return_values = []
                for return_value, is_mocked_file in self.fixture["return_values"]:
                    # If the return value is mocked file look for it in the copied files
                    if is_mocked_file:
                        mocked_return_values.append(
                            copied_output_file_paths[return_value]
                        )
                    # Otherwise just add it
                    else:
                        mocked_return_values.append(return_value)

                # If there is a single value: Return the first element of the list
                if self.return_value_number == 1:
                    mocked_return_value = mocked_return_values[0]
                # Otherwise return a tuple with the list contents
                else:
                    mocked_return_value = tuple(mocked_return_values)

            return mocked_return_value

        return self

    def __exit__(self, *exc):
        """Restores the original function"""

        @wrapt.patch_function_wrapper(self.module_name, self.function_name)
        def original(_mocked, _instance, args, kwargs):
            if self._original_function is not None:
                return self._original_function(*args, **kwargs)

        # Return False, so that any exceptions *exc are propagated to the caller
        return False


class KhiopsTestHelper:
    """Helper functions for the actual tests

    Some of them need to be static so that they can be serialized for multiprocessing.
    """

    @staticmethod
    def get_with_subkey(dictionary, subkey):
        values = []
        for key, value in dictionary.items():
            if not isinstance(key, tuple):
                raise TypeError(type_error_message("key", key, tuple))
            if len(key) < 1:
                raise ValueError("'key' must be  non-empty")
            if subkey in key:
                values.append(value)
        return values

    @staticmethod
    def skip_expensive_test(test_case):
        if "SKIP_EXPENSIVE_TESTS" in os.environ:
            if os.environ["SKIP_EXPENSIVE_TESTS"].lower() == "true":
                test_case.skipTest("Skipping expensive test")

    @staticmethod
    def create_trace():
        """Create empty, updatable, three-level look-up dictionary"""
        return defaultdict(lambda: defaultdict(list))

    @staticmethod
    def wrap_with_parameter_trace(module, function, function_parameters):
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

    @staticmethod
    def wrap_with_runner_attributes_trace(module, function, runner_attributes_trace):
        """Wrap function with runner attributes trace"""

        @wrapt.patch_function_wrapper(module, function)
        def wrapper(wrapped, _instance, args, kwargs):
            # mutate runner_attributes_trace as previously bound / initialized in the
            # outer scope of the `wrap_with_runner_attributes_trace` method by its
            # caller:
            nonlocal runner_attributes_trace

            # Collect all the updated runner attributes
            runner_attributes_trace.update(vars(_instance))

            return wrapped(*args, **kwargs)

        return wrapper

    @staticmethod
    def get_resources_dir():
        """Helper to get the directory containing the fixtures"""
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), "resources")

    @staticmethod
    def get_two_table_data(dataset_path, root_table_name, secondary_table_name):
        """Read two-table data from two CSV files from sample dataset"""
        samples_dir = kh.get_runner().samples_dir
        data_path = os.path.join(samples_dir, dataset_path)
        root_table = pd.read_csv(
            os.path.join(data_path, f"{root_table_name}.txt"), sep="\t"
        )
        secondary_table = pd.read_csv(
            os.path.join(data_path, f"{secondary_table_name}.txt"), sep="\t"
        )
        return (root_table, secondary_table)

    @staticmethod
    def get_monotable_data(dataset_name):
        """Read monotable data from CSV sample dataset"""
        samples_dir = kh.get_runner().samples_dir
        return pd.read_csv(
            os.path.join(samples_dir, dataset_name, f"{dataset_name}.txt"), sep="\t"
        )

    @staticmethod
    def prepare_data(data, target_variable, primary_table=None, y_as_dataframe=False):
        """Prepare training and testing data for automated tests"""
        if primary_table is None:
            data_train, data_test = train_test_split(
                data, test_size=0.3, random_state=1, shuffle=False
            )

            y_test = data_test[target_variable]
            y_train = data_train[target_variable]

            # Create training labels as single-column dataframe
            if y_as_dataframe:
                y_train = pd.DataFrame(y_train, columns=[target_variable])

            x_test = data_test.drop([target_variable], axis=1)
            x_train = data_train.drop([target_variable], axis=1)
        elif isinstance(primary_table, pd.DataFrame):
            root_table_train, root_table_test = train_test_split(
                primary_table, test_size=0.3, random_state=1, shuffle=False
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
    def model_path_of_report_path(report_path):
        return f"{os.path.splitext(report_path)[0]}.model.kdic"
