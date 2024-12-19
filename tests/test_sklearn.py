######################################################################################
# Copyright (c) 2024 Orange. All rights reserved.                                    #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Tests parameter transfer between Khiops sklearn and core APIs"""
import contextlib
import copy
import os
import shutil
import unittest
import warnings

import numpy as np
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import NotFittedError, check_is_fitted

import khiops.core as kh
from khiops.sklearn.estimators import (
    KhiopsClassifier,
    KhiopsCoclustering,
    KhiopsEncoder,
    KhiopsPredictor,
    KhiopsRegressor,
    KhiopsSupervisedEstimator,
)
from tests.test_helper import CoreApiFunctionMock, KhiopsTestHelper

# Disable PEP8 variable names because of scikit-learn X,y conventions
# To capture invalid-names other than X,y run:
#   pylint --disable=all --enable=invalid-names estimators.py
# pylint: disable=invalid-name


class KhiopsSklearnParameterPassingTests(unittest.TestCase):
    """Test that parameters are properly passed from sklearn to core API"""

    @staticmethod
    def assertEqualPath(test_case, path1, path2):
        test_case.assertEqual(
            os.path.abspath(path1),
            os.path.abspath(path2),
            msg=f"Path '{path1}' is not equal to path '{path2}'",
        )

    @staticmethod
    def assertPathHasSuffix(test_case, path, suffix):
        test_case.assertTrue(
            path.endswith(suffix),
            msg=f"Suffix '{suffix}' not found in path '{path}'",
        )

    @staticmethod
    def assertPathHasPrefix(test_case, path, prefix):
        test_case.assertTrue(
            os.path.abspath(path).startswith(os.path.abspath(prefix)),
            msg=f"Prefix '{prefix}' not found in path '{path}'",
        )

    @staticmethod
    def assertEqualAdditionalDataTableNames(
        test_case, additional_data_tables, expected_additional_data_table_keyset
    ):
        test_case.assertEqual(
            set(additional_data_tables.keys()),
            set(expected_additional_data_table_keyset),
        )

    @classmethod
    def setUpClass(cls):
        """Prepare datasets for tests"""
        # Grab output_dir for subsequent deletion
        cls.output_dir = os.path.join(
            "resources", "tmp", "test_sklearn_parameter_transfer"
        )
        if not os.path.isdir(cls.output_dir):
            os.makedirs(cls.output_dir)

        (
            train_multitable_file_dataset_classif,
            test_multitable_file_dataset_classif,
        ) = cls._create_train_test_multitable_file_dataset()

        (
            train_multitable_file_dataset_reg,
            test_multitable_file_dataset_reg,
        ) = cls._create_train_test_multitable_file_dataset(
            transform_for_regression=True
        )

        (
            X_train_multitable_dataframe_classif,
            y_train_multitable_dataframe_classif,
            X_test_multitable_dataframe_classif,
        ) = cls._create_train_test_multitable_dataframe()

        (
            X_train_multitable_dataframe_reg,
            y_train_multitable_dataframe_reg,
            X_test_multitable_dataframe_reg,
        ) = cls._create_train_test_multitable_dataframe(transform_for_regression=True)

        (
            train_monotable_file_dataset_classif,
            test_monotable_file_dataset_classif,
        ) = cls._create_train_test_monotable_file_dataset("class")

        (
            train_monotable_file_dataset_reg,
            test_monotable_file_dataset_reg,
        ) = cls._create_train_test_monotable_file_dataset("age")

        (
            X_train_monotable_dataframe_classif,
            y_train_monotable_dataframe_classif,
            X_test_monotable_dataframe_classif,
        ) = cls._create_train_test_monotable_dataframe("class")

        (
            X_train_monotable_dataframe_reg,
            y_train_monotable_dataframe_reg,
            X_test_monotable_dataframe_reg,
        ) = cls._create_train_test_monotable_dataframe("age")

        (
            _,
            y_train_monotable_dataframe_classif_as_df,
            _,
        ) = cls._create_train_test_monotable_dataframe("class", y_as_dataframe=True)
        (
            _,
            y_train_monotable_dataframe_reg_as_df,
            _,
        ) = cls._create_train_test_monotable_dataframe("age", y_as_dataframe=True)

        # store training and testing datasets, indexed by:
        # - the schema type (montable or multitable),
        # - the data source type (dataframe or file dataset),
        # - the estimator type (KhiopsClassifier, KhiopsEncoder and
        #   KhiopsRegressor, coclustering tests reusing the KhiopsClassifier
        #   data)
        cls.datasets = {
            "multitable": {
                "file_dataset": {
                    KhiopsClassifier: {
                        "train": train_multitable_file_dataset_classif,
                        "test": test_multitable_file_dataset_classif,
                    },
                    KhiopsEncoder: {
                        "train": train_multitable_file_dataset_classif,
                        "test": test_multitable_file_dataset_classif,
                    },
                    KhiopsRegressor: {
                        "train": train_multitable_file_dataset_reg,
                        "test": test_multitable_file_dataset_reg,
                    },
                },
                "dataframe": {
                    KhiopsClassifier: {
                        "X_train": X_train_multitable_dataframe_classif,
                        "y_train": y_train_multitable_dataframe_classif,
                        "X_test": X_test_multitable_dataframe_classif,
                    },
                    KhiopsEncoder: {
                        "X_train": X_train_multitable_dataframe_classif,
                        "y_train": y_train_multitable_dataframe_classif,
                        "X_test": X_test_multitable_dataframe_classif,
                    },
                    KhiopsRegressor: {
                        "X_train": X_train_multitable_dataframe_reg,
                        "y_train": y_train_multitable_dataframe_reg,
                        "X_test": X_test_multitable_dataframe_reg,
                    },
                },
            },
            "monotable": {
                "file_dataset": {
                    KhiopsClassifier: {
                        "train": train_monotable_file_dataset_classif,
                        "test": test_monotable_file_dataset_classif,
                    },
                    KhiopsEncoder: {
                        "train": train_monotable_file_dataset_classif,
                        "test": test_monotable_file_dataset_classif,
                    },
                    KhiopsRegressor: {
                        "train": train_monotable_file_dataset_reg,
                        "test": test_monotable_file_dataset_reg,
                    },
                },
                "dataframe": {
                    KhiopsClassifier: {
                        "X_train": X_train_monotable_dataframe_classif,
                        "y_train": y_train_monotable_dataframe_classif,
                        "X_test": X_test_monotable_dataframe_classif,
                    },
                    KhiopsEncoder: {
                        "X_train": X_train_monotable_dataframe_classif,
                        "y_train": y_train_monotable_dataframe_classif,
                        "X_test": X_test_monotable_dataframe_classif,
                    },
                    KhiopsRegressor: {
                        "X_train": X_train_monotable_dataframe_reg,
                        "y_train": y_train_monotable_dataframe_reg,
                        "X_test": X_test_monotable_dataframe_reg,
                    },
                },
                "dataframe_xy": {
                    KhiopsClassifier: {
                        "X_train": X_train_monotable_dataframe_classif,
                        "y_train": y_train_monotable_dataframe_classif_as_df,
                        "X_test": X_test_monotable_dataframe_classif,
                    },
                    KhiopsEncoder: {
                        "X_train": X_train_monotable_dataframe_classif,
                        "y_train": y_train_monotable_dataframe_classif_as_df,
                        "X_test": X_test_monotable_dataframe_classif,
                    },
                    KhiopsRegressor: {
                        "X_train": X_train_monotable_dataframe_reg,
                        "y_train": y_train_monotable_dataframe_reg_as_df,
                        "X_test": X_test_monotable_dataframe_reg,
                    },
                },
            },
        }

        # mapping between schema type and dataset name; also included is the
        # "not_applicable" schema type, which is used for the coclustering tests
        cls.dataset_of_schema_type = {
            "monotable": "Adult",
            "multitable": "SpliceJunction",
            "not_applicable": "SpliceJunction",
        }

        # Mock objects indexed by estimator type and method, defined as
        # lambda-expressions taking as input a list of resources to be copied
        # from; these resources are then dispatched to each mock as needed
        cls.mocks_table = {
            KhiopsPredictor: {
                "fit": lambda resources: [
                    CoreApiFunctionMock(
                        module_name="khiops.core",
                        function_name="train_predictor",
                        fixture={
                            "output_file_paths": {
                                "report_path": resources["report_path"],
                                "predictor_kdic_path": resources["model_kdic_path"],
                            },
                            "extra_file_paths": {},
                            "return_values": [
                                ("report_path", True),
                                ("predictor_kdic_path", True),
                            ],
                        },
                    ),
                    CoreApiFunctionMock(
                        module_name="khiops.core.api",
                        function_name="export_dictionary_as_json",
                        fixture={
                            "output_file_paths": {
                                "kdicj_path": resources["model_kdicj_path"],
                            },
                            "extra_file_paths": {},
                            "return_values": [("kdicj_path", True)],
                        },
                    ),
                ],
                "predict": lambda resources: (
                    CoreApiFunctionMock(
                        module_name="khiops.core",
                        function_name="deploy_model",
                        fixture={
                            "output_file_paths": {
                                "output_data_table": resources["prediction_table_path"]
                            },
                            "extra_file_paths": {},
                            "return_values": [],
                        },
                    ),
                ),
            },
            KhiopsEncoder: {
                "fit": lambda resources: [
                    CoreApiFunctionMock(
                        module_name="khiops.core",
                        function_name="train_recoder",
                        fixture={
                            "output_file_paths": {
                                "report_path": resources["report_path"],
                                "predictor_kdic_path": resources["model_kdic_path"],
                            },
                            "extra_file_paths": {},
                            "return_values": [
                                ("report_path", True),
                                ("predictor_kdic_path", True),
                            ],
                        },
                    ),
                    CoreApiFunctionMock(
                        module_name="khiops.core.api",
                        function_name="export_dictionary_as_json",
                        fixture={
                            "output_file_paths": {
                                "kdicj_path": resources["model_kdicj_path"],
                            },
                            "extra_file_paths": {},
                            "return_values": [("kdicj_path", True)],
                        },
                    ),
                ],
                "predict": lambda resources: [
                    CoreApiFunctionMock(
                        module_name="khiops.core",
                        function_name="deploy_model",
                        fixture={
                            "output_file_paths": {
                                "output_data_table": resources["prediction_table_path"]
                            },
                            "extra_file_paths": {},
                            "return_values": [],
                        },
                    ),
                ],
            },
            KhiopsCoclustering: {
                "fit": lambda resources: [
                    CoreApiFunctionMock(
                        module_name="khiops.core",
                        function_name="train_coclustering",
                        fixture={
                            "output_file_paths": {
                                "report_path": resources["report_path"]
                            },
                            "extra_file_paths": {
                                "log_file_path": resources["log_file_path"]
                            },
                            "return_values": [("report_path", True)],
                        },
                    ),
                    CoreApiFunctionMock(
                        module_name="khiops.core",
                        function_name="simplify_coclustering",
                        fixture={
                            "output_file_paths": {
                                "report_path": resources["report_path"]
                            },
                            "extra_file_paths": {},
                            "return_values": [],
                        },
                    ),
                    CoreApiFunctionMock(
                        module_name="khiops.core",
                        function_name="build_multi_table_dictionary",
                        fixture={
                            "output_file_paths": {
                                "kdic_path": resources["tmp_model_kdic_path"]
                            },
                            "extra_file_paths": {},
                            "return_values": [],
                        },
                    ),
                    CoreApiFunctionMock(
                        module_name="khiops.core",
                        function_name="prepare_coclustering_deployment",
                        fixture={
                            "output_file_paths": {
                                "deploy_kdic_path": resources["model_kdic_path"]
                            },
                            "extra_file_paths": {},
                            "return_values": [],
                        },
                    ),
                    CoreApiFunctionMock(
                        module_name="khiops.core.api",
                        function_name="export_dictionary_as_json",
                        fixture={
                            "output_file_paths": {
                                "kdicj_path": resources["model_kdicj_path"],
                            },
                            "extra_file_paths": {},
                            "return_values": [
                                ("kdicj_path", True),
                            ],
                        },
                    ),
                ],
                "predict": lambda resources: [
                    CoreApiFunctionMock(
                        module_name="khiops.core",
                        function_name="extract_keys_from_data_table",
                        fixture={
                            "output_file_paths": {
                                "keys_table_path": resources["raw_keys_table_path"],
                            },
                            "extra_file_paths": {},
                            "return_values": [],
                        },
                    ),
                    CoreApiFunctionMock(
                        module_name="khiops.core",
                        function_name="deploy_model",
                        fixture={
                            "output_file_paths": {
                                "output_data_table": resources["prediction_table_path"]
                            },
                            "extra_file_paths": {},
                            "return_values": [],
                        },
                    ),
                ],
            },
        }
        cls.dictionary_domain_kwargs = {
            "not_applicable": {
                ("dataframe",): {
                    KhiopsCoclustering: {
                        "build_multi_table_dictionary": {
                            "expected_n_dictionaries": 1,
                            "expected_main_table_key": None,
                            "expected_main_dictionary_name": "CC_main_table",
                            "expected_additional_data_table_names": [],
                        },
                        "train_coclustering": {
                            "expected_n_dictionaries": 1,
                            "expected_main_table_key": None,
                            "expected_main_dictionary_name": "main_table",
                            "expected_additional_data_table_names": [],
                        },
                        "deploy_model": {
                            "expected_n_dictionaries": 2,
                            "expected_main_table_key": "SampleId",
                            "expected_main_dictionary_name": "CC_Keys_main_table",
                            "expected_additional_data_table_names": ["CC_main_table"],
                        },
                        "extract_keys_from_data_table": {
                            "expected_n_dictionaries": 1,
                            "expected_main_table_key": "SampleId",
                            "expected_main_dictionary_name": "main_table",
                            "expected_additional_data_table_names": [],
                        },
                    }
                },
                ("file_dataset",): {
                    KhiopsCoclustering: {
                        "build_multi_table_dictionary": {
                            "expected_n_dictionaries": 1,
                            "expected_main_table_key": None,
                            "expected_main_dictionary_name": "CC_main_table",
                            "expected_additional_data_table_names": [],
                        },
                        "train_coclustering": {
                            "expected_n_dictionaries": 1,
                            "expected_main_table_key": None,
                            "expected_main_dictionary_name": "main_table",
                            "expected_additional_data_table_names": [],
                        },
                        "deploy_model": {
                            "expected_n_dictionaries": 2,
                            "expected_main_table_key": "SampleId",
                            "expected_main_dictionary_name": ("CC_Keys_main_table"),
                            "expected_additional_data_table_names": ["CC_main_table"],
                        },
                        "extract_keys_from_data_table": {
                            "expected_n_dictionaries": 1,
                            "expected_main_table_key": "SampleId",
                            "expected_main_dictionary_name": "main_table",
                            "expected_additional_data_table_names": [],
                        },
                    }
                },
            },
            "monotable": {
                ("dataframe", "dataframe_xy"): {
                    KhiopsPredictor: {
                        "train_predictor": {
                            "expected_n_dictionaries": 1,
                            "expected_main_table_key": None,
                            "expected_main_dictionary_name": "main_table",
                            "expected_additional_data_table_names": [],
                        },
                        "deploy_model": {
                            "expected_n_dictionaries": 1,
                            "expected_main_table_key": None,
                            "expected_main_dictionary_name": "SNB_main_table",
                            "expected_additional_data_table_names": [],
                        },
                    },
                    KhiopsEncoder: {
                        "train_recoder": {
                            "expected_n_dictionaries": 1,
                            "expected_main_table_key": None,
                            "expected_main_dictionary_name": "main_table",
                            "expected_additional_data_table_names": [],
                        },
                        "deploy_model": {
                            "expected_n_dictionaries": 1,
                            "expected_main_table_key": None,
                            "expected_main_dictionary_name": "R_main_table",
                            "expected_additional_data_table_names": [],
                        },
                    },
                },
                ("file_dataset",): {
                    KhiopsPredictor: {
                        "train_predictor": {
                            "expected_n_dictionaries": 1,
                            "expected_main_table_key": None,
                            "expected_main_dictionary_name": "main_table",
                            "expected_additional_data_table_names": [],
                        },
                        "deploy_model": {
                            "expected_n_dictionaries": 1,
                            "expected_main_table_key": None,
                            "expected_main_dictionary_name": "SNB_main_table",
                            "expected_additional_data_table_names": [],
                        },
                    },
                    KhiopsEncoder: {
                        "train_recoder": {
                            "expected_n_dictionaries": 1,
                            "expected_main_table_key": None,
                            "expected_main_dictionary_name": "main_table",
                            "expected_additional_data_table_names": [],
                        },
                        "deploy_model": {
                            "expected_n_dictionaries": 1,
                            "expected_main_table_key": None,
                            "expected_main_dictionary_name": "R_main_table",
                            "expected_additional_data_table_names": [],
                        },
                    },
                },
            },
            "multitable": {
                ("file_dataset",): {
                    KhiopsPredictor: {
                        "train_predictor": {
                            "expected_n_dictionaries": 2,
                            "expected_main_table_key": "SampleId",
                            "expected_main_dictionary_name": "SpliceJunction",
                            "expected_additional_data_table_names": [
                                "SpliceJunctionDNA"
                            ],
                        },
                        "deploy_model": {
                            "expected_n_dictionaries": 2,
                            "expected_main_table_key": "SampleId",
                            "expected_main_dictionary_name": "SNB_SpliceJunction",
                            "expected_additional_data_table_names": [
                                "SNB_SpliceJunctionDNA"
                            ],
                        },
                    },
                    KhiopsEncoder: {
                        "train_recoder": {
                            "expected_n_dictionaries": 2,
                            "expected_main_table_key": "SampleId",
                            "expected_main_dictionary_name": "SpliceJunction",
                            "expected_additional_data_table_names": [
                                "SpliceJunctionDNA"
                            ],
                        },
                        "deploy_model": {
                            "expected_n_dictionaries": 2,
                            "expected_main_table_key": "SampleId",
                            "expected_main_dictionary_name": "R_SpliceJunction",
                            "expected_additional_data_table_names": [
                                "R_SpliceJunctionDNA"
                            ],
                        },
                    },
                },
                ("dataframe",): {
                    KhiopsPredictor: {
                        "train_predictor": {
                            "expected_n_dictionaries": 2,
                            "expected_main_table_key": "SampleId",
                            "expected_main_dictionary_name": "SpliceJunction",
                            "expected_additional_data_table_names": [
                                "SpliceJunctionDNA"
                            ],
                        },
                        "deploy_model": {
                            "expected_n_dictionaries": 2,
                            "expected_main_table_key": "SampleId",
                            "expected_main_dictionary_name": "SNB_SpliceJunction",
                            "expected_additional_data_table_names": [
                                "SNB_SpliceJunctionDNA"
                            ],
                        },
                    },
                    KhiopsEncoder: {
                        "train_recoder": {
                            "expected_n_dictionaries": 2,
                            "expected_main_table_key": "SampleId",
                            "expected_main_dictionary_name": "SpliceJunction",
                            "expected_additional_data_table_names": [
                                "SpliceJunctionDNA"
                            ],
                        },
                        "deploy_model": {
                            "expected_n_dictionaries": 2,
                            "expected_main_table_key": "SampleId",
                            "expected_main_dictionary_name": "R_SpliceJunction",
                            "expected_additional_data_table_names": [
                                "R_SpliceJunctionDNA"
                            ],
                        },
                    },
                },
            },
        }
        cls.wrapped_functions = {
            KhiopsPredictor: {
                "fit": [("khiops.core", "train_predictor")],
                "predict": [("khiops.core", "deploy_model")],
            },
            KhiopsEncoder: {
                "fit": [("khiops.core", "train_recoder")],
                "predict": [("khiops.core", "deploy_model")],
            },
            KhiopsCoclustering: {
                "fit": [
                    ("khiops.core", "train_coclustering"),
                    ("khiops.core", "read_coclustering_results_file"),
                    ("khiops.core", "build_multi_table_dictionary"),
                    ("khiops.core", "prepare_coclustering_deployment"),
                    ("khiops.core", "simplify_coclustering"),
                ],
                "simplify": [
                    ("khiops.core", "simplify_coclustering"),
                    ("khiops.core", "build_multi_table_dictionary"),
                    ("khiops.core", "prepare_coclustering_deployment"),
                ],
                "predict": [
                    ("khiops.core", "deploy_model"),
                    ("khiops.core", "extract_keys_from_data_table"),
                ],
            },
        }
        cls.expected_args = {
            "not_applicable": {
                ("dataframe",): {
                    KhiopsCoclustering: {
                        "fit": {
                            ("khiops.core", "prepare_coclustering_deployment"): {
                                2: os.path.join(cls.output_dir, "Coclustering.khcj"),
                                3: "CC_main_table",
                                4: "SampleId",
                                5: cls.output_dir,
                            },
                            ("khiops.core", "read_coclustering_results_file"): {
                                0: os.path.join(cls.output_dir, "Coclustering.khcj")
                            },
                            ("khiops.core", "build_multi_table_dictionary"): {
                                2: "CC_main_table"
                            },
                            ("khiops.core", "train_coclustering"): {
                                1: "main_table",
                                3: ("SampleId", "Pos", "Char"),
                                4: cls.output_dir,
                            },
                            ("khiops.core", "simplify_coclustering"): {},
                        },
                        "simplify": {
                            ("khiops.core", "simplify_coclustering"): {
                                2: cls.output_dir
                            },
                            ("khiops.core", "prepare_coclustering_deployment"): {
                                2: os.path.join(cls.output_dir, "Coclustering.khcj"),
                                3: "CC_main_table",
                                4: "SampleId",
                                5: cls.output_dir,
                            },
                            ("khiops.core", "build_multi_table_dictionary"): {
                                2: "CC_main_table"
                            },
                        },
                        "predict": {
                            ("khiops.core", "deploy_model"): {
                                1: "CC_Keys_main_table",
                                3: cls.output_dir,
                            },
                            ("khiops.core", "extract_keys_from_data_table"): {
                                1: "main_table",
                                2: "copy_main_table.txt",
                                3: "keys_main_table.txt",
                            },
                        },
                    },
                },
                ("file_dataset",): {
                    KhiopsCoclustering: {
                        "fit": {
                            ("khiops.core", "prepare_coclustering_deployment"): {
                                2: os.path.join(cls.output_dir, "Coclustering.khcj"),
                                3: "CC_main_table",
                                4: "SampleId",
                                5: cls.output_dir,
                            },
                            ("khiops.core", "read_coclustering_results_file"): {
                                0: os.path.join(cls.output_dir, "Coclustering.khcj")
                            },
                            ("khiops.core", "build_multi_table_dictionary"): {
                                2: "CC_main_table"
                            },
                            ("khiops.core", "train_coclustering"): {
                                1: "main_table",
                                3: ("SampleId", "Pos", "Char"),
                                4: cls.output_dir,
                            },
                        },
                        "predict": {
                            ("khiops.core", "deploy_model"): {
                                1: "CC_Keys_main_table",
                                3: cls.output_dir,
                            },
                            ("khiops.core", "extract_keys_from_data_table"): {
                                1: "main_table",
                                2: "copy_main_table.txt",
                                3: "keys_main_table.txt",
                            },
                        },
                    },
                },
            },
            "monotable": {
                ("dataframe", "dataframe_xy"): {
                    KhiopsRegressor: {
                        "fit": {
                            ("khiops.core", "train_predictor"): {
                                1: "main_table",
                                3: "age",
                            }
                        },
                        "predict": {
                            ("khiops.core", "deploy_model"): {
                                1: "SNB_main_table",
                                2: "main_table.txt",
                                3: cls.output_dir,
                            }
                        },
                    },
                    KhiopsClassifier: {
                        "fit": {
                            ("khiops.core", "train_predictor"): {
                                1: "main_table",
                                2: "main_table.txt",
                                3: "class",
                            }
                        },
                        "predict": {
                            ("khiops.core", "deploy_model"): {
                                1: "SNB_main_table",
                                2: "main_table.txt",
                                3: cls.output_dir,
                            }
                        },
                    },
                    KhiopsEncoder: {
                        "fit": {
                            ("khiops.core", "train_recoder"): {
                                1: "main_table",
                                3: "class",
                            }
                        },
                        "predict": {
                            ("khiops.core", "deploy_model"): {
                                1: "R_main_table",
                                2: "main_table.txt",
                                3: cls.output_dir,
                            }
                        },
                    },
                },
                ("file_dataset",): {
                    KhiopsRegressor: {
                        "fit": {
                            ("khiops.core", "train_predictor"): {
                                1: "main_table",
                                2: "main_table.txt",
                                3: "age",
                                4: cls.output_dir,
                            }
                        },
                        "predict": {
                            ("khiops.core", "deploy_model"): {
                                1: "SNB_main_table",
                                2: "copy_main_table.txt",
                                3: cls.output_dir,
                            }
                        },
                    },
                    KhiopsClassifier: {
                        "fit": {
                            ("khiops.core", "train_predictor"): {
                                1: "main_table",
                                2: "main_table.txt",
                                3: "class",
                                4: cls.output_dir,
                            }
                        },
                        "predict": {
                            ("khiops.core", "deploy_model"): {
                                1: "SNB_main_table",
                                2: "copy_main_table.txt",
                                3: cls.output_dir,
                            }
                        },
                    },
                    KhiopsEncoder: {
                        "fit": {
                            ("khiops.core", "train_recoder"): {
                                1: "main_table",
                                3: "class",
                            }
                        },
                        "predict": {
                            ("khiops.core", "deploy_model"): {
                                1: "R_main_table",
                                2: "copy_main_table.txt",
                                3: cls.output_dir,
                            }
                        },
                    },
                },
            },
            "multitable": {
                ("dataframe",): {
                    KhiopsRegressor: {
                        "fit": {
                            ("khiops.core", "train_predictor"): {
                                1: "SpliceJunction",
                                3: "Class",
                            }
                        },
                        "predict": {
                            ("khiops.core", "deploy_model"): {
                                1: "SNB_SpliceJunction",
                                2: "SpliceJunction.txt",
                                3: cls.output_dir,
                            }
                        },
                    },
                    KhiopsClassifier: {
                        "fit": {
                            ("khiops.core", "train_predictor"): {
                                1: "SpliceJunction",
                                3: "Class",
                            }
                        },
                        "predict": {
                            ("khiops.core", "deploy_model"): {
                                1: "SNB_SpliceJunction",
                                2: "SpliceJunction.txt",
                                3: cls.output_dir,
                            }
                        },
                    },
                    KhiopsEncoder: {
                        "fit": {
                            ("khiops.core", "train_recoder"): {
                                1: "SpliceJunction",
                                3: "Class",
                                4: cls.output_dir,
                            }
                        },
                        "predict": {
                            ("khiops.core", "deploy_model"): {
                                1: "R_SpliceJunction",
                                2: "SpliceJunction.txt",
                                3: cls.output_dir,
                            }
                        },
                    },
                },
                ("file_dataset",): {
                    KhiopsRegressor: {
                        "fit": {
                            ("khiops.core", "train_predictor"): {
                                1: "SpliceJunction",
                                3: "Class",
                                4: cls.output_dir,
                            }
                        },
                        "predict": {
                            ("khiops.core", "deploy_model"): {
                                1: "SNB_SpliceJunction",
                                2: "copy_SpliceJunction.txt",
                                3: cls.output_dir,
                            }
                        },
                    },
                    KhiopsClassifier: {
                        "fit": {
                            ("khiops.core", "train_predictor"): {
                                1: "SpliceJunction",
                                3: "Class",
                                4: cls.output_dir,
                            }
                        },
                        "predict": {
                            ("khiops.core", "deploy_model"): {
                                1: "SNB_SpliceJunction",
                                2: "copy_SpliceJunction.txt",
                                3: cls.output_dir,
                            }
                        },
                    },
                    KhiopsEncoder: {
                        "fit": {
                            ("khiops.core", "train_recoder"): {
                                1: "SpliceJunction",
                                3: "Class",
                            }
                        },
                        "predict": {
                            ("khiops.core", "deploy_model"): {
                                1: "R_SpliceJunction",
                                2: "copy_SpliceJunction.txt",
                                3: cls.output_dir,
                            }
                        },
                    },
                },
            },
        }
        cls.special_arg_checkers = {
            KhiopsPredictor: {
                "fit": {
                    ("khiops.core", "train_predictor"): {
                        2: cls.assertPathHasSuffix,
                        4: cls.assertPathHasPrefix,
                    }
                },
                "predict": {
                    ("khiops.core", "deploy_model"): {
                        2: cls.assertPathHasSuffix,
                        3: cls.assertPathHasPrefix,
                    }
                },
            },
            KhiopsEncoder: {
                "fit": {
                    ("khiops.core", "train_recoder"): {
                        2: cls.assertPathHasSuffix,
                        4: cls.assertPathHasPrefix,
                    }
                },
                "predict": {
                    ("khiops.core", "deploy_model"): {
                        2: cls.assertPathHasSuffix,
                        3: cls.assertPathHasPrefix,
                    }
                },
            },
            KhiopsCoclustering: {
                "fit": {
                    ("khiops.core", "train_coclustering"): {4: cls.assertPathHasPrefix},
                    ("khiops.core", "prepare_coclustering_deployment"): {
                        2: cls.assertEqualPath,
                    },
                    ("khiops.core", "read_coclustering_results_file"): {
                        0: cls.assertEqualPath,
                    },
                },
                "simplify": {
                    ("khiops.core", "prepare_coclustering_deployment"): {
                        2: cls.assertEqualPath,
                    },
                },
                "predict": {
                    ("khiops.core", "deploy_model"): {3: cls.assertPathHasPrefix},
                    ("khiops.core", "extract_keys_from_data_table"): {
                        2: cls.assertPathHasSuffix,
                        3: cls.assertPathHasSuffix,
                    },
                },
            },
        }

        cls.expected_kwargs = {
            "not_applicable": {
                ("dataframe",): {
                    KhiopsCoclustering: {
                        "fit": {
                            ("khiops.core", "prepare_coclustering_deployment"): {
                                "build_cluster_variable": True,
                                "build_distance_variables": False,
                                "build_frequency_variables": False,
                            },
                            ("khiops.core", "read_coclustering_results_file"): {},
                            ("khiops.core", "build_multi_table_dictionary"): {
                                "overwrite_dictionary_file": True
                            },
                            ("khiops.core", "train_coclustering"): {
                                "log_file_path": os.path.join(
                                    cls.output_dir, "khiops_train_cc.log"
                                )
                            },
                            ("khiops.core", "simplify_coclustering"): {
                                "max_part_numbers": {"SampleId": 2},
                            },
                        },
                        "simplify": {
                            ("khiops.core", "simplify_coclustering"): {
                                "max_part_numbers": {"SampleId": 2},
                                "max_preserved_information": 3,
                                "max_cells": 4,
                                "max_total_parts": 1,
                            },
                            ("khiops.core", "prepare_coclustering_deployment"): {
                                "build_cluster_variable": True,
                                "build_distance_variables": False,
                                "build_frequency_variables": False,
                            },
                            ("khiops.core", "build_multi_table_dictionary"): {
                                "overwrite_dictionary_file": True
                            },
                        },
                        "predict": {
                            ("khiops.core", "deploy_model"): {
                                "detect_format": False,
                                "header_line": True,
                                "additional_data_tables": {
                                    "CC_Keys_main_table`CC_main_table"
                                },
                                "log_file_path": os.path.join(
                                    cls.output_dir, "khiops.log"
                                ),
                            },
                            ("khiops.core", "extract_keys_from_data_table"): {
                                "header_line": True,
                                "output_header_line": True,
                            },
                        },
                    },
                },
                ("file_dataset",): {
                    KhiopsCoclustering: {
                        "fit": {
                            ("khiops.core", "prepare_coclustering_deployment"): {
                                "build_cluster_variable": True,
                                "build_distance_variables": False,
                                "build_frequency_variables": False,
                            },
                            ("khiops.core", "read_coclustering_results_file"): {},
                            ("khiops.core", "build_multi_table_dictionary"): {
                                "overwrite_dictionary_file": True
                            },
                            ("khiops.core", "train_coclustering"): {
                                "log_file_path": os.path.join(
                                    cls.output_dir, "khiops_train_cc.log"
                                )
                            },
                        },
                        "predict": {
                            ("khiops.core", "deploy_model"): {
                                "detect_format": False,
                                "header_line": True,
                                "additional_data_tables": {
                                    "CC_Keys_main_table`CC_main_table"
                                },
                                "log_file_path": os.path.join(
                                    cls.output_dir, "khiops.log"
                                ),
                            },
                            ("khiops.core", "extract_keys_from_data_table"): {
                                "header_line": True,
                                "output_header_line": True,
                            },
                        },
                    },
                },
            },
            "monotable": {
                ("dataframe", "dataframe_xy"): {
                    KhiopsClassifier: {
                        "fit": {
                            ("khiops.core", "train_predictor"): {
                                "field_separator": "\t",
                                "detect_format": False,
                                "header_line": True,
                                "max_pairs": 1,
                                "max_trees": 5,
                                "max_selected_variables": 1,
                                "max_evaluated_variables": 3,
                                "specific_pairs": [("age", "race")],
                                "all_possible_pairs": False,
                                "construction_rules": ["TableMode", "TableSelection"],
                                "group_target_value": False,
                                "additional_data_tables": {},
                            }
                        },
                        "predict": {
                            ("khiops.core", "deploy_model"): {
                                "field_separator": "\t",
                                "detect_format": False,
                                "header_line": True,
                                "log_file_path": os.path.join(
                                    cls.output_dir, "khiops.log"
                                ),
                                "additional_data_tables": {},
                            }
                        },
                    },
                    KhiopsRegressor: {
                        "fit": {
                            ("khiops.core", "train_predictor"): {
                                "field_separator": "\t",
                                "detect_format": False,
                                "header_line": True,
                                "max_pairs": 1,
                                "max_trees": 0,
                                "max_selected_variables": 1,
                                "max_evaluated_variables": 3,
                                "specific_pairs": [("age", "race")],
                                "all_possible_pairs": False,
                                "construction_rules": ["TableMode", "TableSelection"],
                                "additional_data_tables": {},
                            }
                        },
                        "predict": {
                            ("khiops.core", "deploy_model"): {
                                "field_separator": "\t",
                                "detect_format": False,
                                "header_line": True,
                                "log_file_path": os.path.join(
                                    cls.output_dir, "khiops.log"
                                ),
                                "additional_data_tables": {},
                            }
                        },
                    },
                    KhiopsEncoder: {
                        "fit": {
                            ("khiops.core", "train_recoder"): {
                                "field_separator": "\t",
                                "detect_format": False,
                                "header_line": True,
                                "max_pairs": 1,
                                "max_trees": 5,
                                "specific_pairs": [("age", "race")],
                                "all_possible_pairs": False,
                                "construction_rules": ["TableMode", "TableSelection"],
                                "informative_variables_only": True,
                                "group_target_value": False,
                                "keep_initial_categorical_variables": False,
                                "keep_initial_numerical_variables": False,
                                "categorical_recoding_method": "part Id",
                                "numerical_recoding_method": "part Id",
                                "pairs_recoding_method": "part Id",
                                "additional_data_tables": {},
                            }
                        },
                        "predict": {
                            ("khiops.core", "deploy_model"): {
                                "field_separator": "\t",
                                "detect_format": False,
                                "header_line": True,
                                "log_file_path": os.path.join(
                                    cls.output_dir, "khiops.log"
                                ),
                                "additional_data_tables": {},
                            }
                        },
                    },
                },
                ("file_dataset",): {
                    KhiopsClassifier: {
                        "fit": {
                            ("khiops.core", "train_predictor"): {
                                "field_separator": "\t",
                                "detect_format": False,
                                "header_line": True,
                                "max_pairs": 1,
                                "max_trees": 5,
                                "max_selected_variables": 1,
                                "max_evaluated_variables": 3,
                                "specific_pairs": [("age", "race")],
                                "all_possible_pairs": False,
                                "construction_rules": ["TableMode", "TableSelection"],
                                "group_target_value": False,
                                "additional_data_tables": {},
                            }
                        },
                        "predict": {
                            ("khiops.core", "deploy_model"): {
                                "field_separator": "\t",
                                "detect_format": False,
                                "header_line": True,
                                "log_file_path": os.path.join(
                                    cls.output_dir, "khiops.log"
                                ),
                                "additional_data_tables": {},
                            }
                        },
                    },
                    KhiopsRegressor: {
                        "fit": {
                            ("khiops.core", "train_predictor"): {
                                "field_separator": "\t",
                                "detect_format": False,
                                "header_line": True,
                                "max_pairs": 1,
                                "max_trees": 0,
                                "max_selected_variables": 1,
                                "max_evaluated_variables": 3,
                                "specific_pairs": [("age", "race")],
                                "all_possible_pairs": False,
                                "construction_rules": ["TableMode", "TableSelection"],
                                "additional_data_tables": {},
                            }
                        },
                        "predict": {
                            ("khiops.core", "deploy_model"): {
                                "field_separator": "\t",
                                "detect_format": False,
                                "header_line": True,
                                "log_file_path": os.path.join(
                                    cls.output_dir, "khiops.log"
                                ),
                                "additional_data_tables": {},
                            }
                        },
                    },
                    KhiopsEncoder: {
                        "fit": {
                            ("khiops.core", "train_recoder"): {
                                "field_separator": "\t",
                                "detect_format": False,
                                "header_line": True,
                                "max_pairs": 1,
                                "max_trees": 5,
                                "specific_pairs": [("age", "race")],
                                "all_possible_pairs": False,
                                "construction_rules": ["TableMode", "TableSelection"],
                                "informative_variables_only": True,
                                "group_target_value": False,
                                "keep_initial_categorical_variables": False,
                                "keep_initial_numerical_variables": False,
                                "categorical_recoding_method": "part Id",
                                "numerical_recoding_method": "part Id",
                                "pairs_recoding_method": "part Id",
                                "additional_data_tables": {},
                            }
                        },
                        "predict": {
                            ("khiops.core", "deploy_model"): {
                                "field_separator": "\t",
                                "detect_format": False,
                                "header_line": True,
                                "log_file_path": os.path.join(
                                    cls.output_dir, "khiops.log"
                                ),
                                "additional_data_tables": {},
                            }
                        },
                    },
                },
            },
            "multitable": {
                ("dataframe",): {
                    KhiopsClassifier: {
                        "fit": {
                            ("khiops.core", "train_predictor"): {
                                "field_separator": "\t",
                                "detect_format": False,
                                "header_line": True,
                                "max_constructed_variables": 10,
                                "max_pairs": 1,
                                "max_trees": 5,
                                "max_selected_variables": 1,
                                "max_evaluated_variables": 3,
                                "specific_pairs": [],
                                "all_possible_pairs": False,
                                "construction_rules": ["TableMode", "TableSelection"],
                                "group_target_value": False,
                                "additional_data_tables": {
                                    "SpliceJunction`SpliceJunctionDNA"
                                },
                            }
                        },
                        "predict": {
                            ("khiops.core", "deploy_model"): {
                                "field_separator": "\t",
                                "detect_format": False,
                                "header_line": True,
                                "log_file_path": os.path.join(
                                    cls.output_dir, "khiops.log"
                                ),
                                "additional_data_tables": {
                                    "SNB_SpliceJunction`SpliceJunctionDNA"
                                },
                            }
                        },
                    },
                    KhiopsRegressor: {
                        "fit": {
                            ("khiops.core", "train_predictor"): {
                                "field_separator": "\t",
                                "detect_format": False,
                                "header_line": True,
                                "max_constructed_variables": 10,
                                "max_pairs": 1,
                                "max_trees": 0,
                                "max_selected_variables": 1,
                                "max_evaluated_variables": 3,
                                "specific_pairs": [],
                                "all_possible_pairs": False,
                                "construction_rules": ["TableMode", "TableSelection"],
                                "additional_data_tables": {
                                    "SpliceJunction`SpliceJunctionDNA"
                                },
                            }
                        },
                        "predict": {
                            ("khiops.core", "deploy_model"): {
                                "field_separator": "\t",
                                "detect_format": False,
                                "header_line": True,
                                "log_file_path": os.path.join(
                                    cls.output_dir, "khiops.log"
                                ),
                                "additional_data_tables": {
                                    "SNB_SpliceJunction`SpliceJunctionDNA"
                                },
                            }
                        },
                    },
                    KhiopsEncoder: {
                        "fit": {
                            ("khiops.core", "train_recoder"): {
                                "field_separator": "\t",
                                "detect_format": False,
                                "header_line": True,
                                "max_constructed_variables": 10,
                                "max_pairs": 1,
                                "max_trees": 5,
                                "specific_pairs": [],
                                "all_possible_pairs": False,
                                "construction_rules": ["TableMode", "TableSelection"],
                                "informative_variables_only": True,
                                "group_target_value": False,
                                "keep_initial_categorical_variables": False,
                                "keep_initial_numerical_variables": False,
                                "categorical_recoding_method": "part Id",
                                "numerical_recoding_method": "part Id",
                                "pairs_recoding_method": "part Id",
                                "additional_data_tables": {
                                    "SpliceJunction`SpliceJunctionDNA"
                                },
                            }
                        },
                        "predict": {
                            ("khiops.core", "deploy_model"): {
                                "field_separator": "\t",
                                "detect_format": False,
                                "header_line": True,
                                "log_file_path": os.path.join(
                                    cls.output_dir, "khiops.log"
                                ),
                                "additional_data_tables": {
                                    "R_SpliceJunction`SpliceJunctionDNA"
                                },
                            }
                        },
                    },
                },
                ("file_dataset",): {
                    KhiopsClassifier: {
                        "fit": {
                            ("khiops.core", "train_predictor"): {
                                "field_separator": "\t",
                                "detect_format": False,
                                "header_line": True,
                                "max_constructed_variables": 10,
                                "max_pairs": 1,
                                "max_trees": 5,
                                "max_selected_variables": 1,
                                "max_evaluated_variables": 3,
                                "specific_pairs": [],
                                "all_possible_pairs": False,
                                "construction_rules": ["TableMode", "TableSelection"],
                                "group_target_value": False,
                                "log_file_path": os.path.join(
                                    cls.output_dir, "khiops.log"
                                ),
                                "additional_data_tables": {
                                    "SpliceJunction`SpliceJunctionDNA"
                                },
                            }
                        },
                        "predict": {
                            ("khiops.core", "deploy_model"): {
                                "field_separator": "\t",
                                "detect_format": False,
                                "header_line": True,
                                "log_file_path": os.path.join(
                                    cls.output_dir, "khiops.log"
                                ),
                                "additional_data_tables": {
                                    "SNB_SpliceJunction`SpliceJunctionDNA"
                                },
                            }
                        },
                    },
                    KhiopsRegressor: {
                        "fit": {
                            ("khiops.core", "train_predictor"): {
                                "field_separator": "\t",
                                "detect_format": False,
                                "header_line": True,
                                "max_constructed_variables": 10,
                                "max_pairs": 1,
                                "max_trees": 0,
                                "max_selected_variables": 1,
                                "max_evaluated_variables": 3,
                                "specific_pairs": [],
                                "all_possible_pairs": False,
                                "construction_rules": ["TableMode", "TableSelection"],
                                "log_file_path": os.path.join(
                                    cls.output_dir, "khiops.log"
                                ),
                                "additional_data_tables": {
                                    "SpliceJunction`SpliceJunctionDNA"
                                },
                            }
                        },
                        "predict": {
                            ("khiops.core", "deploy_model"): {
                                "field_separator": "\t",
                                "detect_format": False,
                                "header_line": True,
                                "log_file_path": os.path.join(
                                    cls.output_dir, "khiops.log"
                                ),
                                "additional_data_tables": {
                                    "SNB_SpliceJunction`SpliceJunctionDNA"
                                },
                            }
                        },
                    },
                    KhiopsEncoder: {
                        "fit": {
                            ("khiops.core", "train_recoder"): {
                                "field_separator": "\t",
                                "detect_format": False,
                                "header_line": True,
                                "max_constructed_variables": 10,
                                "max_pairs": 1,
                                "max_trees": 5,
                                "specific_pairs": [],
                                "all_possible_pairs": False,
                                "construction_rules": ["TableMode", "TableSelection"],
                                "informative_variables_only": True,
                                "group_target_value": False,
                                "keep_initial_categorical_variables": False,
                                "keep_initial_numerical_variables": False,
                                "categorical_recoding_method": "part Id",
                                "numerical_recoding_method": "part Id",
                                "pairs_recoding_method": "part Id",
                                "additional_data_tables": {
                                    "SpliceJunction`SpliceJunctionDNA"
                                },
                            }
                        },
                        "predict": {
                            ("khiops.core", "deploy_model"): {
                                "field_separator": "\t",
                                "detect_format": False,
                                "header_line": True,
                                "log_file_path": os.path.join(
                                    cls.output_dir, "khiops.log"
                                ),
                                "additional_data_tables": {
                                    "R_SpliceJunction`SpliceJunctionDNA"
                                },
                            }
                        },
                    },
                },
            },
        }

        cls.special_kwarg_checkers = {
            KhiopsPredictor: {
                "fit": {
                    ("khiops.core", "train_predictor"): {
                        "log_file_path": cls.assertEqualPath,
                        "additional_data_tables": (
                            cls.assertEqualAdditionalDataTableNames
                        ),
                    }
                },
                "predict": {
                    ("khiops.core", "deploy_model"): {
                        "log_file_path": cls.assertEqualPath,
                        "additional_data_tables": (
                            cls.assertEqualAdditionalDataTableNames
                        ),
                    }
                },
            },
            KhiopsEncoder: {
                "fit": {
                    ("khiops.core", "train_recoder"): {
                        "additional_data_tables": (
                            cls.assertEqualAdditionalDataTableNames
                        )
                    }
                },
                "predict": {
                    ("khiops.core", "deploy_model"): {
                        "log_file_path": cls.assertEqualPath,
                        "additional_data_tables": (
                            cls.assertEqualAdditionalDataTableNames
                        ),
                    }
                },
            },
            KhiopsCoclustering: {
                "fit": {
                    ("khiops.core", "train_coclustering"): {
                        "log_file_path": cls.assertEqualPath
                    }
                },
                "simplify": {},
                "predict": {
                    ("khiops.core", "deploy_model"): {
                        "additional_data_tables": (
                            cls.assertEqualAdditionalDataTableNames
                        ),
                        "log_file_path": cls.assertEqualPath,
                    },
                    ("khiops.core", "extract_keys_from_data_table"): {},
                },
            },
        }

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary resources for this test case: output directory"""
        if os.path.isdir(cls.output_dir):
            shutil.rmtree(cls.output_dir)

        # Enable file-path warnings
        warnings.filterwarnings(
            "always",
            message="File-path dataset input is deprecated and will be removed",
        )

    def _check_dictionary_domain(
        self,
        dictionary_domain,
        expected_n_dictionaries=None,
        expected_main_table_key=None,
        expected_main_dictionary_name=None,
        expected_additional_data_table_names=(),
    ):
        """Check assertions on dictionary domains"""
        self.assertIsInstance(dictionary_domain, kh.DictionaryDomain)
        if expected_n_dictionaries is not None:
            self.assertEqual(
                len(dictionary_domain.dictionaries), expected_n_dictionaries
            )
        for dictionary in dictionary_domain.dictionaries:
            self.assertIsInstance(dictionary, kh.dictionary.Dictionary)
        if expected_main_dictionary_name is not None:
            self.assertEqual(
                dictionary_domain.dictionaries[0].name,
                expected_main_dictionary_name,
            )
        if expected_additional_data_table_names:
            if expected_main_table_key is not None:
                for dictionary in dictionary_domain.dictionaries:
                    self.assertEqual(dictionary.key, [expected_main_table_key])
            self.assertTrue(dictionary_domain.dictionaries[0].root)
            for i, additional_data_table_name in enumerate(
                expected_additional_data_table_names, start=1
            ):
                self.assertEqual(
                    dictionary_domain.dictionaries[i].name,
                    additional_data_table_name,
                )
        else:
            self.assertFalse(dictionary_domain.dictionaries[0].root)
        for dictionary in dictionary_domain.dictionaries[1:]:
            self.assertFalse(dictionary.root)

    def _check_args(self, args, expected_args_with_pos, special_checkers=None):
        """Check assertions on positional arguments"""
        for expected_arg_pos, expected_arg in expected_args_with_pos.items():
            # Check that there is an argument at position
            self.assertGreater(
                len(args),
                expected_arg_pos,
                msg=(
                    "There is no actual argument at expected position "
                    f"'{expected_arg_pos}'"
                ),
            )
            # Normal case
            actual_arg = args[expected_arg_pos]
            if special_checkers is None or expected_arg_pos not in special_checkers:
                self.assertEqual(
                    expected_arg,
                    actual_arg,
                    msg=(
                        f"Expected argument value '{expected_arg}' not in "
                        f"args at expected position '{expected_arg_pos}'."
                    ),
                )
            # Special check case
            else:
                special_checker = special_checkers[expected_arg_pos]
                assert callable(special_checker), (
                    "Special checker for argument at position "
                    f"'{expected_arg_pos}' is not callable."
                )
                special_checker(self, actual_arg, expected_arg)

    def _check_kwargs(
        self,
        kwargs,
        expected_kwargs,
        special_checkers=None,
    ):
        """Check assertions on keyword arguments"""
        # Check that expected keys are keyword arguments
        for key in expected_kwargs.keys():
            self.assertIn(
                key,
                kwargs,
                msg=f"Expected key '{key}' not in kwargs.",
            )

        # Check the expected values; thus, only some kwargs can be checked
        for key, expected_value in expected_kwargs.items():
            actual_value = kwargs[key]

            # Normal case
            if special_checkers is None or key not in special_checkers:
                self.assertEqual(
                    actual_value,
                    expected_value,
                    msg=f"Value for key '{key}' should be '{expected_value}' "
                    f"not '{actual_value}'.",
                )
            # Special check case
            else:
                special_checker = special_checkers[key]
                assert callable(
                    special_checker
                ), f"Special checker for '{key}' is not callable."
                special_checker(self, actual_value, expected_value)

    @classmethod
    def _create_train_test_multitable_file_dataset(cls, transform_for_regression=False):
        (
            root_table_data,
            secondary_table_data,
        ) = KhiopsTestHelper.get_two_table_data(
            "SpliceJunction", "SpliceJunction", "SpliceJunctionDNA"
        )
        root_table_file_name_suffix = ""
        if transform_for_regression:
            root_table_data.replace({"Class": {"EI": 1, "IE": 2, "N": 3}}, inplace=True)
            root_table_file_name_suffix = "_R"
        (root_train_data, root_labels), (
            root_test_data,
            _,
        ) = KhiopsTestHelper.prepare_data(root_table_data, "Class")
        root_train_data["Class"] = root_labels
        secondary_train_data = (
            root_train_data["SampleId"]
            .to_frame()
            .merge(secondary_table_data, on="SampleId")
        )
        secondary_test_data = (
            root_test_data["SampleId"]
            .to_frame()
            .merge(secondary_table_data, on="SampleId")
        )
        root_train_data_path = os.path.join(
            cls.output_dir, f"SpliceJunction_train{root_table_file_name_suffix}.txt"
        )
        root_test_data_path = os.path.join(
            cls.output_dir, f"SpliceJunction_test{root_table_file_name_suffix}.txt"
        )
        secondary_train_data_path = os.path.join(
            cls.output_dir, "SpliceJunctionDNA_train.txt"
        )
        secondary_test_data_path = os.path.join(
            cls.output_dir, "SpliceJunctionDNA_test.txt"
        )
        root_train_data.to_csv(root_train_data_path, sep="\t", header=True, index=False)
        root_test_data.to_csv(root_test_data_path, sep="\t", header=True, index=False)
        secondary_train_data.to_csv(
            secondary_train_data_path, sep="\t", header=True, index=False
        )
        secondary_test_data.to_csv(
            secondary_test_data_path, sep="\t", header=True, index=False
        )
        train_dataset = {
            "main_table": "SpliceJunction",
            "tables": {
                "SpliceJunction": (
                    root_train_data_path,
                    "SampleId",
                ),
                "SpliceJunctionDNA": (
                    secondary_train_data_path,
                    "SampleId",
                ),
            },
            "format": ("\t", True),
        }
        test_dataset = {
            "main_table": "SpliceJunction",
            "tables": {
                "SpliceJunction": (
                    root_test_data_path,
                    "SampleId",
                ),
                "SpliceJunctionDNA": (
                    secondary_test_data_path,
                    "SampleId",
                ),
            },
            "format": ("\t", True),
        }

        return (train_dataset, test_dataset)

    @classmethod
    def _create_train_test_multitable_dataframe(cls, transform_for_regression=False):
        dataset_name = "SpliceJunction"
        (
            root_table_data,
            secondary_table_data,
        ) = KhiopsTestHelper.get_two_table_data(
            dataset_name, "SpliceJunction", "SpliceJunctionDNA"
        )
        if transform_for_regression:
            root_table_data.replace({"Class": {"EI": 1, "IE": 2, "N": 3}}, inplace=True)
        root_train_data, root_test_data = KhiopsTestHelper.prepare_data(
            root_table_data, "Class"
        )
        secondary_train_data = KhiopsTestHelper.prepare_data(
            secondary_table_data, "SampleId", primary_table=root_train_data[0]
        )[0]
        secondary_test_data = KhiopsTestHelper.prepare_data(
            secondary_table_data, "SampleId", primary_table=root_test_data[0]
        )
        X_train_data = {
            "main_table": "SpliceJunction",
            "tables": {
                "SpliceJunction": (root_train_data[0], "SampleId"),
                "SpliceJunctionDNA": (secondary_train_data[0], "SampleId"),
            },
        }
        y_train_data = root_train_data[1]
        X_test_data = {
            "main_table": "SpliceJunction",
            "tables": {
                "SpliceJunction": (root_test_data[0], "SampleId"),
                "SpliceJunctionDNA": (secondary_test_data[0][0], "SampleId"),
            },
        }
        return (X_train_data, y_train_data, X_test_data)

    @classmethod
    def _create_train_test_monotable_file_dataset(cls, label):
        data = KhiopsTestHelper.get_monotable_data("Adult")
        (train_data, train_labels), (
            test_data,
            _,
        ) = KhiopsTestHelper.prepare_data(data, label)

        train_data_path = os.path.join(cls.output_dir, f"Adult_train_for_{label}.txt")
        test_data_path = os.path.join(cls.output_dir, f"Adult_test_for_{label}.txt")
        train_data[label] = train_labels
        train_data.to_csv(train_data_path, sep="\t", header=True, index=False)
        test_data.to_csv(test_data_path, sep="\t", header=True, index=False)
        train_dataset = {
            "main_table": "main_table",
            "tables": {"main_table": (train_data_path, None)},
            "format": ("\t", True),
        }
        test_dataset = {
            "main_table": "main_table",
            "tables": {"main_table": (test_data_path, None)},
            "format": ("\t", True),
        }
        return (train_dataset, test_dataset)

    @classmethod
    def _create_train_test_monotable_dataframe(cls, label, y_as_dataframe=False):
        data = KhiopsTestHelper.get_monotable_data("Adult")
        (train_data, train_labels), (
            test_data,
            _,
        ) = KhiopsTestHelper.prepare_data(data, label, y_as_dataframe=y_as_dataframe)
        return (train_data, train_labels, test_data)

    def _retrieve_data(
        self,
        schema_type,
        source_type,
        estimation_process,
    ):
        return self.datasets[schema_type][source_type][estimation_process]

    def _define_resources(self, dataset, estimator_type, estimator_method):
        # Set the resources directory for the arguments
        head_dir = os.path.join(
            KhiopsTestHelper.get_resources_dir(), "sklearn", "results"
        )
        tail_dir = os.path.join(dataset, estimator_type.__name__)
        ref_reports_dir = os.path.join(head_dir, "ref_json_reports", tail_dir)
        ref_models_dir = os.path.join(head_dir, "ref_models", tail_dir)
        ref_predictions_dir = os.path.join(head_dir, "ref_predictions", tail_dir)

        # Set resources that vary over estimator types
        if estimator_type == KhiopsCoclustering:
            kdic_name = "Coclustering"
            report_name = "Coclustering.khcj"
            tmp_model_kdic_path = os.path.join(
                ref_models_dir, "tmp_cc_deploy_model.kdic"
            )
            raw_keys_table_path = os.path.join(
                ref_predictions_dir, "raw_keys_main_table.txt"
            )
            log_file_path = os.path.join(head_dir, "khiops_train_cc.log")
        else:
            kdic_name = "Modeling"
            report_name = "AllReports.khj"
            tmp_model_kdic_path = None
            raw_keys_table_path = None
            log_file_path = os.path.join(head_dir, "khiops.log")
        report_path = os.path.join(ref_reports_dir, report_name)
        model_kdic_path = os.path.join(ref_models_dir, f"{kdic_name}.kdic")
        model_kdicj_path = os.path.join(ref_models_dir, f"{kdic_name}.kdicj")
        if estimator_type in (KhiopsCoclustering, KhiopsEncoder):
            prediction_table_path = os.path.join(ref_predictions_dir, "transform.txt")
        else:
            if estimator_method == "predict":
                prediction_table_path = os.path.join(ref_predictions_dir, "predict.txt")
            elif estimator_method == "predict_proba":
                prediction_table_path = os.path.join(
                    ref_predictions_dir, "predict_proba.txt"
                )
            else:
                assert estimator_method == "fit", f"Real: {estimator_method}"
                prediction_table_path = ""

        # Buld the resources
        resources = {
            "report_path": report_path,
            "model_kdic_path": model_kdic_path,
            "tmp_model_kdic_path": tmp_model_kdic_path,
            "model_kdicj_path": model_kdicj_path,
            "prediction_table_path": prediction_table_path,
            "raw_keys_table_path": raw_keys_table_path,
            "log_file_path": log_file_path,
        }

        return resources

    def _test_template(
        self,
        estimator_type,
        estimator_method,
        schema_type,
        source_type,
        custom_kwargs=None,
        extra_estimator_kwargs=None,
    ):
        """Test template

        The template is parameterized by:
        - estimator type: class of the estimator
        - estimator method: "fit", "predict" or "simplify" (for KhiopsCoclustering)
        - schema type: "monotable" or "multitable"
        - source type:
          - "dataframe": input data is a Pandas DataFrame, input labels are a Pandas
            Series,
          - "dataframe_xy": input data and labels are both Pandas DataFrames,
          - "file_dataset": input data is specified as paths to files

        The template also takes custom keyword arguments that can be passed
        to specific estimator methods, viz. `fit` and `predict`.

        The template contains the following stages:
        - data retrieval: training (for all tests) and testing (for predict
          tests)
        - resource definitions for mocks
        - mock definitions
        - (localised) mock application, by chaining mock context managers
          and entering them
        - within the mock context managers, wrap tested functions with
          parameter trace
        - run tested functions (`fit` and `predict`, for each estimator)
        - extract parameters traced for the core API functions
        - check these parameters (args and kwargs) with respect to expected
          values
        """
        if estimator_type == KhiopsCoclustering:
            # reuse classifier multitable dataset for coclustering
            data = self._retrieve_data(
                schema_type="multitable",
                source_type=source_type,
                estimation_process=KhiopsClassifier,
            )
            # choose train and test sets according to the source type
            if source_type == "dataframe":
                X_train_data = data["X_train"]["tables"]["SpliceJunctionDNA"][
                    0
                ]  # XXX leaky
                y_train_data = None
                X_test_data = data["X_test"]["tables"]["SpliceJunctionDNA"][
                    0
                ]  # XXX leaky
            else:
                assert source_type == "file_dataset"
                dataset = copy.deepcopy(data)
                X_train_data = dataset["train"]
                del X_train_data["tables"]["SpliceJunction"]  # XXX leaky

                # set the key to None, as the dataset is monotable
                X_train_data["tables"]["SpliceJunctionDNA"] = (
                    X_train_data["tables"]["SpliceJunctionDNA"][0],
                    None,
                )
                X_train_data["main_table"] = "main_table"
                X_train_data["tables"]["main_table"] = X_train_data["tables"][
                    "SpliceJunctionDNA"
                ]
                del X_train_data["tables"]["SpliceJunctionDNA"]

                y_train_data = None

                X_test_data = dataset["test"]
                del X_test_data["tables"]["SpliceJunction"]  # XXX leaky

                # set the key to None, as the dataset is monotable
                X_test_data["tables"]["SpliceJunctionDNA"] = (
                    X_test_data["tables"]["SpliceJunctionDNA"][0],
                    None,
                )
                X_test_data["main_table"] = "main_table"
                X_test_data["tables"]["main_table"] = X_test_data["tables"][
                    "SpliceJunctionDNA"
                ]
                del X_test_data["tables"]["SpliceJunctionDNA"]
        else:
            assert issubclass(estimator_type, KhiopsSupervisedEstimator)
            data = self._retrieve_data(
                schema_type=schema_type,
                source_type=source_type,
                estimation_process=estimator_type,
            )
            if source_type in ("dataframe", "dataframe_xy"):
                X_train_data = data["X_train"]
                y_train_data = data["y_train"]
                X_test_data = data["X_test"]
            else:
                assert source_type == "file_dataset"
                X_train_data = data["train"]
                if schema_type == "multitable":
                    y_train_data = "Class"
                else:
                    assert schema_type == "monotable"
                    if estimator_type == KhiopsRegressor:
                        y_train_data = "age"
                    else:
                        y_train_data = "class"
                X_test_data = data["test"]
        dataset = self.dataset_of_schema_type[schema_type]

        resources = self._define_resources(dataset, estimator_type, estimator_method)

        estimator_type_key = (
            KhiopsPredictor
            if estimator_type in (KhiopsClassifier, KhiopsRegressor)
            else estimator_type
        )

        # Build the mock table
        function_mocks = self.mocks_table[estimator_type_key]["fit"](resources)
        if estimator_method == "predict":
            function_mocks += self.mocks_table[estimator_type_key]["predict"](resources)

        # Enter the context of each function mock
        with contextlib.ExitStack() as stack:
            for function_mock in function_mocks:
                stack.enter_context(function_mock)

            # Set the parameter trace for wrapped functions
            parameter_trace = KhiopsTestHelper.create_parameter_trace()
            for module, function in self.wrapped_functions[estimator_type_key][
                estimator_method
            ]:
                KhiopsTestHelper.wrap_with_parameter_trace(
                    module, function, parameter_trace
                )

            # Train the estimator
            if extra_estimator_kwargs is None:
                extra_estimator_kwargs = {}
            estimator = estimator_type(
                output_dir=self.output_dir, auto_sort=False, **extra_estimator_kwargs
            )
            fit_kwargs = (
                custom_kwargs.get("fit", {}) if custom_kwargs is not None else {}
            )
            estimator.fit(X_train_data, y_train_data, **fit_kwargs)

            # Custom logic for calling additional methods, after `fit`:
            # On a "simplify" test: execute `simplify` on the fitted
            # coclustering estimator
            if estimator_method == "simplify":
                if estimator_type == KhiopsCoclustering:
                    simplify_kwargs = (
                        custom_kwargs.get("simplify", {})
                        if custom_kwargs is not None
                        else {}
                    )
                    estimator.simplify(**simplify_kwargs)
                else:
                    raise NotImplementedError(
                        f"The '{estimator_method}' is not implemented on estimators"
                        f" of type '{estimator_type}'; it is only implemented on"
                        " estimators of type 'KhiopsCoclustering'"
                    )
            # On a "predict" test: execute `predict` / `transform` on the fitted
            # estimator
            elif estimator_method == "predict":
                if estimator_type == KhiopsEncoder:
                    estimator.transform(X_test_data)
                else:
                    estimator.predict(X_test_data)

            # Check the parameters of the traced functions
            for module_name, function_parameters in parameter_trace.items():
                for function_name, parameters in function_parameters.items():
                    # Access the actual args and kwargs of the function
                    first_call_parameters = parameters[0]
                    args = first_call_parameters["args"]
                    kwargs = first_call_parameters["kwargs"]

                    # Check the dictionary domain-specific kwargs
                    dictionary_domain_kwargs_list = (
                        adict.get(estimator_type_key).get(function_name)
                        for adict in KhiopsTestHelper.get_with_subkey(
                            self.dictionary_domain_kwargs.get(schema_type), source_type
                        )
                    )
                    for dictionary_domain_kwargs in dictionary_domain_kwargs_list:
                        if dictionary_domain_kwargs is not None:
                            self._check_dictionary_domain(
                                dictionary_domain=args[0],
                                **dictionary_domain_kwargs,
                            )

                    # Check the function args
                    expected_args_list = (
                        adict.get(estimator_type)
                        .get(estimator_method)
                        .get((module_name, function_name))
                        for adict in KhiopsTestHelper.get_with_subkey(
                            self.expected_args.get(schema_type), source_type
                        )
                    )
                    special_arg_checkers = (
                        self.special_arg_checkers.get(estimator_type_key)
                        .get(estimator_method)
                        .get((module_name, function_name))
                    )
                    for expected_args in expected_args_list:
                        self._check_args(
                            args,
                            expected_args_with_pos=expected_args,
                            special_checkers=special_arg_checkers,
                        )

                    # Check the function kwargs
                    expected_kwargs_list = (
                        adict.get(estimator_type)
                        .get(estimator_method)
                        .get((module_name, function_name))
                        for adict in KhiopsTestHelper.get_with_subkey(
                            self.expected_kwargs.get(schema_type), source_type
                        )
                    )
                    special_kwarg_checkers = (
                        self.special_kwarg_checkers.get(estimator_type_key)
                        .get(estimator_method)
                        .get((module_name, function_name))
                    )
                    for expected_kwargs in expected_kwargs_list:
                        self._check_kwargs(
                            kwargs,
                            expected_kwargs=expected_kwargs,
                            special_checkers=special_kwarg_checkers,
                        )

    def test_parameter_transfer_classifier_fit_from_monotable_dataframe(self):
        """Test parameter transfer from monotable dataframe fit to core API"""
        self._test_template(
            estimator_type=KhiopsClassifier,
            estimator_method="fit",
            schema_type="monotable",
            source_type="dataframe",
            extra_estimator_kwargs={
                "n_pairs": 1,
                "n_trees": 5,
                "n_selected_features": 1,
                "n_evaluated_features": 3,
                "specific_pairs": [("age", "race")],
                "all_possible_pairs": False,
                "construction_rules": ["TableMode", "TableSelection"],
                "group_target_value": False,
            },
        )

    def test_parameter_transfer_classifier_fit_from_monotable_dataframe_with_df_y(
        self,
    ):
        """Test parameter transfer from monotable dataframe fit to core API"""
        self._test_template(
            estimator_type=KhiopsClassifier,
            estimator_method="fit",
            schema_type="monotable",
            source_type="dataframe_xy",
            extra_estimator_kwargs={
                "n_pairs": 1,
                "n_trees": 5,
                "n_selected_features": 1,
                "n_evaluated_features": 3,
                "specific_pairs": [("age", "race")],
                "all_possible_pairs": False,
                "construction_rules": ["TableMode", "TableSelection"],
                "group_target_value": False,
            },
        )

    def test_parameter_transfer_classifier_fit_from_monotable_file_dataset(self):
        """Test parameter transfer from monotable file dataset fit to core API"""
        self._test_template(
            estimator_type=KhiopsClassifier,
            estimator_method="fit",
            schema_type="monotable",
            source_type="file_dataset",
            extra_estimator_kwargs={
                "n_pairs": 1,
                "n_trees": 5,
                "n_selected_features": 1,
                "n_evaluated_features": 3,
                "specific_pairs": [("age", "race")],
                "all_possible_pairs": False,
                "construction_rules": ["TableMode", "TableSelection"],
                "group_target_value": False,
            },
        )

    def test_parameter_transfer_classifier_fit_from_multitable_dataframe(self):
        """Test parameter transfer from multitable dataframe fit to core API"""
        self._test_template(
            estimator_type=KhiopsClassifier,
            estimator_method="fit",
            schema_type="multitable",
            source_type="dataframe",
            extra_estimator_kwargs={
                "n_features": 10,
                "n_pairs": 1,
                "n_trees": 5,
                "n_selected_features": 1,
                "n_evaluated_features": 3,
                "specific_pairs": [],
                "all_possible_pairs": False,
                "construction_rules": ["TableMode", "TableSelection"],
                "group_target_value": False,
            },
        )

    def test_parameter_transfer_classifier_fit_from_multitable_file_dataset(self):
        """Test parameter transfer from file dataset fit to core API"""
        self._test_template(
            estimator_type=KhiopsClassifier,
            estimator_method="fit",
            schema_type="multitable",
            source_type="file_dataset",
            extra_estimator_kwargs={
                "n_features": 10,
                "n_pairs": 1,
                "n_trees": 5,
                "n_selected_features": 1,
                "n_evaluated_features": 3,
                "specific_pairs": [],
                "all_possible_pairs": False,
                "construction_rules": ["TableMode", "TableSelection"],
                "group_target_value": False,
            },
        )

    def test_parameter_transfer_classifier_predict_from_monotable_dataframe(self):
        """Test parameter transfer from monotable dataframe predict to core API"""
        self._test_template(
            estimator_type=KhiopsClassifier,
            estimator_method="predict",
            schema_type="monotable",
            source_type="dataframe",
        )

    def test_parameter_transfer_classifier_predict_from_monotable_file_dataset(
        self,
    ):
        """Test parameter transfer from monotable file dataset predict to core api"""
        self._test_template(
            estimator_type=KhiopsClassifier,
            estimator_method="predict",
            schema_type="monotable",
            source_type="file_dataset",
        )

    def test_parameter_transfer_classifier_predict_from_multitable_dataframe(self):
        """Test parameter transfer from dataframe predict to core API"""
        self._test_template(
            estimator_type=KhiopsClassifier,
            estimator_method="predict",
            schema_type="multitable",
            source_type="dataframe",
        )

    def test_parameter_transfer_classifier_predict_from_multitable_file_dataset(
        self,
    ):
        """Test parameter transfer from file dataset predict to core API"""
        self._test_template(
            estimator_type=KhiopsClassifier,
            estimator_method="predict",
            schema_type="multitable",
            source_type="file_dataset",
        )

    def test_parameter_transfer_encoder_fit_from_monotable_dataframe(self):
        """Test parameter transfer from monotable dataframe fit to core API"""
        self._test_template(
            estimator_type=KhiopsEncoder,
            estimator_method="fit",
            schema_type="monotable",
            source_type="dataframe",
            extra_estimator_kwargs={
                "n_pairs": 1,
                "n_trees": 5,
                "specific_pairs": [("age", "race")],
                "all_possible_pairs": False,
                "construction_rules": ["TableMode", "TableSelection"],
                "informative_features_only": True,
                "group_target_value": False,
                "keep_initial_variables": False,
                "transform_type_categorical": "part_id",
                "transform_type_numerical": "part_id",
                "transform_pairs": "part_id",
            },
        )

    def test_parameter_transfer_encoder_fit_from_monotable_dataframe_with_df_y(
        self,
    ):
        """Test parameter transfer from monotable dataframe fit to core API"""
        self._test_template(
            estimator_type=KhiopsEncoder,
            estimator_method="fit",
            schema_type="monotable",
            source_type="dataframe_xy",
            extra_estimator_kwargs={
                "n_pairs": 1,
                "n_trees": 5,
                "specific_pairs": [("age", "race")],
                "all_possible_pairs": False,
                "construction_rules": ["TableMode", "TableSelection"],
                "informative_features_only": True,
                "group_target_value": False,
                "keep_initial_variables": False,
                "transform_type_categorical": "part_id",
                "transform_type_numerical": "part_id",
                "transform_pairs": "part_id",
            },
        )

    def test_parameter_transfer_encoder_fit_from_monotable_file_dataset(self):
        """Test parameter transfer from monotable file dataset fit to core API"""
        self._test_template(
            estimator_type=KhiopsEncoder,
            estimator_method="fit",
            schema_type="monotable",
            source_type="file_dataset",
            extra_estimator_kwargs={
                "n_pairs": 1,
                "n_trees": 5,
                "specific_pairs": [("age", "race")],
                "all_possible_pairs": False,
                "construction_rules": ["TableMode", "TableSelection"],
                "informative_features_only": True,
                "group_target_value": False,
                "keep_initial_variables": False,
                "transform_type_categorical": "part_id",
                "transform_type_numerical": "part_id",
                "transform_pairs": "part_id",
            },
        )

    def test_parameter_transfer_encoder_fit_from_multitable_dataframe(self):
        """Test parameter transfer from multitable dataframe fit to core API"""
        self._test_template(
            estimator_type=KhiopsEncoder,
            estimator_method="fit",
            schema_type="multitable",
            source_type="dataframe",
            extra_estimator_kwargs={
                "n_features": 10,
                "n_pairs": 1,
                "n_trees": 5,
                "specific_pairs": [],
                "all_possible_pairs": False,
                "construction_rules": ["TableMode", "TableSelection"],
                "informative_features_only": True,
                "group_target_value": False,
                "keep_initial_variables": False,
                "transform_type_categorical": "part_id",
                "transform_type_numerical": "part_id",
                "transform_pairs": "part_id",
            },
        )

    def test_parameter_transfer_encoder_fit_from_multitable_file_dataset(self):
        """Test parameter transfer from multitable file dataset fit to core API"""
        self._test_template(
            estimator_type=KhiopsEncoder,
            estimator_method="fit",
            schema_type="multitable",
            source_type="file_dataset",
            extra_estimator_kwargs={
                "n_features": 10,
                "n_pairs": 1,
                "n_trees": 5,
                "specific_pairs": [],
                "all_possible_pairs": False,
                "construction_rules": ["TableMode", "TableSelection"],
                "informative_features_only": True,
                "group_target_value": False,
                "keep_initial_variables": False,
                "transform_type_categorical": "part_id",
                "transform_type_numerical": "part_id",
                "transform_pairs": "part_id",
            },
        )

    def test_parameter_transfer_encoder_predict_from_monotable_dataframe(self):
        """Test parameter transfer from monotable dataframe predict to core API"""
        self._test_template(
            estimator_type=KhiopsEncoder,
            estimator_method="predict",
            schema_type="monotable",
            source_type="dataframe",
        )

    def test_parameter_transfer_encoder_predict_from_monotable_file_dataset(self):
        """Test parameter transfer from monotable file dataset predict to core API"""
        self._test_template(
            estimator_type=KhiopsEncoder,
            estimator_method="predict",
            schema_type="monotable",
            source_type="file_dataset",
        )

    def test_parameter_transfer_encoder_predict_from_multitable_dataframe(self):
        """Test parameter transfer from dataframe predict to core API"""
        self._test_template(
            estimator_type=KhiopsEncoder,
            estimator_method="predict",
            schema_type="multitable",
            source_type="dataframe",
        )

    def test_parameter_transfer_encoder_predict_from_multitable_file_dataset(self):
        """Test parameter transfer from file dataset predict to core API"""
        self._test_template(
            estimator_type=KhiopsEncoder,
            estimator_method="predict",
            schema_type="multitable",
            source_type="file_dataset",
        )

    def test_parameter_transfer_regressor_fit_from_monotable_dataframe(self):
        """Test parameter transfer from monotable dataframe fit to core API"""
        self._test_template(
            estimator_type=KhiopsRegressor,
            estimator_method="fit",
            schema_type="monotable",
            source_type="dataframe",
            extra_estimator_kwargs={
                "n_pairs": 1,
                "n_selected_features": 1,
                "n_evaluated_features": 3,
                "specific_pairs": [("age", "race")],
                "all_possible_pairs": False,
                "construction_rules": ["TableMode", "TableSelection"],
            },
        )

    def test_parameter_transfer_regressor_fit_from_monotable_dataframe_with_df_y(
        self,
    ):
        """Test parameter transfer from monotable dataframe fit to core API"""
        self._test_template(
            estimator_type=KhiopsRegressor,
            estimator_method="fit",
            schema_type="monotable",
            source_type="dataframe_xy",
            extra_estimator_kwargs={
                "n_pairs": 1,
                "n_selected_features": 1,
                "n_evaluated_features": 3,
                "specific_pairs": [("age", "race")],
                "all_possible_pairs": False,
                "construction_rules": ["TableMode", "TableSelection"],
            },
        )

    def test_parameter_transfer_regressor_fit_from_monotable_file_dataset(self):
        """Test parameter transfer from monotable file dataset fit to core API"""
        self._test_template(
            estimator_type=KhiopsRegressor,
            estimator_method="fit",
            schema_type="monotable",
            source_type="file_dataset",
            extra_estimator_kwargs={
                "n_pairs": 1,
                "n_selected_features": 1,
                "n_evaluated_features": 3,
                "specific_pairs": [("age", "race")],
                "all_possible_pairs": False,
                "construction_rules": ["TableMode", "TableSelection"],
            },
        )

    def test_parameter_transfer_regressor_fit_from_multitable_dataframe(self):
        """Test parameter transfer from multitable dataframe fit to core API"""
        self._test_template(
            estimator_type=KhiopsRegressor,
            estimator_method="fit",
            schema_type="multitable",
            source_type="dataframe",
            extra_estimator_kwargs={
                "n_features": 10,
                "n_pairs": 1,
                "n_trees": 0,
                "n_selected_features": 1,
                "n_evaluated_features": 3,
                "specific_pairs": [],
                "all_possible_pairs": False,
                "construction_rules": ["TableMode", "TableSelection"],
            },
        )

    def test_parameter_transfer_regressor_fit_from_multitable_file_dataset(self):
        """Test parameter transfer from file dataset fit to core API"""
        self._test_template(
            estimator_type=KhiopsRegressor,
            estimator_method="fit",
            schema_type="multitable",
            source_type="file_dataset",
            extra_estimator_kwargs={
                "n_features": 10,
                "n_pairs": 1,
                "n_trees": 0,
                "n_selected_features": 1,
                "n_evaluated_features": 3,
                "specific_pairs": [],
                "all_possible_pairs": False,
                "construction_rules": ["TableMode", "TableSelection"],
            },
        )

    def test_parameter_transfer_regressor_predict_from_monotable_dataframe(self):
        """Test parameter transfer from monotable dataframe predict to core API"""
        self._test_template(
            estimator_type=KhiopsRegressor,
            estimator_method="predict",
            schema_type="monotable",
            source_type="dataframe",
        )

    def test_parameter_transfer_regressor_predict_from_monotable_file_dataset(self):
        """Test parameter transfer from monotable file dataset predict to core API"""
        self._test_template(
            estimator_type=KhiopsRegressor,
            estimator_method="predict",
            schema_type="monotable",
            source_type="file_dataset",
        )

    def test_parameter_transfer_regressor_predict_from_multitable_dataframe(self):
        """Test parameter transfer from dataframe predict to core API"""
        self._test_template(
            estimator_type=KhiopsRegressor,
            estimator_method="predict",
            schema_type="multitable",
            source_type="dataframe",
        )

    def test_parameter_transfer_regressor_predict_from_multitable_file_dataset(
        self,
    ):
        """Test parameter transfer from file dataset predict to core API"""
        self._test_template(
            estimator_type=KhiopsRegressor,
            estimator_method="predict",
            schema_type="multitable",
            source_type="file_dataset",
        )

    def test_parameter_transfer_coclustering_fit_from_dataframe(self):
        """Test parameter transfer from dataframe coclustering fit to core API"""
        self._test_template(
            estimator_type=KhiopsCoclustering,
            estimator_method="fit",
            schema_type="not_applicable",
            source_type="dataframe",
            custom_kwargs={
                "fit": {
                    "columns": ("SampleId", "Pos", "Char"),
                    "id_column": "SampleId",
                    "max_part_numbers": {"SampleId": 2},
                }
            },
        )

    def test_parameter_transfer_coclustering_fit_from_file_dataset(self):
        """Test parameter transfer from file dataset coclustering fit to core API"""
        self._test_template(
            estimator_type=KhiopsCoclustering,
            estimator_method="fit",
            schema_type="not_applicable",
            source_type="file_dataset",
            custom_kwargs={
                "fit": {
                    "columns": ("SampleId", "Pos", "Char"),
                    "id_column": "SampleId",
                }
            },
        )

    def test_parameter_transfer_coclustering_simplify_from_dataframe(self):
        """Test parameter transfer from dataframe coclustering simplify to core API"""
        self._test_template(
            estimator_type=KhiopsCoclustering,
            estimator_method="simplify",
            schema_type="not_applicable",
            source_type="dataframe",
            custom_kwargs={
                "fit": {
                    "columns": ("SampleId", "Pos", "Char"),
                    "id_column": "SampleId",
                },
                "simplify": {
                    "max_part_numbers": {"SampleId": 2},
                    "max_preserved_information": 3,
                    "max_cells": 4,
                    "max_total_parts": 1,
                },
            },
        )

    def test_parameter_transfer_coclustering_predict_from_dataframe(self):
        # prepare two-table data for coclustering fit from dataframe
        # wrap core coclustering predict with parameter trace
        # (namely, two relevant core API methods)
        # launch sklearn coclustering fit to build a coclustering model
        # probe parameters passed to core's coclustering predict
        self._test_template(
            estimator_type=KhiopsCoclustering,
            estimator_method="predict",
            schema_type="not_applicable",
            source_type="dataframe",
            custom_kwargs={
                "fit": {
                    "columns": ("SampleId", "Pos", "Char"),
                    "id_column": "SampleId",
                }
            },
        )

    def test_parameter_transfer_coclustering_predict_from_file_dataset(self):
        # prepare two-table data for coclustering fit from file dataset
        # wrap core coclustering predict with parameter trace
        # (namely, two relevant core API methods)
        # launch sklearn coclustering fit to build a coclustering model
        # probe parameters passed to core's coclustering predict
        self._test_template(
            estimator_type=KhiopsCoclustering,
            estimator_method="predict",
            schema_type="not_applicable",
            source_type="file_dataset",
            custom_kwargs={
                "fit": {
                    "columns": ("SampleId", "Pos", "Char"),
                    "id_column": "SampleId",
                }
            },
        )


class KhiopsSklearnEstimatorStandardTests(unittest.TestCase):
    """Tests to comply with `sklearn.util.estimator_checks.check_estimator`"""

    def test_sklearn_check_estimator(self):
        # Set the estimators to test
        # Notes:
        # - We use n_trees=0 so the tests execute faster
        # - We omit KhiopsCoclustering because he needs special inputs to work well
        #   and sklearn's check_estimator method does not accept them.
        # - KhiopsEncoder:
        #   - We set it with transform_type_numerical="0-1_normalization" as the tests
        #   expect numeric types
        #   - We set it with informative_features_only=False so it always have output
        #   columns (sklearn estimator checks expect non-empty encoders)
        khiops_estimators = [
            KhiopsClassifier(n_trees=0),
            KhiopsRegressor(n_trees=0),
            KhiopsEncoder(
                n_trees=0,
                informative_features_only=False,
                transform_type_numerical="0-1_normalization",
            ),
        ]

        # Ignore the "No informative variables" warnings
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=r"[\S\n\t\v ]+no informative variables"
            )
            warnings.filterwarnings(
                "ignore", message=r"[\S\n\t\v ]+No informative input variable"
            )

            # Execute sklearn's estimator test battery
            print("")
            for khiops_estimator in khiops_estimators:
                for estimator, check in check_estimator(
                    khiops_estimator, generate_only=True
                ):
                    check_name = check.func.__name__
                    if check_name == "check_n_features_in_after_fitting":
                        continue
                    print(
                        f">>> Executing {check_name} on "
                        f"{estimator.__class__.__name__}... ",
                        end="",
                    )
                    with self.subTest(
                        sklearn_check_name=check_name,
                        sklearn_check_kwargs=check.keywords,
                    ):
                        check(estimator)
                    print("Done")


class KhiopsSklearnVariousTests(unittest.TestCase):
    """Miscelanous sklearn classes tests"""

    def assertNotFit(self, estimator):
        """Asserts that an estimator is not in 'fit' state"""
        try:
            check_is_fitted(estimator)
            self.fail(
                f"Expected {estimator.__class__.__name__} not to be in 'fit' state."
            )
        except NotFittedError:
            pass

    def test_khiops_encoder_no_output_variables_implies_not_fit(self):
        """Test that KhiopsEncoder is not fit when there are no output columns"""
        # Obtain the features of Iris
        df = KhiopsTestHelper.get_monotable_data("Iris")
        X = df.drop("Class", axis=1)

        # Create a noise target
        rng = np.random.default_rng(seed=123)
        y = rng.binomial(1, 0.5, size=X.shape[0])

        # Fit a KhiopsEncoder and check we get a warning about no having output columns
        khe = KhiopsEncoder()
        with self.assertWarnsRegex(
            UserWarning, "Encoder is not fit because Khiops didn't create any output"
        ):
            khe.fit(X, y)

        # Check that the encoder is not fit
        self.assertNotFit(khe)
