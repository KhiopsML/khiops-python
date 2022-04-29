###############################################################################
# Copyright (c) 2022 Orange - All Rights Reserved
# * This software is the confidential and proprietary information of Orange.
# * You shall not disclose such Restricted Information and shall use it only in
#   accordance with the terms of the license agreement you entered into with
#   Orange named the "Khiops - Python Library Evaluation License".
# * Unauthorized copying of this file, via any medium is strictly prohibited.
# * See the "LICENSE.md" file for more details.
###############################################################################
"""Tests parameter transfer between PyKhiops sklearn and core APIs"""
import contextlib
import copy
import os
import shutil
import unittest

import pykhiops.core as pk
from pykhiops.sklearn.estimators import (
    KhiopsClassifier,
    KhiopsCoclustering,
    KhiopsEncoder,
    KhiopsPredictor,
    KhiopsRegressor,
)
from tests.test_helper import Mock, PyKhiopsSklearnTestsHelper

# pylint: disable=too-many-lines
# pylint: disable=invalid-name


class PyKhiopsSklearnParameterPassingTests(
    unittest.TestCase, PyKhiopsSklearnTestsHelper
):
    """Test that parameters are properly passed from sklearn to core API"""

    @staticmethod
    def assertEqualPath(test_case, path1, path2, msg=""):
        test_case.assertEqual(os.path.abspath(path1), os.path.abspath(path2), msg=msg)

    @staticmethod
    def assertPathHasSuffix(test_case, path, suffix, msg=""):
        test_case.assertTrue(path.endswith(suffix), msg=msg)

    @staticmethod
    def assertPathHasPrefix(test_case, path, prefix, msg=""):
        test_case.assertTrue(
            os.path.abspath(path).startswith(os.path.abspath(prefix)), msg=msg
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
            os.curdir, "resources", "tmp", "test_sklearn_parameter_transfer"
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
                "fit": lambda resources: (
                    Mock(
                        module="pykhiops.core.api",
                        function="export_dictionary_as_json",
                        resources=[resources[2]],
                        output_dir_grabber=lambda *x: os.path.dirname(x[1]),
                        output_file_names_map={
                            "Modeling.kdicj": lambda *x: os.path.basename(x[1])
                        },
                    ),
                    Mock(
                        module="pykhiops.core",
                        function="train_predictor",
                        resources=resources[:-1],
                        output_dir_grabber=cls._output_dir_grabber_train,
                    ),
                ),
                "predict": lambda resources: (
                    Mock(
                        module="pykhiops.core",
                        function="deploy_model",
                        resources=resources,
                        output_dir_grabber=cls._output_dir_grabber_deploy,
                    ),
                ),
            },
            KhiopsEncoder: {
                "fit": lambda resources: (
                    Mock(
                        module="pykhiops.core.api",
                        function="export_dictionary_as_json",
                        resources=[resources[2]],
                        output_dir_grabber=lambda *x: os.path.dirname(x[1]),
                        output_file_names_map={
                            "Modeling.kdicj": lambda *x: os.path.basename(x[1])
                        },
                    ),
                    Mock(
                        module="pykhiops.core",
                        function="train_recoder",
                        resources=resources[:-1],
                        output_dir_grabber=cls._output_dir_grabber_train,
                    ),
                ),
                "predict": lambda resources: (
                    Mock(
                        module="pykhiops.core",
                        function="deploy_model",
                        resources=resources,
                        output_dir_grabber=cls._output_dir_grabber_deploy,
                    ),
                ),
            },
            KhiopsCoclustering: {
                "fit": lambda resources: (
                    Mock(
                        module="pykhiops.core.api",
                        function="export_dictionary_as_json",
                        resources=[resources[-1]],
                        output_dir_grabber=lambda *x: os.path.dirname(x[1]),
                        output_file_names_map={
                            "Coclustering.kdicj": lambda *x: os.path.basename(x[1])
                        },
                    ),
                    Mock(
                        module="pykhiops.core",
                        function="train_coclustering",
                        resources=[resources[0]],
                        output_dir_grabber=cls._output_dir_grabber_train,
                    ),
                    Mock(
                        module="pykhiops.core",
                        function="build_multi_table_dictionary",
                        resources=[resources[1]],
                        output_dir_grabber=cls._output_dir_grabber_deploy,
                    ),
                    Mock(
                        module="pykhiops.core",
                        function="prepare_coclustering_deployment",
                        resources=[resources[2]],
                        output_dir_grabber=lambda *x: x[5],
                    ),
                ),
                "predict": lambda resources: (
                    Mock(
                        module="pykhiops.core",
                        function="deploy_model",
                        resources=[resources[0]],
                        output_dir_grabber=cls._output_dir_grabber_deploy,
                    ),
                    Mock(
                        module="pykhiops.core",
                        function="extract_keys_from_data_table",
                        resources=resources[1:],
                        output_dir_grabber=cls._output_dir_grabber_deploy,
                    ),
                ),
            },
        }
        cls.dictionary_domain_kwargs = {
            "not_applicable": {
                "dataframe": {
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
                "file_dataset": {
                    KhiopsCoclustering: {
                        "build_multi_table_dictionary": {
                            "expected_n_dictionaries": 1,
                            "expected_main_table_key": None,
                            "expected_main_dictionary_name": "CC_SpliceJunctionDNA",
                            "expected_additional_data_table_names": [],
                        },
                        "train_coclustering": {
                            "expected_n_dictionaries": 1,
                            "expected_main_table_key": None,
                            "expected_main_dictionary_name": "SpliceJunctionDNA",
                            "expected_additional_data_table_names": [],
                        },
                        "deploy_model": {
                            "expected_n_dictionaries": 2,
                            "expected_main_table_key": "SampleId",
                            "expected_main_dictionary_name": (
                                "CC_Keys_SpliceJunctionDNA"
                            ),
                            "expected_additional_data_table_names": [
                                "CC_SpliceJunctionDNA"
                            ],
                        },
                        "extract_keys_from_data_table": {
                            "expected_n_dictionaries": 1,
                            "expected_main_table_key": "SampleId",
                            "expected_main_dictionary_name": "SpliceJunctionDNA",
                            "expected_additional_data_table_names": [],
                        },
                    }
                },
            },
            "monotable": {
                "dataframe": {
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
                "file_dataset": {
                    KhiopsPredictor: {
                        "train_predictor": {
                            "expected_n_dictionaries": 1,
                            "expected_main_table_key": None,
                            "expected_main_dictionary_name": "Adult",
                            "expected_additional_data_table_names": [],
                        },
                        "deploy_model": {
                            "expected_n_dictionaries": 1,
                            "expected_main_table_key": None,
                            "expected_main_dictionary_name": "SNB_Adult",
                            "expected_additional_data_table_names": [],
                        },
                    },
                    KhiopsEncoder: {
                        "train_recoder": {
                            "expected_n_dictionaries": 1,
                            "expected_main_table_key": None,
                            "expected_main_dictionary_name": "Adult",
                            "expected_additional_data_table_names": [],
                        },
                        "deploy_model": {
                            "expected_n_dictionaries": 1,
                            "expected_main_table_key": None,
                            "expected_main_dictionary_name": "R_Adult",
                            "expected_additional_data_table_names": [],
                        },
                    },
                },
            },
            "multitable": {
                "file_dataset": {
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
                "dataframe": {
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
                "fit": [("pykhiops.core", "train_predictor")],
                "predict": [("pykhiops.core", "deploy_model")],
            },
            KhiopsEncoder: {
                "fit": [("pykhiops.core", "train_recoder")],
                "predict": [("pykhiops.core", "deploy_model")],
            },
            KhiopsCoclustering: {
                "fit": [
                    ("pykhiops.core", "train_coclustering"),
                    ("pykhiops.core", "read_coclustering_results_file"),
                    ("pykhiops.core", "build_multi_table_dictionary"),
                    ("pykhiops.core", "prepare_coclustering_deployment"),
                ],
                "predict": [
                    ("pykhiops.core", "deploy_model"),
                    ("pykhiops.core", "extract_keys_from_data_table"),
                ],
            },
        }
        cls.expected_args = {
            "not_applicable": {
                "dataframe": {
                    KhiopsCoclustering: {
                        "fit": {
                            ("pykhiops.core", "prepare_coclustering_deployment"): {
                                2: os.path.join(cls.output_dir, "Coclustering.khcj"),
                                3: "CC_main_table",
                                4: "SampleId",
                                5: cls.output_dir,
                            },
                            ("pykhiops.core", "read_coclustering_results_file"): {
                                0: os.path.join(cls.output_dir, "Coclustering.khcj")
                            },
                            ("pykhiops.core", "build_multi_table_dictionary"): {
                                2: "CC_main_table"
                            },
                            ("pykhiops.core", "train_coclustering"): {
                                1: "main_table",
                                3: ("SampleId", "Pos", "Char"),
                                4: cls.output_dir,
                            },
                        },
                        "predict": {
                            ("pykhiops.core", "deploy_model"): {
                                1: "CC_Keys_main_table",
                                3: cls.output_dir,
                            },
                            ("pykhiops.core", "extract_keys_from_data_table"): {
                                1: "main_table",
                                2: "sorted_main_table.txt",
                                3: "keys_main_table.txt",
                            },
                        },
                    },
                },
                "file_dataset": {
                    KhiopsCoclustering: {
                        "fit": {
                            ("pykhiops.core", "prepare_coclustering_deployment"): {
                                2: os.path.join(cls.output_dir, "Coclustering.khcj"),
                                3: "CC_SpliceJunctionDNA",
                                4: "SampleId",
                                5: cls.output_dir,
                            },
                            ("pykhiops.core", "read_coclustering_results_file"): {
                                0: os.path.join(cls.output_dir, "Coclustering.khcj")
                            },
                            ("pykhiops.core", "build_multi_table_dictionary"): {
                                2: "CC_SpliceJunctionDNA"
                            },
                            ("pykhiops.core", "train_coclustering"): {
                                1: "SpliceJunctionDNA",
                                3: ("SampleId", "Pos", "Char"),
                                4: cls.output_dir,
                            },
                        },
                        "predict": {
                            ("pykhiops.core", "deploy_model"): {
                                1: "CC_Keys_SpliceJunctionDNA",
                                3: cls.output_dir,
                            },
                            ("pykhiops.core", "extract_keys_from_data_table"): {
                                1: "SpliceJunctionDNA",
                                2: "sorted_SpliceJunctionDNA.txt",
                                3: "keys_SpliceJunctionDNA.txt",
                            },
                        },
                    },
                },
            },
            "monotable": {
                "dataframe": {
                    KhiopsRegressor: {
                        "fit": {
                            ("pykhiops.core", "train_predictor"): {
                                1: "main_table",
                                3: "age",
                            }
                        },
                        "predict": {
                            ("pykhiops.core", "deploy_model"): {
                                1: "SNB_main_table",
                                2: "main_table.txt",
                                3: cls.output_dir,
                            }
                        },
                    },
                    KhiopsClassifier: {
                        "fit": {
                            ("pykhiops.core", "train_predictor"): {
                                1: "main_table",
                                2: "main_table.txt",
                                3: "class",
                            }
                        },
                        "predict": {
                            ("pykhiops.core", "deploy_model"): {
                                1: "SNB_main_table",
                                2: "main_table.txt",
                                3: cls.output_dir,
                            }
                        },
                    },
                    KhiopsEncoder: {
                        "fit": {
                            ("pykhiops.core", "train_recoder"): {
                                1: "main_table",
                                3: "class",
                            }
                        },
                        "predict": {
                            ("pykhiops.core", "deploy_model"): {
                                1: "R_main_table",
                                2: "main_table.txt",
                                3: cls.output_dir,
                            }
                        },
                    },
                },
                "file_dataset": {
                    KhiopsRegressor: {
                        "fit": {
                            ("pykhiops.core", "train_predictor"): {
                                1: "Adult",
                                2: "Adult.txt",
                                3: "age",
                                4: cls.output_dir,
                            }
                        },
                        "predict": {
                            ("pykhiops.core", "deploy_model"): {
                                1: "SNB_Adult",
                                2: "sorted_Adult.txt",
                                3: cls.output_dir,
                            }
                        },
                    },
                    KhiopsClassifier: {
                        "fit": {
                            ("pykhiops.core", "train_predictor"): {
                                1: "Adult",
                                2: "Adult.txt",
                                3: "class",
                                4: cls.output_dir,
                            }
                        },
                        "predict": {
                            ("pykhiops.core", "deploy_model"): {
                                1: "SNB_Adult",
                                2: "sorted_Adult.txt",
                                3: cls.output_dir,
                            }
                        },
                    },
                    KhiopsEncoder: {
                        "fit": {
                            ("pykhiops.core", "train_recoder"): {1: "Adult", 3: "class"}
                        },
                        "predict": {
                            ("pykhiops.core", "deploy_model"): {
                                1: "R_Adult",
                                2: "sorted_Adult.txt",
                                3: cls.output_dir,
                            }
                        },
                    },
                },
            },
            "multitable": {
                "dataframe": {
                    KhiopsRegressor: {
                        "fit": {
                            ("pykhiops.core", "train_predictor"): {
                                1: "SpliceJunction",
                                3: "Class",
                            }
                        },
                        "predict": {
                            ("pykhiops.core", "deploy_model"): {
                                1: "SNB_SpliceJunction",
                                2: "SpliceJunction.txt",
                                3: cls.output_dir,
                            }
                        },
                    },
                    KhiopsClassifier: {
                        "fit": {
                            ("pykhiops.core", "train_predictor"): {
                                1: "SpliceJunction",
                                3: "Class",
                            }
                        },
                        "predict": {
                            ("pykhiops.core", "deploy_model"): {
                                1: "SNB_SpliceJunction",
                                2: "SpliceJunction.txt",
                                3: cls.output_dir,
                            }
                        },
                    },
                    KhiopsEncoder: {
                        "fit": {
                            ("pykhiops.core", "train_recoder"): {
                                1: "SpliceJunction",
                                3: "Class",
                                4: cls.output_dir,
                            }
                        },
                        "predict": {
                            ("pykhiops.core", "deploy_model"): {
                                1: "R_SpliceJunction",
                                2: "SpliceJunction.txt",
                                3: cls.output_dir,
                            }
                        },
                    },
                },
                "file_dataset": {
                    KhiopsRegressor: {
                        "fit": {
                            ("pykhiops.core", "train_predictor"): {
                                1: "SpliceJunction",
                                3: "Class",
                                4: cls.output_dir,
                            }
                        },
                        "predict": {
                            ("pykhiops.core", "deploy_model"): {
                                1: "SNB_SpliceJunction",
                                2: "sorted_SpliceJunction.txt",
                                3: cls.output_dir,
                            }
                        },
                    },
                    KhiopsClassifier: {
                        "fit": {
                            ("pykhiops.core", "train_predictor"): {
                                1: "SpliceJunction",
                                3: "Class",
                                4: cls.output_dir,
                            }
                        },
                        "predict": {
                            ("pykhiops.core", "deploy_model"): {
                                1: "SNB_SpliceJunction",
                                2: "sorted_SpliceJunction.txt",
                                3: cls.output_dir,
                            }
                        },
                    },
                    KhiopsEncoder: {
                        "fit": {
                            ("pykhiops.core", "train_recoder"): {
                                1: "SpliceJunction",
                                3: "Class",
                            }
                        },
                        "predict": {
                            ("pykhiops.core", "deploy_model"): {
                                1: "R_SpliceJunction",
                                2: "sorted_SpliceJunction.txt",
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
                    ("pykhiops.core", "train_predictor"): {
                        2: cls.assertPathHasSuffix,
                        4: cls.assertPathHasPrefix,
                    }
                },
                "predict": {
                    ("pykhiops.core", "deploy_model"): {
                        2: cls.assertPathHasSuffix,
                        3: cls.assertPathHasPrefix,
                    }
                },
            },
            KhiopsEncoder: {
                "fit": {
                    ("pykhiops.core", "train_recoder"): {
                        2: cls.assertPathHasSuffix,
                        4: cls.assertPathHasPrefix,
                    }
                },
                "predict": {
                    ("pykhiops.core", "deploy_model"): {
                        2: cls.assertPathHasSuffix,
                        3: cls.assertPathHasPrefix,
                    }
                },
            },
            KhiopsCoclustering: {
                "fit": {
                    ("pykhiops.core", "train_coclustering"): {
                        4: cls.assertPathHasPrefix
                    },
                    ("pykhiops.core", "prepare_coclustering_deployment"): {
                        2: cls.assertEqualPath,
                    },
                    ("pykhiops.core", "read_coclustering_results_file"): {
                        0: cls.assertEqualPath,
                    },
                },
                "predict": {
                    ("pykhiops.core", "deploy_model"): {3: cls.assertPathHasPrefix},
                    ("pykhiops.core", "extract_keys_from_data_table"): {
                        2: cls.assertPathHasSuffix,
                        3: cls.assertPathHasSuffix,
                    },
                },
            },
        }

        cls.expected_kwargs = {
            "not_applicable": {
                "dataframe": {
                    KhiopsCoclustering: {
                        "fit": {
                            ("pykhiops.core", "prepare_coclustering_deployment"): {
                                "build_cluster_variable": True,
                                "build_distance_variables": False,
                                "build_frequency_variables": False,
                                "max_part_numbers": None,
                            },
                            ("pykhiops.core", "read_coclustering_results_file"): {},
                            ("pykhiops.core", "build_multi_table_dictionary"): {
                                "overwrite_dictionary_file": True
                            },
                            ("pykhiops.core", "train_coclustering"): {
                                "log_file_path": os.path.join(
                                    cls.output_dir, "khiops_train_cc.log"
                                )
                            },
                        },
                        "predict": {
                            ("pykhiops.core", "deploy_model"): {
                                "detect_format": False,
                                "header_line": True,
                                "additional_data_tables": {
                                    "CC_Keys_main_table`CC_main_table"
                                },
                                "log_file_path": os.path.join(
                                    cls.output_dir, "khiops.log"
                                ),
                            },
                            ("pykhiops.core", "extract_keys_from_data_table"): {
                                "header_line": True,
                                "output_header_line": True,
                            },
                        },
                    },
                },
                "file_dataset": {
                    KhiopsCoclustering: {
                        "fit": {
                            ("pykhiops.core", "prepare_coclustering_deployment"): {
                                "build_cluster_variable": True,
                                "build_distance_variables": False,
                                "build_frequency_variables": False,
                                "max_part_numbers": None,
                            },
                            ("pykhiops.core", "read_coclustering_results_file"): {},
                            ("pykhiops.core", "build_multi_table_dictionary"): {
                                "overwrite_dictionary_file": True
                            },
                            ("pykhiops.core", "train_coclustering"): {
                                "log_file_path": os.path.join(
                                    cls.output_dir, "khiops_train_cc.log"
                                )
                            },
                        },
                        "predict": {
                            ("pykhiops.core", "deploy_model"): {
                                "detect_format": False,
                                "header_line": True,
                                "additional_data_tables": {
                                    "CC_Keys_SpliceJunctionDNA`CC_SpliceJunctionDNA"
                                },
                                "log_file_path": os.path.join(
                                    cls.output_dir, "khiops.log"
                                ),
                            },
                            ("pykhiops.core", "extract_keys_from_data_table"): {
                                "header_line": True,
                                "output_header_line": True,
                            },
                        },
                    },
                },
            },
            "monotable": {
                "dataframe": {
                    KhiopsPredictor: {
                        "fit": {
                            ("pykhiops.core", "train_predictor"): {
                                "field_separator": "\t",
                                "detect_format": False,
                                "header_line": True,
                                "additional_data_tables": {},
                            }
                        },
                        "predict": {
                            ("pykhiops.core", "deploy_model"): {
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
                            ("pykhiops.core", "train_recoder"): {
                                "field_separator": "\t",
                                "detect_format": False,
                                "header_line": True,
                                "additional_data_tables": {},
                                "keep_initial_categorical_variables": False,
                                "keep_initial_numerical_variables": False,
                                "categorical_recoding_method": "part Id",
                                "numerical_recoding_method": "part Id",
                            }
                        },
                        "predict": {
                            ("pykhiops.core", "deploy_model"): {
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
                "file_dataset": {
                    KhiopsPredictor: {
                        "fit": {
                            ("pykhiops.core", "train_predictor"): {
                                "field_separator": "\t",
                                "detect_format": False,
                                "header_line": True,
                                "additional_data_tables": {},
                            }
                        },
                        "predict": {
                            ("pykhiops.core", "deploy_model"): {
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
                            ("pykhiops.core", "train_recoder"): {
                                "field_separator": "\t",
                                "detect_format": False,
                                "header_line": True,
                                "additional_data_tables": {},
                                "keep_initial_categorical_variables": False,
                                "keep_initial_numerical_variables": False,
                                "categorical_recoding_method": "part Id",
                                "numerical_recoding_method": "part Id",
                            }
                        },
                        "predict": {
                            ("pykhiops.core", "deploy_model"): {
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
                "dataframe": {
                    KhiopsPredictor: {
                        "fit": {
                            ("pykhiops.core", "train_predictor"): {
                                "field_separator": "\t",
                                "detect_format": False,
                                "header_line": True,
                                "additional_data_tables": {
                                    "SpliceJunction`SpliceJunctionDNA"
                                },
                            }
                        },
                        "predict": {
                            ("pykhiops.core", "deploy_model"): {
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
                            ("pykhiops.core", "train_recoder"): {
                                "field_separator": "\t",
                                "detect_format": False,
                                "header_line": True,
                                "additional_data_tables": {
                                    "SpliceJunction`SpliceJunctionDNA"
                                },
                                "keep_initial_categorical_variables": False,
                                "keep_initial_numerical_variables": False,
                                "categorical_recoding_method": "part Id",
                                "numerical_recoding_method": "part Id",
                            }
                        },
                        "predict": {
                            ("pykhiops.core", "deploy_model"): {
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
                "file_dataset": {
                    KhiopsPredictor: {
                        "fit": {
                            ("pykhiops.core", "train_predictor"): {
                                "field_separator": "\t",
                                "detect_format": False,
                                "header_line": True,
                                "log_file_path": os.path.join(
                                    cls.output_dir, "khiops.log"
                                ),
                                "additional_data_tables": {
                                    "SpliceJunction`SpliceJunctionDNA"
                                },
                            }
                        },
                        "predict": {
                            ("pykhiops.core", "deploy_model"): {
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
                            ("pykhiops.core", "train_recoder"): {
                                "field_separator": "\t",
                                "detect_format": False,
                                "header_line": True,
                                "additional_data_tables": {
                                    "SpliceJunction`SpliceJunctionDNA"
                                },
                                "keep_initial_categorical_variables": False,
                                "keep_initial_numerical_variables": False,
                                "categorical_recoding_method": "part Id",
                                "numerical_recoding_method": "part Id",
                            }
                        },
                        "predict": {
                            ("pykhiops.core", "deploy_model"): {
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
                    ("pykhiops.core", "train_predictor"): {
                        "log_file_path": cls.assertEqualPath,
                        "additional_data_tables": (
                            cls.assertEqualAdditionalDataTableNames
                        ),
                    }
                },
                "predict": {
                    ("pykhiops.core", "deploy_model"): {
                        "log_file_path": cls.assertEqualPath,
                        "additional_data_tables": (
                            cls.assertEqualAdditionalDataTableNames
                        ),
                    }
                },
            },
            KhiopsEncoder: {
                "fit": {
                    ("pykhiops.core", "train_recoder"): {
                        "additional_data_tables": (
                            cls.assertEqualAdditionalDataTableNames
                        )
                    }
                },
                "predict": {
                    ("pykhiops.core", "deploy_model"): {
                        "log_file_path": cls.assertEqualPath,
                        "additional_data_tables": (
                            cls.assertEqualAdditionalDataTableNames
                        ),
                    }
                },
            },
            KhiopsCoclustering: {
                "fit": {
                    ("pykhiops.core", "train_coclustering"): {
                        "log_file_path": cls.assertEqualPath
                    }
                },
                "predict": {
                    ("pykhiops.core", "deploy_model"): {
                        "additional_data_tables": (
                            cls.assertEqualAdditionalDataTableNames
                        ),
                        "log_file_path": cls.assertEqualPath,
                    },
                    ("pykhiops.core", "extract_keys_from_data_table"): {},
                },
            },
        }

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary resources for this test case: output directory"""

        if os.path.isdir(cls.output_dir):
            shutil.rmtree(cls.output_dir)

    @classmethod
    def _output_dir_grabber_train(cls, *args):
        # grab output directory from train predictor / coclustering args
        return args[4]  # results_dir

    @classmethod
    def _output_dir_grabber_deploy(cls, *args):
        # grab output directory from deploy model args
        return os.path.dirname(args[3])  # output_data_table_path

    def _check_dictionary_domain(
        self,
        dictionary_domain,
        expected_n_dictionaries=None,
        expected_main_table_key=None,
        expected_main_dictionary_name=None,
        expected_additional_data_table_names=(),
    ):
        """Check assertions on dictionary domains"""
        self.assertIsInstance(dictionary_domain, pk.dictionary.DictionaryDomain)
        if expected_n_dictionaries is not None:
            self.assertEqual(
                len(dictionary_domain.dictionaries), expected_n_dictionaries
            )
        for dictionary in dictionary_domain.dictionaries:
            self.assertIsInstance(dictionary, pk.dictionary.Dictionary)
        if expected_main_dictionary_name is not None:
            self.assertEqual(
                dictionary_domain.dictionaries[0].name, expected_main_dictionary_name
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
                    dictionary_domain.dictionaries[i].name, additional_data_table_name
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
        (root_table_data, secondary_table_data) = cls.get_two_table_data(
            "SpliceJunction", "SpliceJunction", "SpliceJunctionDNA"
        )
        root_table_file_name_suffix = ""
        if transform_for_regression:
            root_table_data.replace({"Class": {"EI": 1, "IE": 2, "N": 3}}, inplace=True)
            root_table_file_name_suffix = "_R"
        (root_train_data, root_labels), (root_test_data, _) = cls.prepare_data(
            root_table_data, "Class"
        )
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
        root_table_data, secondary_table_data = cls.get_two_table_data(
            dataset_name, "SpliceJunction", "SpliceJunctionDNA"
        )
        if transform_for_regression:
            root_table_data.replace({"Class": {"EI": 1, "IE": 2, "N": 3}}, inplace=True)
        root_train_data, root_test_data = cls.prepare_data(root_table_data, "Class")
        secondary_train_data = cls.prepare_data(
            secondary_table_data, "SampleId", primary_table=root_train_data[0]
        )[0]
        secondary_test_data = cls.prepare_data(
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
        data = cls.get_monotable_data("Adult")
        (train_data, train_labels), (test_data, _) = cls.prepare_data(data, label)

        train_data_path = os.path.join(cls.output_dir, f"Adult_train_for_{label}.txt")
        test_data_path = os.path.join(cls.output_dir, f"Adult_test_for_{label}.txt")
        train_data[label] = train_labels
        train_data.to_csv(train_data_path, sep="\t", header=True, index=False)
        test_data.to_csv(test_data_path, sep="\t", header=True, index=False)
        train_dataset = {
            "main_table": "Adult",
            "tables": {"Adult": (train_data_path, "Label")},
            "format": ("\t", True),
        }
        test_dataset = {
            "main_table": "Adult",
            "tables": {"Adult": (test_data_path, "Label")},
            "format": ("\t", True),
        }
        return (train_dataset, test_dataset)

    @classmethod
    def _create_train_test_monotable_dataframe(cls, label):
        data = cls.get_monotable_data("Adult")
        (train_data, train_labels), (test_data, _) = cls.prepare_data(data, label)
        return (train_data, train_labels, test_data)

    def _retrieve_data(
        self,
        schema_type,
        source_type,
        estimation_process,
    ):
        return self.datasets[schema_type][source_type][estimation_process]

    def _define_resources(self, dataset, estimator_type, source_type, schema_type):
        resources = {}
        report_name = (
            "Coclustering.khcj"
            if estimator_type == KhiopsCoclustering
            else "AllReports.khj"
        )
        model_name = (
            "Coclustering" if estimator_type == KhiopsCoclustering else "Modeling"
        )

        report = os.path.join(
            self.get_resources_dir(),
            "sklearn",
            "results",
            "ref_json_reports",
            dataset,
            estimator_type.__name__,
            source_type,
            report_name,
        )

        model_kdic = os.path.join(
            self.get_resources_dir(),
            "sklearn",
            "results",
            "ref_models",
            dataset,
            estimator_type.__name__,
            source_type,
            f"{model_name}.kdic",
        )

        tmp_model_kdic = (
            os.path.join(
                self.get_resources_dir(),
                "sklearn",
                "results",
                "ref_models",
                dataset,
                estimator_type.__name__,
                source_type,
                "tmp_cc_deploy_model.kdic",
            )
            if estimator_type == KhiopsCoclustering
            else None
        )

        model_json = os.path.join(  # XXX refactor with previous
            self.get_resources_dir(),
            "sklearn",
            "results",
            "ref_models",
            dataset,
            estimator_type.__name__,
            source_type,
            f"{model_name}.kdicj",
        )

        predictions = os.path.join(
            self.get_resources_dir(),
            "sklearn",
            "results",
            "ref_predictions",
            dataset,
            estimator_type.__name__,
            source_type,
            "transformed.txt",
        )
        keys_main_table_file = (
            "main_table"
            if source_type == "dataframe"
            else f"{self.dataset_of_schema_type[schema_type]}DNA"
        )
        keys_main_table = (
            os.path.join(
                self.get_resources_dir(),
                "sklearn",
                "results",
                "ref_predictions",
                dataset,
                estimator_type.__name__,
                source_type,
                f"keys_{keys_main_table_file}.txt",
            )
            if estimator_type == KhiopsCoclustering
            else None
        )
        raw_keys_main_table = (
            os.path.join(
                self.get_resources_dir(),
                "sklearn",
                "results",
                "ref_predictions",
                dataset,
                estimator_type.__name__,
                source_type,
                f"raw_keys_{keys_main_table_file}.txt",
            )
            if estimator_type == KhiopsCoclustering
            else None
        )

        resources["fit"] = [
            res
            for res in (report, model_kdic, tmp_model_kdic, model_json)
            if res is not None
        ]

        resources["predict"] = [
            res
            for res in (predictions, keys_main_table, raw_keys_main_table)
            if res is not None
        ]

        return resources

    def _test_template(
        self,
        estimator_type,
        estimator_method,
        schema_type,
        source_type,
        custom_kwargs=None,
    ):
        """Test template

        The template is parameterized by:
        - estimator type: class of the estimator
        - estimator method: "fit" or "predict"
        - schema type: "monotable" or "multitable"
        - source type: "dataframe" or "file_dataset".

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
        # * retrieve data:
        #   - training data
        #   - testing data (if needed)
        # * define resources for mocks
        # * define mocks
        # * enter mock context manager
        # * wrap relevant methods with parameter trace
        # * run methods
        # * retrieve parameters
        # * check dictionary domain
        # * check args:
        #   - define special checkers
        #   - perform check
        # * check kwargs:
        #   - define special checkers
        #   - perform check
        #
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
                # file_dataset
                dataset = copy.deepcopy(data)
                X_train_data = dataset["train"]
                del X_train_data["tables"]["SpliceJunction"]  # XXX leaky
                y_train_data = None
                X_test_data = dataset["test"]
                del X_test_data["tables"]["SpliceJunction"]  # XXX leaky
        else:
            data = self._retrieve_data(
                schema_type=schema_type,
                source_type=source_type,
                estimation_process=estimator_type,
            )
            if source_type == "dataframe":
                X_train_data = data["X_train"]
                y_train_data = data["y_train"]
                X_test_data = data["X_test"]
            else:
                # file_dataset
                X_train_data = data["train"]
                if schema_type == "multitable":
                    y_train_data = "Class"
                else:
                    # monotable
                    if estimator_type == KhiopsRegressor:
                        y_train_data = "age"
                    else:
                        y_train_data = "class"
                X_test_data = data["test"]
        dataset = self.dataset_of_schema_type[schema_type]

        resources = self._define_resources(
            dataset, estimator_type, source_type, schema_type
        )

        estimator_type_key = (
            KhiopsPredictor
            if estimator_type in (KhiopsClassifier, KhiopsRegressor)
            else estimator_type
        )

        mocks = self.mocks_table[estimator_type_key]["fit"](resources["fit"])

        # if in a predict test, then add relevant predict mocks
        if estimator_method == "predict":
            mocks += self.mocks_table[estimator_type_key][estimator_method](
                resources[estimator_method]
            )
        # enter the mocks as context managers, to contain them:
        with contextlib.ExitStack() as stack:
            for mock in mocks:
                stack.enter_context(mock)
            # trace parameters for wrapped functions
            parameter_trace = self.create_parameter_trace()
            for (module, function) in self.wrapped_functions[estimator_type_key][
                estimator_method
            ]:
                self.wrap_with_parameter_trace(module, function, parameter_trace)
            # train estimator
            estimator = PyKhiopsSklearnTestsHelper.fit_helper(
                estimator_type,
                data=(X_train_data, y_train_data),
                pickled=False,
                fit_kwargs=custom_kwargs["fit"] if custom_kwargs is not None else None,
                output_dir=self.output_dir,
            )

            # if in a predict test, then call the predict method on the estimator
            if estimator_method == "predict":
                PyKhiopsSklearnTestsHelper.predict_helper(
                    data=(estimator, (X_test_data, None)),
                    kind="simple",
                    pickled=False,
                )
            # loop over parameter trace, to get the relevant module
            # and functions associated to each module
            for module, functions in parameter_trace.items():
                # for each function and its parameters, test their conformance
                # to the expectected parameter values
                for function, parameters in functions.items():
                    # get the actual args and kwargs of `function`
                    args_kwargs = parameters[0]
                    args = args_kwargs["args"]
                    kwargs = args_kwargs["kwargs"]
                    # get dictionary domain-specific kwargs and check them
                    dictionary_domain_kwargs = self.dictionary_domain_kwargs[
                        schema_type
                    ][source_type][estimator_type_key].get(function)
                    if dictionary_domain_kwargs is not None:
                        self._check_dictionary_domain(
                            dictionary_domain=args[0],
                            **dictionary_domain_kwargs,
                        )
                    # check the args of `function`, some of which have special
                    # checkers
                    expected_args = self.expected_args[schema_type][source_type][
                        estimator_type
                    ][estimator_method][(module, function)]
                    special_arg_checkers = self.special_arg_checkers[
                        estimator_type_key
                    ][estimator_method].get((module, function))
                    self._check_args(
                        args,
                        expected_args_with_pos=expected_args,
                        special_checkers=special_arg_checkers,
                    )
                    # check the kwargs of `function`, some of which have
                    # special checkers
                    expected_kwargs = self.expected_kwargs[schema_type][source_type][
                        estimator_type_key
                    ][estimator_method][(module, function)]
                    special_kwarg_checkers = self.special_kwarg_checkers[
                        estimator_type_key
                    ][estimator_method].get((module, function))
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
        )

    def test_parameter_transfer_classifier_fit_from_monotable_file_dataset(self):
        """Test parameter transfer from monotable file dataset fit to core API"""
        self._test_template(
            estimator_type=KhiopsClassifier,
            estimator_method="fit",
            schema_type="monotable",
            source_type="file_dataset",
        )

    def test_parameter_transfer_classifier_fit_from_multitable_dataframe(self):
        """Test parameter transfer from multitable dataframe fit to core API"""
        self._test_template(
            estimator_type=KhiopsClassifier,
            estimator_method="fit",
            schema_type="multitable",
            source_type="dataframe",
        )

    def test_parameter_transfer_classifier_fit_from_multitable_file_dataset(self):
        """Test parameter transfer from file dataset fit to core API"""
        self._test_template(
            estimator_type=KhiopsClassifier,
            estimator_method="fit",
            schema_type="multitable",
            source_type="file_dataset",
        )

    def test_parameter_transfer_classifier_predict_from_monotable_dataframe(self):
        """Test parameter transfer from monotable dataframe predict to core API"""
        self._test_template(
            estimator_type=KhiopsClassifier,
            estimator_method="predict",
            schema_type="monotable",
            source_type="dataframe",
        )

    def test_parameter_transfer_classifier_predict_from_monotable_file_dataset(self):
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

    def test_parameter_transfer_classifier_predict_from_multitable_file_dataset(self):
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
        )

    def test_parameter_transfer_encoder_fit_from_monotable_file_dataset(self):
        """Test parameter transfer from monotable file dataset fit to core API"""
        self._test_template(
            estimator_type=KhiopsEncoder,
            estimator_method="fit",
            schema_type="monotable",
            source_type="file_dataset",
        )

    def test_parameter_transfer_encoder_fit_from_multitable_dataframe(self):
        """Test parameter transfer from multitable dataframe fit to core API"""
        self._test_template(
            estimator_type=KhiopsEncoder,
            estimator_method="fit",
            schema_type="multitable",
            source_type="dataframe",
        )

    def test_parameter_transfer_encoder_fit_from_multitable_file_dataset(self):
        """Test parameter transfer from multitable file dataset fit to core API"""
        self._test_template(
            estimator_type=KhiopsEncoder,
            estimator_method="fit",
            schema_type="multitable",
            source_type="file_dataset",
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
        )

    def test_parameter_transfer_regressor_fit_from_monotable_file_dataset(self):
        """Test parameter transfer from monotable file dataset fit to core API"""
        self._test_template(
            estimator_type=KhiopsRegressor,
            estimator_method="fit",
            schema_type="monotable",
            source_type="file_dataset",
        )

    def test_parameter_transfer_regressor_fit_from_multitable_dataframe(self):
        """Test parameter transfer from multitable dataframe fit to core API"""
        self._test_template(
            estimator_type=KhiopsRegressor,
            estimator_method="fit",
            schema_type="multitable",
            source_type="dataframe",
        )

    def test_parameter_transfer_regressor_fit_from_multitable_file_dataset(self):
        """Test parameter transfer from file dataset fit to core API"""
        self._test_template(
            estimator_type=KhiopsRegressor,
            estimator_method="fit",
            schema_type="multitable",
            source_type="file_dataset",
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

    def test_parameter_transfer_regressor_predict_from_multitable_file_dataset(self):
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
