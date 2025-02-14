######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Tests for checking the output types of predictors"""
import os
import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from sklearn import datasets
from sklearn.utils.validation import column_or_1d

from khiops import core as kh
from khiops.sklearn.estimators import KhiopsClassifier, KhiopsRegressor

# Disable PEP8 variable names because of scikit-learn X,y conventions
# To capture invalid-names other than X,y run:
#   pylint --disable=all --enable=invalid-names estimators.py
# pylint: disable=invalid-name


def create_iris():
    """Returns a mono table iris dataset"""
    X_iris_array, y_iris_array = datasets.load_iris(return_X_y=True)
    X_iris_df = pd.DataFrame(
        X_iris_array, columns=["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]
    )
    y_iris_series = pd.Series(y_iris_array, name="Class")
    return X_iris_df, y_iris_series


def create_iris_mt():
    """Returns a multitable table iris dataset"""
    X_iris_df, y_iris_series = create_iris()
    X_iris_df["Id"] = X_iris_df.index
    X_iris_sec_df = X_iris_df.melt(
        id_vars=["Id"], var_name="Measurement", value_name="Value"
    )
    X_iris_df = X_iris_df.drop(
        ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"], axis=1
    )
    return X_iris_df, X_iris_sec_df, y_iris_series


class KhiopsSklearnOutputTypes(unittest.TestCase):
    """Tests for checking the output types of predictors"""

    _env_khiops_proc_number = None

    @classmethod
    def setUpClass(cls):
        # Set the number of processes to 1: Lots of test on small datasets
        kh.get_runner()  # Just to activate the lazy initialization
        cls._env_khiops_proc_number = os.environ["KHIOPS_PROC_NUMBER"]
        os.environ["KHIOPS_PROC_NUMBER"] = "1"

    @classmethod
    def tearDownClass(cls):
        # Restore the original number of processes
        os.environ["KHIOPS_PROC_NUMBER"] = cls._env_khiops_proc_number

    def _replace(self, array, replacement_dict):
        return np.array([replacement_dict[value] for value in array])

    def test_classifier_output_types(self):
        """Test the KhiopsClassifier output types and classes of predict* methods"""
        # Create the references for the combinations of mono/multi-table and
        # binary/multiclass
        X, y = create_iris()
        raw_X_main_mt, raw_X_sec_mt, _ = create_iris_mt()
        X_mt = {
            "main_table": "iris_main",
            "tables": {
                "iris_main": (raw_X_main_mt, "Id"),
                "iris_sec": (raw_X_sec_mt, "Id"),
            },
            "relations": [("iris_main", "iris_sec")],
        }
        khc = KhiopsClassifier(n_trees=0)
        khc.fit(X, y)
        y_pred = khc.predict(X)
        y_bin = y.replace({0: 0, 1: 0, 2: 1})
        khc.fit(X, y_bin)
        y_bin_pred = khc.predict(X)
        khc.fit(X_mt, y)
        khc.export_report_file("report.khj")
        y_mt_pred = khc.predict(X_mt)
        khc.fit(X_mt, y_bin)
        y_mt_bin_pred = khc.predict(X_mt)

        # Create the fixtures
        fixtures = {
            "ys": {
                "int": y,
                "int binary": y_bin,
                "string": self._replace(y, {0: "se", 1: "vi", 2: "ve"}),
                "string binary": self._replace(y_bin, {0: "vi_or_se", 1: "ve"}),
                "int as string": self._replace(y, {0: "8", 1: "9", 2: "10"}),
                "int as string binary": self._replace(y_bin, {0: "89", 1: "10"}),
                "cat int": pd.Series(y).astype("category"),
                "cat string": pd.Series(
                    self._replace(y, {0: "se", 1: "vi", 2: "ve"})
                ).astype("category"),
            },
            "y_type_check": {
                "int": pd.api.types.is_integer_dtype,
                "int binary": pd.api.types.is_integer_dtype,
                "string": pd.api.types.is_string_dtype,
                "string binary": pd.api.types.is_string_dtype,
                "int as string": pd.api.types.is_string_dtype,
                "int as string binary": pd.api.types.is_string_dtype,
                "cat int": pd.api.types.is_integer_dtype,
                "cat string": pd.api.types.is_string_dtype,
            },
            "expected_classes": {
                "int": column_or_1d([0, 1, 2]),
                "int binary": column_or_1d([0, 1]),
                "string": column_or_1d(["se", "ve", "vi"]),
                "string binary": column_or_1d(["ve", "vi_or_se"]),
                "int as string": column_or_1d(["10", "8", "9"]),
                "int as string binary": column_or_1d(["10", "89"]),
                "cat int": column_or_1d([0, 1, 2]),
                "cat string": column_or_1d(["se", "ve", "vi"]),
            },
            "expected_y_preds": {
                "mono": {
                    "int": y_pred,
                    "int binary": y_bin_pred,
                    "string": self._replace(y_pred, {0: "se", 1: "vi", 2: "ve"}),
                    "string binary": self._replace(
                        y_bin_pred, {0: "vi_or_se", 1: "ve"}
                    ),
                    "int as string": self._replace(y_pred, {0: "8", 1: "9", 2: "10"}),
                    "int as string binary": self._replace(
                        y_bin_pred, {0: "89", 1: "10"}
                    ),
                    "cat int": y_pred,
                    "cat string": self._replace(y_pred, {0: "se", 1: "vi", 2: "ve"}),
                },
                "multi": {
                    "int": y_mt_pred,
                    "int binary": y_mt_bin_pred,
                    "string": self._replace(y_mt_pred, {0: "se", 1: "vi", 2: "ve"}),
                    "string binary": self._replace(
                        y_mt_bin_pred, {0: "vi_or_se", 1: "ve"}
                    ),
                    "int as string": self._replace(
                        y_mt_pred, {0: "8", 1: "9", 2: "10"}
                    ),
                    "int as string binary": self._replace(
                        y_mt_bin_pred, {0: "89", 1: "10"}
                    ),
                    "cat int": y_mt_pred,
                    "cat string": self._replace(y_mt_pred, {0: "se", 1: "vi", 2: "ve"}),
                },
            },
            "Xs": {
                "mono": X,
                "multi": X_mt,
            },
        }

        # Test for each fixture configuration
        for y_type, y in fixtures["ys"].items():
            y_type_check = fixtures["y_type_check"][y_type]
            expected_classes = fixtures["expected_classes"][y_type]
            for dataset_type, X in fixtures["Xs"].items():
                with self.subTest(
                    y_type=y_type,
                    dataset_type=dataset_type,
                    estimator=KhiopsClassifier.__name__,
                ):
                    # Train the classifier
                    khc = KhiopsClassifier(n_trees=0)
                    khc.fit(X, y)

                    # Check the expected classes
                    assert_array_equal(khc.classes_, expected_classes)

                    # Check the return type of predict
                    y_pred = khc.predict(X)
                    self.assertTrue(
                        y_type_check(y_pred),
                        f"'{y_type_check.__name__}' was False for "
                        f"dtype '{y_pred.dtype}'.",
                    )

                    # Check the predictions match
                    expected_y_pred = fixtures["expected_y_preds"][dataset_type][y_type]
                    assert_array_equal(y_pred, expected_y_pred)

                    # Check the dimensions of predict_proba
                    y_probas = khc.predict_proba(X)
                    self.assertEqual(len(y_probas.shape), 2)
                    self.assertEqual(y_probas.shape[1], len(khc.classes_))

    def test_regression_output_types(self):
        """Test the KhiopsRegressor output types of the predict method"""
        X, y = create_iris()
        X_mt, X_sec_mt, _ = create_iris_mt()

        fixtures = {
            "ys": {"int": y, "float": y.astype(float)},
            "Xs": {
                "mono": X,
                "multi": {
                    "main_table": "iris_main",
                    "tables": {
                        "iris_main": (X_mt, "Id"),
                        "iris_sec": (X_sec_mt, "Id"),
                    },
                },
            },
        }

        for y_type, y in fixtures["ys"].items():
            for dataset_type, X in fixtures["Xs"].items():
                with self.subTest(
                    y_type=y_type,
                    dataset_type=dataset_type,
                    estimator=KhiopsClassifier.__name__,
                ):
                    # Train the classifier
                    khr = KhiopsRegressor()
                    khr.fit(X, y)

                    # Check the return type of predict
                    y_pred = khr.predict(X)
                    self.assertTrue(
                        pd.api.types.is_float_dtype(y_pred),
                        f"Invalid predict return type {y_pred.dtype}.",
                    )
