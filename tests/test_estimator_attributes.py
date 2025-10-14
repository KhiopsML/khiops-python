######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Test consistency of the estimator's attributes with the output reports"""
import unittest
import warnings
from os import path

import pandas as pd

from khiops import core as kh
from khiops.sklearn.estimators import KhiopsClassifier, KhiopsEncoder, KhiopsRegressor

# Disable PEP8 variable names because of scikit-learn X,y conventions
# To capture invalid-names other than X,y run:
#   pylint --disable=all --enable=invalid-names estimators.py
# pylint: disable=invalid-name


class EstimatorAttributesTests(unittest.TestCase):
    """Test consistency of the estimator's attributes with Khiops's output reports

    The following tests allow to verify that:
    - The values of each estimator's attributes are consistent with the the reports
      (PreparationReport.xls and ModelingReport.xls) produced by Khiops post training.
    - The attributes are tested for all supervised estimators: KhiopsClassifier,
     KhiopsRegressor and KhiopsEncoder.
    - Two datasets are used for the tests:
        - Adult (mono-table)
        - Accidents (multitable).
    """

    def _create_multitable_input(self, size=None):
        # Load `Accidents` into dataframes
        accidents_dataset_path = path.join(kh.get_samples_dir(), "Accidents")
        accidents_df = pd.read_csv(
            path.join(accidents_dataset_path, "Accidents.txt"), sep="\t"
        )
        users_df = pd.read_csv(path.join(accidents_dataset_path, "Users.txt"), sep="\t")
        vehicles_df = pd.read_csv(
            path.join(accidents_dataset_path, "Vehicles.txt"), sep="\t"
        )
        places_df = pd.read_csv(
            path.join(accidents_dataset_path, "Places.txt"), sep="\t", low_memory=False
        )

        # Set the sample size
        if size is None:
            size = len(accidents_df)

        # Create the multi-table dataset spec
        X = {
            "main_table": (
                accidents_df.drop("Gravity", axis=1)[:size],
                ["AccidentId"],
            ),
            "additional_data_tables": {
                "Vehicles": (vehicles_df, ["AccidentId", "VehicleId"]),
                "Vehicles/Users": (users_df, ["AccidentId", "VehicleId"]),
                "Places": (places_df, ["AccidentId"], True),
            },
        }
        y = accidents_df["Gravity"][:size]

        return X, y

    def assert_attribute_values_ok(self, model, X, y):
        # Special checks for KhiopsClassifier
        if isinstance(model, KhiopsClassifier):
            self.assertEqual(model.classes_.tolist(), sorted(y.unique()))
            self.assertEqual(model.n_classes_, len(y.unique()))
            self.assertEqual(model.n_features_in_, len(X.columns))

    def test_classifier_attributes_monotable(self):
        """Test consistency of KhiopsClassifier's attributes with the output reports

        - This test verifies that the values of a trained KhiopsClassifier, on a
         a monotable dataset (Adult), are consistent with the reports produced by Khiops
         post training.
        """
        adult_dataset_path = path.join(kh.get_samples_dir(), "Adult", "Adult.txt")
        adult_df = pd.read_csv(adult_dataset_path, sep="\t")
        X = adult_df.drop("class", axis=1)
        y = adult_df["class"]
        khc_adult = KhiopsClassifier()
        khc_adult.fit(X, y)

        self.assert_attribute_values_ok(khc_adult, X, y)
        self.assertFalse(khc_adult.is_multitable_model_)

    def test_classifier_attributes_multitable(self):
        """Test consistency of KhiopsClassifier's attributes with the output reports

        - This test verifies that the values of a trained KhiopsClassifier, on a
         a multitable dataset (Accidents), are consistent with the reports produced
         by Khiops post training.
        """
        X, y = self._create_multitable_input()
        khc_accidents = KhiopsClassifier(n_trees=0, n_text_features=0, n_pairs=10)
        khc_accidents.fit(X, y)
        self.assert_attribute_values_ok(khc_accidents, X["main_table"][0], y)
        self.assertTrue(khc_accidents.is_multitable_model_)

    def test_regressor_attributes_monotable(self):
        """Test consistency of KhiopsRegressor's attributes with the output reports

        - This test verifies that the values of a trained KhiopsRegressor, on a
         a monotable dataset (Adult), are consistent with the reports produced by Khiops
         post training.
        """
        adult_dataset_path = path.join(kh.get_samples_dir(), "Adult", "Adult.txt")
        adult_df = pd.read_csv(adult_dataset_path, sep="\t").sample(750)
        X = adult_df.drop("age", axis=1)
        y = adult_df["age"]
        khr_adult = KhiopsRegressor(n_trees=0, n_text_features=0)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore",
                category=UserWarning,
                message="Khiops ended correctly but there were minor issues",
            )
            khr_adult.fit(X, y)

        self.assert_attribute_values_ok(khr_adult, X, None)
        self.assertFalse(khr_adult.is_multitable_model_)

    def test_regressor_attributes_multitable(self):
        """Test consistency of KhiopsRegressor's attributes with the output reports

        - This test verifies that the values of a trained KhiopsRegressor, on a
         a multitable dataset (Accidents), are consistent with the reports produced
         by Khiops post training.
        """
        X, _ = self._create_multitable_input(750)
        y = X["main_table"][0]["Commune"]
        X["main_table"][0].drop("Commune", axis=1, inplace=True)
        khr_accidents = KhiopsRegressor(n_trees=0, n_text_features=0)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore",
                category=UserWarning,
                message="Khiops ended correctly but there were minor issues",
            )
            khr_accidents.fit(X, y)

        self.assert_attribute_values_ok(khr_accidents, X["main_table"][0], None)
        self.assertTrue(khr_accidents.is_multitable_model_)

    def test_encoder_attributes_monotable(self):
        """Test consistency of KhiopsEncoder's attributes with the output reports

        - This test verifies that the values of a trained KhiopsEncoder, on a
         a monotable dataset (Adult), are consistent with the reports produced
         by Khiops post training.
        """
        adult_dataset_path = path.join(kh.get_samples_dir(), "Adult", "Adult.txt")
        adult_df = pd.read_csv(adult_dataset_path, sep="\t")
        X = adult_df.drop("class", axis=1)
        y = adult_df["class"]
        khe_adult = KhiopsEncoder()
        khe_adult.fit(X, y)

        self.assert_attribute_values_ok(khe_adult, X, None)
        self.assertFalse(khe_adult.is_multitable_model_)

    def test_encoder_attributes_multitable(self):
        """Test consistency of KhiopsEncoder's attributes with the output reports

        - This test verifies that the values of a trained KhiopsEncoder, on a
         a multitable dataset (Accidents), are consistent with the reports produced
         by Khiops post training.
        """
        X, y = self._create_multitable_input()
        khe_accidents = KhiopsEncoder(n_trees=5, n_text_features=5000)
        khe_accidents.fit(X, y)

        self.assert_attribute_values_ok(khe_accidents, X, None)
        self.assertTrue(khe_accidents.is_multitable_model_)
