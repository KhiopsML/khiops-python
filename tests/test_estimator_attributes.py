######################################################################################
# Copyright (c) 2024 Orange. All rights reserved.                                    #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Test consistency of the estimator's attributes with the output reports"""
import unittest
from os import path

import numpy as np
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

    def are_lists_equal(self, list1, list2):
        return sorted(list1) == sorted(list2)

    def check_model_attribute_values(self, model, preparation_report, modeling_report):

        self.assertEqual(
            model.n_features_evaluated_, preparation_report.evaluated_variable_number
        )
        self.assertTrue(
            self.are_lists_equal(
                model.feature_evaluated_names_.tolist(),
                preparation_report.get_variable_names(),
            )
        )
        feature_evaluated_importances_report = [
            [preparation_report.get_variable_statistics(var).level]
            for var in preparation_report.get_variable_names()
        ]
        self.assertTrue(
            self.are_lists_equal(
                model.feature_evaluated_importances_.tolist(),
                feature_evaluated_importances_report,
            )
        )
        if not isinstance(model, KhiopsEncoder):
            feature_used_names = np.array(
                [
                    [var.name]
                    for var in modeling_report.get_snb_predictor().selected_variables
                ]
            )
            self.assertTrue(
                self.are_lists_equal(
                    model.feature_used_names_.tolist(), feature_used_names.tolist()
                )
            )
            feature_used_importances_report = [
                [var.weight, var.importance, var.level]
                for var in modeling_report.get_snb_predictor().selected_variables
            ]
            self.assertTrue(
                self.are_lists_equal(
                    model.feature_used_importances_.tolist(),
                    feature_used_importances_report,
                )
            )
            self.assertEqual(
                model.n_feature_used_, len(feature_used_importances_report)
            )
        self.assertEqual(model.is_fitted_, True)

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

        self.assertTrue(
            self.are_lists_equal(khc_adult.classes_.tolist(), y.unique().tolist())
        )
        self.assertEqual(khc_adult.n_classes_, len(y.unique()))
        self.assertEqual(khc_adult.n_features_in_, len(X.columns))
        preparation_report = khc_adult.model_report_.preparation_report
        modeling_report = khc_adult.model_report_.modeling_report
        self.check_model_attribute_values(
            khc_adult, preparation_report, modeling_report
        )
        self.assertEqual(khc_adult.is_multitable_model_, False)

    def test_classifier_attributes_multitable(self):
        """Test consistency of KhiopsClassifier's attributes with the output reports

        - This test verifies that the values of a trained KhiopsClassifier, on a
         a multitable dataset (Accidents), are consistent with the reports produced
         by Khiops post training.
        """
        accidents_dataset_path = path.join(kh.get_samples_dir(), "Accidents")
        accidents_df = pd.read_csv(
            path.join(accidents_dataset_path, "Accidents.txt"),
            sep="\t",
            encoding="latin1",
        )
        users_df = pd.read_csv(
            path.join(accidents_dataset_path, "Users.txt"), sep="\t", encoding="latin1"
        )
        vehicles_df = pd.read_csv(
            path.join(accidents_dataset_path, "Vehicles.txt"),
            sep="\t",
            encoding="latin1",
        )
        places_df = pd.read_csv(
            path.join(accidents_dataset_path, "Places.txt"), sep="\t", encoding="latin1"
        )

        X = {
            "main_table": "Accidents",
            "tables": {
                "Accidents": (accidents_df, "AccidentId"),
                "Vehicles": (vehicles_df, ["AccidentId", "VehicleId"]),
                "Users": (
                    users_df.drop("Gravity", axis=1),
                    ["AccidentId", "VehicleId"],
                ),
                "Places": (places_df, ["AccidentId"]),
            },
            "relations": [
                ("Accidents", "Vehicles"),
                ("Vehicles", "Users"),
                ("Accidents", "Places", True),
            ],
        }

        y = pd.read_csv(
            path.join(kh.get_samples_dir(), "AccidentsSummary", "Accidents.txt"),
            sep="\t",
            encoding="latin1",
        )["Gravity"]

        khc_accidents = KhiopsClassifier(n_trees=0)
        khc_accidents.fit(X, y)

        self.assertTrue(
            self.are_lists_equal(khc_accidents.classes_.tolist(), y.unique().tolist())
        )
        self.assertEqual(khc_accidents.n_classes_, len(y.unique()))
        self.assertEqual(khc_accidents.n_features_in_, len(accidents_df.columns))

        preparation_report = khc_accidents.model_report_.preparation_report
        modeling_report = khc_accidents.model_report_.modeling_report
        self.check_model_attribute_values(
            khc_accidents, preparation_report, modeling_report
        )
        self.assertEqual(khc_accidents.is_multitable_model_, True)

    def test_regressor_attributes_monotable(self):
        """Test consistency of KhiopsRegressor's attributes with the output reports

        - This test verifies that the values of a trained KhiopsRegressor, on a
         a monotable dataset (Adult), are consistent with the reports produced by Khiops
         post training.
        """
        adult_dataset_path = path.join(kh.get_samples_dir(), "Adult", "Adult.txt")
        adult_df = pd.read_csv(adult_dataset_path, sep="\t").sample(1000)
        X = adult_df.drop("age", axis=1)
        y = adult_df["age"]
        khr_adult = KhiopsRegressor()
        khr_adult.fit(X, y)

        self.assertEqual(khr_adult.n_features_in_, len(X.columns))

        preparation_report = khr_adult.model_report_.preparation_report
        modeling_report = khr_adult.model_report_.modeling_report
        self.check_model_attribute_values(
            khr_adult, preparation_report, modeling_report
        )
        self.assertEqual(khr_adult.is_multitable_model_, False)

    def test_regressor_attributes_multitable(self):
        """Test consistency of KhiopsRegressor's attributes with the output reports

        - This test verifies that the values of a trained KhiopsRegressor, on a
         a multitable dataset (Accidents), are consistent with the reports produced
         by Khiops post training.
        """
        accidents_dataset_path = path.join(kh.get_samples_dir(), "Accidents")
        accidents_df = pd.read_csv(
            path.join(accidents_dataset_path, "Accidents.txt"),
            sep="\t",
            encoding="latin1",
        ).sample(1000)
        users_df = pd.read_csv(
            path.join(accidents_dataset_path, "Users.txt"), sep="\t", encoding="latin1"
        )
        vehicles_df = pd.read_csv(
            path.join(accidents_dataset_path, "Vehicles.txt"),
            sep="\t",
            encoding="latin1",
        )
        places_df = pd.read_csv(
            path.join(accidents_dataset_path, "Places.txt"), sep="\t", encoding="latin1"
        )

        X = {
            "main_table": "Accidents",
            "tables": {
                "Accidents": (accidents_df.drop("Commune", axis=1), "AccidentId"),
                "Vehicles": (vehicles_df, ["AccidentId", "VehicleId"]),
                "Users": (
                    users_df.drop("Gravity", axis=1),
                    ["AccidentId", "VehicleId"],
                ),
                "Places": (places_df, ["AccidentId"]),
            },
            "relations": [
                ("Accidents", "Vehicles"),
                ("Vehicles", "Users"),
                ("Accidents", "Places", True),
            ],
        }

        y = accidents_df["Commune"]

        khr_accidents = KhiopsRegressor(n_trees=0)
        khr_accidents.fit(X, y)

        self.assertEqual(
            khr_accidents.n_features_in_, len(X["tables"]["Accidents"][0].columns)
        )

        preparation_report = khr_accidents.model_report_.preparation_report
        modeling_report = khr_accidents.model_report_.modeling_report
        self.check_model_attribute_values(
            khr_accidents, preparation_report, modeling_report
        )
        self.assertEqual(khr_accidents.is_multitable_model_, True)

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

        self.assertEqual(khe_adult.n_features_in_, len(X.columns))
        preparation_report = khe_adult.model_report_.preparation_report
        modeling_report = khe_adult.model_report_.modeling_report
        self.check_model_attribute_values(
            khe_adult, preparation_report, modeling_report
        )
        self.assertEqual(khe_adult.is_multitable_model_, False)

    def test_encoder_attributes_multitable(self):
        """Test consistency of KhiopsEncoder's attributes with the output reports

        - This test verifies that the values of a trained KhiopsEncoder, on a
         a multitable dataset (Accidents), are consistent with the reports produced
         by Khiops post training.
        """
        accidents_dataset_path = path.join(kh.get_samples_dir(), "Accidents")
        accidents_df = pd.read_csv(
            path.join(accidents_dataset_path, "Accidents.txt"),
            sep="\t",
            encoding="latin1",
        )
        users_df = pd.read_csv(
            path.join(accidents_dataset_path, "Users.txt"), sep="\t", encoding="latin1"
        )
        vehicles_df = pd.read_csv(
            path.join(accidents_dataset_path, "Vehicles.txt"),
            sep="\t",
            encoding="latin1",
        )
        places_df = pd.read_csv(
            path.join(accidents_dataset_path, "Places.txt"), sep="\t", encoding="latin1"
        )

        X = {
            "main_table": "Accidents",
            "tables": {
                "Accidents": (accidents_df, "AccidentId"),
                "Vehicles": (vehicles_df, ["AccidentId", "VehicleId"]),
                "Users": (
                    users_df.drop("Gravity", axis=1),
                    ["AccidentId", "VehicleId"],
                ),
                "Places": (places_df, ["AccidentId"]),
            },
            "relations": [
                ("Accidents", "Vehicles"),
                ("Vehicles", "Users"),
                ("Accidents", "Places", True),
            ],
        }

        y = pd.read_csv(
            path.join(kh.get_samples_dir(), "AccidentsSummary", "Accidents.txt"),
            sep="\t",
            encoding="latin1",
        )["Gravity"]

        khe_accidents = KhiopsClassifier(n_trees=0)
        khe_accidents.fit(X, y)

        self.assertEqual(khe_accidents.n_features_in_, len(accidents_df.columns))

        preparation_report = khe_accidents.model_report_.preparation_report
        modeling_report = khe_accidents.model_report_.modeling_report
        self.check_model_attribute_values(
            khe_accidents, preparation_report, modeling_report
        )
        self.assertEqual(khe_accidents.is_multitable_model_, True)
