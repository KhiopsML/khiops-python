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

import os
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split

import pykhiops.core as pk
from pykhiops.core.common import is_iterable
from pykhiops.sklearn.estimators import KhiopsEncoder, KhiopsEstimator


class PyKhiopsSklearnTestsHelper:
    """Helper functions for the actual tests.
    They need to be static so that they can be serialized for multiprocessing.
    """

    def get_two_table_data(self, dataset_path, root_table_name, secondary_table_name):
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

    def get_monotable_data(self, dataset_name):
        """Read monotable data from CSV sample dataset"""
        samples_dir = pk.get_runner().samples_dir
        return pd.read_csv(
            os.path.join(samples_dir, dataset_name, f"{dataset_name}.txt"), sep="\t"
        )

    def prepare_data(self, data, target_variable, primary_table=None):
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
        estimator_class, observations, labels, **estimator_kwargs
    ):
        estimator = estimator_class.__call__(**estimator_kwargs)
        if labels is not None:
            fitted_estimator = estimator.fit(X=observations, y=labels)
        else:
            fitted_estimator = estimator.fit(X=observations)
        return fitted_estimator

    @staticmethod
    def fit_helper(estimator_class, data, pickled=True, **estimator_kwargs):
        """Creates a fitted estimator from a class and a dataset"""
        assert isinstance(
            estimator_class, KhiopsEstimator.__class__
        ), "'estimator_class' should be a class derived from KhiopsEstimator"

        # Unpickle training data if necessary
        training_data = pickle.loads(data) if pickled else data
        assert (
            is_iterable(training_data) and len(training_data) == 2
        ), "'training_data' should be an iterabl with 2 elements"

        # Build a fitted estimator
        fitted_estimator = PyKhiopsSklearnTestsHelper._create_and_fit_estimator_to_data(
            estimator_class,
            training_data[0],  # observations
            training_data[1],  # labels
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
