######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Tests for executing Khiops in parallel"""

import pickle
from functools import partial
from multiprocessing import Pool
from unittest import TestCase

from khiops.core.internals.common import is_iterable
from khiops.sklearn.estimators import (
    KhiopsClassifier,
    KhiopsCoclustering,
    KhiopsEncoder,
    KhiopsRegressor,
)
from tests.test_helper import KhiopsTestHelper


class KhiopsParallelRunningTests(TestCase, KhiopsTestHelper):
    """Test if Khiops estimator instances can be built and run in parallel"""

    _n_estimator_instances = 2
    _n_cpus = -1

    def setUp(self):
        KhiopsTestHelper.skip_long_test(self)

    def _parallelise(self, callback, arg_sequence, n_cpus):
        """Parallelisation driver"""
        # Check argument types
        if not is_iterable(arg_sequence):
            raise TypeError("'arg_sequence' must be an iterable.")
        if not isinstance(n_cpus, int):
            raise TypeError("'n_cpus' must be integer.")

        # Parallel execution with explicit numbers of cpus
        if n_cpus > 1:
            with Pool(n_cpus) as pool:
                return pool.map(callback, (pickle.dumps(arg) for arg in arg_sequence))
        # Parallel execution with the number of *available* cpus
        elif n_cpus == -1:
            with Pool(None) as pool:
                return pool.map(callback, (pickle.dumps(arg) for arg in arg_sequence))
        # Sequential execution
        elif n_cpus == 1:
            return [callback(arg, pickled=False) for arg in arg_sequence]
        else:
            raise ValueError(f"'n_cpus' must be positive or -1. It is: '{n_cpus}'.")

    def test_parallel_classifier_fit(self):
        """Create many KhiopsClassifier and fit them"""
        print("\n>>> Testing parallel estimator fit")

        # Obtain the "Adult" dataset with "class" as target
        data = self.get_monotable_data("Adult")
        train_data = self.prepare_data(data, "class")[0]

        # Create many classifiers and fit them in parallel
        n_classifiers = self._n_estimator_instances
        estimators = self._parallelise(
            callback=partial(KhiopsTestHelper.fit_helper, KhiopsClassifier),
            arg_sequence=(train_data,) * n_classifiers,
            n_cpus=self._n_cpus,
        )
        # Check that the expected number of classifiers was created
        self.assertEqual(len(estimators), n_classifiers)

    def test_parallel_regressor_fit(self):
        """Create many KhiopsRegressor and fit them"""
        print("\n>>> Testing parallel regressor fit")

        # Obtain the "Adult" dataset with "age" as target
        data = self.get_monotable_data("Adult")
        train_data = self.prepare_data(data, "age")[0]

        # Create many regressors and fit them in parallel
        n_regressors = self._n_estimator_instances
        regressors = self._parallelise(
            callback=partial(KhiopsTestHelper.fit_helper, KhiopsRegressor),
            arg_sequence=(train_data,) * n_regressors,
            n_cpus=self._n_cpus,
        )

        # Check that the expected number of regressors was created
        self.assertEqual(len(regressors), n_regressors)

    def test_parallel_coclustering_fit(self):
        """Create many KhiopsCoclustering and fit them"""
        print("\n>>> Testing parallel coclustering fit")

        # Obtain the "SpliceJunction" dataset
        root_table_data, secondary_table_data = self.get_two_table_data(
            "SpliceJunction", "SpliceJunction", "SpliceJunctionDNA"
        )
        root_train_data = self.prepare_data(root_table_data, "Class")[0][0]
        secondary_train_data = self.prepare_data(
            secondary_table_data, "SampleId", primary_table=root_train_data
        )[0]

        # Create many clusterers and fit them in parallel
        n_clusterers = self._n_estimator_instances
        clusterers = self._parallelise(
            callback=partial(
                KhiopsTestHelper.fit_helper,
                KhiopsCoclustering,
                key="SampleId",
                variables=["SampleId", "Pos", "Char"],
            ),
            arg_sequence=(secondary_train_data,) * n_clusterers,
            n_cpus=self._n_cpus,
        )

        # Check that the expected number of clusterers was created
        self.assertEqual(len(clusterers), n_clusterers)

    def test_parallel_encoder_fit(self):
        """Create many KhiopsEncoder and fit them"""
        print("\n>>> Testing parallel encoder fit")
        n_encoders = self._n_estimator_instances
        data = self.get_monotable_data("Adult")
        train_data = self.prepare_data(data, "class")[0]
        estimators = self._parallelise(
            callback=partial(KhiopsTestHelper.fit_helper, KhiopsEncoder),
            arg_sequence=(train_data,) * n_encoders,
            n_cpus=self._n_cpus,
        )
        # Check that the expected number of encoders was created
        self.assertEqual(len(estimators), n_encoders)

    def test_parallel_classifier_fit_predict(self):
        """Create many KhiopsClassifier fit them and predict on the test data"""
        print("\n>>> Testing parallel estimator fit and predict")

        # Obtain the "Adult" dataset with "class" as target
        data = self.get_monotable_data("Adult")
        train_data, test_data = self.prepare_data(data, "class")

        # Create many classifiers in parallel
        n_classifiers = self._n_estimator_instances
        estimators = self._parallelise(
            callback=partial(KhiopsTestHelper.fit_helper, KhiopsClassifier),
            arg_sequence=(train_data,) * n_classifiers,
            n_cpus=self._n_cpus,
        )

        # Fit and predict all classifiers in parallel
        prediction_tables = self._parallelise(
            callback=partial(KhiopsTestHelper.predict_helper, kind="simple"),
            arg_sequence=zip(estimators, (test_data,) * n_classifiers),
            n_cpus=self._n_cpus,
        )

        # Fit and predict probabilities for all classifiers in parallel
        probability_tables = self._parallelise(
            callback=partial(KhiopsTestHelper.predict_helper, kind="proba"),
            arg_sequence=zip(estimators, (test_data,) * n_classifiers),
            n_cpus=self._n_cpus,
        )

        # Check that:
        # - the expected number of clusterers was created
        # - all output tables have the expected number of rows
        self.assertEqual(len(estimators), n_classifiers)
        expected_n_rows = test_data[1].shape[0]
        for i, prediction_table in enumerate(prediction_tables):
            actual_n_rows = prediction_table.shape[0]
            self.assertEqual(
                actual_n_rows,
                expected_n_rows,
                f"Prediction table #{i} has wrong number of rows",
            )
        for i, probability_table in enumerate(probability_tables):
            actual_n_rows = probability_table.shape[0]
            self.assertEqual(
                actual_n_rows,
                expected_n_rows,
                f"Probability table #{i} has wrong number of rows",
            )

    def test_parallel_coclustering_fit_predict(self):
        """Create many KhiopsCoclustering fit them and predict on the test data"""
        print("\n>>> Testing parallel coclustering fit and predict")

        # Obtain "SpliceJunction" dataset
        (root_table_data, secondary_table_data) = self.get_two_table_data(
            "SpliceJunction", "SpliceJunction", "SpliceJunctionDNA"
        )
        root_train_data, _ = self.prepare_data(root_table_data, "Class")[0]
        secondary_train_data, secondary_test_data = self.prepare_data(
            secondary_table_data, "SampleId", primary_table=root_train_data
        )

        # Create mayn clusterers in parallel
        n_clusterers = self._n_estimator_instances
        clusterers = self._parallelise(
            callback=partial(
                KhiopsTestHelper.fit_helper,
                KhiopsCoclustering,
                key="SampleId",
                variables=["SampleId", "Pos", "Char"],
            ),
            arg_sequence=(secondary_train_data,) * n_clusterers,
            n_cpus=self._n_cpus,
        )

        # Fit and predict all clusterers in parallel
        cluster_tables = self._parallelise(
            callback=KhiopsTestHelper.predict_helper,
            arg_sequence=zip(clusterers, (secondary_test_data,) * n_clusterers),
            n_cpus=self._n_cpus,
        )
        # Check that:
        # - the expected number of clusterers was created
        # - all output tables have the expected number of rows
        self.assertEqual(len(cluster_tables), n_clusterers)
        expected_n_rows = secondary_test_data[0]["SampleId"].drop_duplicates().shape[0]
        for i, cluster_table in enumerate(cluster_tables):
            actual_n_rows = cluster_table.shape[0]
            self.assertEqual(
                actual_n_rows,
                expected_n_rows,
                f"Cluster table #{i} has wrong number of rows",
            )

    def test_parallel_regressor_fit_predict(self):
        """Create many KhiopsRegressor fit them and predict on the test data"""
        print("\n>>> Testing parallel regressor fit and predict")

        # Obtain the "Adult" dataset with "age" as target
        data = self.get_monotable_data("Adult")
        train_data, test_data = self.prepare_data(data, "age")

        # Create many encoders in parallel
        n_regressors = self._n_estimator_instances
        regressors = self._parallelise(
            callback=partial(KhiopsTestHelper.fit_helper, KhiopsRegressor),
            arg_sequence=(train_data,) * n_regressors,
            n_cpus=self._n_cpus,
        )

        # Fit all encoders in parallel
        prediction_tables = self._parallelise(
            callback=KhiopsTestHelper.predict_helper,
            arg_sequence=zip(regressors, (test_data,) * n_regressors),
            n_cpus=self._n_cpus,
        )

        # Check that:
        # - the expected number of regressors was created
        # - all output tables have the expected number of rows
        self.assertEqual(len(regressors), n_regressors)
        expected_n_rows = test_data[1].shape[0]
        for i, prediction_table in enumerate(prediction_tables):
            actual_n_rows = prediction_table.shape[0]
            self.assertEqual(
                actual_n_rows,
                expected_n_rows,
                f"Prediction table #{i} has wrong number of rows",
            )

    def test_parallel_encoder_fit_predict(self):
        """Create many KhiopsEncoder fit them and transform the test data"""
        print("\n>>> Testing parallel encoder fit and predict")

        # Obtain the "Adult" dataset with "class" as target
        data = self.get_monotable_data("Adult")
        train_data, test_data = self.prepare_data(data, "class")

        # Create many encoders in parallel
        n_encoders = self._n_estimator_instances
        encoders = self._parallelise(
            callback=partial(KhiopsTestHelper.fit_helper, KhiopsEncoder),
            arg_sequence=(train_data,) * n_encoders,
            n_cpus=self._n_cpus,
        )

        # Fit all encoders in parallel
        encoded_tables = self._parallelise(
            callback=KhiopsTestHelper.predict_helper,
            arg_sequence=zip(encoders, (test_data,) * n_encoders),
            n_cpus=self._n_cpus,
        )

        # Check that:
        # - the expected number of encoders was created
        # - all output tables have the expected number of rows
        self.assertEqual(len(encoded_tables), n_encoders)
        expected_n_rows = test_data[1].shape[0]
        for i, encoded_table in enumerate(encoded_tables):
            actual_n_rows = encoded_table.shape[0]
            self.assertEqual(
                actual_n_rows,
                expected_n_rows,
                f"Encoded table #{i} has the wrong number of rows",
            )
