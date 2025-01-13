######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Scikit-Learn interface to the Khiops AutoML suite"""
from khiops.sklearn.estimators import (
    KhiopsClassifier,
    KhiopsCoclustering,
    KhiopsEncoder,
    KhiopsRegressor,
)
from khiops.sklearn.helpers import train_test_split_dataset
