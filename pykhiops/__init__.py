######################################################################################
# Copyright (c) 2018 - 2022 Orange - All Rights Reserved                             #
# * This software is the confidential and proprietary information of Orange.         #
# * You shall not disclose such Restricted Information and shall use it only in      #
#   accordance with the terms of the license agreement you entered into with Orange  #
#   named the "Khiops - Python Library Evaluation License".                          #
# * Unauthorized copying of this file, via any medium is strictly prohibited.        #
# * See the "LICENSE.md" file for more details.                                      #
######################################################################################
"""
The main package for Khiops support for Python.

Example:
   from pykhiops import core as pk
   pk.train_predictor(...)

The available sub-modules inside the package are:
    core/core: main API to execute Khiops in its native way
    core/dictionary: Classes to manipulate Khiops dictionaries JSON files
      (extension ".kdicj")
    core/analysis_results: Classes to inspect Khiops JSON report files
      (extension ".khj")
    core/coclustering_results: Classes to instpect Khiops Coclustering report files
      (extension ".khcj")
    sklearn: Scikit-Learn classes to execute Khiops

"""
from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
