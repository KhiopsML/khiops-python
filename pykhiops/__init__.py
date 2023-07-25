######################################################################################
# Copyright (c) 2023 Orange. All rights reserved.                                    #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""
The main package for Khiops support for Python.

Example:
   from pykhiops import core as pk
   pk.train_predictor(...)

The available sub-modules inside the package are:

- core/api: main API to execute Khiops in its native way
- core/dictionary: Classes to manipulate Khiops dictionaries JSON files
  (extension ".kdicj")
- core/analysis_results: Classes to inspect Khiops JSON report files
  (extension ".khj")
- core/coclustering_results: Classes to instpect Khiops Coclustering report files
  (extension ".khcj")
- sklearn: Scikit-Learn classes to execute Khiops

**Deprecated** will be removed in Khiops 11, use ``khiops`` top-level package instead.

"""
import sys
import warnings

# Make sure khiops is in sys.modules
import khiops

# Deprecate pykhiops in favor of khiops
from khiops.core.internals.common import deprecation_message

warnings.warn(
    deprecation_message(
        deprecated_feature="pykhiops", replacement="khiops", deadline_version="11.0"
    )
)

# Link pykhiops to khiops
sys.modules[__name__] = __import__("khiops")
