######################################################################################
# Copyright (c) 2023 Orange. All rights reserved.                                    #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""
The main module of the Khiops Python package.

Example:
   from khiops import core as kh
   kh.train_predictor(...)

The available sub-modules inside the package are:

- core/api: main API to execute Khiops in its native way
- core/dictionary: Classes to manipulate Khiops dictionaries JSON files
  (extension ".kdicj")
- core/analysis_results: Classes to inspect Khiops JSON report files
  (extension ".khj")
- core/coclustering_results: Classes to instpect Khiops Coclustering report files
  (extension ".khcj")
- sklearn: Scikit-Learn estimator classes to learn and use Khiops models
"""
import importlib
import importlib.util
import os
import sys
import warnings
from copy import copy
from pathlib import Path

from khiops._version import get_versions
from khiops.core.internals.version import KhiopsVersion

__version__ = get_versions()["version"]
del get_versions


def get_compatible_khiops_version():
    """Returns the latest Khiops version compatible with this package's version"""

    # Define auxiliary function to remove trailing chars
    def remove_snapshot_trailing_chars(version_part):
        if "+" in version_part:
            clean_version_part = version_part[: version_part.index("+")]
        else:
            clean_version_part = version_part
        return clean_version_part

    # Build the Khiops Python version without the snapshot part
    khiops_version_parts = __version__.split(".")
    if len(khiops_version_parts) < 2:
        raise ValueError(f"Invalid Khiops Python version '{__version__}'")
    khiops_version_major = remove_snapshot_trailing_chars(khiops_version_parts[0])
    khiops_version_minor = remove_snapshot_trailing_chars(khiops_version_parts[1])
    return KhiopsVersion(f"{khiops_version_major}.{khiops_version_minor}")
