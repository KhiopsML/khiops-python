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

- core/api: main API to execute Khiops in its native way
- core/dictionary: Classes to manipulate Khiops dictionaries JSON files
  (extension ".kdicj")
- core/analysis_results: Classes to inspect Khiops JSON report files
  (extension ".khj")
- core/coclustering_results: Classes to instpect Khiops Coclustering report files
  (extension ".khcj")
- sklearn: Scikit-Learn classes to execute Khiops
"""
from pykhiops._version import get_versions
from pykhiops.core.common import KhiopsVersion

__version__ = get_versions()["version"]
del get_versions


def get_compatible_khiops_version():
    """Returns the latest Khiops version compatible with this version of pyKhiops"""
    # Define auxiliary function to remove trailing chars
    def remove_snapshot_trailing_chars(version_part):
        if "+" in version_part:
            clean_version_part = version_part[: version_part.index("+")]
        else:
            clean_version_part = version_part
        return clean_version_part

    # Build the pyKhiops version without the snapshot part
    pykhiops_version_parts = __version__.split(".")
    if len(pykhiops_version_parts) < 2:
        raise ValueError(f"Invalid pyKhiops version '{__version__}'")
    khiops_version_major = remove_snapshot_trailing_chars(pykhiops_version_parts[0])
    khiops_version_minor = remove_snapshot_trailing_chars(pykhiops_version_parts[1])
    return KhiopsVersion(f"{khiops_version_major}.{khiops_version_minor}")
