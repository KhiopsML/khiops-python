######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Khiops Python exception classes"""


class KhiopsJSONError(Exception):
    """Parsing error for Khiops-generated JSON files"""


class KhiopsRuntimeError(Exception):
    """Khiops execution related errors"""


class KhiopsEnvironmentError(Exception):
    """Khiops execution environment error

    Example: Khiops binary not found.
    """
