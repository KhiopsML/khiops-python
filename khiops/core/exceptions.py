######################################################################################
# Copyright (c) 2023 Orange. All rights reserved.                                    #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Khiops Python exception classes"""
import warnings


class KhiopsJSONError(Exception):
    """Parsing error for Khiops-generated JSON files"""


class KhiopsRuntimeError(Exception):
    """Khiops execution related errors"""


class KhiopsEnvironmentError(Exception):
    """Khiops execution environment error

    Example: Khiops binary not found.
    """


######################
# Deprecated Classes #
######################

# Note: We don't use deprecation_message to avoid a circular import


class PyKhiopsJSONError(KhiopsJSONError):
    """Deprecated

    See `KhiopsJSONError`.
    """

    def __init__(self):
        super().__init__()
        warnings.warn(
            "'PyKhiopsJSONError' is deprecated and will be removed by "
            "version 11.0.0. Use 'KhiopsJSONError' instead."
        )


class PyKhiopsRuntimeError(KhiopsRuntimeError):
    """Deprecated

    See `KhiopsRuntimeError`.
    """

    def __init__(self):
        super().__init__()
        warnings.warn(
            "'PyKhiopsRuntimeError' is deprecated and will be removed "
            "by version 11.0.0. Use 'KhiopsRuntimeError' instead."
        )


class PyKhiopsEnvironmentError(KhiopsEnvironmentError):
    """Deprecated

    See `KhiopsEnvironmentError`.
    """

    def __init__(self):
        super().__init__()
        warnings.warn(
            "'PyKhiopsEnvironmentError' is deprecated and will be removed "
            "by version 11.0.0. Use 'KhiopsEnvironmentError' instead."
        )
