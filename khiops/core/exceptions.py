######################################################################################
# Copyright (c) 2023 Orange. All rights reserved.                                    #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Khiops Python exception classes"""
import warnings

######################
# Deprecated Errors  #
######################

# Note: We don't use deprecation_message to avoid a circular import


class PyKhiopsJSONError(Exception):
    """Deprecated

    See `KhiopsJSONError`.
    """

    def __init__(self, *args):
        super().__init__(*args)
        if self.__class__.__name__ == "PyKhiopsJSONError":
            warnings.warn(
                "'PyKhiopsJSONError' is deprecated and will be removed "
                "by version 11.0.0. Use 'KhiopsJSONError' instead."
            )


class PyKhiopsRuntimeError(Exception):
    """Deprecated

    See `KhiopsRuntimeError`.
    """

    def __init__(self, *args):
        super().__init__(*args)
        if self.__class__.__name__ == "PyKhiopsRuntimeError":
            warnings.warn(
                "'PyKhiopsRuntimeError' is deprecated and will be removed "
                "by version 11.0.0. Use 'KhiopsRuntimeError' instead."
            )


class PyKhiopsEnvironmentError(Exception):
    """Deprecated

    See `KhiopsEnvironmentError`.
    """

    def __init__(self, *args):
        super().__init__(*args)
        if self.__class__.__name__ == "PyKhiopsEnvironmentError":
            warnings.warn(
                "'PyKhiopsEnvironmentError' is deprecated and will be removed "
                "by version 11.0.0. Use 'KhiopsEnvironmentError' instead."
            )


#################
# Active Errors #
#################

# Note: We need to put them after the deprecated ones because the latter ones need to be
# defined.


class KhiopsJSONError(PyKhiopsJSONError):
    """Parsing error for Khiops-generated JSON files"""


class KhiopsRuntimeError(PyKhiopsRuntimeError):
    """Khiops execution related errors"""


class KhiopsEnvironmentError(PyKhiopsEnvironmentError):
    """Khiops execution environment error

    Example: Khiops binary not found.
    """
