######################################################################################
# Copyright (c) 2023 Orange. All rights reserved.                                    #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Miscellaneous utility tools

.. warning::
    Entry point functions in this module use `sys.exit`. They are not designed to be
    called from another program or python shell.
"""
import sys
import warnings

import khiops.core as kh
from khiops.core.internals.common import deprecation_message

# Note: We dont include these tools in coverage


def pk_status_entry_point():  # pragma: no cover
    """Entry point of the pk-status command
    **Deprecated** will be removed in Khiops 11, use `kh_status_entry_point` instead.
    """
    warnings.warn(
        deprecation_message(
            deprecated_feature="pk-status",
            replacement="kh-status",
            deadline_version="11.0",
        )
    )
    return kh_status_entry_point()


def kh_status_entry_point():  # pragma: no cover
    """Entry point of the kh-status command"""
    try:
        kh.get_runner().print_status()
        print("\nKhiops Python installation OK")
        sys.exit(0)
    except kh.KhiopsEnvironmentError as error:
        print(
            f"Khiops Python backend ERROR: {error}"
            "\nCheck https://www.khiops.com to install the Khiops app in your computer"
        )
        sys.exit(1)
