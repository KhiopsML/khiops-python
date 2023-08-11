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
from khiops.samples import samples as samples_core
from khiops.samples import samples_sklearn

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


def kh_samples_entry_point():  # pragma: no cover
    """Entry point of the kh-samples command"""
    try:
        # For technical reasons, the arguments to the script can only be read
        # from `sys.argv`, as entry point script functions cannot have parameters
        args = sys.argv[1:]

        # Check which samples to execute
        if len(args) > 0 and args[0] in ("core", "sklearn"):
            submodule = args.pop(0)
        # By default, execute khiops.core samples
        else:
            submodule = "core"

        # Execute the required samples
        if submodule == "sklearn":
            argument_parser = samples_sklearn.build_argument_parser(
                prog="kh-samples sklearn",
                description=(
                    "Examples of use of the sklearn submodule of the "
                    "Khiops Python library"
                ),
            )
            samples_sklearn.execute_samples(argument_parser.parse_args(args))
        # The khiops.core samples are used
        else:
            argument_parser = samples_sklearn.build_argument_parser(
                prog="kh-samples[ core]",
                description=(
                    "Examples of use of the core submodule of the "
                    "Khiops Python library"
                ),
            )
            samples_core.execute_samples(argument_parser.parse_args(args))
    except kh.KhiopsRuntimeError as error:
        print(f"Khiops Python backend ERROR: {error}")
        sys.exit(1)
