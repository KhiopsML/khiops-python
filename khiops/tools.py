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
import argparse
import os
import pathlib
import shutil
import sys
import tempfile
import urllib.request
import warnings
import zipfile

import khiops.core as kh
from khiops.core.internals.common import deprecation_message
from khiops.samples import samples as samples_core

# We deactivate the warnings to not show a deprecation warning from sklearn
# pylint: disable=wrong-import-position
warnings.simplefilter("ignore")
from khiops.samples import samples_sklearn

warnings.resetwarnings()
# pylint: enable=wrong-import-position

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
        # Catch runtime warnings
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            kh.get_runner().print_status()
            if len(caught_warnings) == 0:
                print("khiops-python installation OK")
            else:
                print("khiops-python installation OK, with warnings:")
                # Print the warning message and remember any related to sample datasets
                provide_dataset_info = False
                for warning in caught_warnings:
                    print(f"warning: {warning.message}")
                    if "Sample datasets" in str(warning.message):
                        provide_dataset_info = True

                # Show additional info for the datasets if the were related warnings
                if provide_dataset_info:
                    print(
                        "You may install the datasets by executing kh-download-datasets"
                    )
            warnings.resetwarnings()
            sys.exit(0)
    except kh.KhiopsEnvironmentError as error:
        print(
            f"Khiops core installation error: {error}"
            "\nCheck https://www.khiops.org to set-up Khiops on your computer"
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
                prog="kh-samples core",
                description=(
                    "Examples of use of the core submodule of the "
                    "Khiops Python library"
                ),
            )
            samples_core.execute_samples(argument_parser.parse_args(args))
    except kh.KhiopsRuntimeError as error:
        print(f"Khiops Python backend ERROR: {error}")
        sys.exit(1)


def kh_download_datasets_entry_point():
    """Entry point for the download samples helper script"""
    samples_dir = pathlib.Path.home() / "khiops_data" / "samples"
    arg_parser = argparse.ArgumentParser(
        prog="kh-download-datasets",
        description=f"Downloads the Khiops samples dataset to {samples_dir}.",
    )
    arg_parser.add_argument(
        "-v", "--version", default="10.1.1", help="Sample datasets version"
    )
    arg_parser.add_argument(
        "-o",
        "--overwrite",
        default=False,
        action="store_true",
        help=f"Overwrites any existent directory at {samples_dir}",
    )
    args = arg_parser.parse_args()
    _download_datasets(samples_dir, args.version, overwrite=args.overwrite)


def _download_datasets(samples_dir, version, overwrite=False):
    """Download the khiops sample datasets for a given version"""
    # Check if the home sample dataset location is available and build it if necessary
    write_samples_dir = True
    if samples_dir.exists() and not overwrite:
        write_samples_dir = _query_user(f"Overwrite {samples_dir} ?")
        if write_samples_dir:
            shutil.rmtree(samples_dir)

    # Write if the check went ok
    if write_samples_dir:
        # Create the samples dataset directory
        os.makedirs(samples_dir, exist_ok=True)

        # Set the sample dataset zip URL
        samples_repo_url = "https://github.com/KhiopsML/khiops-samples"
        samples_zip_file = f"khiops-samples-v{version}.zip"
        samples_zip_url = (
            f"{samples_repo_url}/releases/download/v{version}/{samples_zip_file}"
        )

        # Download the sample zip file and extracted to the home dataset dir
        with tempfile.NamedTemporaryFile() as temp_zip_file, urllib.request.urlopen(
            samples_zip_url
        ) as zip_request:
            temp_zip_file.write(zip_request.read())
            temp_zip_file.seek(0)
            with zipfile.ZipFile(temp_zip_file) as temp_zip:
                temp_zip.extractall(samples_dir)
            print(f"Samples dataset successfully downloaded to {samples_dir}")


def _query_user(question, default="no"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " (y/n) "
    elif default == "yes":
        prompt = " (Y/n) "
    elif default == "no":
        prompt = " (y/N) "
    else:
        raise ValueError(f"invalid default answer: '{default}'")
    while True:
        print(question + prompt, end="")
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            print("Please respond with 'yes' or 'no' (or 'y' or 'n')", end="")
