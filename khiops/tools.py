######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Miscellaneous utility tools

.. warning::
    The entry point functions in this module use `sys.exit`. They are not designed to be
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
from khiops.core.internals.runner import get_default_samples_dir
from khiops.samples import samples as samples_core

# We deactivate the warnings to not show a deprecation warning from sklearn
# pylint: disable=wrong-import-position
warnings.simplefilter("ignore")
from khiops.samples import samples_sklearn

warnings.resetwarnings()
# pylint: enable=wrong-import-position

# Note: We dont include these tools in coverage


def kh_status_entry_point():  # pragma: no cover
    """Entry point of the kh-status command"""
    status_code = kh.get_runner().print_status()
    sys.exit(status_code)


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
        if submodule == "core":
            samples_module = samples_core
        else:
            samples_module = samples_sklearn

        argument_parser = samples_module.build_argument_parser(
            prog="kh-samples [core|sklearn]",
            description=(
                "Executes the sample code snippets of the Khiops Python library"
            ),
        )
        samples_module.execute_samples(argument_parser.parse_args(args))
    except kh.KhiopsRuntimeError as error:
        print(f"khiops engine error: {error}")
        sys.exit(1)


# Samples version: To be updated when khiops-samples does
DEFAULT_SAMPLES_VERSION = "11.0.0"


def kh_download_datasets_entry_point():
    """Entry point for the download samples helper script"""
    samples_dir = pathlib.Path.home() / "khiops_data" / "samples"
    arg_parser = argparse.ArgumentParser(
        prog="kh-download-datasets",
        description=f"Downloads the Khiops samples dataset to {samples_dir}.",
    )
    arg_parser.add_argument(
        "-v",
        "--version",
        default=DEFAULT_SAMPLES_VERSION,
        help="Sample datasets version",
    )
    arg_parser.add_argument(
        "-f",
        "--force-overwrite",
        default=False,
        action="store_true",
        help=f"Overwrites any existent directory at {samples_dir}",
    )
    args = arg_parser.parse_args()
    with warnings.catch_warnings(record=True) as caught_warnings:
        download_datasets(
            force_overwrite=args.force_overwrite,
            version=args.version,
            _called_from_shell=True,
        )
        if caught_warnings:
            for warning in caught_warnings:
                print(f"warning: {warning.message}")
    sys.exit(0)


def download_datasets(
    force_overwrite=False, version=DEFAULT_SAMPLES_VERSION, _called_from_shell=False
):
    """Downloads the Khiops sample datasets for a given version

    The datasets are downloaded to:
        - all systems: ``KHIOPS_SAMPLES_DIR/khiops_data/samples`` if
          ``KHIOPS_SAMPLES_DIR`` is defined and non-empty
        - Windows:
            - ``%PUBLIC%\\khiops_data\\samples`` if ``%PUBLIC%`` is defined
            - ``%USERPROFILE%\\khiops_data\\samples`` otherwise
        - Linux/macOS: ``$HOME/khiops_data/samples``

    Parameters
    ==========
    force_overwrite : bool, default ``False``
        If ``True`` it always overwrites the local samples directory even if it exists.
    version : str, default "10.2.0"
        The version of the samples datasets.
    """
    # Note: The hidden parameter _called_from_shell is just to change the user messages.
    samples_dir = get_default_samples_dir()
    if os.path.exists(samples_dir) and not force_overwrite:
        if _called_from_shell:
            instructions = "Execute with '--force-overwrite' to overwrite it"
        else:
            instructions = "Set 'force_overwrite=True' to overwrite it"
        warnings.warn(
            "Download not executed since the sample datasets directory "
            f"already exists. {instructions}. Path: {samples_dir}"
        )
    else:
        # Create the samples dataset directory
        if os.path.exists(samples_dir):
            shutil.rmtree(samples_dir)
        os.makedirs(samples_dir, exist_ok=True)

        # Set the sample dataset zip URL
        samples_repo_url = "https://github.com/KhiopsML/khiops-samples"
        samples_zip_file = f"khiops-samples-{version}.zip"
        samples_zip_url = (
            f"{samples_repo_url}/releases/download/{version}/{samples_zip_file}"
        )

        # Download the sample zip file and extracted to the home dataset dir
        print(f"Downloading samples from {samples_zip_url}")
        with tempfile.NamedTemporaryFile() as temp_zip_file, urllib.request.urlopen(
            samples_zip_url
        ) as zip_request:
            temp_zip_file.write(zip_request.read())
            temp_zip_file.seek(0)
            with zipfile.ZipFile(temp_zip_file) as temp_zip:
                temp_zip.extractall(samples_dir)
            print(f"Samples dataset successfully downloaded to {samples_dir}")
