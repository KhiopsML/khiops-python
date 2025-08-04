######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################

"""Extract dependencies from pyproject.toml file"""

import argparse
import codecs

try:
    import tomllib as tomli
except ModuleNotFoundError:
    import tomli


def main(args):
    with open(args.pyproject_file, "rb") as pyproject_file_descriptor:
        project_metadata = tomli.load(pyproject_file_descriptor)["project"]
        dependencies = project_metadata["dependencies"]

        # Optional dependencies listed as per-group dependencies
        for dependency_group in project_metadata["optional-dependencies"].values():
            dependencies += list(dependency_group)
    print(args.dependency_separator.join(dependencies))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python extract_dependencies_from_pyproject_toml.py",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Extracts the dependencies from a pyproject.toml file",
    )
    parser.add_argument(
        "-f",
        "--pyproject-file",
        metavar="FILE",
        help="Path of the pyproject.toml file",
    )

    parser.add_argument(
        "-s",
        "--dependency-separator",
        default=" ",
        type=lambda s: codecs.decode(s, "unicode_escape") if s == "\\n" else s,
        help="Dependency separator (default: ' ')",
    )
    main(parser.parse_args())
