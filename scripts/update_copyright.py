"""Updates the copyright notice of the input files"""
import argparse
from datetime import datetime

# pylint: disable=line-too-long
copyright_blob = f"""######################################################################################
# Copyright (c) {datetime.today().year} Orange. All rights reserved.                                    #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""
# pylint: enable=line-too-long


def main(args):
    """Main method"""
    for file_path in args.file_paths:
        update_copyright(file_path)


def update_copyright(file_path):
    """Updates the copyright notice of a file"""
    print(f"Updating {file_path}")
    with open(file_path, encoding="utf8") as file:
        lines = file.readlines()
    skipped_copyright = False
    with open(file_path, "w", encoding="utf8") as file:
        file.write(copyright_blob)
        for line in lines:
            if line.startswith("#") and not skipped_copyright:
                continue
            elif not line.startswith("#") and not skipped_copyright:
                skipped_copyright = True
                file.write(line)
            elif skipped_copyright:
                file.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python update_copyright.py",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Updates the copyright notice of the input files",
    )
    parser.add_argument(
        "file_paths",
        metavar="FILE",
        nargs="+",
        help="Location of the pykhiops-tutorial directory",
    )
    main(parser.parse_args())
