######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Updates the copyright notice of the input files"""

import argparse
from datetime import datetime

# pylint: disable=line-too-long
copyright_blob = f"""######################################################################################
# Copyright (c) 2023-{datetime.today().year} Orange. All rights reserved.                               #
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
        for line_number, line in enumerate(lines, start=1):
            # Write any shebang
            if line.startswith("#!") and line_number == 1:
                file.write(line)
            # Ignore a previous copyright banner
            elif line.startswith("#") and not skipped_copyright:
                continue
            # After reading the old banner write the new and any line after
            elif not line.startswith("#") and not skipped_copyright:
                skipped_copyright = True
                file.write(copyright_blob)
                file.write(line)
            elif skipped_copyright:
                file.write(line)

        # Write copyright if in not in skipped_copyright state at the end of the file
        # This case is for files with only the banner (ex. __init__.py)
        if not skipped_copyright:
            file.write(copyright_blob)


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
        help="Location of the khiops-tutorial directory",
    )
    main(parser.parse_args())
