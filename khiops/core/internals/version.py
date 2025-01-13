######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Class to handle Khiops version strings"""

import re


def _is_simple_number(string):
    """Tests if a string contains only characters [0-9] and no left zeroes

    note::
        We do not use str.isdigit() because it returns ``True`` for digit-like UTF-8
        characters (fractions and superscripts for example).
    """
    if string:
        all_chars = all(char in "0123456789" for char in string)
        no_left_zeroes = not string.startswith("0") or string == "0"
        is_simple_number = all_chars and no_left_zeroes
    else:
        is_simple_number = False
    return is_simple_number


class KhiopsVersion:
    """Encapsulates the Khiops version string

    Implements comparison operators.
    """

    def __init__(self, version_str):
        # Save the raw version string
        self._version_str = version_str

        # Remove the "v" prefix if present
        raw_parts = re.sub("^v", "", self._version_str).split(".", maxsplit=2)

        # Check the Khiops version format: MAJOR.MINOR.PATCH[-PRE_RELEASE]
        if len(raw_parts) < 3:
            self._raise_init_error(
                "Version must have the format MAJOR.MINOR.PATCH[-PRE_RELEASE]",
                version_str,
            )
        self._major, self._minor, patch_and_pre_release = raw_parts

        # Check MAJOR and MINOR are numeric
        if _is_simple_number(self._major):
            self._major = int(self._major)
        else:
            self._raise_init_error("MAJOR part of version isn't numeric", version_str)
        if _is_simple_number(self._minor):
            self._minor = int(self._minor)
        else:
            self._raise_init_error("MINOR part of version isn't numeric", version_str)

        # Check that the third part:
        # - Follows the PATCH[-PRE_RELEASE] format
        # - PATCH is numeric
        # - If PRE_RELEASE is present that is a,b or rc followed by a number
        # Third part with only digits
        if _is_simple_number(patch_and_pre_release):
            self._patch = int(patch_and_pre_release)
            self._pre_release_id = None
            self._pre_release_increment = None
        # Third part with not only digits
        else:
            # Check that it has only one dash and store the delimited parts
            if patch_and_pre_release.count("-") != 1:
                self._raise_init_error(
                    "PATCH-PRE_RELEASE version part must contain a single '-'",
                    version_str,
                )
            if patch_and_pre_release.count(".") > 1:
                self._raise_init_error(
                    "PATCH-PRE_RELEASE version part must contain at most a single '.'",
                    version_str,
                )
            self._patch, _pre_release = patch_and_pre_release.split("-")

            # Store only the patch version part if there are only digits
            if _is_simple_number(self._patch):
                self._patch = int(self._patch)
            else:
                self._raise_init_error("PATCH version part isn't numeric", version_str)

            # Store the pre-release id (alpha, beta or release candidate)
            self._pre_release_id = None
            for pre_release_id in ("a", "b", "rc"):
                if _pre_release.startswith(pre_release_id):
                    self._pre_release_id = pre_release_id
            if self._pre_release_id is None:
                self._raise_init_error(
                    "PRE_RELEASE version part must start with 'a', 'b' or 'rc'",
                    version_str,
                )

            # Store the rest of the prerelease (if any) and check it is a number
            # We accept not having a "." in the pre-release increment for backward
            # compatibility.
            self._pre_release_increment = _pre_release.replace(
                self._pre_release_id, ""
            ).replace(".", "")
            if _is_simple_number(self._pre_release_increment):
                self._pre_release_increment = int(self._pre_release_increment)
            else:
                self._raise_init_error(
                    "PRE_RELEASE version part increment is not numeric", version_str
                )

    def _raise_init_error(self, msg, version_str):
        raise ValueError(f"{msg}. Version string: '{version_str}'.")

    @property
    def major(self):
        """int : The version's major number"""
        return self._major

    @property
    def minor(self):
        """int : The version's minor number"""
        return self._minor

    @property
    def patch(self):
        """int : The version's patch number"""
        return self._patch

    @property
    def pre_release(self):
        """str : The version's pre-release tag

        Returns: either 'a', 'b' or 'rc' followed by '.' and a number or None.
        """
        if self._pre_release_id is None:
            return None
        else:
            return f"{self._pre_release_id}.{self._pre_release_increment}"

    def __repr__(self):
        return self._version_str

    def __eq__(self, version):
        return (
            self.major == version.major
            and self.minor == version.minor
            and self.patch == version.patch
            and self._pre_release_id == version._pre_release_id
            and self._pre_release_increment == version._pre_release_increment
        )

    def __gt__(self, version):
        """Normal versioning order

        Not having a pre-release part has more priority than having it.
        For example: 10.1.0-b1 < 10.1.0.
        """
        if self.major > version.major:
            is_greater = True
        elif self.major < version.major:
            is_greater = False
        elif self.minor > version.minor:
            is_greater = True
        elif self.minor < version.minor:
            is_greater = False
        elif self.patch > version.patch:
            is_greater = True
        elif self.patch < version.patch:
            is_greater = False
        elif self.pre_release is None and version.pre_release is not None:
            is_greater = True
        elif self.pre_release is not None and version.pre_release is None:
            is_greater = False
        elif self.pre_release is not None and version.pre_release is not None:
            if self._pre_release_id > version._pre_release_id:
                is_greater = True
            elif self._pre_release_id < version._pre_release_id:
                is_greater = False
            else:
                is_greater = (
                    self._pre_release_increment > version._pre_release_increment
                )
        else:
            assert self.__eq__(version)
            is_greater = False

        return is_greater

    def __ge__(self, version):
        return self.__eq__(version) or self.__gt__(version)

    def __lt__(self, version):
        return not self.__ge__(version)

    def __le__(self, version):
        return not self.__gt__(version)

    def __hash__(self):
        return self.__repr__().__hash__()
