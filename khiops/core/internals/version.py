######################################################################################
# Copyright (c) 2023 Orange. All rights reserved.                                    #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Class to handle Khiops version strings"""

import re


class KhiopsVersion:
    """Encapsulates the Khiops version string

    Implements comparison operators.
    """

    def __init__(self, version):
        # Save the version and its parts
        self.version = version
        self._parts = None
        self._trailing_char = None
        self._allowed_chars = ["c", "i", "a", "b"]

        # Remove the "v" prefix if present
        # Check that :
        # - each part besides the last is numeric
        # - the last part is alphanumeric
        raw_parts = re.sub("^v", "", self.version).split(".")
        for i, part in enumerate(raw_parts, start=1):
            if i < len(raw_parts) and not part.isnumeric():
                raise ValueError(
                    f"Component #{i} of version string '{version}' " "must be numeric."
                )
            if i == len(raw_parts):
                if not part.isalnum():
                    raise ValueError(
                        f"Component #{i} of version string '{version}' "
                        "must be alphanumeric."
                    )
                if not part[0].isnumeric():
                    raise ValueError(
                        f"Component #{i} of version string '{version}' "
                        "must start with a numeric character"
                    )

                must_be_alpha = False
                for char in part:
                    if char.isnumeric() and must_be_alpha:
                        raise ValueError(
                            f"Component #{i} of version string '{version}' "
                            "must have alphabetic characters only at the end"
                        )
                    elif char.isalpha() and not must_be_alpha:
                        must_be_alpha = True

        # Save the numeric parts
        self._parts = []
        for part in raw_parts[:-1]:
            self._parts.append(int(part))

        # Save the numeric portion of the last part and any remaining chars
        last_part = raw_parts[-1]
        if last_part.isnumeric():
            self._parts.append(int(last_part))
        else:
            for i, char in enumerate(last_part):
                if char.isalpha():
                    self._parts.append(int(last_part[:i]))
                    self._trailing_char = last_part[i:]
                    break
        if (
            self._trailing_char is not None
            and self._trailing_char not in self._allowed_chars
        ):
            raise ValueError(
                f"Trailing char of version string '{version}' "
                f"must be one of {self._allowed_chars}"
            )

        # Transform numeric parts to tuple
        self._parts = tuple(self._parts)

        assert isinstance(self._parts, tuple)

    @property
    def major(self):
        """int : The major number of this version"""
        return self._parts[0]

    @property
    def minor(self):
        """int : The minor number of this version"""
        if len(self._parts) < 2:
            minor_number = 0
        else:
            minor_number = self._parts[1]
        return minor_number

    @property
    def patch(self):
        """int : The patch number of this version"""
        if len(self._parts) < 3:
            patch_number = 0
        else:
            patch_number = self._parts[2]
        return patch_number

    def __str__(self):
        return self.version

    def __repr__(self):
        return self.version

    def __eq__(self, version):
        # Pad with zeros the versions if necessary
        if len(self._parts) > len(version._parts):
            padded_parts = version._parts + (0,) * (
                len(self._parts) - len(version._parts)
            )
            this_padded_parts = self._parts
        elif len(self._parts) < len(version._parts):
            this_padded_parts = self._parts + (0,) * (
                len(version._parts) - len(self._parts)
            )
            padded_parts = version._parts
        else:
            this_padded_parts = self._parts
            padded_parts = version._parts

        # Compare the padded parts and the trailing chars
        if (
            this_padded_parts == padded_parts
            and self._trailing_char == version._trailing_char
        ):
            return True
        return False

    def __gt__(self, version):
        if self == version:
            return False
        elif self._parts > version._parts:
            return True
        elif self._parts < version._parts:
            return False
        else:
            if self._trailing_char is None and version._trailing_char is not None:
                return True
            elif self._trailing_char is not None and version._trailing_char is None:
                return False
            else:
                self_index = self._allowed_chars.index(self._trailing_char)
                version_index = self._allowed_chars.index(version._trailing_char)
                return self_index > version_index

    def __ge__(self, version):
        return self.__eq__(version) or self.__gt__(version)

    def __lt__(self, version):
        return not self.__ge__(version)

    def __le__(self, version):
        return not self.__gt__(version)

    def __hash__(self):
        return self.version.__hash__()
