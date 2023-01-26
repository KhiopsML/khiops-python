######################################################################################
# Copyright (c) 2018 - 2022 Orange - All Rights Reserved                             #
# * This software is the confidential and proprietary information of Orange.         #
# * You shall not disclose such Restricted Information and shall use it only in      #
#   accordance with the terms of the license agreement you entered into with Orange  #
#   named the "Khiops - Python Library Evaluation License".                          #
# * Unauthorized copying of this file, via any medium is strictly prohibited.        #
# * See the "LICENSE.md" file for more details.                                      #
######################################################################################
"""Common pyKhiops utility functions and classes"""
import io
import json
import os
import platform
import warnings
from collections.abc import Iterable, Mapping, Sequence
from urllib.parse import urlparse

from pykhiops.core import filesystems as fs

##############
# Exceptions #
##############


class PyKhiopsJSONError(Exception):
    """Parsing error for Khiops-generated JSON files"""


class PyKhiopsRuntimeError(Exception):
    """Khiops execution related errors"""


class PyKhiopsEnvironmentError(Exception):
    """PyKhiops execution environment error

    Example: Khiops binary not found.
    """


############
# Messages #
############


def type_error_message(variable_name, variable, *target_types):
    """Formats a type error message

    Parameters
    ----------
    variable_name : str
        Name of the variable for whom the type error is signaled.
    variable : any
        Actual variable for whom the type error is signaled.
    target_types : list
        Expected types for ``variable``, either as a type or as a string.

    Returns
    -------
    str
        The type error message.

    """
    assert len(target_types) > 0, "At least one target type must be provided"
    assert all(
        isinstance(target_type, (type, str)) for target_type in target_types
    ), "All target types must be 'type' or 'str'"
    assert isinstance(variable_name, str), "'variable_name' must be 'str'"
    assert len(variable_name) > 0, "'variable_name' should not be empty"

    # Transform to 'type' the string arguments
    typed_target_types = []
    for target_type in target_types:
        if isinstance(target_type, str):
            typed_target_types.append(type(target_type, (), {}))
        else:
            typed_target_types.append(target_type)

    # Build the type error message
    if len(typed_target_types) == 1:
        target_type_str = f"'{typed_target_types[0].__name__}'"
    elif len(typed_target_types) == 2:
        target_type_str = (
            f"either '{typed_target_types[0].__name__}' "
            f"or '{typed_target_types[1].__name__}'"
        )
    else:
        target_types_str = " or ".join(
            f"'{target_type.__name__}'" for target_type in typed_target_types
        )
        target_type_str = f"one of {target_types_str}"

    if len(variable_name.strip().split(" ")) == 1:
        variable_name_str = f"'{variable_name}'"
    else:
        variable_name_str = variable_name

    return (
        f"{variable_name_str} type must be {target_type_str}, "
        f"not '{type(variable).__name__}'"
    )


def removal_message(removed_feature, since, replacement=None):
    """Formats a feature removal message"""
    message = f"'{removed_feature}' removed since {since}. "
    if replacement:
        message += f"Use '{replacement}'."
    else:
        message += "There is no replacement."
    return message


def renaming_message(renamed_feature, new_name, since):
    """Formats a feature renaming message"""
    return f"Ignoring '{renamed_feature}': renamed to '{new_name}' since {since}."


def invalid_keys_message(kwargs):
    """Formats an invalid keyword parameter message"""
    return f"Ignoring invalid parameter(s): {','.join(kwargs.keys())}."


def deprecation_message(
    deprecated_feature, deadline_version, replacement=None, quote=True
):
    """Formats a deprecation message"""
    if quote:
        message = f"'{deprecated_feature}' is deprecated "
    else:
        message = f"{deprecated_feature} is deprecated "
    message += f"and will be removed by version {deadline_version}."
    if replacement is not None:
        if quote:
            message += f" Prefer '{replacement}'."
        else:
            message += f" Prefer {replacement}."
    else:
        message += " There will be no replacement when removed."
    return message


###############
# Type checks #
###############


def is_string_like(test_object):
    """Returns True if a string is a valid pyKhiops string"""
    return isinstance(test_object, (str, bytes))


def is_list_like(list_like):
    """Returns True if an object is list-like

    An object is ``list-like`` if and only if inherits from `collections.abc.Sequence`
    and it is not `string-like <is_string_like>`
    """
    return isinstance(list_like, Sequence) and not is_string_like(list_like)


def is_dict_like(test_object):
    """Returns True if an object is dict-like

    An object is ``dict-like`` if and only if inherits from the
    `collections.abc.Mapping`.
    """
    return isinstance(test_object, Mapping)


def is_iterable(test_object):
    """Return True if a container object is iterable, but not string-like"""
    return isinstance(test_object, (Sequence, Iterable)) and not is_string_like(
        test_object
    )


#######################
# Khiops specific I/O #
#######################


def encode_file_path(file_path):
    """Encodes a file path

    This is custom path encoding for Khiops scenarios that is platform dependent. The
    encoding is done only if file_path is of type str.

    Parameters
    ----------
    file_path : str or bytes
        The path of a file.

    Returns
    -------
    `bytes`
        If ``file_path`` is str
            - In Windows : The path decoded to UTF-8 excepting the "ANSI" Unicode
              characters.
            - In Linux/Unix/Mac : The path decoded to UTF-8.
        If ``file_path`` is `bytes`:
            It just returns the input ``file_path``

    Raises
    ------
    `TypeError`
        If ``file_path`` is not str or bytes
    """
    # Check input type
    if not is_string_like(file_path):
        raise TypeError(type_error_message("file_path", file_path, str, bytes))

    # Return as-is if it is a byte sequence
    if not isinstance(file_path, str):
        assert isinstance(file_path, bytes)
        return file_path

    # Platform *nix: return UTF-8 encoded path
    if platform.system() != "Windows":
        return bytes(file_path, encoding="utf8")

    # Platform Windows:
    # - Return ANSI encoded chars if they over the 128-255 window
    # - Return UTF8 encoded chars otherwise
    decoded_bytes = bytearray()
    for char in file_path:
        if char in all_ansi_unicode_chars:
            decoded_bytes.extend(all_ansi_unicode_chars_to_ansi[char])
        else:
            decoded_bytes.extend(bytearray(char, "utf8"))
    return bytes(decoded_bytes)


def create_unambiguous_khiops_path(path):
    """Creates a path that is unambiguous for Khiops

    Khiops needs that a non absolute path starts with "." so not use the path of an
    internally saved state as reference point.

    For example: if we open the data table "/some/path/to/data.txt" and then set the
    results directory simply as "results" the effective location of the results
    directory will be "/some/path/to/results" instead of "$CWD/results". This behavior
    is a feature in Khiops but it is undesirable when using it as a library.

    This function returns a path so the library behave as expected: a path relative to
    the $CWD if it is a non absolute path.
    """
    # Check for string
    if not isinstance(path, (str, bytes)):
        raise TypeError(type_error_message("path", path, str, bytes))

    # Empty path returned as-is
    if not path:
        return path

    # Add a "." to a local path if necessary. It is *not* necessary when:
    # - `path` is an URI
    # - `path` is an absolute path
    # - `path` is a path starting with "."
    dot = "."
    empty = ""
    if isinstance(path, bytes):
        dot = bytes(dot, encoding="ascii")
        empty = bytes(empty, encoding="ascii")
    uri_info = urlparse(path, allow_fragments=False)
    if os.path.isabs(path) or path.startswith(dot) or uri_info.scheme != empty:
        return path
    else:
        return os.path.join(dot, path)


class KhiopsJSONObject:
    """Represents the contents of a Khiops JSON file

    Parameters
    ----------
    json_data : dict, optional
        Python dictionary representing the data of a Khiops JSON file. If None an empty
        it returns an empty object.

    Raises
    ------
    `PyKhiopsJSONError`
        If the JSON data is invalid.

    Attributes
    ----------
    tool : str
        Name of the Khiops tool that generated the file.
    version : str
        Version of the Khiops tool that generated the file.
    khiops_encoding : str, optional
        Custom encoding used by Khiops in the file. Valid values:
            - ``None`` : for backwards compatibility
            - "ascii": ASCII encoding
            - "ansi": ANSI encoding
            - "utf8": UTF-8 encoding
            - "mixed_ansi_utf8": Mixed characters from UTF-8 and ANSI but no collision.
            - "colliding_ansi_utf8" : Colliding characters from UTF-8 and ANSI.
    sub_tool : str, optional
        Identifies the tool that originated the JSON file. Used by tools of the Khiops
        family such as PataText or Enneade.
    """

    def __init__(self, json_data=None):
        """See class docstring"""
        # Initialize empty object attributes
        self.version = ""
        self.tool = ""
        self.sub_tool = None
        self.khiops_encoding = "utf8"
        self.ansi_chars = []
        self.colliding_utf8_chars = []

        # Initialize from json data
        if json_data is not None:
            # Check the type of json_data
            if not isinstance(json_data, dict):
                raise TypeError(type_error_message("json_data", json_data, dict))

            # Input check
            if "tool" not in json_data:
                raise PyKhiopsJSONError(
                    "Khiops JSON file does not have a 'tool' field "
                )
            if "version" not in json_data:
                raise PyKhiopsJSONError(
                    "Khiops JSON file does not have a 'version' field "
                )

            if "khiops_encoding" not in json_data:
                warnings.warn(
                    "Khiops JSON file does not have a 'khiops_encoding' field "
                    "(generated with Khiops older than 10.0.1?). "
                    "Exported dictionary/report files may be corrupted. ",
                    stacklevel=6,
                )
            elif json_data["khiops_encoding"] == "colliding_ansi_utf8":
                if "colliding_utf8_chars" in json_data:
                    colliding_chars_message = "Colliding chars: " + ", ".join(
                        json_data["colliding_utf8_chars"]
                    )
                else:
                    colliding_chars_message = ""
                warnings.warn(
                    "Khiops JSON file contains colliding characters. "
                    "Exported dictionary/report files may be unusable. "
                    f"{colliding_chars_message}",
                    stacklevel=6,
                )
            elif json_data["khiops_encoding"] not in [
                "ascii",
                "ansi",
                "utf8",
                "mixed_ansi_utf8",
            ]:
                raise PyKhiopsJSONError(
                    "Khiops JSON file khiops_encoding field value must be 'ascii', "
                    "'ansi', 'utf8', 'mixed_ansi_utf8' or 'colliding_ansi_utf8', "
                    f"""not '{json_data["khiops_encoding"]}'."""
                )

            # Initialize attributes from data
            self.tool = json_data["tool"]
            self.version = json_data["version"]
            self.sub_tool = json_data.get("subTool")
            self.khiops_encoding = json_data.get("khiops_encoding")
            if self.khiops_encoding == "colliding_ansi_utf8":
                self.ansi_chars = json_data["ansi_chars"]
                if "colliding_utf8_chars" in json_data:
                    self.colliding_utf8_chars = json_data["colliding_utf8_chars"]

    def create_output_file_writer(self, stream):
        """Creates an output file with the proper encoding settings

        Parameters
        ----------
        stream : `io.IOBase`
            An output stream object.

        Returns
        -------
        `.PyKhiopsOutputWriter`
            An output file object.
        """
        if self.khiops_encoding is None or self.khiops_encoding in ["ascii", "utf8"]:
            return PyKhiopsOutputWriter(stream)
        else:
            if self.khiops_encoding == "colliding_ansi_utf8":
                return PyKhiopsOutputWriter(
                    stream, force_ansi=True, ansi_unicode_chars=self.ansi_chars
                )
            else:
                return PyKhiopsOutputWriter(stream, force_ansi=True)

    def load_khiops_json_file(self, json_file_path):
        """Initializes the object from a Khiops JSON file

        Parameters
        ----------
        json_file_path : str
            Path of the Khiops JSON file.

        Raises
        ------
        `.PyKhiopsJSONError`
            If the file is an invalid JSON, Khiops JSON or if it is not UTF-8.
        """
        json_file_res = fs.create_resource(json_file_path)
        with io.BytesIO(json_file_res.read()) as json_file_stream:
            try:
                json_data = json.load(json_file_stream)
                first_load_failed = False
            except UnicodeDecodeError as error:
                warnings.warn(
                    "Khiops JSON file raised UnicodeDecodeError, "
                    "probably because the file is not encoded in UTF-8. "
                    "The file will be loaded with replacement characters on "
                    "decoding errors and this may generate problems downstream. "
                    "To avoid any problem, regenerate the file with Khiops 10.0 "
                    f"or newer.\nKhiops JSON file: {json_file_path}.\n"
                    f"UnicodeDecodeError message:\n{error}"
                )
                first_load_failed = True

        # Try a second time with flexible errors
        if first_load_failed:
            with io.StringIO(
                json_file_res.read().decode("utf8", errors="replace")
            ) as json_file_stream:
                json_data = json.load(json_file_stream)
        try:
            self.__init__(json_data)
        except PyKhiopsJSONError as error:
            raise PyKhiopsJSONError(
                f"Could not load Khiops JSON file: {json_file_path}"
            ) from error


class PyKhiopsOutputWriter:
    """Output writer with additional services to handle Khiops special encodings

    Parameters
    ----------
    stream : `io.IOBase`
        A writable output stream. Special text transformations in buffers inheriting
        from `io.TextIOBase` are ignored.
    force_ansi : bool, default False
        All output written will be transformed back ANSI characters in that range that
        were recoded to UTF-8.
    ansi_unicode_chars : list of str, optional
        A list of UTF-8 characters with equivalents in the ANSI 128-256 range which will
        be encoded back to ANSI when writing with ``force_ansi`` is ``True``. By default
        all UTF-8 equivalents of the ANSI 128-256 will be encoded back.
    """

    def __init__(self, stream, force_ansi=False, ansi_unicode_chars=None):
        # Set the output stream
        if isinstance(stream, io.IOBase):
            if not stream.writable():
                raise ValueError("'stream' must be writable")
            if isinstance(stream, io.TextIOBase):
                self.stream = stream.buffer
            else:
                self.stream = stream
        else:
            raise ValueError(type_error_message("stream", stream, io.IOBase))

        # Set the force_ansi parameter
        self.force_ansi = force_ansi
        if ansi_unicode_chars is None:
            self.ansi_unicode_chars = all_ansi_unicode_chars
        else:
            for char in ansi_unicode_chars:
                if char not in all_ansi_unicode_chars:
                    raise ValueError(
                        f"Unicode char '{char}' does not have an equivalent "
                        "in the 128-256 ANSI range"
                    )
            self.ansi_unicode_chars = ansi_unicode_chars

    def write(self, string):
        if not is_string_like(string):
            raise TypeError(type_error_message("string", string, "string-like"))

        if isinstance(string, str):
            if self.force_ansi:
                self._write_ansi(string)
            else:
                self.stream.write(bytes(string, "utf8"))
        else:
            self.stream.write(string)

    def writeln(self, string):
        self.write(string)
        self.write(os.linesep)

    def close(self):
        self.stream.close()

    def _write_ansi(self, string):
        for char in string:
            if char in self.ansi_unicode_chars:
                self.stream.write(all_ansi_unicode_chars_to_ansi[char])
            else:
                self.stream.write(bytes(char, "utf8"))


# Mapping ANSI -> Unicode as a list of 128 Unicode characters.
# Only non-ASCII characters are represented, that is the range 128-255.
all_ansi_unicode_chars = [
    "\u20AC",
    "\u0081",
    "\u201A",
    "\u0192",
    "\u201E",
    "\u2026",
    "\u2020",
    "\u2021",
    "\u02C6",
    "\u2030",
    "\u0160",
    "\u2039",
    "\u0152",
    "\u008D",
    "\u017D",
    "\u008F",
    "\u0090",
    "\u2018",
    "\u2019",
    "\u201C",
    "\u201D",
    "\u2022",
    "\u2013",
    "\u2014",
    "\u02DC",
    "\u2122",
    "\u0161",
    "\u203A",
    "\u0153",
    "\u009D",
    "\u017E",
    "\u0178",
    "\u00A0",
    "\u00A1",
    "\u00A2",
    "\u00A3",
    "\u00A4",
    "\u00A5",
    "\u00A6",
    "\u00A7",
    "\u00A8",
    "\u00A9",
    "\u00AA",
    "\u00AB",
    "\u00AC",
    "\u00AD",
    "\u00AE",
    "\u00AF",
    "\u00B0",
    "\u00B1",
    "\u00B2",
    "\u00B3",
    "\u00B4",
    "\u00B5",
    "\u00B6",
    "\u00B7",
    "\u00B8",
    "\u00B9",
    "\u00BA",
    "\u00BB",
    "\u00BC",
    "\u00BD",
    "\u00BE",
    "\u00BF",
    "\u00C0",
    "\u00C1",
    "\u00C2",
    "\u00C3",
    "\u00C4",
    "\u00C5",
    "\u00C6",
    "\u00C7",
    "\u00C8",
    "\u00C9",
    "\u00CA",
    "\u00CB",
    "\u00CC",
    "\u00CD",
    "\u00CE",
    "\u00CF",
    "\u00D0",
    "\u00D1",
    "\u00D2",
    "\u00D3",
    "\u00D4",
    "\u00D5",
    "\u00D6",
    "\u00D7",
    "\u00D8",
    "\u00D9",
    "\u00DA",
    "\u00DB",
    "\u00DC",
    "\u00DD",
    "\u00DE",
    "\u00DF",
    "\u00E0",
    "\u00E1",
    "\u00E2",
    "\u00E3",
    "\u00E4",
    "\u00E5",
    "\u00E6",
    "\u00E7",
    "\u00E8",
    "\u00E9",
    "\u00EA",
    "\u00EB",
    "\u00EC",
    "\u00ED",
    "\u00EE",
    "\u00EF",
    "\u00F0",
    "\u00F1",
    "\u00F2",
    "\u00F3",
    "\u00F4",
    "\u00F5",
    "\u00F6",
    "\u00F7",
    "\u00F8",
    "\u00F9",
    "\u00FA",
    "\u00FB",
    "\u00FC",
    "\u00FD",
    "\u00FE",
    "\u00FF",
]

# Map the ANSI 128-255 range coded in Unicode back to its ANSI encoding
all_ansi_unicode_chars_to_ansi = {
    char: bytes([128 + i]) for i, char in enumerate(all_ansi_unicode_chars)
}


##################
# Khiops Version #
##################


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

        # Check that :
        # - each part besides the last is numeric
        # - the last part is alphanumeric
        raw_parts = version.split(".")
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
