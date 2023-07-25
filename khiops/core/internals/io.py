######################################################################################
# Copyright (c) 2023 Orange. All rights reserved.                                    #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Classes to handle Khiops specific I/O"""
import copy
import io
import json
import os
import platform
import warnings

import khiops.core.internals.filesystems as fs
from khiops.core.exceptions import KhiopsJSONError
from khiops.core.internals.common import is_string_like, type_error_message


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


class KhiopsJSONObject:
    """Represents the contents of a Khiops JSON file

    Parameters
    ----------
    json_data : dict, optional
        Python dictionary representing the data of a Khiops JSON file. If None an empty
        it returns an empty object.

    Raises
    ------
    `~.KhiopsJSONError`
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
        self.json_data = None

        # Initialize from json data
        if json_data is not None:
            # Check the type of json_data
            if not isinstance(json_data, dict):
                raise TypeError(type_error_message("json_data", json_data, dict))

            # Input check
            if "tool" not in json_data:
                raise KhiopsJSONError("Khiops JSON file does not have a 'tool' field")
            if "version" not in json_data:
                raise KhiopsJSONError(
                    "Khiops JSON file does not have a 'version' field"
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
                raise KhiopsJSONError(
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

            # Store a copy of the data to be able to write copies of it
            self.json_data = copy.deepcopy(json_data)

    def create_output_file_writer(self, stream):
        """Creates an output file with the proper encoding settings

        Parameters
        ----------
        stream : `io.IOBase`
            An output stream object.

        Returns
        -------
        `.KhiopsOutputWriter`
            An output file object.
        """
        if self.khiops_encoding is None or self.khiops_encoding in ["ascii", "utf8"]:
            return KhiopsOutputWriter(stream)
        else:
            if self.khiops_encoding == "colliding_ansi_utf8":
                return KhiopsOutputWriter(
                    stream, force_ansi=True, ansi_unicode_chars=self.ansi_chars
                )
            else:
                return KhiopsOutputWriter(stream, force_ansi=True)

    def load_khiops_json_file(self, json_file_path):
        """Initializes the object from a Khiops JSON file

        Parameters
        ----------
        json_file_path : str
            Path of the Khiops JSON file.

        Raises
        ------
        `.KhiopsJSONError`
            If the file is an invalid JSON, Khiops JSON or if it is not UTF-8.
        """
        with io.BytesIO(fs.read(json_file_path)) as json_file_stream:
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
                fs.read(json_file_path).decode("utf8", errors="replace")
            ) as json_file_stream:
                json_data = json.load(json_file_stream)
        try:
            self.__init__(json_data)
        except KhiopsJSONError as error:
            raise KhiopsJSONError(
                f"Could not load Khiops JSON file: {json_file_path}"
            ) from error

    def write_khiops_json_file(self, json_file_path):
        """Write the JSON data of the object to a Khiops JSON file

        Parameters
        ----------
        json_file_path : str
            Path to the Khiops JSON file.
        """
        if self.json_data is not None:
            # Serialize JSON data to string
            # Do not escape non-ASCII Unicode characters
            json_string = json.dumps(self.json_data, ensure_ascii=False)
            with io.BytesIO() as json_stream:
                writer = self.create_output_file_writer(json_stream)
                writer.write(json_string)
                fs.write(uri_or_path=json_file_path, data=json_stream.getvalue())


class KhiopsOutputWriter:
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
