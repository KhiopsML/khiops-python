######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
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
from khiops.core.internals.common import (
    deprecation_message,
    is_string_like,
    type_error_message,
)


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


def flexible_json_load(json_file_path):
    """Loads flexibly a JSON file

    First it tries a vanilla read, then if that fails it warns and then loads the files
    replacing the errors.

    Parameters
    ----------
    json_file_path : str
        Path of the Khiops JSON file.

    Returns
    -------
    dict
        The in-memory representation of the JSON file.
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

    return json_data


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
    json_key_sort_spec : dict, optional
        Dictionary that specifies the order of the keys in the Khiops JSON report.
        Its values are `None`, except when they are dictionaries themselves.

        .. note::
            This is a class attribute that can be set in subclasses, to specify
            a key order when serializing the report in a JSON file, via the
            ``write_khiops_json_file`` method.

    json_data : dict
        Python dictionary extracted from the Khiops JSON report file.
        **Deprecated** will be removed in Khiops 12.
    """

    # Set default JSON key sort specification attribute
    # Can be set in classes that specialize this class
    json_key_sort_spec = None

    def __init__(self, json_data=None):
        """See class docstring"""
        # Check the type of the json_key_sort_spec class attribute
        assert self.json_key_sort_spec is None or isinstance(
            self.json_key_sort_spec, dict
        ), type_error_message("key_sort_spec", self.json_key_sort_spec, dict)

        # Check the type of json_data
        if json_data is not None and not isinstance(json_data, dict):
            raise TypeError(type_error_message("json_data", json_data, dict))

        # Input check
        if json_data is None:
            json_data = {}
        else:
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
                    "Khiops JSON file 'khiops_encoding' field value must be 'ascii', "
                    "'ansi', 'utf8', 'mixed_ansi_utf8' or 'colliding_ansi_utf8', "
                    f"""not '{json_data["khiops_encoding"]}'."""
                )

        # Initialize attributes from data
        # `tool` and `version` need to be strings by default, so that they can
        # be written
        self.tool = json_data.get("tool", "")
        self.version = json_data.get("version", "")
        self.sub_tool = json_data.get("subTool")

        # Obtain encoding fields
        self.khiops_encoding = json_data.get("khiops_encoding")
        if self.khiops_encoding is not None:
            self.ansi_chars = json_data.get("ansi_chars")
            self.colliding_utf8_chars = json_data.get("colliding_utf8_chars")
        # To support Khiops < 10.0.1
        else:
            self.ansi_chars = []
            self.colliding_utf8_chars = []

        # Store a copy of the data to be able to write copies of it
        self._json_data = copy.deepcopy(json_data)

    @property
    def json_data(self):
        warnings.warn(deprecation_message("'json_data'", "12.0.0", quote=False))
        return self._json_data

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

    def to_dict(self):
        """Serialize object instance to the Khiops JSON format"""
        report = {
            "tool": self.tool,
            "version": self.version,
            "khiops_encoding": self.khiops_encoding,
        }
        if self.ansi_chars is not None:
            report["ansi_chars"] = self.ansi_chars
        if self.colliding_utf8_chars is not None:
            report["colliding_utf8_chars"] = self.colliding_utf8_chars
        if self.sub_tool is not None:
            report["subTool"] = self.sub_tool
        return report

    def _json_key_sort_by_spec(self, jdict, key_sort_spec=None):
        # json_key_sort_spec must be set before using this method
        assert self.json_key_sort_spec is not None

        # Handle the base case with non-None key_sort_spec
        sorted_jdict = {}
        if key_sort_spec is None:
            key_sort_spec = self.json_key_sort_spec

        # Iterate over the current fields and recurse if necessary
        for spec_key, spec_value in key_sort_spec.items():
            if not (spec_value is None or isinstance(spec_value, (dict, list))):
                raise ValueError(
                    type_error_message(
                        "specification value",
                        spec_value,
                        "'None' or dict or list",
                    )
                )
            if spec_key in jdict:
                json_value = jdict[spec_key]

                # If json_value is not a dict, then:
                # - if not list-like, then add it as such to the output dict
                # - else, iterate on the list-like value
                # else, recurse on the dict structure
                if not isinstance(json_value, dict):
                    if not isinstance(json_value, list):
                        sorted_jdict[spec_key] = json_value
                    else:
                        sorted_jdict[spec_key] = []
                        for json_el in json_value:
                            if not isinstance(json_el, dict):
                                sorted_jdict[spec_key].append(json_el)
                            else:
                                if isinstance(spec_value, list):
                                    sorted_jdict[spec_key].append(
                                        self._json_key_sort_by_spec(
                                            json_el, key_sort_spec=spec_value[0]
                                        )
                                    )
                else:
                    sorted_jdict[spec_key] = self._json_key_sort_by_spec(
                        json_value, key_sort_spec=spec_value
                    )
        return sorted_jdict

    def write_khiops_json_file(self, json_file_path, _ensure_ascii=False):
        """Write the JSON data of the object to a Khiops JSON file

        The JSON keys are sorted according to the
        ``KhiopsJSONObject.json_key_sort_spec`` class attribute, if set.
        Otherwise, the JSON keys are not sorted.

        Parameters
        ----------
        json_file_path : str
            Path to the Khiops JSON file.
        _ensure_ascii : bool, default False
            If True, then non-ASCII characters in the report are escaped. Otherwise,
            they are dumped as-is.
        """
        # Serialize JSON data to string
        # Do not escape non-ASCII Unicode characters
        json_dict = self.to_dict()
        if self.json_key_sort_spec is not None:
            json_dict = self._json_key_sort_by_spec(json_dict)
            json_string = json.dumps(json_dict, indent=4, ensure_ascii=_ensure_ascii)
        else:
            json_string = json.dumps(json_dict, indent=4, ensure_ascii=_ensure_ascii)
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
