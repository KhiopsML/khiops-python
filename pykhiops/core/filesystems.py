######################################################################################
# Copyright (c) 2018 - 2022 Orange - All Rights Reserved                             #
# * This software is the confidential and proprietary information of Orange.         #
# * You shall not disclose such Restricted Information and shall use it only in      #
#   accordance with the terms of the license agreement you entered into with Orange  #
#   named the "Khiops - Python Library Evaluation License".                          #
# * Unauthorized copying of this file, via any medium is strictly prohibited.        #
# * See the "LICENSE.md" file for more details.                                      #
######################################################################################
"""Classes to interact with local and remote filesystems"""

import json
import os
import platform
import shutil
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from urllib.parse import urlparse

# Disable pylint UPPER_CASE convention: these are module variables not constants
# pylint: disable=invalid-name

# Import boto3 if available
# Delay the raising of an ImportError to an instantation of a AmazonS3Resource
try:
    import boto3
    import boto3.session
    from botocore.exceptions import ClientError

    boto3_import_error = None
except ImportError as import_error:
    boto3_import_error = import_error

# Import google.could if available
# Delay the raising of an ImportError to an instantation of a GoogleCloudStorageResource
try:
    from google.cloud import storage

    gcs_import_error = None
except ImportError as import_error:
    gcs_import_error = import_error

# pylint: enable=invalid-name


def is_local_resource(uri_or_path):
    r"""Checks if a URI or path is effectively a local path

    .. note::

        An URI with scheme of size 1 will be considered a local path. This is to take
        into account Windows paths such as ``C:\Some\Windows\Path``.

    Returns
    -------
    `bool`
        `True` if a URI refers to a local path
    """
    uri_info = urlparse(uri_or_path, allow_fragments=False)
    return len(uri_info.scheme) <= 1 or uri_info.scheme == "file"


def create_resource(uri_or_path):
    """Factory method to create a FilesystemResource from an URI

    Parameters
    ----------
    uri : str
        The resource's URI . Supported protocols/schemes:

        - ``file`` or empty: Local filesystem resource
        - ``s3``: Amazon S3 resource
        - ``gs``: Google Cloud Storage resource

    Returns
    -------
    `FilesystemResource`
        The URI resource object, its class depends on the URI.
    """
    uri_info = urlparse(uri_or_path, allow_fragments=False)
    if uri_info.scheme == "s3":
        return AmazonS3Resource(uri_or_path)
    elif uri_info.scheme == "gs":
        return GoogleCloudStorageResource(uri_or_path)
    elif is_local_resource(uri_or_path):
        return LocalFilesystemResource(uri_or_path)
    else:
        raise ValueError(f"Unsupported URI scheme {uri_info.scheme}")


def parent_path(path):
    r"""Returns the parent of the specified path

    Notes
    -----
    This function always return a posix path ("/" as separator). For example for the
    windows path::

        C:\Program Files\khiops

    this method returns::

        C:/Program Files
    """
    return Path(path).parent.as_posix()


def parent_uri_info(uri_info):
    """Creates the parent for the input URI info

    Parameters
    ----------
    uri_info : `urllib.parse.ParseResult`
        URI info structure (output of `urllib.parse.urlparse`)

    Returns
    -------
    `urllib.parse.ParseResult`
        URI info structure for the parent URI

    """
    return uri_info._replace(path=parent_path(uri_info.path))


def child_path(path, child_name):
    r"""Creates a path with a child appended

    Notes
    -----
    This function always return a posix path ("/" as separator). For example for the
    windows path and child name::

        parent: C:\Program Files
        child:  khiops

    this method returns::
        C:/Program Files/khiops
    """
    return Path(path).joinpath(child_name).as_posix()


def child_uri_info(uri_info, child_name):
    r"""Creates a URI info with a path with child appended

    Parameters
    ----------
    uri_info : `urllib.parse.ParseResult`
        URI info structure (output of `urllib.parse.urlparse`)

    child_name : str
        Name of the new childe node

    Returns
    -------
    `urllib.parse.ParseResult`
        URI info structure for the child URI
    """
    return uri_info._replace(path=child_path(uri_info.path, child_name))


class FilesystemResource(ABC):
    """Abstract Filesystem Resource"""

    def __init__(self, uri):
        self.uri = uri
        self.uri_info = urlparse(uri, allow_fragments=False)

    @abstractmethod
    def read(self, size=None):
        """Reads the entire file unless size parameter is specified

        Parameters
        ----------
        size : int, optional
            Number of bytes to read

        Returns
        -------
        `bytes`
            A buffer containing the contents read
        """

    @abstractmethod
    def write(self, data):
        """Writes data in binary mode the provided data as a complete file

        Parameters
        ----------
        data : str or `bytes`
            The data to be written

        Returns
        -------
        `bool`
            True if the write succeeded
        """

    @abstractmethod
    def exists(self):
        """Checks the existence of the resource

        Returns
        -------
        `bool`
            True if the resource exists
        """

    @abstractmethod
    def remove(self):
        """Removes the resource

        Returns
        -------
        `bool`
            `True` if the operation succeeded
        """

    @abstractmethod
    def copy_from_local(self, local_path):
        """Copies the content of a local file to the resource

        Parameters
        ----------
        local_path : str
            The path of the local file to be copied

        Returns
        -------
        `bool`
            If the operation succeeded
        """

    @abstractmethod
    def copy_to_local(self, local_path):
        """Copies the content of the resource to a local path

        Parameters
        ----------
        local_path : str
            The path of the local file to be copied

        Returns
        -------
        `bool`
            `True` if the operation succeeded
        """

    @abstractmethod
    def list_dir(self):
        """List a directory resource contents

        Returns
        -------
        list
            A list of paths or URIs
        """

    @abstractmethod
    def make_dir(self):
        """Creates a directory at the resource's path

        Returns
        -------
        bool
          True if the operation succeeded
        """

    @abstractmethod
    def create_child(self, file_name):
        """Creates a resource representing a child of this instance

        Returns
        -------
        `FilesystemResource`
            The specific resource type is that of the caller
        """

    @abstractmethod
    def create_parent(self, file_name):
        """Creates a resource representing the parent of this instance

        Returns
        -------
        `FilesystemResource`
            The specific resource type is that of the caller
        """


class LocalFilesystemResource(FilesystemResource):
    """A local filesystem resource"""

    def __init__(self, uri):
        super().__init__(uri)

        # Obtain the local from the URI
        # Case where the scheme is in fact a windows drive
        #   => Build the proper path with drive
        if (
            len(self.uri_info.scheme) == 1
            and self.uri_info.scheme.isalpha()
            and platform.system() == "Windows"
        ):
            self.path = os.path.join(f"{self.uri_info.scheme}:\\", self.uri_info.path)
        # Case of the "file" scheme
        elif self.uri_info.scheme == "file":
            # If invalid second colon in path (eg. "/C:/Users"):
            #   => drive of a windows path
            #   => eliminate initial slash
            if self.uri_info.path[2] == ":":
                self.path = self.uri_info.path[1:]
            # Otherwise copy as-is
            else:
                self.path = self.uri_info.path
        # Case of the empty scheme
        #  => copy as-is
        elif not self.uri_info.scheme:
            self.path = self.uri_info.path
        # Invalid scheme
        else:
            raise ValueError(f"Invalid local path or URI: {self.uri}")

        # Normalize to platform: This ensures a single separator
        self.path = os.path.normpath(self.path)

    def read(self, size=None):
        with open(self.path, "rb") as local_file:
            if size is None:
                return local_file.read()
            else:
                return local_file.read(size)

    def write(self, data):
        with open(self.path, "wb") as output_file:
            output_file.write(data)
        return True

    def exists(self):
        return os.path.exists(self.path)

    def remove(self):
        if os.path.isdir(self.path):
            os.rmdir(self.path)
        else:
            os.remove(self.path)
        return True

    def copy_from_local(self, local_path):
        directory = os.path.dirname(self.path)
        if not os.path.isdir(directory):
            os.makedirs(directory)
        return shutil.copy(local_path, self.path)

    def copy_to_local(self, local_path):
        return shutil.copy(self.path, local_path)

    def list_dir(self):
        return os.listdir(self.path)

    def make_dir(self):
        os.makedirs(self.path)
        return True

    def create_child(self, file_name):
        return create_resource(os.path.join(self.path, file_name))

    def create_parent(self):
        return create_resource(parent_path(self.path))


class GoogleCloudStorageResource(FilesystemResource):
    """Google Cloud Storage Resource

    By default it reads the configuration from standard location.
    """

    def __init__(self, uri):
        # Stop initialization if google.cloud module is not available
        if gcs_import_error is not None:
            warnings.warn(
                "Could not import google.cloud module. "
                "Make sure you have installed the google-cloud-storage package to "
                "access Google Cloud Storage files."
            )
            raise gcs_import_error
        super().__init__(uri)
        self.gcs_client = storage.Client()

    def read(self, size=None):
        end = None
        if size is not None:
            end = size - 1
        return (
            self.gcs_client.bucket(self.uri_info.netloc)
            .blob(self.uri_info.path[1:])
            .download_as_bytes(end=end, checksum=None)
        )

    def write(self, data):
        self.gcs_client.bucket(self.uri_info.netloc).blob(
            self.uri_info.path[1:]
        ).upload_from_string(data)
        return True

    def exists(self):
        return (
            self.gcs_client.bucket(self.uri_info.netloc)
            .blob(self.uri_info.path[1:])
            .exists()
        )

    def remove(self):
        return (
            self.gcs_client.bucket(self.uri_info.netloc)
            .blob(self.uri_info.path[1:])
            .delete()
        )

    def copy_from_local(self, local_path):
        return (
            self.gcs_client.bucket(self.uri_info.netloc)
            .blob(self.uri_info.path[1:])
            .upload_from_filename(local_path)
        )

    def copy_to_local(self, local_path):
        return (
            self.gcs_client.bucket(self.uri_info.netloc)
            .blob(self.uri_info.path[1:])
            .download_to_filename(local_path)
        )

    def list_dir(self):
        # Add an extra slash to the path to treat it as a folder
        dir_path = self.uri_info.path
        if not dir_path.endswith("/"):
            dir_path += "/"

        # Retrieve the file list via the GCS client
        blobs = self.gcs_client.list_blobs(
            self.uri_info.netloc, prefix=dir_path[1:], delimiter="/"
        )
        paths = []
        for blob in blobs:
            if not blob.name.endswith("/"):
                paths.append(os.path.basename(blob.name))

        return paths

    def make_dir(self):
        warnings.warn(
            "'make_dir' is a non-operation on Google Cloud Storage. "
            "See the documentation at https://cloud.google.com/storage/docs/folders"
        )
        return True

    def create_child(self, file_name):
        return create_resource(child_uri_info(self.uri_info, file_name).geturl())

    def create_parent(self):
        return create_resource(parent_uri_info(self.uri_info).geturl())


# Avoid pylint complaining on dynamic class returned by boto3 API
# pylint: disable=no-member
class AmazonS3Resource(FilesystemResource):
    """Amazon Simple Storage Service (S3) Resource

    The default configuration and credentials are read from the paths

    - ``~/.aws/configuration``
    - ``~/.aws/credentials``

    The location of the configuration and credentials files may be overridden using
    the following environment variables:

    - ``AWS_CONFIG_FILE``: location of the configuration file
    - ``AWS_SHARED_CREDENTIALS_FILE``: location of the credentials file

    If no configuration/credentials files are usable, Amazon SDK defaults apply.
    Individual settings such as endpoint URL or region can be used to override any of
    the available settings.

    Other relevant environment variables::

    - AWS_S3_ENDPOINT_URL: sets the service endpoint URL
    - AWS_DEFAULT_REGION: sets the region to send requests to

    .. note::
        Operations with the s3 client are only verified by checking that the HTTP
        response code is in the 200 range.

    """

    def __init__(self, uri):
        # Stop initialization if boto3 could not be imported
        if boto3_import_error is not None:
            warnings.warn(
                "Could not import boto3 python library, "
                "make sure you it installed to access S3 files."
            )
            raise boto3_import_error

        # Initialization
        super().__init__(uri)
        s3_config = boto3.session.Session()._session.get_scoped_config()
        endpoint_url = None
        region_name = None
        if s3_config.get("s3"):
            endpoint_url = s3_config["s3"].get("endpoint_url")
            region_name = s3_config["s3"].get("region")
        endpoint_url = os.getenv("AWS_S3_ENDPOINT_URL", endpoint_url)
        region_name = os.getenv("AWS_DEFAULT_REGION", region_name)
        self.s3_client = boto3.resource(
            service_name="s3",
            endpoint_url=endpoint_url,
            region_name=region_name,
            use_ssl=True,
        )
        self.s3_object = self.s3_client.Object(
            self.uri_info.netloc, self.uri_info.path[1:]
        )

    def read(self, size=None):
        # Set the number of bytes to read if specified
        get_kwargs = {}
        if size is not None:
            get_kwargs = {"Range": f"bytes=0-{size}"}

        # Execute the request
        response = self.s3_object.get(**get_kwargs)
        read_ok = 200 <= response["ResponseMetadata"]["HTTPStatusCode"] <= 299

        # Read the contents if ok
        if read_ok:
            with response["Body"] as body:
                return body.read()
        else:
            raise ValueError(
                f"Failed to read of S3 object {self.uri}: {json.dumps(response)}"
            )

    def write(self, data):
        response = self.s3_object.put(Body=data)
        write_ok = 200 <= response["ResponseMetadata"]["HTTPStatusCode"] <= 299
        return write_ok

    def exists(self):
        # Test the existence by loading the object
        try:
            self.s3_object.load()
            return True
        # The object does not exist
        except ClientError as error:
            if error.response["Error"]["Code"] == "404":
                return False
            # Something else has gone wrong.
            else:
                raise

    def remove(self):
        # Execute remove request
        response = self.s3_object.delete()

        # Check exit status
        remove_ok = 200 <= response["ResponseMetadata"]["HTTPStatusCode"] <= 299
        if not remove_ok:
            warnings.warn(f"S3 remove failed {self.uri}: {json.dumps(response)}")
        return remove_ok

    def copy_from_local(self, local_path):
        response = self.s3_client.Bucket(self.uri_info.netloc).upload_file(
            local_path, self.uri_info.path[1:]
        )
        copy_ok = 200 <= response["ResponseMetadata"]["HTTPStatusCode"] <= 299
        if not copy_ok:
            warnings.warn(f"S3 upload failed {self.uri}: {json.dumps(response)}")
        return copy_ok

    def copy_to_local(self, local_path):
        response = self.s3_client.Bucket(self.uri_info.netloc).download_file(
            local_path, self.uri_info.path[1:]
        )
        copy_ok = 200 <= response["ResponseMetadata"]["HTTPStatusCode"] <= 299
        if not copy_ok:
            warnings.warn(f"S3 download failed {self.uri}: {json.dumps(response)}")
        return copy_ok

    def list_dir(self):
        # Add an extra slash to the path to treat it as a folder
        dir_path = self.uri_info.path
        if not dir_path.endswith("/"):
            dir_path += "/"

        # Retrieve the file list via the S3 client
        bucket = self.s3_client.Bucket(self.uri_info.netloc)
        bucket_objects = bucket.objects.filter(Prefix=dir_path[1:], Delimiter="/")

        return [os.path.basename(bucket_object.key) for bucket_object in bucket_objects]

    def make_dir(self):
        warnings.warn(
            "'make_dir' is a non-operation on Amazon S3. "
            "See the documentation at "
            "https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-folders.html"
        )
        return True

    def create_child(self, file_name):
        return create_resource(child_uri_info(self.uri_info, file_name).geturl())

    def create_parent(self):
        return create_resource(parent_uri_info(self.uri_info).geturl())


# pylint: enable=no-member
