######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Runner to interact with a HTTP gateway to Khiops"""

import io
import json
import os
import ssl
import tempfile
import uuid
import warnings
from urllib.request import Request, urlopen

import khiops.core.internals.filesystems as fs
from khiops import core as kh
from khiops import get_compatible_khiops_version
from khiops.core.exceptions import KhiopsEnvironmentError
from khiops.core.internals.common import type_error_message
from khiops.core.internals.runner import KhiopsRunner
from khiops.core.internals.task import KhiopsTask


class KhiopsDockerRunner(KhiopsRunner):
    """Implementation of a dockerized remote Khiops runner

    Requires a running docker Khiops instance

    Parameters
    ----------

    url : str
        URL for the Docker Khiops server.
    shared_dir : str
        Location of the shared directory. May be an URL/URI.
    insecure : bool, default ``False``
        If ``True`` the target server an HTTPS URL connection requires a certificate.
    """

    def __init__(self, url, shared_dir, insecure=False):
        # Check the parameters
        if not isinstance(url, str):
            raise TypeError(type_error_message("url", url, str))
        if not isinstance(shared_dir, str):
            raise TypeError(type_error_message("shared_dir", shared_dir, str))
        elif fs.is_local_resource(shared_dir) and not fs.exists(shared_dir):
            raise KhiopsEnvironmentError(
                f"'shared_dir' does not exists. Path: {shared_dir}."
            )

        # Call parent constructor
        super().__init__()

        # Initialize this class specific members
        if not url.endswith("/"):
            url += "/"
        self._url = url
        self._insecure = insecure
        self.shared_dir = shared_dir
        # Define SSL context only for https
        if url.startswith("https"):
            # Compare URL host and certificate contents
            ctx = ssl._create_unverified_context()
            ctx.check_hostname = True
            ctx.verify_mode = ssl.CERT_REQUIRED
            # Ignore SSL certificate errors
            if self._insecure:
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
            self._ctx = ctx
        else:
            self._ctx = None

    def _initialize_khiops_version(self):
        self._khiops_version = get_compatible_khiops_version()

    def _get_khiops_version(self):
        if self._khiops_version is None:
            self._initialize_khiops_version()
        return self._khiops_version

    def _fetch(self, request):
        # HTTPS mode
        if self._ctx:
            with urlopen(request, context=self._ctx) as response_obj:
                response = response_obj.read().decode()
        # HTTP mode
        else:
            with urlopen(request) as response_obj:
                response = response_obj.read().decode()
        return response

    def _create_scenario_file(self, task):
        assert isinstance(task, KhiopsTask)
        return self._create_local_temp_file(f"{task.name}_", "._kh")

    def _run(
        self,
        tool_name,
        scenario_path,
        command_line_options,
        trace,
    ):
        # Check arguments
        if command_line_options.output_scenario_path:
            output_scenario_dir = fs.get_parent_path(
                command_line_options.output_scenario_path
            )
            if not fs.exists(output_scenario_dir):
                fs.make_dir(output_scenario_dir)
            fs.copy_from_local(command_line_options.output_scenario_path, scenario_path)
        if command_line_options.task_file_path is not None:
            warnings.warn(
                "Ignoring unsupported KhiopsDockerRunner parameter 'task_file_path'"
            )

        # Submit the job  for remote execution
        url = f"{self._url}v1/batch"
        if trace:
            print(f"Docker runner URL: {url}")
        tool_flag = "KHIOPS"
        if tool_name == "khiops_coclustering":
            tool_flag = "KHIOPS_COCLUSTERING"
        shared_scenario_path = self.create_temp_file("scenario", "._kh")
        fs.copy_from_local(shared_scenario_path, scenario_path)
        post_fields = {"tool": tool_flag, "scenario_path": shared_scenario_path}
        request = Request(url, data=bytes(json.dumps(post_fields), encoding="utf-8"))
        json_response = self._fetch(request)
        response = json.loads(json_response)

        # If trace is on: display call arguments
        if trace:
            print(f"Docker runner response: {json_response}")

        # Raise an exception if the job could not be sent
        if "error" in response:
            error_message = (
                f"{tool_name} execution failed with return code "
                f"{response['error']['code']}, "
                f"msg = {response['error']['message']}"
            )
            raise kh.KhiopsRuntimeError(
                f"KhiopsDockerRunner could not send job "
                f"to server. Error:\n{error_message}"
            )

        # Repeatedly query operation status, waiting until it is 'done'
        job_id = response["name"]
        if trace:
            print(f"KhiopsDockerRunner job id: {job_id}")
        while not bool(response["done"]):
            url = f"{self._url}v1/operation/{job_id}:wait?timeout=10s"
            request = Request(url)
            json_response = self._fetch(request)
            response = json.loads(json_response)
            if trace:
                print(f"Docker runner fetch job status: {json_response}")

        # Once done obtain the Khiops return code, stderr
        # The stdout ("output" field) is written to the specified log file
        # Then report (warn or exception) any errors
        return_code = response["response"]["status"]
        stderr = response["response"]["error"]
        with io.StringIO() as log_file_stream:
            log_file_stream.write(response["response"]["output"])
            fs.write(
                command_line_options.log_file_path,
                log_file_stream.getvalue().encode("utf8", errors="replace"),
            )

        # Delete the finished operation from the server
        url = f"{self._url}v1/operation/{job_id}:delete"
        request = Request(url, method="DELETE")
        json_response = self._fetch(request)
        if trace:
            print(f"Docker runner delete job: {json_response}")

        # We return empty stdout because the 'output' field of the JSON response are the
        # contents of the log
        return return_code, "", stderr

    def _create_local_temp_file(self, prefix, suffix):
        """Creates a unique temporary file in the runner's root temporary directory

        Parameters
        ----------
        prefix : str
            Prefix for the file's name.

        suffix : str
            Suffix for the file's name.

        Returns
        -------
        str
            The path of the created file.
        """
        tmp_file_fd, tmp_file_path = tempfile.mkstemp(
            prefix=prefix, suffix=suffix, dir=self.root_temp_dir
        )
        os.close(tmp_file_fd)
        return tmp_file_path

    # Override methods from base class
    def create_temp_file(self, prefix, suffix):
        """Creates a unique temporary file name in the runner's shared directory
        The actual file is not created on disk.

        Parameters
        ----------
        prefix : str
            Prefix for the file's name.

        suffix : str
            Suffix for the file's name.

        Returns
        -------
        str
            The path of the file to be created.
        """
        return fs.get_child_path(self.shared_dir, f"{prefix}{uuid.uuid4()}{suffix}")

    def create_temp_dir(self, prefix):
        """Creates a unique directory in the runner's shared directory

        Parameters
        ----------
        prefix : str
            Prefix for the directory's name.

        Returns
        -------
        str
            The path of the created directory.
        """
        tmp_dir_path = fs.get_child_path(
            self.shared_dir, f"{prefix}{uuid.uuid4()}{os.path.sep}"
        )
        fs.make_dir(tmp_dir_path)
        return tmp_dir_path
