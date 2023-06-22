######################################################################################
# Copyright (c) 2018 - 2022 Orange - All Rights Reserved                             #
# * This software is the confidential and proprietary information of Orange.         #
# * You shall not disclose such Restricted Information and shall use it only in      #
#   accordance with the terms of the license agreement you entered into with Orange  #
#   named the "Khiops - Python Library Evaluation License".                          #
# * Unauthorized copying of this file, via any medium is strictly prohibited.        #
# * See the "LICENSE.md" file for more details.                                      #
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

from pykhiops import core as pk
from pykhiops.core import filesystems as fs
from pykhiops.core.common import KhiopsVersion
from pykhiops.core.runner import PyKhiopsRunner


class PyKhiopsDockerRunner(PyKhiopsRunner):
    """Implementation of a dockerized remote Khiops runner

    Requires a running docker Khiops instance
    """

    def __init__(self, url, shared_dir, insecure=False):
        # Parent constructor
        super().__init__()

        # Initialize this class specific members
        if not url.endswith("/"):
            url += "/"
        self._url = url
        self._insecure = insecure
        self.shared_dir = shared_dir
        if self.shared_dir is None or not fs.create_resource(self.shared_dir).exists():
            raise ValueError(
                "'shared_dir' parameter is required to connect with the Khiops service"
            )

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
        self._khiops_version = KhiopsVersion("10.1")

    def _fetch(self, request):
        # HTTPS mode
        response = None
        if self._ctx:
            response = urlopen(request, context=self._ctx).read().decode()
        # HTTP mode
        else:
            response = urlopen(request).read().decode()
        return response

    def _create_scenario_file(self, scenario, force_ansi_scenario=False):
        # If there are search/replace keywords use them to create the execution scenario
        scenario_path = self._create_local_temp_file(
            f"{scenario.template_name}_", scenario.template_ext
        )
        scenario.write_file(scenario_path, force_ansi=force_ansi_scenario)
        return scenario_path

    def _run(
        self,
        tool_name,
        scenario_path,
        batch_mode,
        log_file_path,
        output_scenario_path,
        task_file_path,
        trace,
    ):
        # Check arguments
        if not batch_mode:
            warnings.warn(
                "Ignoring unsupported PyKhiopsDockerRunner parameter 'batch_mode'",
                stacklevel=3,
            )
        if output_scenario_path:
            output_scenario_file = fs.create_resource(output_scenario_path)
            output_scenario_dir = output_scenario_file.create_parent()
            if not output_scenario_dir.exists():
                output_scenario_dir.make_dir()
            output_scenario_file.copy_from_local(scenario_path)
        if task_file_path:
            warnings.warn(
                "Ignoring unsupported PyKhiopsDockerRunner parameter 'task_file_path'."
            )

        # Submit the job  for remote execution
        url = f"{self._url}v1/batch"
        if trace:
            print(f"Docker runner URL: {url}")
        tool_flag = "KHIOPS"
        if tool_name == "khiops_coclustering":
            tool_flag = "KHIOPS_COCLUSTERING"
        shared_scenario_path = self.create_temp_file("scenario", "._kh")
        scenario_file_res = fs.create_resource(shared_scenario_path)
        scenario_file_res.copy_from_local(scenario_path)
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
            raise pk.PyKhiopsRuntimeError(
                f"PyKhiopsDockerRunner could not send job "
                f"to server. Error:\n{error_message}"
            )

        # Repeatedly query operation status, waiting until it is 'done'
        job_id = response["name"]
        if trace:
            print(f"PyKhiopsDockerRunner job id: {job_id}")
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
        log_file_res = fs.create_resource(log_file_path)
        with io.StringIO() as log_file_stream:
            log_file_stream.write(response["response"]["output"])
            log_file_res.write(
                log_file_stream.getvalue().encode("utf8", errors="replace")
            )

        # Delete the finished operation from the server
        url = f"{self._url}v1/operation/{job_id}:delete"
        request = Request(url, method="DELETE")
        json_response = self._fetch(request)
        response = json.loads(json_response)
        if trace:
            print(f"Docker runner delete job: {json_response}")

        return return_code, stderr

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
        shared_dir_res = fs.create_resource(self.shared_dir)
        tmp_file_path = shared_dir_res.create_child(
            f"{prefix}{uuid.uuid4()}{suffix}"
        ).uri
        return tmp_file_path

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
        shared_dir_res = fs.create_resource(self.shared_dir)
        tmp_dir_path_res = shared_dir_res.create_child(
            f"{prefix}{uuid.uuid4()}{os.path.sep}"
        )
        tmp_dir_path_res.make_dir()
        return tmp_dir_path_res.uri
