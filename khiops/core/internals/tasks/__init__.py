######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Tasks module

This is a "read-only" module that registers all the task specs at __init__ time.
"""
import json
import os
import pathlib
from pathlib import Path

from khiops.core.internals.task import get_task_registry

base_folder = os.path.join(
    Path(__file__).parent.parent,
    "khiops-api-specifications",
)

task_registry = get_task_registry()

# process all the declarations and templates
declarations_folder = os.path.join(base_folder, "task_declarations")
templates_folder = os.path.join(base_folder, "task_templates")
for declaration_name in os.listdir(declarations_folder):
    function_name = pathlib.Path(declaration_name).stem
    print(f"Loading declaration and template for {function_name}")
    with (
        open(
            os.path.join(declarations_folder, declaration_name), "r"
        ) as declaration_file,
        open(
            os.path.join(templates_folder, f"{function_name}.kht"), "r"
        ) as template_file,
    ):
        json_data = json.load(declaration_file)
        task_registry.register_task(json_data["name"], json_data, template_file.read())
