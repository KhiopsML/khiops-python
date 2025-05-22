######################################################################################
# Copyright (c) 2023-2025 Orange. All rights reserved.                               #
# This software is distributed under the BSD 3-Clause-clear License, the text of     #
# which is available at https://spdx.org/licenses/BSD-3-Clause-Clear.html or         #
# see the "LICENSE.md" file for more details.                                        #
######################################################################################
"""Tasks module

This is a "read-only" module that registers all the task definition at __init__ time.
"""
from khiops.core.internals.task import KhiopsTask, get_task_registry
from khiops.core.internals.tasks import (
    build_deployed_dictionary,
    build_dictionary_from_data_table,
    check_database,
    deploy_model,
    detect_data_table_format,
    detect_data_table_format_with_dictionary,
    evaluate_predictor,
    export_dictionary_as_json,
    extract_clusters,
    extract_keys_from_data_table,
    interpret_predictor,
    prepare_coclustering_deployment,
    reinforce_predictor,
    simplify_coclustering,
    sort_data_table,
    train_coclustering,
    train_instance_variable_coclustering,
    train_predictor,
    train_recoder,
)

# Register the tasks in each module
task_modules = [
    build_deployed_dictionary,
    build_dictionary_from_data_table,
    check_database,
    deploy_model,
    detect_data_table_format,
    detect_data_table_format_with_dictionary,
    evaluate_predictor,
    export_dictionary_as_json,
    extract_clusters,
    extract_keys_from_data_table,
    interpret_predictor,
    reinforce_predictor,
    prepare_coclustering_deployment,
    simplify_coclustering,
    sort_data_table,
    train_coclustering,
    train_instance_variable_coclustering,
    train_predictor,
    train_recoder,
]
for task_module in task_modules:
    assert isinstance(task_module.TASKS, list)
    for task in task_module.TASKS:
        assert isinstance(task, KhiopsTask)
        task_registry = get_task_registry()
        task_registry.register_task(task)
