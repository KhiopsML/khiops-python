######################################################################################
# Copyright (c) 2018 - 2023 Orange - All Rights Reserved                             #
# * This software is the confidential and proprietary information of Orange.         #
# * You shall not disclose such Restricted Information and shall use it only in      #
#   accordance with the terms of the license agreement you entered into with Orange  #
#   named the "Khiops - Python Library Evaluation License".                          #
# * Unauthorized copying of this file, via any medium is strictly prohibited.        #
# * See the "LICENSE.md" file for more details.                                      #
######################################################################################
"""Tasks module

This is a "read-only" module that registers all the task definition at __init__ time.
"""
from pykhiops.core.internals.task import KhiopsTask, get_task_registry
from pykhiops.core.internals.tasks import (
    build_deployed_dictionary,
    build_dictionary_from_data_table,
    build_multi_table_dictionary,
    check_database,
    deploy_model,
    detect_data_table_format,
    detect_data_table_format_with_dictionary,
    evaluate_predictor,
    export_dictionary_as_json,
    extract_clusters,
    extract_keys_from_data_table,
    prepare_coclustering_deployment,
    simplify_coclustering,
    sort_data_table,
    train_coclustering,
    train_predictor,
    train_recoder,
)

# Register the tasks in each module
task_modules = [
    build_deployed_dictionary,
    build_dictionary_from_data_table,
    build_multi_table_dictionary,
    check_database,
    deploy_model,
    detect_data_table_format,
    detect_data_table_format_with_dictionary,
    evaluate_predictor,
    export_dictionary_as_json,
    extract_clusters,
    extract_keys_from_data_table,
    prepare_coclustering_deployment,
    simplify_coclustering,
    sort_data_table,
    train_coclustering,
    train_predictor,
    train_recoder,
]
for task_module in task_modules:
    assert isinstance(task_module.TASKS, list)
    for task in task_module.TASKS:
        assert isinstance(task, KhiopsTask)
        task_registry = get_task_registry()
        task_registry.register_task(task)

task_registry.set_task_end_version("build_multi_table_dictionary", "11.0.0")
