######################################################################################
# Copyright (c) 2018 - 2022 Orange - All Rights Reserved                             #
# * This software is the confidential and proprietary information of Orange.         #
# * You shall not disclose such Restricted Information and shall use it only in      #
#   accordance with the terms of the license agreement you entered into with Orange  #
#   named the "Khiops - Python Library Evaluation License".                          #
# * Unauthorized copying of this file, via any medium is strictly prohibited.        #
# * See the "LICENSE.md" file for more details.                                      #
######################################################################################
"""Sklearn Utilities module"""

import os

import pykhiops.core.filesystems as fs


def is_convertible_to_number(value):
    """Helper method to test if a value is convertible to a number

    Parameters
    ----------
    value : Any
        Value whose convertibility to a number is checked.

    Returns
    -------
    bool
        `True` if ``value`` is convertible.
    """
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def log_missmatch(prefix, expected, found, suffix=""):
    """
    Build an error message from str and expected/found values
    :param prefix: str
    :param expected: any
    :param found: any
    :param suffix: str
    :return: str
    """
    return f"{prefix}: `{expected}` expected but `{found}` found. {suffix}"


def remove_extension(file_name):
    """Removes the extension of filename"""
    return os.path.splitext(file_name)[0]


def temp_folder_rmtree(uri_or_path, keep_root=False):
    """Cleans the temporary folder"""
    dir_resource = fs.create_resource(uri_or_path)
    for file_name in dir_resource.list_dir():
        file_resource = dir_resource.create_child(file_name)
        file_resource.remove()
    if not keep_root:
        dir_resource.remove()
