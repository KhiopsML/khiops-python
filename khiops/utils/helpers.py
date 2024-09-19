"""General helper functions"""

import os
import platform
import subprocess


def os_open(path):
    """Opens a file or directory with its default application

    Parameters
    ----------
    path : str
        The path of the file to be open.

    """
    try:
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.call(["open", path])
        else:
            subprocess.call(["xdg-open", path])
    except OSError as error:
        print(f"Could not open file: {error}. Path: {path}")
