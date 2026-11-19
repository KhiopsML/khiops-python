"""Run installed-package checks in a non-activated Conda child process."""

import os
import subprocess
import sys


def normalized(path):
    return os.path.normcase(os.path.realpath(path))


prefix = normalized(sys.prefix)
environment = {
    key: value for key, value in os.environ.items() if "CONDA" not in key.upper()
}
# Remove every directory within the environment, including Windows' prefix,
# Scripts and Library/bin entries. Keep unrelated OS executable directories.
environment["PATH"] = os.pathsep.join(
    entry
    for entry in os.get_exec_path()
    if normalized(entry) != prefix and not normalized(entry).startswith(prefix + os.sep)
)
CHECK = """
import os
import sys
assert not any('CONDA' in key.upper() for key in os.environ)
prefix = os.path.normcase(os.path.realpath(sys.prefix))
for entry in os.get_exec_path():
    entry = os.path.normcase(os.path.realpath(entry))
    assert entry != prefix and not entry.startswith(prefix + os.sep), entry
import khiops.core as kh
from khiops.core.internals.runner import _infer_khiops_installation_method
assert _infer_khiops_installation_method() == 'conda-based'
"""
subprocess.run([sys.executable, "-c", CHECK], env=environment, check=True)
subprocess.run([sys.executable, *sys.argv[1:]], env=environment, check=True)
