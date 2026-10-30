# Copyright (c) 2023-2026 Orange. All rights reserved.
# This software is distributed under the BSD 3-Clause-clear License, the text
# of which is available at
# https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file
# for more details.

import sys
from importlib.metadata import PackageNotFoundError, version

# Pip may be absent (e.g. in a uv virtual environment): fall back to the
# 'packaging' package, and to None if neither is available
try:
    from pip._vendor.packaging.version import Version
except ImportError:
    try:
        from packaging.version import Version
    except ImportError:
        Version = None


def installed_version(package_name):
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


minimal_pip_version, minimal_certifi_version = sys.argv[1:3]

pip_version = installed_version("pip")
certifi_version = installed_version("certifi")

# Use Pip's vendored Certifi when Certifi is not installed
if certifi_version is None and pip_version is not None:
    from pip._vendor import certifi

    certifi_version = certifi.__version__

version_check_failure = False
for package_name, current_version, minimal_version in [
    ("pip", pip_version, minimal_pip_version),
    ("certifi", certifi_version, minimal_certifi_version),
]:
    # An absent package cannot be older than the minimal version
    if current_version is None:
        print(f"'{package_name}' is not installed: nothing to check")
        continue
    if Version is None:
        print(
            f"::error::Neither Pip nor 'packaging' is available to compare "
            f"the '{package_name}' version"
        )
        version_check_failure = True
        continue
    if Version(current_version) < Version(minimal_version):
        print(
            f"::error::'{package_name}' {current_version} found, >= "
            f"{minimal_version} required"
        )
        version_check_failure = True
    else:
        print(
            f"'{package_name}' version is OK: {current_version} (>= "
            f"{minimal_version})"
        )
sys.exit(1 if version_check_failure else 0)
