#!/bin/bash

# Set-up the shell to behave more like a general-purpose programming language
set -euo pipefail

# Build the Khiops Python package in the base directory
$PYTHON -m pip install . --no-deps --ignore-installed --no-cache-dir --no-build-isolation -vvv
