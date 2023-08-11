# !/bin/bash

# Set-up the shell to behave more like a general-purpose programming language
set -euo pipefail

# Clone Khiops sources (we change working directory there)
git clone --depth 1 https://github.com/khiopsml/khiops.git khiops-core
cd khiops-core
git checkout ${KHIOPS_REVISION}

# Copy License file
cp ./LICENSE ..


# Build MODL and MODL_Coclustering
# Note on macOS we need the macOS SDK 10.10 for this conda build to work
if [[ "$(uname)" == "Darwin" ]]
then
  CMAKE_PRESET="macos-clang-release"
else
  CMAKE_PRESET="linux-gcc-release"
fi
cmake --preset $CMAKE_PRESET -DBUILD_JARS=OFF
cmake --build --preset $CMAKE_PRESET --parallel --target MODL MODL_Coclustering

# Copy the MODL binaries to the Conda PREFIX path
cp ./build/$CMAKE_PRESET/bin/MODL $PREFIX/bin
cp ./build/$CMAKE_PRESET/bin/MODL_Coclustering $PREFIX/bin

# Build the Khiops Python package in the base directory
cd ..
$PYTHON -m pip install . --no-deps --ignore-installed --no-cache-dir --no-build-isolation -vvv
