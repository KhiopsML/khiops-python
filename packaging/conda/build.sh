# !/bin/bash

# Set-up the shell to behave more like a general-purpose programming language
set -euo pipefail

# Clone Khiops sources
git clone --depth 1 https://github.com/khiopsml/khiops.git khiops_bin
cd khiops_bin/
git checkout ${KHIOPS_REVISION}
cd ..

# Copy relevant Khiops files to current directory
cp -R khiops_bin/{src,test} .
cp -R khiops_bin/packaging/common ./packaging/
cp khiops_bin/{CMakeLists.txt,CMakePresets.json,LICENSE} .
cp khiops_bin/packaging/{install,packaging}.cmake ./packaging/

# Build the Khiops binaries
# On macOS, we have to build with the compiler outside conda. With Conda's clang the following error occurs:
# ld: unsupported tapi file type '!tapi-tbd' in YAML file '/Applications/Xcode_14.2.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/lib/libSystem.tbd' for architecture x86_64
# The binary location depends on the preset name used at the configure step
if [ "$(uname)" == "Darwin" ]
then
    cmake --preset macos-clang-release -DBUILD_JARS=OFF -DTESTING=OFF -DCMAKE_CXX_COMPILER=/usr/bin/clang++
    cmake --build --preset macos-clang-release --parallel --target MODL MODL_Coclustering
    BUILD_DIR="macos-clang-release"
else
    cmake --preset linux-gcc-release -DBUILD_JARS=OFF -DTESTING=OFF
    cmake --build --preset linux-gcc-release --parallel --target MODL MODL_Coclustering
    BUILD_DIR="linux-gcc-release"
fi

# Copy the MODL binaries to the Conda PREFIX path
mkdir -p $PREFIX/bin
cp build/$BUILD_DIR/bin/MODL $PREFIX/bin
cp build/$BUILD_DIR/bin/MODL_Coclustering $PREFIX/bin

# Build the Khiops Python package
$PYTHON -m pip install . --no-deps --ignore-installed --no-cache-dir --no-build-isolation -vvv
