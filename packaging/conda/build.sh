# !/bin/bash

# Echo all output
set -x

cp -R $CWD/khiops_bin/{src,test} .
cp -R $CWD/khiops_bin/packaging/common ./packaging/
cp $CWD/khiops_bin/{CMakeLists.txt,CMakePresets.json,LICENSE} .
cp $CWD/khiops_bin/packaging/{install,packaging}.cmake ./packaging/

# On macOS, we have to build with the compiler outside conda. With teh conda's clang the following error occurs:
# ld: unsupported tapi file type '!tapi-tbd' in YAML file '/Applications/Xcode_14.2.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/lib/libSystem.tbd' for architecture x86_64
# The binary location depends on the preset name used at the configure step (Cf. build.sh)
#

function copy_modl_binaries_to_miniconda_prefix_path () {
    # Copy the MODL binaries to the Miniconda PREFIX path
    local build_dir="$1"
    local bin_dir="$2"
    mkdir -p $PREFIX/$bin_dir
    cp build/$build_dir/bin/MODL $PREFIX/$bin_dir
    cp build/$build_dir/bin/MODL_Coclustering $PREFIX/$bin_dir
}

if [ "$(uname)" == "Darwin" ]
then
    cmake --preset macos-clang-release -DBUILD_JARS=OFF -DTESTING=OFF -DCMAKE_CXX_COMPILER=/usr/bin/clang++
    cmake --build --preset macos-clang-release --parallel --target MODL MODL_Coclustering
    copy_modl_binaries_to_miniconda_prefix_path "macos-clang-release" "bin"

    # Cross-compile to ARM64
    cmake --preset macos-clang-release -DCMAKE_OSX_ARCHITECTURES=arm64 -DBUILD_JARS=OFF -DTESTING=OFF -DCMAKE_CXX_COMPILER=/usr/bin/clang++
    cmake --build --preset macos-clang-release -DCMAKE_OSX_ARCHITECTURES=arm64 --parallel --target MODL MODL_Coclustering
    # Also copy the ARM64 MODL binaries, which *override* the x86 binaries
    copy_modl_binaries_to_miniconda_prefix_path "macos-clang-release" "khiops_arm64"
else
    cmake --preset linux-gcc-release -DBUILD_JARS=OFF -DTESTING=OFF
    cmake --build --preset linux-gcc-release --parallel --target MODL MODL_Coclustering
    copy_modl_binaries_to_miniconda_prefix_path "linux-gcc-release" "bin"
fi

$PYTHON -m pip install . --no-deps --ignore-installed --no-cache-dir --no-build-isolation -vvv
