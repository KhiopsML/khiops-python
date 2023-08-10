# !/bin/bash

# On MAC OS, if on ARM, move relevant binaries to bin (overwrite X86 binaries)
KHIOPS_ARM64_DIR="khiops_arm64"
if [[ "$(uname)" == "Darwin" && "$(uname -m)" == "arm64" ]]
then
    mv $PREFIX/$KHIOPS_ARM64_DIR/MODL $PREFIX/bin/
    mv $PREFIX/$KHIOPS_ARM64_DIR/MODL_Coclustering $PREFIX/bin/
    rm -fr $PREFIX/$KHIOPS_ARM64_DIR
else
    # Remove whole directory containing ARM-specific builds, if it exists
    rm -fr $PREFIX/$KHIOPS_ARM64_DIR
fi

