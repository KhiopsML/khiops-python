# Khiops Conda Packaging Scripts
In conda, `khiops-python` is packaged with the Khiops executables. These executables are built from
source during the packaging process.

## How to Build
You'll need `conda-build` installed in your system. The environment variable `KHIOPS_REVISION` must
be set to a Git hash/branch/tag of the `khiops` repository.

```bash
# At the root of the repo
# These commands will leave a ready to use conda channel in `./khiops-conda-build`
KHIOPS_REVISION=10.2.0

# Windows
conda build --output-folder ./khiops-conda-build packaging/conda

# Linux/macOS
# Note: We need the conda-forge channel to obtain the pinned versions of MPICH
conda build --channel conda-forge --output-folder ./khiops-conda-build packaging/conda
```

### Executable Signatures in macOS
The script can sign the Khiops binaries. This is to avoid annoying firewall pop-ups. To enable this
set the following environment variable:
- `KHIOPS_APPLE_CERTIFICATE_COMMON_NAME`: The common name of the signing certificate.

Additionally a certificate file encoded in base64 may be provided by setting the following
environment variables:
- `KHIOPS_APPLE_CERTIFICATE_BASE64`: The base64 encoding of the signing certificate.
-`KHIOPS_APPLE_CERTIFICATE_PASSWORD`: The password of the signing certificate.
- `KHIOPS_APPLE_TMP_KEYCHAIN_PASSWORD` : A password for the temporary keychain created in the process.

For more details see the comments in the signing section of `build.sh`.



