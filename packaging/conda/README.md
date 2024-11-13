# Khiops Conda Packaging Scripts

## How to Build
You'll need `conda-build` installed in your system. The environment variable `KHIOPS_REVISION` must
be set to a Git tag of the `khiops` repository.

```bash
# At the root of the repo
# These commands will leave a ready to use conda channel in `./khiops-conda-build`
KHIOPS_REVISION=10.2.0

# Windows
conda build --output-folder ./khiops-conda-build packaging/conda

# Linux/macOS
# Note: We use the conda-forge channel so the khiops-core package obtains the pinned MPICH versions
conda build --channel conda-forge --output-folder ./khiops-conda-build packaging/conda
```
