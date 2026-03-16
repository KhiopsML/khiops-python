# Khiops Conda Packaging Scripts

## Purpose
The current folder helps you build locally a conda package from the sources. 

## How to Build
You'll need the package `conda-build` installed in your conda environment.
Optionally if installed, the package `conda-verify` will find obvious packaging bugs. 

```bash
# At the root of the project repository

# Windows
conda build --output-folder ./khiops-conda-build packaging/conda

# Linux/macOS
# Note: We use the conda-forge channel so the khiops-core package obtains the pinned MPICH versions
conda build --channel conda-forge --output-folder ./khiops-conda-build packaging/conda
```

## How to Install

The freshly built package can be installed in your conda environment

```bash
conda install --channel ./khiops-conda-build khiops
```