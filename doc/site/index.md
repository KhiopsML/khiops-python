# Khiops Python API Docs

Welcome to the Khiops Python API documentation page.

## Installation

Khiops can be installed in a Python virtual environment using `pip`,
under Linux and macOS (in a `bash` shell):

```bash
python -m venv khiops-venv
source khiops-venv/bin/activate
pip install -U khiops
```

under Windows (in a `powershell` shell):

```powershell
python -m venv khiops-venv
khiops-venv\Scripts\activate
pip install -U khiops
```

Alternatively, you can install Khiops with the [Conda package manager](https://docs.conda.io/en/latest/):

```bash
conda create -n khiops-env
conda activate khiops-env
conda install -c conda-forge khiops
```

More details and other installation methods are documented at the [Khiops website](https://www.khiops.org/setup).

## Main Submodules

This package contains the following main submodules.

### `sklearn` submodule

The [sklearn](sklearn/index.md) module is a [Scikit-learn](https://scikit-learn.org) based interface to
Khiops. Use it if you are just started using Khiops and are familiar with the Scikit-learn workflow
based on dataframes and estimator classes.

### `core` submodule

The [core](core/index.md) module is a pure Python library exposing all Khiops functionalities. Use it if
you are familiar with the Khiops workflow based on plain-text tabular data files and dictionary
files (`.kdic`).
