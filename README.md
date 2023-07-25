# Khiops Python

**Khiops Python** is a Python library for the [Khiops AutoML suite][khiops].

## Description
Khiops is a robust AutoML suite for constructing supervised models (classifiers, regressors and
encoders) and unsupervised models (coclusterings). With this package you can use Khiops via Python
in two ways:
- with the module `khiops.core`: To use Khiops in its native way (Khiops dictionary files +
  tabular data files as input)
- with the module `khiops.sklearn`: To use Khiops with Scikit-Learn estimator objects (Pandas
  dataframes or Numpy arrays as input)

## Installation

[TO BE COMPLETED]

### Requirements
- [The Khiops native binaries][khiops] (>=9.0)
- [Python][python] (>=3.8)
- [Pandas][pandas] (>=0.25.3)
- [Scikit-Learn][sklearn] (>=0.22.2)

If you have proxy problems with the installation or want a specific configuration see [Advanced
Installations](#advanced-installations) section below.

## Documentation
The main documentation of the Khiops Python library is available [here][pykhiops-doc].

The library itself is documented with docstrings: for example to obtain help on the
`train_predictor` function you can use:
```python
from khiops.sklearn import KhiopsClassifier
help(KhiopsClassifier)

from khiops import core as pk
help(pk.train_predictor)
```

## Advanced Installations
### Light Installations

[TO BE UPDATED]

### "Extras" installation
If you want to have support for the Amazon S3 and/or Google Cloud Storage filesystems in the
`sklearn` module you must install their extra dependencies. This can be done in two ways:

- via Conda (the recommended way):
```bash
conda install boto3
conda install google-cloud-storage
```

- via Pip (if the native Khiops binaries are already installed and accessible in the executable
  path):
```bash
pip install khiops --extras s3
pip install khiops --extras gcs
```

##### Requirements for the "Extras" installation
The requirements of the normal installation plus:
- s3 : [boto3][boto3] (>=1.17.39)
- gcs : [google-cloud-storage][gcs] (>=1.37.0)


### Migrating from Khiops 9 to Khiops 10
If you have script using the Khiops 9 module you can fix most, if not all,  problems with the
script `convert-pk10` installed with the Khiops Python library.
```
usage: convert-pk10 [-h] INPYFILE OUTPYFILE

converts a Khiops 9 script to a Khiops 10 script

positional arguments:
  INPYFILE    input python script path
  OUTPYFILE   output python script path

optional arguments:
  -h, --help  show this help message and exit

```
Usually after installation this script is available in the user's `PATH` and its precise
location can be found with `pip show -f khiops`. If for any reason you can't find the
script an alternative is to open a python terminal and execute:
```python
>>> from khiops import tools
>>> tools.convert_pk10("my_input.py", "my_output.py")
```
## License
See the [LICENSE.md](./LICENSE.md) file of this repository or in the `-info` directory
of the python package installation.

## Credits
The Khiops Python library has been developed at [Orange Labs][olabs-ai].

Current contributors:
- Yassine Nair Benrekia
- Felipe Olmos
- Vladimir Popescu

Past contributors:
- Alexis Bondu
- Enzo Bonnal
- Marc Boullé
- Pierre Nodet
- Stéphane Gouache

[khiops]: https://www.khiops.com
[pykhiops-doc]: https://www.khiops.com/html/pyKhiops
[python]: https://www.python.org
[pandas]: https://pandas.pydata.org
[sklearn]: https://scikit-learn.org/stable
[boto3]: https://github.com/boto/boto3
[gcs]: https://github.com/googleapis/python-storage
[olabs-ai]: https://hellofuture.orange.com/en/artificial-intelligence
