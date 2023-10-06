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

```bash
conda install khiops -c conda-forge -c khiops
```

### Requirements
- [Python][python] (>=3.8)

If you have proxy problems with the installation or want a specific configuration see [Advanced
Installations](#advanced-installations) section below.

## Documentation
The main documentation of the Khiops Python library is available [here][khiops-python-doc].

The library itself is documented with docstrings: for example to obtain help on the
`train_predictor` function you can use:
```python
from khiops.sklearn import KhiopsClassifier
help(KhiopsClassifier)

from khiops import core as kh
help(kh.train_predictor)
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

## License
This software is distributed under the BSD 3-Clause-clear License, the text of which is available at
https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the [LICENSE.md](./LICENSE.md) for more
details.

## Credits
The Khiops Python library is currently developed at [Orange Innovation][o-innov] by the Khiops
Team: khiops.team@orange.com .

[khiops]: https://www.khiops.com
[khiops-python-doc]: https://www.khiops.com/html/pyKhiops
[python]: https://www.python.org
[pandas]: https://pandas.pydata.org
[sklearn]: https://scikit-learn.org/stable
[boto3]: https://github.com/boto/boto3
[gcs]: https://github.com/googleapis/python-storage
[o-innov]: https://hellofuture.orange.com/en/
