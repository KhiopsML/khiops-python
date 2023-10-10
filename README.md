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
conda install khiops -c khiops

# For Apple Silicon users
conda install khiops -c khiops -c conda-forge
```
Other install method are documented at the [Khiops website][khiops-install].

### Requirements
- [Python][python] (>=3.8)

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

## License
This software is distributed under the BSD 3-Clause-clear License, the text of which is available at
https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the [LICENSE.md](./LICENSE.md) for more
details.

## Credits
The Khiops Python library is currently developed at [Orange Innovation][o-innov] by the Khiops
Team: khiops.team@orange.com .

[khiops]: https://khiops.org
[khiops-install]: https://khiops.org/setup/installation
[khiops-python-doc]: https://khiopsml.github.io/khiops-python
[python]: https://www.python.org
[pandas]: https://pandas.pydata.org
[sklearn]: https://scikit-learn.org/stable
[boto3]: https://github.com/boto/boto3
[gcs]: https://github.com/googleapis/python-storage
[o-innov]: https://hellofuture.orange.com/en/
