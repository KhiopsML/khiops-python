# Khiops Python Library

This is the repository of the **Khiops Python Library** for the [Khiops AutoML suite][khiops].

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
# Windows
conda install khiops -c khiops

# Linux/macOS
conda install khiops -c khiops -c conda-forge
```
Other install method are documented at the [Khiops website][khiops-install].

### Requirements
- [Python][python] (>=3.8)
- [Pandas][pandas] (>=0.25.3)
- [Scikit-Learn][sklearn] (>=0.22.2)

[pandas]: https://pandas.pydata.org
[sklearn]: https://scikit-learn.org/stable

## Documentation
The API Docs for the Khiops Python library are available [here][khiops-api-docs]. Other
documentation (algorithms, installation, etc) may be found in the [Khiops site][khiops].

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
[khiops-install]: https://khiops.org/setup
[khiops-api-docs]: https://khiopsml.github.io/khiops-python
[python]: https://www.python.org
[pandas]: https://pandas.pydata.org
[sklearn]: https://scikit-learn.org/stable
[boto3]: https://github.com/boto/boto3
[gcs]: https://github.com/googleapis/python-storage
[o-innov]: https://hellofuture.orange.com/en/
