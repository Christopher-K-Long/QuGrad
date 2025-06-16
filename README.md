# QuGrad
A Python package for quantum optimal control.

[![Unit Tests](https://github.com/Christopher-K-Long/QuGrad/actions/workflows/test-python-package.yml/badge.svg)](https://github.com/Christopher-K-Long/QuGrad/actions/workflows/test-python-package.yml)

## Installation

The python package can be installed with pip as follows:
```bash
pip install qugrad
```

If on Linux and using a conda environment you may encounter an error
```
version `GLIBCXX_...' not found
```
to fix this you also need to execute:
```bash
conda install -c conda-forge libstdcxx-ng
```

### Requirements

Requires:
- [PySTE](https://PySTE.readthedocs.io)
- [TensorFlow](https://www.tensorflow.org)
- [NumPy](https://numpy.org)

#### Additional requirements for testing

- [toml](https://github.com/uiri/toml)
- [PyYAML](https://pyyaml.org/)

## Documentation

Documentation including worked examples can be found at: [https://QuGrad.readthedocs.io](https://QuGrad.readthedocs.io)

## Source Code

Source code can be found at: [https://github.com/Christopher-K-Long/QuGrad](https://github.com/Christopher-K-Long/QuGrad)

## Version and Changes

The current version is [`1.0.1`](ChangeLog.md#release-101). Please see the [Change Log](ChangeLog.md) for more details. QuGrad uses [semantic versioning](https://semver.org/).