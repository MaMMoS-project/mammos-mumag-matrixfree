# mammos-mumag-matrixfree

`mammos-mumag-matrixfree` is a finite-element micromagnetic simulation tool capable of simulating hysteresis loops of magnetic materials with multiple grains. This software exploits 

| Description   | Badge                                                                                                                                                                                                          |
|---------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Tests         | [![Test package](https://github.com/MaMMoS-project/mammos-mumag-matrixfree/actions/workflows/test.yml/badge.svg)](https://github.com/MaMMoS-project/mammos-mumag-matrixfree/actions/workflows/test.yml)        |
| Linting       | [![pre-commit.ci status](https://results.pre-commit.ci/badge/github/MaMMoS-project/mammos-mumag-matrixfree/main.svg)](https://results.pre-commit.ci/latest/github/MaMMoS-project/mammos-mumag-matrixfree/main) |
| Releases      | [![PyPI version](https://badge.fury.io/py/mammos-mumag-matrixfree.svg)](https://badge.fury.io/py/mammos-mumag-matrixfree)                                                                                      |
| Documentation | TODO                                                                                                                                                                                                           |
| Binder        | TODO                                                                                                                                                                                                           |
| License       | [![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)                                                                                                           |
| DOI           | TODO                                                                                                                                                                                                           |


## Try it in the cloud
Try `mammos-mumag-matrixfree` without installing it locally by directly accessing it directly in the cloud
via Binder.

Simply click the badge in the table above to get started.

Sessions are temporary and may time out after a period of inactivity, and any files
created or modified during your session will not be saved.
To avoid losing your work, please remember to download any files you create or edit
before your session ends.

## Documentation

See [Quickstart.md](Quickstart.md).

## Installation

To install `mammos-mumag-matrixfree`, you can use `pip install mammos-mumag-matrixfree` inside a Python environment.
For more details refer to the documentation.

### With `pixi` (recommended)

Requirements: `pixi` (https://pixi.sh).

Pixi will install Python and `mammos-mumag-matrixfree`.

```
pixi init
pixi workspace channel add set3mah
pixi add python>=3.11.0 neper>=4.6.0 povray>=3.7.0.8
pixi add --pypi "mammos-mumag-matrixfree @ git+https://github.com/MaMMoS-project/mammos-mumag-matrixfree"
```

To install NVIDIA GPU support for `jax`, install instead:

```
pixi add --pypi "mammos-mumag-matrixfree[cuda] @ git+https://github.com/MaMMoS-project/mammos-mumag-matrixfree"
```

## How to cite

TODO

## Acknowledgements

This software has been supported by the European Union’s Horizon Europe research and innovation programme under grant agreement No 101135546 [MaMMoS](https://mammos-project.github.io/).
