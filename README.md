# PyKeyATM

PyKeyATM is a Python package for fitting keyword-assisted topic models and related LDA models. Its Python interface is backed by a C++ sampler.

## Install

A C++17 compiler is required to build the extension. On Windows, install the Visual Studio C++ Build Tools; on macOS or Linux, install the platform's C++ toolchain.

From a source checkout:

```bash
python -m pip install .
```

This installs the core package dependencies. Plotting helpers are optional:

```bash
python -m pip install ".[plot]"
```

## Quick start

```python
import pandas as pd
from pykeyatm import keyATM, keyATM_read

dtm = pd.DataFrame(
    [[3, 1, 0, 2], [0, 2, 3, 1], [2, 0, 1, 3], [1, 2, 0, 2]],
    columns=["policy", "health", "market", "public"],
)
docs = keyATM_read(dtm)
fit = keyATM(
    docs=docs,
    model="base",
    no_keyword_topics=1,
    keywords={"policy": ["policy"], "health": ["health"]},
    options={"seed": 42, "iterations": 100, "verbose": False, "use_cache": False},
)
print(fit["theta"])
```

See [`examples/`](examples/) for runnable base, covariate, and dynamic models. They use small synthetic inputs for API demonstrations.

## Models

`keyATM()` supports `base`, `covariates`, and `dynamic` models. `weightedLDA()` provides corresponding no-keyword LDA models. Count weighting is controlled by model options; see [the API guide](docs/API.md) for input requirements and available settings.

## Tests

After installation, run:

```bash
python -m unittest discover -s tests -v
```

## License

PyKeyATM is distributed under GPL-3.0-only. The C++ extension includes Eigen, which is separately available under MPL-2.0; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
