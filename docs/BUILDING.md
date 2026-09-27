# Build and install

PyKeyATM builds a pybind11 C++17 extension during installation. Install the compiler toolchain for your platform before running `python -m pip install .`.

The C++ extension source is in `cpp/`; the Python package uses a `src/` layout. Eigen headers are vendored under `cpp/vendor/eigen/Eigen/`.
