# Installation

You can install Coordax from source using pip:
```
pip install coordax
```

Coordax currently has core dependencies on Jax, NumPy and treescope.
Optional dependencies include:
* [`chex`](https://github.com/google-deepmind/chex): for `coordax.testing`.
* [`xarray`](https://github.com/pydata/xarray): for conversion to/from Xarray objects.
* [`jax-datetime`](https://github.com/google/jax-datetime): for integration with datetime objects.

To install all optional dependencies, use `pip install coordax[complete]`.
