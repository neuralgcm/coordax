# Coordax: Coordinate axes for scientific computing in JAX

Coordax is a Python library for labeled axes with [JAX](https://github.com/jax-ml/jax/).
Our approach is reminiscent of [Xarray](https://github.com/pydata/xarray),
but tailored to meet the needs of simulation codes in scientific computing.

Compared to other libraries for labeled arrays, Coordax provides a handful of key
features:

1. First class integration with JAX, including support for arbitrary JAX transformations
   that introduce or remove dimensions (e.g., `vmap` and `scan`).
2. Easy wrapping of code not written for labeled arrays with `cmap`
   (originally forked from Daniel Johnson's [Penzai](https://penzai.readthedocs.io/en/stable/notebooks/named_axes.html))
3. Support for propagating arbitrary `Coordinate` objects through computations
   (e.g., to keep track of discretization details).
4. Lossless conversion to and from [Xarray](https://github.com/pydata/xarray)
   data structures (e.g., for serialization and data analysis).

Read on for more details!

## Contents

```{toctree}
:maxdepth: 1
why_coordax.md
installation.md
fields.ipynb
cmap.ipynb
jax_transformations.ipynb
coordinates.ipynb
xarray.ipynb
api.md
```

## Questions?

The best place to ask for help or report bugs is
[on GitHub](https://github.com/neuralgcm/coordax/issues).
