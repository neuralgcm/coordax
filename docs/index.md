# Coordax: Coordinate Axes for JAX

Coordax is a Python library for labeled axes with [JAX](https://github.com/jax-ml/jax/).
Our approach is reminiscent of [Xarray](https://github.com/pydata/xarray),
but tailored to meet the needs of modern physics- and AI-based simulation codes
written in JAX, such as [NeuralGCM](https://github.com/neuralgcm/neuralgcm).

Compared to other libraries for labeled arrays, Coordax provides a handful of key
features:

1. First class integration with JAX, including support for arbitrary JAX transformations
2. Easy wrapping of code not written for labeled arrays with `cmap`,
   inspired by [Penzai](https://penzai.readthedocs.io/en/stable/notebooks/named_axes.html)
3. Optional `Coordinate` objects, for advanced use-cases
4. Lossless conversion to and from [Xarray](https://github.com/pydata/xarray),
   for serialization and data analysis

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
