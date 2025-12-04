![](https://github.com/neuralgcm/coordax/raw/main/docs/_static/coordax_logo_light.png)

# Coordax: Coordinate Axes for JAX

Authors: Dmitrii Kochkov and Stephan Hoyer

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

Coordax was developed to meet the needs of
[NeuralGCM](https://github.com/neuralgcm/neuralgcm), but we hope it will be
useful more broadly!

For more details, **[read the documentation](https://coordax.readthedocs.io/)**.

## Disclaimer

Coordax is an experiment that we are sharing with the outside world in the    hope
that it will be useful. It is not a supported Google product. We welcome feedback,
bug reports and code contributions, but cannot guarantee they will be addressed.
