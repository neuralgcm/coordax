# Why Coordax?

As authors of [NeuralGCM](https://github.com/neuralgcm/neuralgcm),
[JAX-CFD](https://github.com/google/jax-cfd), and [Xarray](https://github.com/pydata/xarray),
we are big believers in both JAX and labeled arrays for scientific computing.

Coordax is our attempt to bridge the gap between these two worlds. It provides
a light-weight, JAX-native alternative to Xarray that is tailored to the needs
of simulation codes.

## Support for JAX transformations

The JAX ecosystem is built around robust support for code transformations
(`jit`, `vmap`, `grad`, etc).

These can largely be supported for arbitrary objects by declaring them to be
[JAX pytrees](https://docs.jax.dev/en/latest/pytrees.html), but transformations
that add or remove dimensions (e.g., [`vmap`](https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html)
and [`scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html)) are tricky
for data models that required every dimension to be named (such as `xarray.DataArray`).

Coordax robustly supports such transformations, while still propagating coordinates,
by allowing the insertion or removal of unnamed leading dimensions in pytree unflattening.

## Wrapping of non-labeled code

A key challenge for libraries that extend the array model for numerical computing is
the ability to work with and extend code written without the extended data model in mind.
Xarray solved this challenge the hard way with a gigantic API surface, wrapping hundreds
of routines from NumPy, Pandas and SciPy to add support for dimension names.

For Coordax, we take a different approach, largely copied from
[nmap](https://penzai.readthedocs.io/en/stable/notebooks/named_axes.html) in
Daniel Johnson's Penzai (from which parts of Coordax were originally forked).
Instead of expecting every routine to be wrapped, we provide
[simple transformations](cmap.ipynb) for converting functions written for unlabeled
arrays to labeled arrays.

## Optional coordinate objects 

There is an inherent tension between the appealing simplicity of string-based dimension
names and the need to keep track of detailed information about coordinate systems.
Coordax attempts a middle path:

- String axis-names can be used instead of coordinate objects in almost every case.
- [Coordinate objects](coordinates.ipynb) are also supported for more advanced use-cases.

Coordax's use of coordinate _objects_ rather than only keeping track of coordinate _arrays_
(like Xarray) provides natural support for complex coordinate systems, such as those
described by multiple variables or [coordinate reference systems](https://en.wikipedia.org/wiki/Spatial_reference_system).

## Compatibility with Xarray

Xarray has a rich API for working with labeled arrays, so Coordax makes it as
easy as possible to [convert your data](xarray.ipynb) back and forth.

## Why not use Xarray directly?

Xarray is great, but its APIs were designed for easy data analysis rather than
writing robust and efficient simulation codes. For example, Xarray operations attempt to
automatically align coordinate arrays and skip missing values in aggregations,
both of which can be quite expensive. Despite significant effort to support
[numpy-like arrays](https://docs.xarray.dev/en/latest/user-guide/duckarrays.html) in Xarray,
the implementation of Xarray still reflects its origins as a library wrapping NumPy arrays,
with many functions implemented via NumPy and Pandas that cannot be easily extended to JAX.
We believe Coordax is a better choice for most JAX-native codes.

The good news is that if you disagree, Xarray does in fact have nascent support for wrapping
JAX arrays in Xarray data structures, both directly via the `__array_namespace__`
protocol and through the [xarray_jax](https://github.com/google-deepmind/xarray_jax)
library (which wraps Xarray data structures into pytrees).
