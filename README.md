# Coordax: coordinate Axes for scientific computing in JAX

Authors: Dmitrii Kochkov and Stephan Hoyer

Coordax makes it easy to associate array dimensions with coordinates in the
context of scientific simulation codes written in JAX. This allows for efficient
and expressive manipulation of data defined on structured grids, enabling
operations like differentiation and interpolation with respect to physical
coordinates.

Coordax was designed to meet the needs of
[NeuralGCM](https://github.com/neuralgcm/neuralgcm), but we hope it will be
useful more broadly!

## Key features

1. Compute on locally-positional axes via coordinate map (`cmap`)
2. Coordinate objects that carry discretization details and custom methods
3. Lossless conversion to and from [Xarray](https://github.com/pydata/xarray)
   data structures (e.g., for serialization)

Coordax is particularly well-suited for scientific simulations where it is
crucial to propagate discretization details and associated objects throughout
the computation, such as Earth system modeling of fluid dynamics. The approach
to labeled dimensions was originally forked from Daniel Johnson's
[Penzai](https://penzai.readthedocs.io/en/stable/notebooks/named_axes.html),
which may be a better fit for simpler use-cases.

## Why not use Xarray?

Xarray does indeed support putting JAX arrays into Xarray data structures, an
approach used by the
[GraphCast codebase](https://github.com/google-deepmind/graphcast/blob/v0.1.1/graphcast/xarray_jax.py).
This works reasonably well, but wrapping JAX in Xarray will always be at least a
little bit awkward, because Xarray was designed for the needs of data analysis
rather than modeling, and cannot build core functionality on top of JAX power
features such as `vmap`. For JAX-native simulations, we believe Coordax is a
better choice.

## Documentation and examples

Coming soon!
