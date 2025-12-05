# Why Coordax?

As authors of [NeuralGCM](https://github.com/neuralgcm/neuralgcm),
[JAX-CFD](https://github.com/google/jax-cfd), and [Xarray](https://github.com/pydata/xarray),
we are big believers in both JAX and labeled arrays for scientific computing.

Coordax is our attempt to bridge the gap between these two worlds. It provides
a light-weight, JAX-native alternative to Xarray that is tailored to the needs
of modern physics- and AI-based simulation codes.

## Support for JAX transformations

The JAX ecosystem is built around robust support for code transformations
(`jit`, `vmap`, `grad`, etc).

These can largely be supported for arbitrary objects by declaring them to be
[JAX pytrees](https://docs.jax.dev/en/latest/pytrees.html), but transformations
that add or remove dimensions (e.g., [`vmap`](https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html)
and [`scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html)) are tricky
for data models that require every dimension to be named (such as `xarray.DataArray`).

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

- String axis names can be used instead of coordinate objects in almost every case.
- [Coordinate objects](coordinates.ipynb) are also supported for more advanced use-cases.

Coordax's use of coordinate _objects_ rather than only keeping track of coordinate _arrays_
(like Xarray) provides natural support for complex coordinate systems, such as those
described by multiple variables or [coordinate reference systems](https://en.wikipedia.org/wiki/Spatial_reference_system).

## Compatibility with Xarray

Xarray has a rich API for working with labeled arrays, so Coordax makes it as
easy as possible to [convert your data](xarray.ipynb) back and forth.

## Alternatives

### Why not Xarray?

Xarray is great, but its APIs were designed for easy data analysis rather than
writing robust and efficient simulation codes. For example, Xarray operations attempt to
automatically align coordinate arrays and skip missing values in aggregations,
both of which can be quite expensive. Despite significant effort to support
[numpy-like arrays](https://docs.xarray.dev/en/latest/user-guide/duckarrays.html) in Xarray,
the implementation of Xarray still reflects its origins as a library wrapping NumPy arrays,
with many functions implemented via NumPy and Pandas that cannot be easily extended to JAX.
It's also hard to support arbitrary JAX transformations in Xarray because its data
model requires names for all dimensions, so dimensionality-changing operations
like `vmap` require a custom wrapper.

We believe Coordax is a better choice for most JAX-native codes, but if you
disagree, Xarray does in fact have nascent support for wrapping JAX arrays in
Xarray data structures, both directly via the `__array_namespace__`
protocol and through the [xarray_jax](https://github.com/google-deepmind/xarray_jax)
library (which wraps Xarray data structures into pytrees).

### Why not Penzai?

We really liked the transformation-based approach to implementing
label-propagating operations in
[penzai.named_axes](https://penzai.readthedocs.io/en/stable/notebooks/named_axes.html) with `tag`/`untag`/`nmap`, so much so that we
forked it as the basis of Coordax's `tag`/`untag`/`cmap` (thanks Daniel!).
We didn't love that Penzai uses two different labeled array types (`NamedArray`
and `NamedArrayView`), with different data models and indirection for
axis names with `data_axis_for_logical_axis` and `data_axis_for_name`. This
adds a layer of friction when using transformations not designed for labeled
arrays, because users need to know what type of named arrays they have. Penzai
also does not expose APIs for controlling axis order in underlying unlabeled
arrays.

{py:class}`coordax.Field` uses a simpler data model, with just one labeled array
type, with a tuple of names (`dims`) for keeping track of dimensions (like
Xarray). This makes it much easier to drop down a level of abstraction to
working with unlabeled arrays, which we found to be quite important in practice.
As for the two array types, we were able to get dimensionality changing JAX
transformations to work like Penzai with a single `Field` type with a clever
[tree_unflatten](https://github.com/neuralgcm/coordax/blob/b91dcfa0b5417cf2aff890608ce993d3b8d51a5d/coordax/named_axes.py#L700-L729)
method, which allows leading `dims` to be padded or trimmed.

We also needed the ability to keep track of full coordinate information
(e.g., arrays of latitude and longitude coordinates), not only string names,
so we layered on `Coordinate` objects as an optional feature.

### Why not Haliax?

[Haliax](https://github.com/marin-community/haliax) is another compelling
option for labeled arrays in JAX. The main differentiator of Haliax versus both
Coordax and Penzai is that Haliax's API "reimplements the world," similar to
Xarray. This is convenient if you like Haliax's style for model building, but
it also makes the labeled array library much more intrusive in your codebase.
Practically, it's also much more work for the authors of the array library,
because there's a lot of stuff to wrap! For example, if you don't like Haliax's
choice of neural net library (Equinox), you'd want to write your own wrapper.

Penzai also has a [nice discussion](https://penzai.readthedocs.io/en/stable/notebooks/named_axes.html#penzai-core-named-axes-vs-haliax) of the trade-offs between Penzai and
Haliax, most of which applies equally well for Coordax.

## What's missing?

Coordax is intentionally simple and limited in scope, with the idea that
[`cmap`](cmap.ipynb) means we didn't need to build-in a full implementation of
labeled array operations. Users can write those themselves, as needed, and a
small core makes Coordax easy to hack on.

This means that some array library features that you might expect
(e.g., indexing and concatenating arrays) are not built-in -- but are easy to
implement yourself with `cmap`, e.g.,

```python
import coordax as cx
import jax.numpy as jnp

def index(field: cx.Field, axis: str | cx.Coordinate, value: int) -> cx.Field:
  """Integer indexing like xarray.DataArray.isel({axis: value})."""
  return cx.cmap(lambda x: x[value])(field.untag(axis))

def concat(fields: list[cx.Field], axis: str) -> cx.Field:
  """Concatenate arrays along an existing axis."""
  return cx.cmap(jnp.concatenate)(cx.untag(fields, axis)).tag(axis)
```

That said, we are not opposed to adding frequently-used convenience functions in
principle, we were just lazy! If you're interested in helping out, please
[reach out on Github](https://github.com/neuralgcm/coordax/issues).
