# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Defines the Field class, which is the main data structure for Coordax.

``Field`` objects keep track of positional and named dimensions of an array.
Named dimensions of a ``Field`` are associated with coordinates that describe
their discretization.
"""
from __future__ import annotations

import collections
import functools
import operator
import types
from typing import Any, Callable, Literal, Self, TYPE_CHECKING, TypeAlias, TypeGuard, TypeVar
import warnings

from coordax import coordinate_systems
from coordax import named_axes as named_axes_lib
from coordax import ndarrays
from coordax import utils
import jax
import jax.numpy as jnp
import numpy as np
import treescope
from treescope import lowering
from treescope import rendering_parts

if TYPE_CHECKING:
  import xarray


Pytree: TypeAlias = Any
Sequence = collections.abc.Sequence

T = TypeVar('T')

Coordinate = coordinate_systems.Coordinate
LabeledAxis = coordinate_systems.LabeledAxis
DummyAxis = coordinate_systems.DummyAxis

Array = ndarrays.Array
ArrayLike = ndarrays.ArrayLike


def _dimension_names(
    *names: str | Coordinate | types.EllipsisType,
) -> tuple[str | types.EllipsisType, ...]:
  """Returns a tuple of dimension names from a list of names or coordinates."""
  dims_or_name_tuple = lambda x: x.dims if isinstance(x, Coordinate) else (x,)
  return sum([dims_or_name_tuple(c) for c in names], start=tuple())


def _axes_attrs(field: Field) -> str:
  """Returns a string representation of the coordinate attributes."""

  def _coord_name(c: coordinate_systems.Coordinate):
    if isinstance(c, coordinate_systems.SelectedAxis):
      return f'SelectedAxis({c.coordinate.__class__.__name__}, axis={c.axis})'
    return c.__class__.__name__

  coord_names = {k: _coord_name(c) for k, c in field.axes.items()}
  return '{' + ', '.join(f'{k!r}: {v}' for k, v in coord_names.items()) + '}'


@utils.export
def new_axis_name(field: Field, excluded_names: set[str] | None = None) -> str:
  """Returns axis name that is not present in ``field`` or ``excluded_names``.

  Args:
    field: The field to generate a new axis name for.
    excluded_names: Optional set of names to exclude.

  Returns:
    A new axis name.

  Examples:
    >>> import coordax as cx
    >>> import jax.numpy as jnp
    >>> field = cx.field(jnp.zeros((2, 3)))
    >>> field2 = field.tag(cx.new_axis_name(field), ...)
    >>> field2
    <Field dims=('axis_0', None) shape=(2, 3) axes={} >
    >>> field3 = field2.tag(cx.new_axis_name(field2))
    >>> field3
    <Field dims=('axis_0', 'axis_1') shape=(2, 3) axes={} >
  """
  excluded_names = excluded_names or set()
  for i in range(field.ndim + len(excluded_names) + 1):
    name = f'axis_{i}'
    if name not in excluded_names and name not in field.named_dims:
      return name
  assert False  # unreachable


@utils.export
def tmp_axis_name(field: Field, excluded_names: set[str] | None = None) -> str:
  """Deprecated alias for ``new_axis_name``."""
  warnings.warn(
      'cx.tmp_axis_name() is deprecated, use cx.new_axis_name() instead',
      DeprecationWarning,
      stacklevel=2,
  )
  return new_axis_name(field, excluded_names)


@utils.export
def cmap(
    fun: Callable[..., Any],
    out_axes: (
        dict[str, int] | Literal['leading', 'trailing', 'same_as_input']
    ) = 'trailing',
    *,
    vmap: Callable = jax.vmap,  # pylint: disable=g-bare-generic
) -> Callable[..., Any]:
  # fmt: off
  """Vectorizes ``fun`` over coordinate dimensions of ``Field`` inputs.

  ``cmap`` is a "coordinate vectorizing map". It wraps an ordinary
  positional-axis-based function so that it accepts ``Field`` objects as input
  and produces ``Field`` objects as output, and vectorizes over all named
  dimensions using ``jax.vmap``.

  Unlike ``jax.vmap``, the axes to vectorize over are inferred automatically
  from the named dimensions in the ``Field`` inputs, rather than being specified
  as part of the mapping transformation. Specifically, each dimension name that
  appears in any of the arguments is vectorized over jointly across all
  arguments that include that dimension, and is then included as a named
  dimension in the output. To make an axis visible to ``fun``, you can call
  ``untag`` on the argument and pass the axis name(s) of interest; ``fun`` will
  then see those axes as positional axes instead of mapping over them.

  ``untag`` and ``cmap`` are together the primary ways to apply individual
  operations to axes of a ``Field``. ``tag`` can then be used on the result to
  re-bind names to positional axes.

  Within ``fun``, any mapped-over axes will be accessible using standard JAX
  collective operations like ``psum``, although using this is usually
  unnecessary.

  Args:
    fun: Function to vectorize by name. This can take arbitrary arguments (even
      non-JAX-arraylike arguments or "static" axis sizes), but must produce a
      PyTree of JAX ArrayLike outputs.
    out_axes: Specifies strategy for choosing labeled axis positions in the
      outputs. Options include:

      - dict[str, int]: mapping from dimension name to axis position. Keys must
        include all named dimensions present in the inputs. Axis positions must
        be unique and either all positive or all negative.
      - 'leading': dimension names will appear as the leading axes on every
        output, in order of their appearance on the inputs.
      - 'trailing': dimension names will appear as the trailing axes on every
        output, in order of their appearance on the inputs.
      - 'same_as_input': dimension names will appear in the same order as in the
        inputs, where the inputs must all have the same named axes and the same
        number of dimensions as the outputs.

    vmap: Vectorizing transformation to use when mapping over named axes.
      Defaults to ``jax.vmap``. A different implementation can be used to make
      coordax compatible with custom objects (e.g. neural net modules).

  Returns:
    A vectorized version of ``fun`` that applies original ``fun`` to locally
    positional dimensions in inputs, while vectorizing over all coordinate
    dimensions. All dimensions over which ``fun`` is vectorized will be present
    in every output.

  Examples:
    >>> import coordax as cx
    >>> import jax.numpy as jnp

    Named axes are trailing by default:

    >>> field = cx.field(jnp.ones((2, 3, 1)), 'x', None, 'y')
    >>> cx.cmap(jnp.sin)(field).dims
    (None, 'x', 'y')
    >>> cx.cmap(jnp.sin, out_axes='leading')(field).dims
    ('x', 'y', None)
    >>> cx.cmap(jnp.sin, out_axes='same_as_input')(field).dims
    ('x', None, 'y')

    Multiple field arguments result in all input axes in the outputs, in order
    of appearence:

    >>> a = cx.field(jnp.ones((2, 3)), 'x', 'y')
    >>> b = cx.field(jnp.ones((3, 4)), 'y', 'z')
    >>> cx.cmap(jnp.add)(a, b).dims
    ('x', 'y', 'z')

    ``cmap`` leverages JAX's pytree machinery, so arbitrarily nested inputs and
    outputs are supported, as well as keyword arguments:

    >>> z2 = cx.field(jnp.ones((2, 4)))
    >>> z3 = cx.field(jnp.ones((3, 4)))
    >>> cx.cmap(jnp.concat)([z2, z3], axis=0)
    <Field dims=(None, None) shape=(5, 4) axes={} >

  See also:
    :func:`coordax.cpmap`
    :meth:`coordax.Field.tag`
    :meth:`coordax.Field.untag`
  """
  # fmt: on
  if hasattr(fun, '__name__'):
    fun_name = fun.__name__
  else:
    fun_name = repr(fun)
  if hasattr(fun, '__doc__'):
    fun_doc = fun.__doc__
  else:
    fun_doc = None
  return _cmap_with_doc(fun, fun_name, fun_doc, out_axes, vmap=vmap)


@utils.export
def cpmap(
    fun: Callable[..., Any],
    *,
    vmap: Callable = jax.vmap,  # pylint: disable=g-bare-generic
) -> Callable[..., Any]:
  """Coordinate preserving cmap.

  ``cpmap(fun)`` is an alias for ``cmap(fun, out_axes='same_as_input')``.

  Primary use case is applying a function over positional axes while preserving
  the dimensionality and the coordinate order.

  Args:
    fun: Function to apply over positional axes of the inputs.
    vmap: Vectorizing transformation to use when mapping over named axes.
      Defaults to jax.vmap. A different implementation can be used to make
      coordax compatible with custom objects (e.g. neural net modules).

  Returns:
    A function that applies ``fun`` to positional axes of the inputs while
    vectorizing over all coordinate dimensions. The coordinate order of the
    inputs is preserved in the outputs.

  Examples:
    >>> import coordax as cx
    >>> import jax.numpy as jnp

    >>> field = cx.field(jnp.ones((2, 3, 4)), 'x', None, 'y')
    >>> cx.cpmap(lambda x: x**2)(field).dims
    ('x', None, 'y')

    ``cpmap`` requires all inputs to have the same named axes ordering:

    >>> a = cx.field(jnp.ones((2, 3)), 'x', 'y')
    >>> b = cx.field(jnp.ones((3, 2)), 'y', 'x')
    >>> cx.cpmap(jnp.add)(a, b)  # doctest: +NORMALIZE_WHITESPACE
    Traceback (most recent call last):
    ...
    ValueError: 'same_as_input' for out_axes requires all NamedArray inputs with
    named axes to have the same named_axes. Found multiple distinct
    named_axes on inputs:
    [{'x': 0, 'y': 1}, {'y': 0, 'x': 1}]

  See also:
    :func:`coordax.cmap`
  """
  return cmap(fun, out_axes='same_as_input', vmap=vmap)


def _cmap_with_doc(
    fun: Callable[..., Any],
    fun_name: str,
    fun_doc: str | None = None,
    out_axes: (
        dict[str, int] | Literal['leading', 'trailing', 'same_as_input']
    ) = 'trailing',
    *,
    vmap: Callable = jax.vmap,  # pylint: disable=g-bare-generic
) -> Callable[..., Any]:
  """Builds a coordinate-vectorized wrapped function with a docstring."""

  @functools.wraps(fun)
  def wrapped_fun(*args, **kwargs):
    leaves, treedef = jax.tree.flatten((args, kwargs), is_leaf=is_field)
    field_leaves = [leaf for leaf in leaves if is_field(leaf)]
    all_axes = {}
    for field in field_leaves:
      for dim_name, c in field.axes.items():
        if dim_name in all_axes and all_axes[dim_name] != c:
          other = all_axes[dim_name]
          raise ValueError(f'Coordinates {c=} != {other=} use same {dim_name=}')
        else:
          all_axes[dim_name] = c
    named_array_leaves = [x.named_array if is_field(x) else x for x in leaves]
    fun_on_named_arrays = named_axes_lib.nmap(fun, out_axes=out_axes, vmap=vmap)
    na_args, na_kwargs = jax.tree.unflatten(treedef, named_array_leaves)
    result = fun_on_named_arrays(*na_args, **na_kwargs)

    def _wrap_field(leaf):
      return Field.from_namedarray(
          named_array=leaf,
          axes={k: all_axes[k] for k in leaf.dims if k in all_axes},
      )

    return jax.tree.map(
        _wrap_field, result, is_leaf=named_axes_lib.is_namedarray
    )

  docstr = (
      f'Dimension-vectorized version of ``{fun_name}``. Takes similar arguments'
      f' as ``{fun_name}`` but accepts and returns Fields in place of arrays.'
  )
  if fun_doc:
    docstr += f'\n\nOriginal documentation:\n\n{fun_doc}'
  wrapped_fun.__doc__ = docstr
  return wrapped_fun


def _check_valid(
    named_array: named_axes_lib.NamedArray, axes: dict[str, Coordinate]
) -> None:
  """Checks that the field coordinates and dimension names are consistent."""

  # internal consistency of coordinates
  for dim, coord in axes.items():
    if coord.ndim > 1:
      raise ValueError(
          f'all coordinates in the axes dict must be 1D, got {coord} for '
          f'dimension {dim}. Consider using Field.tag() instead to associate '
          'multi-dimensional coordinates.'
      )
    if (dim,) != coord.dims:
      raise ValueError(
          f'coordinate under key {dim!r} in the axes dict must have '
          f'dims={(dim,)!r} but got {coord.dims=}'
      )

  data_dims = set(named_array.named_dims)
  keys_dims = set(_remove_dummy_axes(axes).keys())
  if not keys_dims <= data_dims:
    raise ValueError(
        'axis keys must be a subset of the named dimensions of the '
        f'underlying named array, got axis keys {keys_dims} vs '
        f'data dimensions {data_dims}'
    )

  for dim, coord in axes.items():
    if named_array.named_shape[dim] != coord.sizes[dim]:
      raise ValueError(
          f'inconsistent size for dimension {dim!r} between data and'
          f' coordinates: {named_array.named_shape[dim]} vs'
          f' {coord.sizes[dim]} on named array vs'
          f' coordinate:\n{named_array}\n{coord}'
      )


def _remove_dummy_axes(axes: dict[str, Coordinate]) -> dict[str, Coordinate]:
  """Removes dummy axes from a dict of coordinates."""
  return {k: v for k, v in axes.items() if not isinstance(v, DummyAxis)}


def _swapped_binop(binop):
  """Swaps the order of operations for a binary operation."""

  def swapped(x, y):
    return binop(y, x)

  return swapped


def _wrap_scalar_conversion(scalar_conversion):
  """Wraps a scalar conversion operator on a Field."""

  def wrapped_scalar_conversion(self: Field):
    if self.shape:
      raise ValueError(
          f'Cannot convert a non-scalar Field with {scalar_conversion}'
      )
    return scalar_conversion(self.data)

  return wrapped_scalar_conversion


def _wrap_array_method(name):
  """Wraps an array method on a Field."""

  def func(array, *args, **kwargs):
    return getattr(array, name)(*args, **kwargs)

  array_method = getattr(jax.Array, name)
  wrapped_func = cmap(func)
  functools.update_wrapper(
      wrapped_func,
      array_method,
      assigned=('__name__', '__qualname__', '__annotations__'),
      updated=(),
  )
  wrapped_func.__module__ = __name__
  wrapped_func.__doc__ = (
      'Name-vectorized version of array method'
      f' ``{name} <numpy.ndarray.{name}>``. Takes similar arguments as'
      f' ``{name} <numpy.ndarray.{name}>`` but accepts and returns Fields'
      ' in place of regular arrays.'
  )
  return wrapped_func


def _in_treescope_abbreviation_mode() -> bool:
  """Returns True if treescope.abbreviation is set by context or globally."""
  return treescope.abbreviation_threshold.get() is not None


@utils.export
@jax.tree_util.register_pytree_node_class
class Field:
  # pylint: disable=line-too-long
  # fmt: off
  """An array with optional named dimensions and associated coordinates.

  Examples:
    >>> import coordax as cx
    >>> import jax.numpy as jnp
    >>> x = cx.SizedAxis('x', 2)
    >>> field = cx.Field(jnp.ones((2, 3, 4)), dims=('x', 'y', None), axes={'x': x})
    >>> field
    <Field dims=('x', 'y', None) shape=(2, 3, 4) axes={'x': SizedAxis} >
    >>> field.data  # doctest: +ELLIPSIS
    Array([[[1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.]],
    <BLANKLINE>
           [[1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.]]], dtype=float32)
    >>> field.dims
    ('x', 'y', None)
    >>> field.axes
    {'x': coordax.SizedAxis('x', size=2)}
    >>> field.shape
    (2, 3, 4)
    >>> field.named_shape
    {'x': 2, 'y': 3}
    >>> field.positional_shape
    (4,)
    >>> field.named_dims
    ('x', 'y')
    >>> field.named_axes
    {'x': 0, 'y': 1}
    >>> field.coord_fields
    {}
    >>> field.coordinate
    CartesianProduct(coordinates=(coordax.SizedAxis('x', size=2), coordax.DummyAxis('y', size=3), coordax.DummyAxis(None, size=4)))
  """
  # fmt on

  _named_array: named_axes_lib.NamedArray
  _axes: dict[str, Coordinate]

  def __init__(
      self,
      data: ArrayLike,
      dims: tuple[str | None, ...] | None = None,
      axes: dict[str, Coordinate] | None = None,
  ):
    """Construct a Field.

    Args:
      data: the underlying data array.
      dims: optional tuple of dimension names, with the same length as
        ``data.ndim``. Strings indicate named axes, and may not be repeated.
        ``None`` indicates positional axes. If ``dims`` is not provided, all
        axes are positional.
      axes: optional mapping from dimension names to associated
        ``coordax.Coordinate`` objects.

    Examples:
      >>> import coordax as cx
      >>> import jax.numpy as jnp
      >>> cx.Field(jnp.ones((2, 3)), dims=('x', 'y'))
      <Field dims=('x', 'y') shape=(2, 3) axes={} >

    See also:
      :func:`coordax.field`
    """
    self._named_array = named_axes_lib.NamedArray(data, dims)
    if axes is None:
      self._axes = {}
    else:
      _check_valid(self._named_array, axes)
      self._axes = _remove_dummy_axes(axes)

  @classmethod
  def from_namedarray(
      cls,
      named_array: named_axes_lib.NamedArray,
      axes: dict[str, Coordinate] | None = None,
  ) -> Self:
    """Creates a Field from a named array."""
    return cls(named_array.data, named_array.dims, axes)

  @classmethod
  def from_xarray(
      cls,
      data_array: xarray.DataArray,
      coord_types: Sequence[type[Coordinate]] = (LabeledAxis, DummyAxis),
  ) -> Field:
    """Deprecated alias for coordax.from_xarray."""
    warnings.warn(
        'cx.Field.from_xarray() is deprecated, use cx.from_xarray() instead',
        DeprecationWarning,
        stacklevel=2,
    )
    return from_xarray(data_array, coord_types)

  def to_xarray(self) -> xarray.DataArray:
    """Convert this Field to an xarray.DataArray with NumPy array data.

    Returns:
      An xarray.DataArray object with the same data as the input coordax.Field.
      This DataArray will still be wrapping a jax.Array, and have operations
      implemented on jax.Array objects using the Python Array API interface.

    Examples:
      >>> import coordax as cx
      >>> import jax.numpy as jnp
      >>> import numpy as np
      >>> field = cx.field(jnp.zeros((2, 3)), 'x', 'y')
      >>> field.to_xarray()  # doctest: +ELLIPSIS
      <xarray.DataArray (x: 2, y: 3)>...
      array([[0., 0., 0.],
             [0., 0., 0.]], dtype=float32)
      Dimensions without coordinates: x, y
    """
    import xarray

    if not all(isinstance(dim, str) for dim in self.dims):
      raise ValueError(
          'can only convert Field objects with fully named dimensions to '
          f'xarray.DataArray objects, got dimensions {self.dims!r}'
      )

    # TODO(shoyer): Consider making this conversion optional, for use cases
    # where it is desirable to wrap jax.Array objects inside Xarray.
    data = ndarrays.to_numpy_array(self.data)

    coords = {}
    for coord in self.axes.values():
      for name, variable in coord.to_xarray().items():
        if name in coords and not variable.identical(coords[name]):
          raise ValueError(
              f'inconsistent coordinate fields for {name!r}:\n'
              f'{variable}\nvs\n{coords[name]}'
          )
        coords[name] = variable

    return xarray.DataArray(data=data, dims=self.dims, coords=coords)

  @property
  def named_array(self) -> named_axes_lib.NamedArray:
    """The value of the underlying named array."""
    return self._named_array

  @property
  def axes(self) -> dict[str, Coordinate]:
    """This field's coordinates, as a dict of 1d coordinates."""
    return self._axes

  @property
  def coordinate(self) -> Coordinate:
    """This field's coordinate, as a single Coordinate object."""
    return get_coordinate(self)

  @property
  def data(self) -> Array:
    """The value of the underlying data array."""
    return self.named_array.data

  @property
  def dtype(self) -> np.dtype | None:
    """The dtype of the field."""
    return self.named_array.dtype

  @property
  def named_shape(self) -> dict[str, int]:
    """A mapping of axis names to their sizes."""
    return self.named_array.named_shape

  @property
  def positional_shape(self) -> tuple[int, ...]:
    """A tuple of axis sizes for any anonymous axes."""
    return self.named_array.positional_shape

  @property
  def shape(self) -> tuple[int, ...]:
    """A tuple of axis sizes of the underlying data array."""
    return self.named_array.shape

  @property
  def ndim(self) -> int:
    """Number of dimensions in the underlying data array."""
    return len(self.dims)

  @property
  def dims(self) -> tuple[str | None, ...]:
    """Named and unnamed dimensions of this array."""
    return self.named_array.dims

  @property
  def named_dims(self) -> tuple[str, ...]:
    """Named dimensions of this array."""
    return self.named_array.named_dims

  @property
  def named_axes(self) -> dict[str, int]:
    """Mapping from dimension names to axis positions."""
    return self.named_array.named_axes

  @property
  def coord_fields(self) -> dict[str, Field]:
    """A mapping from coordinate field names to their values."""
    return functools.reduce(
        operator.or_, [c.fields for c in self.axes.values()], {}
    )

  def tree_flatten(self):
    """Flatten this object for JAX pytree operations."""
    return [self.named_array], tuple(sorted(self.axes.items()))

  @classmethod
  def tree_unflatten(cls, axes, leaves) -> Self:
    """Unflatten this object for JAX pytree operations."""
    [named_array] = leaves
    result = object.__new__(cls)
    result._named_array = named_array
    result._axes = dict(axes)
    if isinstance(named_array.data, Array):
      _check_valid(result.named_array, result.axes)
    return result

  def unwrap(self, *names: str | Coordinate) -> Array:
    """Extracts underlying data from a field without named dimensions.

    This is effectively syntactic sugar for ``assert field.named_dims ==
    names``,
    followed by returning ``field.data``.

    Args:
      *names: Names of dimensions to check against.

    Returns:
      The underlying data array.

    Raises:
      ValueError: If the field has named dimensions that do not match ``names``.

    Examples:
      >>> import coordax as cx
      >>> import jax.numpy as jnp
      >>> field = cx.field(jnp.ones((2, 3)), 'x', 'y')
      >>> field.unwrap('x', 'y')
      Array([[1., 1., 1.], [1., 1., 1.]], dtype=float32)

      >>> field.unwrap('y', 'x')  # doctest: +NORMALIZE_WHITESPACE
      Traceback (most recent call last):
      ...
      ValueError: Field has self.named_dims=('x', 'y') but names=('y', 'x') were
      requested.
    """
    names = _dimension_names(*names)
    if names != self.named_dims:
      raise ValueError(
          f'Field has {self.named_dims=} but {names=} were requested.'
      )
    return self.data

  def _validate_matching_coords(
      self,
      dims_or_coords: Sequence[str | Coordinate | types.EllipsisType],
  ):
    """Validate that given coordinates are all found on this field."""
    axes = []
    for part in dims_or_coords:
      if isinstance(part, Coordinate) and not isinstance(part, DummyAxis):
        axes.extend(part.axes)

    for c in axes:
      [dim] = c.dims
      if dim not in self.axes:
        raise ValueError(
            f'coordinate not found on this field:\n{c}\n'
            f'not found in coordinates {list(self.axes)}'
        )
      if self.axes[dim] != c:
        raise ValueError(
            'coordinate not equal to the corresponding coordinate on this'
            f' field:\n{c}\nvs\n{self.axes[dim]}'
        )

  def untag(self, *axis_order: str | Coordinate) -> Field:
    """Returns a view of the field with the requested axes made positional.

    Args:
      *axis_order: Names or coordinates of the axes to untag.

    Returns:
      A new Field with the specified axes converted to positional dimensions.

    Examples:
      >>> import coordax as cx
      >>> import jax.numpy as jnp
      >>> x = cx.SizedAxis('x', 2)
      >>> field = cx.field(jnp.ones((2, 3)), x, 'y')

      Untag by name:

      >>> field.untag('x')
      <Field dims=(None, 'y') shape=(2, 3) axes={} >
      >>> field.untag('y')
      <Field dims=('x', None) shape=(2, 3) axes={'x': SizedAxis} >

      Untag by coordinate (validates that the coordinate matches):

      >>> field.untag(x)
      <Field dims=(None, 'y') shape=(2, 3) axes={} >

    See also:
      :meth:`coordax.Field.tag`
      :func:`coordax.untag`
    """
    self._validate_matching_coords(axis_order)
    untag_dims = _dimension_names(*axis_order)
    named_array = self.named_array.untag(*untag_dims)
    axes = {k: v for k, v in self.axes.items() if k not in untag_dims}
    result = Field.from_namedarray(named_array=named_array, axes=axes)
    return result

  def tag(self, *names: str | Coordinate | ellipsis | None) -> Field:
    """Returns a Field with attached coordinates to the positional axes.

    Args:
      *names: Names or coordinates to assign to the positional axes. The total
        number of dimensions corresponding to these objects must match the
        number of positional axes in the field unless ``...`` is used to
        indicate positions of untagged axes.

    Returns:
      A new Field with the specified dimensions tagged.

    Examples:
      >>> import coordax as cx
      >>> import jax.numpy as jnp
      >>> field = cx.Field(jnp.ones((2, 3)))
      >>> field.tag('x', 'y')
      <Field dims=('x', 'y') shape=(2, 3) axes={} >

      Tagging with ``Coordinate`` objects adds them to the field:

      >>> x = cx.SizedAxis('x', 2)
      >>> field.tag(x, 'y')
      <Field dims=('x', 'y') shape=(2, 3) axes={'x': SizedAxis} >

      ``None`` leaves a dimension untagged:

      >>> field.tag('x', None)
      <Field dims=('x', None) shape=(2, 3) axes={} >

      Ellipsis ``...`` can be indicate dimensions that should not be named:

      >>> field.tag(..., 'y')
      <Field dims=(None, 'y') shape=(2, 3) axes={} >

      Tagging with the wrong number of arguments raises an error:

      >>> field.tag('x')  # doctest: +NORMALIZE_WHITESPACE
      Traceback (most recent call last):
      ...
      ValueError: there must be exactly as many dimensions given to ``tag`` as
      there are positional axes in the array, but got ('x',) for 2 positional
      axes.

      You can also tag with multi-dimensional coordinates corresponding to
      multiple array axes. If the multi-dimensional coordinate is a
      :class:`coordax.CartesianProduct`, it is unpacked:

      >>> x = cx.SizedAxis('x', 2)
      >>> y = cx.SizedAxis('y', 3)
      >>> xy = cx.compose(x, y)
      >>> xy
      CartesianProduct(axes={'x': SizedAxis, 'y': SizedAxis})
      >>> field = cx.Field(jnp.zeros((2, 3)))
      >>> field.tag(xy)
      <Field dims=('x', 'y') shape=(2, 3) axes={'x': SizedAxis, 'y': SizedAxis}
      >

    See also:
      :meth:`coordax.Field.untag`
      :func:`coordax.tag`
    """
    tag_dims = _dimension_names(*names)
    tagged_array = self.named_array.tag(*tag_dims)
    axes = {}
    axes.update(self.axes)
    for c in names:
      if isinstance(c, Coordinate):
        for dim, axis in zip(c.dims, c.axes):
          # TODO(shoyer): consider raising an error if an unnamed axis has the
          # wrong size.
          if dim is not None:
            axes[dim] = axis
    result = Field.from_namedarray(tagged_array, axes)
    return result

  # Note: Can't call this "transpose" like Xarray, to avoid conflicting with the
  # positional only ndarray method.
  def order_as(
      self, *axis_order: str | Coordinate | types.EllipsisType
  ) -> Field:
    """Returns a field with the axes in the given order.

    Args:
      *axis_order: The desired order of axes, specified by name or coordinate.
        ``...`` may be used once, to indicate all other dimensions in order of
        appearance on this array.

    Returns:
      A new Field with the axes permuted to match the requested order.

    Examples:
      >>> import coordax as cx
      >>> import jax.numpy as jnp
      >>> field = cx.field(jnp.ones((2, 3)), 'x', 'y')
      >>> field.order_as('y', 'x')
      <Field dims=('y', 'x') shape=(3, 2) axes={} >
      >>> field.order_as(..., 'x')
      <Field dims=('y', 'x') shape=(3, 2) axes={} >
    """
    self._validate_matching_coords(axis_order)
    ordered_dims = _dimension_names(*axis_order)
    ordered_array = self.named_array.order_as(*ordered_dims)
    result = Field.from_namedarray(ordered_array, self.axes)
    return result

  def broadcast_like(self, other: Self | Coordinate) -> Self:
    """Returns a field broadcasted like ``other``.

    Args:
      other: The field or coordinate to broadcast to.

    Returns:
      A new Field broadcasted to match ``other``.

    Examples:
      >>> import coordax as cx
      >>> import jax.numpy as jnp
      >>> field = cx.field(jnp.zeros((2,)), 'x')
      >>> other = cx.field(jnp.zeros((2, 3)), 'x', 'y')
      >>> field.broadcast_like(other)
      <Field dims=('x', 'y') shape=(2, 3) axes={} >
    """
    if isinstance(other, Coordinate):
      other = shape_struct_field(other)
    for k, v in self.axes.items():
      if other.axes.get(k) != v:
        raise ValueError(
            'cannot broadcast field because axes corresponding to dimension '
            f'{k!r} do not match: {v} vs {other.axes.get(k)}'
        )
    return Field.from_namedarray(
        self.named_array.broadcast_like(other.named_array), other.axes
    )

  def __repr__(self):
    if _in_treescope_abbreviation_mode():
      return treescope.render_to_text(self)
    else:
      with treescope.abbreviation_threshold.set_scoped(1):
        with treescope.using_expansion_strategy(9, 80):
          return treescope.render_to_text(self)

  def __treescope_repr__(self, path: str | None, subtree_renderer: Any):
    """Treescope handler for Field."""

    def _make_label():
      # reuse dim/shape summary from the underlying NamedArray.
      attrs, summary, _ = named_axes_lib.attrs_summary_type(
          self.named_array, False
      )
      axes_attrs = _axes_attrs(self)
      attrs = ' '.join([attrs, f'axes={axes_attrs}'])

      return rendering_parts.summarizable_condition(
          summary=rendering_parts.abbreviation_color(  # non-expanded repr.
              rendering_parts.text(f'<{type(self).__name__} {attrs} {summary}>')
          ),
          detail=rendering_parts.siblings(
              rendering_parts.text(f'<{type(self).__name__} ('),
          ),
      )

    children = rendering_parts.build_field_children(
        self,
        path,
        subtree_renderer,
        fields_or_attribute_names=('dims', 'shape', 'axes'),
    )
    indented_children = rendering_parts.indented_children(children)
    return rendering_parts.build_custom_foldable_tree_node(
        contents=rendering_parts.summarizable_condition(
            detail=rendering_parts.siblings(indented_children, ')')
        ),
        label=lowering.maybe_defer_rendering(
            main_thunk=lambda _: _make_label(),
            placeholder_thunk=_make_label,
        ),
        path=path,
        expand_state=rendering_parts.ExpandState.WEAKLY_COLLAPSED,
    )

  def __treescope_ndarray_adapter__(self):
    """Treescope handler for named arrays."""

    def _summary_fn(field, inspect_data):
      attrs, array_summary, data_type = named_axes_lib.attrs_summary_type(
          field.named_array, inspect_data
      )
      axes_attrs = _axes_attrs(field)
      attrs = ', '.join([attrs, f'axes={axes_attrs}'])
      return attrs, array_summary, data_type

    return named_axes_lib.NamedArrayAdapter(_summary_fn)

  # Convenience wrappers: Elementwise infix operators.
  __lt__ = _cmap_with_doc(operator.lt, 'jax.Array.__lt__')
  __le__ = _cmap_with_doc(operator.le, 'jax.Array.__le__')
  __eq__ = _cmap_with_doc(operator.eq, 'jax.Array.__eq__')
  __ne__ = _cmap_with_doc(operator.ne, 'jax.Array.__ne__')
  __ge__ = _cmap_with_doc(operator.ge, 'jax.Array.__ge__')
  __gt__ = _cmap_with_doc(operator.gt, 'jax.Array.__gt__')

  __add__ = _cmap_with_doc(operator.add, 'jax.Array.__add__')
  __sub__ = _cmap_with_doc(operator.sub, 'jax.Array.__sub__')
  __mul__ = _cmap_with_doc(operator.mul, 'jax.Array.__mul__')
  __truediv__ = _cmap_with_doc(operator.truediv, 'jax.Array.__truediv__')
  __floordiv__ = _cmap_with_doc(operator.floordiv, 'jax.Array.__floordiv__')
  __mod__ = _cmap_with_doc(operator.mod, 'jax.Array.__mod__')
  __divmod__ = _cmap_with_doc(divmod, 'jax.Array.__divmod__')
  __pow__ = _cmap_with_doc(operator.pow, 'jax.Array.__pow__')
  __lshift__ = _cmap_with_doc(operator.lshift, 'jax.Array.__lshift__')
  __rshift__ = _cmap_with_doc(operator.rshift, 'jax.Array.__rshift__')
  __and__ = _cmap_with_doc(operator.and_, 'jax.Array.__and__')
  __or__ = _cmap_with_doc(operator.or_, 'jax.Array.__or__')
  __xor__ = _cmap_with_doc(operator.xor, 'jax.Array.__xor__')

  __radd__ = _cmap_with_doc(_swapped_binop(operator.add), 'jax.Array.__radd__')
  __rsub__ = _cmap_with_doc(_swapped_binop(operator.sub), 'jax.Array.__rsub__')
  __rmul__ = _cmap_with_doc(_swapped_binop(operator.mul), 'jax.Array.__rmul__')
  __rtruediv__ = _cmap_with_doc(
      _swapped_binop(operator.truediv), 'jax.Array.__rtruediv__'
  )
  __rfloordiv__ = _cmap_with_doc(
      _swapped_binop(operator.floordiv), 'jax.Array.__rfloordiv__'
  )
  __rmod__ = _cmap_with_doc(_swapped_binop(operator.mod), 'jax.Array.__rmod__')
  __rdivmod__ = _cmap_with_doc(_swapped_binop(divmod), 'jax.Array.__rdivmod__')
  __rpow__ = _cmap_with_doc(_swapped_binop(operator.pow), 'jax.Array.__rpow__')
  __rlshift__ = _cmap_with_doc(
      _swapped_binop(operator.lshift), 'jax.Array.__rlshift__'
  )
  __rrshift__ = _cmap_with_doc(
      _swapped_binop(operator.rshift), 'jax.Array.__rrshift__'
  )
  __rand__ = _cmap_with_doc(_swapped_binop(operator.and_), 'jax.Array.__rand__')
  __ror__ = _cmap_with_doc(_swapped_binop(operator.or_), 'jax.Array.__ror__')
  __rxor__ = _cmap_with_doc(_swapped_binop(operator.xor), 'jax.Array.__rxor__')

  __abs__ = _cmap_with_doc(operator.abs, 'jax.Array.__abs__')
  __neg__ = _cmap_with_doc(operator.neg, 'jax.Array.__neg__')
  __pos__ = _cmap_with_doc(operator.pos, 'jax.Array.__pos__')
  __invert__ = _cmap_with_doc(operator.inv, 'jax.Array.__invert__')

  # Convenience wrappers: Scalar conversions.
  __bool__ = _wrap_scalar_conversion(bool)
  __complex__ = _wrap_scalar_conversion(complex)
  __int__ = _wrap_scalar_conversion(int)
  __float__ = _wrap_scalar_conversion(float)
  __index__ = _wrap_scalar_conversion(operator.index)

  # elementwise operations
  astype = _wrap_array_method('astype')
  clip = _wrap_array_method('clip')
  conj = _wrap_array_method('conj')
  conjugate = _wrap_array_method('conjugate')
  imag = _wrap_array_method('imag')
  real = _wrap_array_method('real')
  round = _wrap_array_method('round')
  view = _wrap_array_method('view')

  # Intentionally not included: anything that acts on a subset of axes or takes
  # an axis as an argument (e.g., mean). It is ambiguous whether these should
  # act over positional or named axes.
  # TODO(shoyer): re-write some of these with explicit APIs similar to xarray.

  # maybe include some of below with names that signify positional nature?
  # reshape = _wrap_array_method('reshape')
  # squeeze = _wrap_array_method('squeeze')
  # transpose = _wrap_array_method('transpose')
  # T = _wrap_array_method('T')
  # mT = _wrap_array_method('mT')  # pylint: disable=invalid-name


@utils.export
def field(array: ArrayLike, *names: str | Coordinate | None) -> Field:
  """Wraps a positional array as a ``Field``.

  ``cx.field(array, *names)`` is a shortcut for ``cx.Field(array).tag(*names)``.

  Args:
    array: the array to wrap.
    *names: the name or coordinates to attach to the array.

  Returns:
    A Field object.

  Examples:
    >>> import coordax as cx
    >>> import jax.numpy as jnp
    >>> cx.field(jnp.ones((2, 3)), 'x', 'y')
    <Field dims=('x', 'y') shape=(2, 3) axes={} >

  See also:
    :class:`coordax.Field`
    :meth:`coordax.Field.tag`
  """
  field_ = Field(array)
  if names:
    field_ = field_.tag(*names)
  return field_


@utils.export
def wrap(array: ArrayLike, *names: str | Coordinate | None) -> Field:
  """Deprecated alias for ``cx.field``."""
  warnings.warn(
      'cx.wrap() is deprecated, use cx.field() instead',
      DeprecationWarning,
      stacklevel=2,
  )
  return field(array, *names)


@utils.export
def wrap_like(array: ArrayLike, other: Field) -> Field:
  """Wraps ``array`` with the same coordinates as ``other``."""
  warnings.warn(
      'cx.wrap_like() is deprecated, use cx.field(array, other.coordinate) '
      'instead of cx.wrap_like(array, other)',
      DeprecationWarning,
      stacklevel=2,
  )
  if isinstance(array, jax.typing.ArrayLike):
    array = jnp.asarray(array)
  if array.shape != other.shape:
    raise ValueError(f'{array.shape=} and {other.shape=} must be equal')
  return Field(array, other.dims, other.axes)


@utils.export
def from_xarray(
    data_array: xarray.DataArray,
    coord_types: Sequence[type[Coordinate]] = (LabeledAxis, DummyAxis),
) -> Field:
  # pylint: disable=g-import-not-at-top,line-too-long
  # fmt: off
  """Create a coordax.Field from an xarray.DataArray.

  Args:
    data_array: xarray.DataArray to convert into a Field.
    coord_types: sequence of ``coordax.Coordinate`` subclasses with
      ``from_xarray`` methods defined. The first coordinate class that returns a
      coordinate object (indicating a match) will be used. By default,
      coordinates will use only generic ``LabeledAxis`` and ``DummyAxis``
      objects.

  Returns:
    A coordax.Field object with the same data as the input xarray.DataArray.

  Examples:
    >>> import coordax as cx
    >>> import xarray as xr
    >>> import numpy as np
    >>> da = xr.DataArray(np.zeros((2, 3)), dims=('x', 'y'), coords={'x': [1, 2]})
    >>> cx.from_xarray(da)
    <Field dims=('x', 'y') shape=(2, 3) axes={'x': LabeledAxis} >

  See also:
    :func:`coordax.coords.from_xarray`
  """
  # fmt: on
  data = data_array.data
  coord = coordinate_systems.from_xarray(data_array, coord_types)
  return field(data, coord)


@utils.export
def is_field(value) -> TypeGuard[Field]:
  """Returns True if ``value`` is of type ``Field``."""
  return isinstance(value, Field)


@utils.export
def shape_struct_field(*axes: Coordinate) -> Field:
  """Returns a Field with ``axes`` and a ShapeDtypeStruct in place of data.

  Args:
    *axes: The coordinates to use for the field.

  Returns:
    A Field with ``ShapeDtypeStruct`` data and the given axes.

  Examples:
    >>> import coordax as cx
    >>> import jax
    >>> axis = cx.SizedAxis('x', 5)
    >>> cx.shape_struct_field(axis)
    <Field dims=('x',) shape=(5,) axes={'x': SizedAxis} >
  """
  coordinate = coordinate_systems.compose(*axes)

  def _materialize_dummy_field() -> Field:
    return field(jnp.zeros(coordinate.shape), coordinate)

  return jax.eval_shape(_materialize_dummy_field)


MissingAxes = Literal['error', 'dummy', 'skip']


@utils.export
def get_coordinate(
    field: Field, *, missing_axes: MissingAxes = 'dummy'
) -> Coordinate:
  # fmt: off
  """Returns a single coordinate for a field.

  Args:
    field: coordax.Field from which the coordinate will be extracted.
    missing_axes: controls how axes without coorinates are handled. Options are:

      * ``'dummy'``: uses ``DummyAxis`` for dimensions without a coordinate.
      * ``'skip'``: ignores dimensions without a coordinate.
      * ``'error'``: raises if dimensions without a coordinate are present.

  Returns:
    Coordinate associated with the ``field``.

  Examples:
    >>> import coordax as cx
    >>> import jax.numpy as jnp
    >>> x = cx.SizedAxis('x', 2)
    >>> field = cx.field(jnp.zeros((2, 3)), x, 'y')
    >>> cx.get_coordinate(field)
    CartesianProduct(coordinates=(coordax.SizedAxis('x', size=2), coordax.DummyAxis('y', size=3)))
  """
  # fmt: on
  if missing_axes not in ('dummy', 'skip', 'error'):
    raise ValueError(
        'missing axes must be one of "dummy", "skip", or "error", got'
        f' {missing_axes!r}'
    )
  axes = []
  for d, s in zip(field.dims, field.shape, strict=True):
    if d in field.axes:
      axes.append(field.axes[d])
    elif missing_axes == 'dummy':
      axes.append(coordinate_systems.DummyAxis(d, s))
    elif missing_axes == 'error':
      raise ValueError(f'{field.dims=} has unnamed dims and {missing_axes=}')
  return coordinate_systems.compose(*axes)


PyTree = Any


@utils.export
def tag(tree: PyTree, *dims: str | Coordinate | ellipsis | None) -> PyTree:
  """Tag dimensions on all fields in a PyTree.

  Args:
    tree: The PyTree of fields.
    *dims: Names or coordinates to tag the positional axes with.

  Returns:
    A new PyTree with all fields tagged.

  Examples:
    >>> import coordax as cx
    >>> import jax.numpy as jnp
    >>> tree = {'a': cx.Field(jnp.zeros((2,)))}
    >>> tree
    {'a': <Field dims=(None,) shape=(2,) axes={} >}
    >>> cx.tag(tree, 'x')
    {'a': <Field dims=('x',) shape=(2,) axes={} >}

  See also:
    :meth:`coordax.Field.tag`
  """
  tag_arrays = lambda x: x.tag(*dims) if is_field(x) else x
  return jax.tree.map(tag_arrays, tree, is_leaf=is_field)


@utils.export
def untag(tree: PyTree, *dims: str | Coordinate) -> PyTree:
  """Untag dimensions from all fields in a PyTree.

  Args:
    tree: The PyTree of fields.
    *dims: The axes to untag.

  Returns:
    A new PyTree with all fields untagged.

  Examples:
    >>> import coordax as cx
    >>> import jax.numpy as jnp
    >>> tree = {'a': cx.field(jnp.zeros((2,)), 'x')}
    >>> tree
    {'a': <Field dims=('x',) shape=(2,) axes={} >}
    >>> cx.untag(tree, 'x')
    {'a': <Field dims=(None,) shape=(2,) axes={} >}

  See also:
    :meth:`coordax.Field.untag`
  """
  untag_arrays = lambda x: x.untag(*dims) if is_field(x) else x
  return jax.tree.map(untag_arrays, tree, is_leaf=is_field)
