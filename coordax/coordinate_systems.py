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
"""Coordinate systems for use on coordax.Field objects.

``Coordinate`` objects define a discretization schema, dimension names and
provide methods & coordinate field values to facilitate computations.
"""

from __future__ import annotations

import abc
import collections
from collections.abc import Callable, Iterable
import dataclasses
import functools
import itertools
import typing
from typing import Any, Literal, Self, TYPE_CHECKING, Type, TypeAlias, TypeGuard, TypeVar
import warnings

from coordax import utils
import jax
import numpy as np

if TYPE_CHECKING:
  # import only under TYPE_CHECKING to avoid circular dependency
  # pylint: disable=g-bad-import-order
  from coordax import fields  # pylint: disable=unused-import
  import xarray


Pytree: TypeAlias = Any
Sequence = collections.abc.Sequence

SelMethod: TypeAlias = Literal['nearest'] | None


@functools.partial(utils.export, module='coordax.coords')
@dataclasses.dataclass(frozen=True)
class NoCoordinateMatch:
  """For use when no Coordax coordinate matches xarray coordinate."""

  reason: str


def normalize_indexers(
    indexers: dict[str | Coordinate, Any] | None,
    **indexers_kwargs,
) -> dict[str | Coordinate, Any]:
  """Returns indexers replacing sequence with arrays, checks spec mode."""
  if indexers is None:  # only kwargs is allowed.
    normalized_indexers = dict(indexers_kwargs)
  else:
    if not isinstance(indexers, dict):
      raise ValueError(f'Indexers must be a dict, got {type(indexers)}.')
    if indexers_kwargs:
      raise ValueError(
          'Using dict and kwarg indexers simultaneously is dangerous and not'
          f' supported, got {indexers=}, {indexers_kwargs=}.'
      )
    normalized_indexers = indexers
  seq_to_array = lambda x: np.asarray(x) if isinstance(x, Sequence) else x
  return {k: seq_to_array(v) for k, v in normalized_indexers.items()}


@utils.export
class Coordinate(abc.ABC):
  """Abstract class for coordinate objects.

  Coordinate subclasses are expected to obey several invariants:
  1. Dimension names may not be repeated: ``len(set(dims)) == len(dims)``
  2. All dimensions must be named: ``len(shape) == len(dims)``

  Every non-abstract Coordinate subclass must be registered as a "static"
  pytree node, e.g., by decorating the class with
  ``@jax.tree_util.register_static``. Static pytrees nodes must implement
  ``__hash__`` and ``__eq__`` according to the requirements of keys in Python
  dictionaries. This is easiest to acheive with frozen dataclasses, but care
  must be taken when working with np.ndarray attributes.
  """

  @property
  @abc.abstractmethod
  def dims(self) -> tuple[str | None, ...]:
    """Dimension names of the coordinate.

    All subclasses must return a tuple of dimension names as strings, with the
    exception of ``DummyAxis``.
    """
    raise NotImplementedError()

  @property
  @abc.abstractmethod
  def shape(self) -> tuple[int, ...]:
    """Shape of the coordinate."""
    raise NotImplementedError()

  @property
  def fields(self) -> dict[str, 'fields.Field']:
    """Optional dict that maps from field names to their values."""
    return {}

  @property
  def sizes(self) -> dict[str, int]:
    """Sizes of all dimensions on this coordinate."""
    return {
        dim: size for dim, size in zip(self.dims, self.shape) if dim is not None
    }

  @property
  def ndim(self) -> int:
    """Dimensionality of the coordinate."""
    return len(self.dims)

  @property
  def axes(self) -> tuple[Coordinate, ...]:
    """Tuple of one-dimensional Coordinate objects for each dimension."""
    if self.ndim == 1:
      return (self,)
    else:
      return tuple(SelectedAxis(self, i) for i in range(self.ndim))

  def isel(
      self,
      indexers: dict[str | Coordinate, Any] | None = None,
      **indexers_kwargs,
  ) -> Coordinate:
    """Returns a new coordinate with the given integer indexers applied.

    Note: This is an experimental feature, and may be changed or completely
    removed in the future.

    Note that ``isel`` should only be used with integer or slice indexers, or
    array-like objects with integer or slice values. All keys provided to `isel`
    are expected to be present in the dimension names or axes of a the
    coordinate. By default, empty and ``slice(None)`` indexers return the
    original coordinate. Otherwise a subclass-specific ``_isel`` is called.

    For label-based selection, use ``sel`` instead that is implemented using
    ``map_indexers`` in subclasses that allows mapping from label indexers to
    integer indexers used by ``isel``.

    Args:
      indexers: A mapping from dimensions to indices, slices, or arrays.
      **indexers_kwargs: The keyword arguments form of ``indexers``.

    Returns:
      A new coordinate with the selection applied.

    Examples:
      >>> import coordax as cx
      >>> import numpy as np
      >>> x = cx.SizedAxis('x', 5)
      >>> x.isel(x=0)
      Scalar()
      >>> x.isel(x=slice(1, 4))
      coordax.SizedAxis('x', size=3)
      >>> x.isel(x=[0, 2, 4])
      coordax.SizedAxis('x', size=3)

      >>> y = cx.LabeledAxis('y', np.arange(5))
      >>> y.isel(y=slice(0, 2))
      coordax.LabeledAxis('y', ticks=array([0, 1]))
    """
    indexers = normalize_indexers(indexers, **indexers_kwargs)
    self._validate_indexers(indexers)

    indexer = indexers.get(self)
    if not indexers or (isinstance(indexer, slice) and indexer == slice(None)):
      return self

    return self._isel(indexers)

  def _isel(self, indexers: dict[str | Coordinate, Any]) -> Coordinate:
    """Returns a new coordinate with the given indexers applied.

    Args:
      indexers: A mapping from dimensions to indices, slices, or arrays.
        Guaranteed to be non-empty and contain only valid keys for this
        coordinate.
    """
    raise NotImplementedError(f'{type(self).__name__} does not implement _isel')

  def _validate_indexers(self, indexers: dict[str | Coordinate, Any]) -> None:
    """Validates that the integer indexers are valid for this coordinate."""
    unknown_dims = set(indexers.keys()) - set(self.axes) - set(self.dims)
    if unknown_dims:
      for k in unknown_dims:
        if is_coord(k) and k.ndim > 1:
          raise ValueError(
              f'Indexing with coordinate {k=} with ndim > 1 is not supported.'
          )
      raise ValueError(
          f'Dimensions {unknown_dims} do not exist in coordinate'
          f' {type(self).__name__}'
      )

  def sel(
      self,
      indexers: dict[str | Coordinate, Any] | None = None,
      method: Literal['nearest'] | None = None,
      **indexers_kwargs,
  ) -> Coordinate:
    """Returns a new coordinate with the given selection applied.

    Note: This is an experimental feature, and may be changed or completely
    removed in the future.

    ``sel`` is designed to work with label-based indexers, which may include
    indexers that are not in the coordinate dimensions. The selection is
    accomplished by mapping label-based indexers to integer-based indexers via
    ``map_indexers``, which can be customized in subclasses. Indexers that are
    not processed by ``map_indexers`` are considered "unused" and will raise an
    error.

    Args:
      indexers: A mapping specifying labels to be selected from the coordinate.
      method: Optional method to use for inexact matches. Cannot be used when
        ``indexers`` contain slices. Default is `None`.
      **indexers_kwargs: The keyword arguments form of ``indexers``.

    Returns:
      A new coordinate with the selection applied.

    Examples:
      >>> import coordax as cx
      >>> import numpy as np
      >>> x = cx.LabeledAxis('x', np.array([10, 20, 30]))
      >>> x.sel(x=20)
      Scalar()
      >>> x.sel(x=slice(10, 20))
      coordax.LabeledAxis('x', ticks=array([10, 20]))
      >>> x.sel(x=12, method='nearest')
      Scalar()
    """
    sel_indexers = normalize_indexers(indexers, **indexers_kwargs)
    if not sel_indexers:
      return self

    unpacked_indexers, unpacked_c = unpack_and_validate_indexers(sel_indexers)
    mapped_indexers, consumed = self.map_indexers(
        unpacked_indexers, method=method
    )

    final_consumed = set()
    for c in consumed:
      if c in unpacked_c:
        final_consumed.add(unpacked_c[c])
      else:
        final_consumed.add(c)

    unused_sel_indexers = set(sel_indexers.keys()) - final_consumed
    if unused_sel_indexers:
      raise ValueError(
          f'Indexers {unused_sel_indexers} were not processed by any component'
          f' in {self}'
      )

    return self.isel(mapped_indexers)

  def map_indexers(
      self,
      indexers: dict[str | Coordinate, Any],
      method: Literal['nearest'] | None = None,
  ) -> tuple[dict[str | Coordinate, Any], set[str | Coordinate]]:
    """Maps label-based indexers to integer-based indexers (indices/slices)."""
    del method  # unused.
    # support `sel` on composite coords as long as they are not being indexed.
    if not indexers or not any(dim in indexers for dim in self.dims + (self,)):
      return {}, set()
    raise NotImplementedError(
        f'{type(self).__name__} does not implement map_indexers'
    )

  def to_xarray(self) -> dict[str, xarray.Variable]:
    """Convert this coordinate into xarray variables."""
    import xarray  # pylint: disable=g-import-not-at-top

    variables = {}
    dims_set = {dim for dim in self.dims if dim is not None}
    for name, coord_field in self.fields.items():
      if set(coord_field.dims) <= dims_set:
        # xarray.DataArray coordinate dimensions must be a subset of the
        # dimensions of the associated DataArray, which is not necessarily a
        # constraint for coordax.Field.
        variables[name] = xarray.Variable(coord_field.dims, coord_field.data)
    return variables

  @classmethod
  def from_xarray(
      cls, dims: tuple[str, ...], coords: xarray.Coordinates
  ) -> Self | NoCoordinateMatch:
    """Construct a matching Coordax coordinate from xarray, if possible.

    Args:
      dims: tuple of dimension names. Only the leading dimensions should be
        checks for a match.
      coords: xarray.Coordinates object providing dimension sizes and coordinate
        values.

    Returns:
      A matching instance of this coordinate or ``NoCoordinateMatch`` if this
      coordinate type does not match the xarray dimensions and coordinates.

    See also:
      :func:`coordax.from_xarray`
      :func:`coordax.coords.from_xarray`
    """
    raise NotImplementedError('from_xarray not implemented')


def unpack_and_validate_indexers(
    indexers: dict[str | Coordinate, Any],
) -> tuple[dict[str | Coordinate, Any], dict[str | Coordinate, Coordinate]]:
  """Unpacks multidimensional indexers and raises if slice.step is not None."""
  unpacked_indexers = {}
  unpacked_coords = {}
  # pytype: disable=attribute-error
  for k, v in indexers.items():
    if is_coord(k) and k.ndim > 1:
      key_coord = k
      if is_coord(v):
        if key_coord.dims != v.dims:
          raise ValueError(
              f'{key_coord.dims=} do not match indexer dimensions {v.dims=}.'
          )
        for ax, v_ax in zip(key_coord.axes, v.axes):
          unpacked_indexers[ax] = v_ax
          assert isinstance(ax, Coordinate)
          assert isinstance(key_coord, Coordinate)
          unpacked_coords[ax] = key_coord
      else:
        raise ValueError(
            f'Indexer for {key_coord=} with {key_coord.ndim=} > 0 must be a'
            f' coordinate, got {v}.'
        )
    else:
      if isinstance(v, slice) and v.step is not None:
        raise ValueError(
            f'Indexer for {k=} uses slice with {v.step=} != None, which is '
            'not supported.'
        )
      unpacked_indexers[k] = v
  # pytype: enable=attribute-error
  return unpacked_indexers, unpacked_coords


@functools.partial(utils.export, module='coordax.coords')
@dataclasses.dataclass(frozen=True)
class ArrayKey:
  """Wrapper for a numpy array to make it hashable."""

  value: np.ndarray

  def __eq__(self, other):
    return (
        isinstance(self, ArrayKey)
        and self.value.dtype == other.value.dtype
        and self.value.shape == other.value.shape
        and (self.value == other.value).all()
    )

  def __hash__(self) -> int:
    return hash((self.value.shape, self.value.tobytes()))


@utils.export
def is_coord(obj: Any) -> TypeGuard[Coordinate]:
  """Returns True if obj is a Coordinate."""
  return isinstance(obj, Coordinate)


def contains_dims(
    coord: Coordinate,
    *dims: str | Coordinate,
) -> bool:
  """Returns True if coordinate contains the given dimensions or coordinates."""
  dim_names = filter(lambda x: isinstance(x, str), dims)
  dim_coords = filter(is_coord, dims)
  dim_axes = _concat_tuples((c.axes for c in dim_coords))  # pytype: disable=attribute-error
  contain_names = set(dim_names).issubset(coord.dims)
  contain_coords = set(dim_axes).issubset(coord.axes)
  return contain_names and contain_coords


@utils.export
@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class Scalar(Coordinate):
  """Zero dimensional sentinel coordinate used to label scalar fields."""

  @property
  def dims(self) -> tuple[str, ...]:
    return ()

  @property
  def shape(self) -> tuple[int, ...]:
    return ()

  def map_indexers(
      self,
      indexers: dict[str | Coordinate, Any],
      method: Literal['nearest'] | None = None,
  ) -> tuple[dict[str | Coordinate, Any], set[str | Coordinate]]:
    del indexers, method  # unused.
    return {}, set()


@utils.export
@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class SelectedAxis(Coordinate):
  """Coordinate that exposes one dimension of a multidimensional coordinate."""

  coordinate: Coordinate
  axis: int

  def __post_init__(self):
    if self.axis >= self.coordinate.ndim:
      raise ValueError(
          f'Dimension {self.axis=} of {self.coordinate=} is out of bounds'
      )
    if self.coordinate.dims[self.axis] is None:
      raise ValueError(
          f'dimension {self.axis=} of {self.coordinate=} is not named'
      )

  def map_indexers(
      self,
      indexers: dict[str | Coordinate, Any],
      method: Literal['nearest'] | None = None,
  ) -> tuple[dict[str | Coordinate, Any], set[str | Coordinate]]:
    return self.coordinate.map_indexers(indexers, method=method)

  def _isel(self, indexers: dict[str | Coordinate, Any]) -> Coordinate:
    start_ndim = self.coordinate.ndim
    new_coord = self.coordinate.isel(indexers)
    if new_coord.ndim == start_ndim:
      return SelectedAxis(new_coord, self.axis)
    return Scalar()

  @property
  def dims(self) -> tuple[str, ...]:
    """Dimension names of the coordinate."""
    dim = self.coordinate.dims[self.axis]
    assert dim is not None
    return (dim,)

  @property
  def shape(self) -> tuple[int, ...]:
    """Shape of the coordinate."""
    return (self.coordinate.shape[self.axis],)

  @property
  def fields(self) -> dict[str, 'fields.Field']:
    """A maps from field names to their values."""
    return self.coordinate.fields

  def __repr__(self):
    return f'coordax.SelectedAxis({self.coordinate!r}, axis={self.axis})'

  def to_xarray(self) -> dict[str, xarray.Variable]:
    """Convert this coordinate into xarray variables."""
    # Override the default method to avoid restricting variables to only those
    # along the selected axis.
    return self.coordinate.to_xarray()


def _expand_coordinates(*coordinates: Coordinate) -> tuple[Coordinate, ...]:
  """Expands coordinates, removing CartesianProducts and Scalars."""
  expanded = []
  for c in coordinates:
    if isinstance(c, CartesianProduct):
      expanded.extend(c.coordinates)
    elif isinstance(c, Scalar):
      pass
    else:
      expanded.append(c)
  return tuple(expanded)


def _consolidate_coordinates(
    *coordinates: Coordinate,
) -> tuple[Coordinate, ...]:
  """Consolidates coordinates, removing SelectedAxes when possible."""
  axes = []
  result = []

  def reset_axes():
    result.extend(axes)
    axes[:] = []

  def append_axis(c):
    axes.append(c)
    if len(axes) == c.coordinate.ndim:
      # sucessful consolidation
      result.append(c.coordinate)
      axes[:] = []

  for c in coordinates:
    if isinstance(c, SelectedAxis) and c.axis == 0:
      # new SelectedAxis to consider consolidating
      reset_axes()
      append_axis(c)
    elif (
        isinstance(c, SelectedAxis)
        and axes
        and c.axis == len(axes)
        and c.coordinate == axes[-1].coordinate
    ):
      # continued SelectedAxis to consolidate
      append_axis(c)
    else:
      # coordinate cannot be consolidated
      reset_axes()
      result.append(c)

  reset_axes()

  return tuple(result)


@functools.partial(utils.export, module='coordax.coords')
def canonicalize(*coordinates: Coordinate) -> tuple[Coordinate, ...]:
  """Canonicalize coordinates into a minimum equivalent collection.

  Args:
    *coordinates: The coordinates to canonicalize.

  Returns:
    A tuple of canonicalized coordinates, where ``CartesianProduct`` objects are
    flattened, ``Scalar`` objects are removed and ``SelectedAxis`` objects are
    merged back into the original coordinate if all axes are selected in order.

  Examples:
    >>> import coordax as cx
    >>> x = cx.SizedAxis('x', 2)
    >>> y = cx.SizedAxis('y', 3)
    >>> cx.coords.canonicalize(x, y)
    (coordax.SizedAxis('x', size=2), coordax.SizedAxis('y', size=3))

    >>> xy = cx.coords.compose(x, y)
    >>> cx.coords.canonicalize(xy)
    (coordax.SizedAxis('x', size=2), coordax.SizedAxis('y', size=3))
  """
  coordinates = _expand_coordinates(*coordinates)
  coordinates = _consolidate_coordinates(*coordinates)
  existing_dims = collections.Counter()
  for c in coordinates:
    existing_dims.update([d for d in c.dims if d is not None])
  repeated_dims = [dim for dim, count in existing_dims.items() if count > 1]
  if repeated_dims:
    raise ValueError(f'coordinates contain {repeated_dims=}')
  return coordinates


T = TypeVar('T')


def _concat_tuples(tuples: Iterable[tuple[T, ...]]) -> tuple[T, ...]:
  """Concatenates tuples."""
  return tuple(itertools.chain(*tuples))


K = TypeVar('K')
V = TypeVar('V')


def _merge_dicts(dicts: Iterable[dict[K, V]]) -> dict[K, V]:
  """Merges dicts."""
  result = {}
  for d in dicts:
    result.update(d)
  return result


@utils.export
@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class CartesianProduct(Coordinate):
  """Coordinate defined as the outer product of independent coordinates.

  To construct a ``CartesianProduct``, use :func:`coordax.coords.compose`.
  """

  coordinates: tuple[Coordinate, ...]

  def __post_init__(self):
    coordinates = canonicalize(*self.coordinates)
    object.__setattr__(self, 'coordinates', coordinates)

  def __eq__(self, other):
    # TODO(shoyer): require exact equality of coordinate types?
    if not isinstance(other, CartesianProduct):
      return len(self.coordinates) == 1 and self.coordinates[0] == other
    return isinstance(other, CartesianProduct) and self.axes == other.axes

  @property
  def dims(self) -> tuple[str | None, ...]:
    return _concat_tuples(c.dims for c in self.coordinates)

  @property
  def shape(self) -> tuple[int, ...]:
    """Returns the shape of the coordinate axes."""
    return _concat_tuples(c.shape for c in self.coordinates)

  @property
  def fields(self) -> dict[str, 'fields.Field']:
    """Returns a mapping from field names to their values."""
    return _merge_dicts(c.fields for c in self.coordinates)

  @property
  def axes(self) -> tuple[Coordinate, ...]:
    """Returns a tuple of Axis objects for each dimension."""
    return _concat_tuples(c.axes for c in self.coordinates)

  def _isel(self, indexers: dict[str | Coordinate, Any]) -> Coordinate:
    new_coords = []
    for c in self.coordinates:  # already canonicalized.
      c_indexers = {k: v for k, v in indexers.items() if contains_dims(c, k)}
      new_coords.append(c.isel(c_indexers))
    return compose(*new_coords)

  def map_indexers(
      self,
      indexers: dict[str | Coordinate, Any],
      method: Literal['nearest'] | None = None,
  ) -> tuple[dict[str | Coordinate, Any], set[str | Coordinate]]:
    mapped = {}
    consumed = set()
    for c in self.coordinates:
      sub_mapped, sub_consumed = c.map_indexers(indexers, method=method)
      mapped.update(sub_mapped)
      consumed.update(sub_consumed)
    return mapped, consumed


@utils.export
@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class SizedAxis(Coordinate):
  """One dimensional coordinate with fixed size but no associated fields."""

  name: str
  size: int

  @property
  def dims(self) -> tuple[str, ...]:
    return (self.name,)

  @property
  def shape(self) -> tuple[int, ...]:
    return (self.size,)

  def map_indexers(
      self,
      indexers: dict[str | Coordinate, Any],
      method: Literal['nearest'] | None = None,
  ) -> tuple[dict[str | Coordinate, Any], set[str | Coordinate]]:
    if indexers and (self.name in indexers or self in indexers):
      key = self.name if self.name in indexers else self
      assert isinstance(key, (str, Coordinate))
      if indexers[key] == self:
        return {key: slice(None)}, {key}
      raise ValueError(f'{type(self).__name__} does not support `sel`.')
    return {}, set()

  def _isel(self, indexers: dict[str | Coordinate, Any]) -> Coordinate:
    key = self.name if self.name in indexers else self
    indexer = indexers[key]
    if isinstance(indexer, int):
      return Scalar()

    if isinstance(indexer, slice):
      start, stop, step = indexer.indices(self.size)
      new_size = len(range(start, stop, step))
      return SizedAxis(self.name, new_size)

    if hasattr(indexer, '__len__'):  # array or list.
      return SizedAxis(self.name, len(indexer))

    raise ValueError(f'Unsupported indexer type {type(indexer)}')

  def __repr__(self):
    return f'coordax.SizedAxis({self.name!r}, size={self.size})'

  @classmethod
  def from_xarray(
      cls, dims: tuple[str, ...], coords: xarray.Coordinates
  ) -> Self | NoCoordinateMatch:
    dim = dims[0]
    if dim in coords:
      return NoCoordinateMatch(
          'can only reconstruct SizedAxis objects from xarray dimensions'
          ' without associated coordinate variables, but found a coordinate'
          f' variable for dimension {dim!r}'
      )
    for name, coord in coords.variables.items():
      if dim in coord.dims:
        return NoCoordinateMatch(
            'can only reconstruct SizedAxis objects from xarray dimensions'
            ' if the dimensions is not found on any coordinate variables, but '
            f' found a coordinate variable for dimension {dim!r} on {name!r}'
        )
    return cls(dim, size=coords.sizes[dim])


@utils.export
@dataclasses.dataclass(frozen=True)
class DummyAxis(Coordinate):
  """Dummy coordinate for dimensions without associated coordinate values.

  DummyAxis are placeholders for dimensions that do not have associated
  coordinate values. They are automatically dropped from the Field constructor,
  but are useful for specifying how to construct fields with missing dimension
  names and/or coordinates.
  """

  name: str | None
  size: int

  @property
  def dims(self) -> tuple[str | None, ...]:
    return (self.name,)

  @property
  def shape(self) -> tuple[int, ...]:
    return (self.size,)

  def map_indexers(
      self,
      indexers: dict[str | Coordinate, Any],
      method: Literal['nearest'] | None = None,
  ) -> tuple[dict[str | Coordinate, Any], set[str | Coordinate]]:
    if indexers and (self.name in indexers or self in indexers):
      key = self.name if self.name in indexers else self
      assert isinstance(key, (str, Coordinate))
      if indexers[key] == self:
        return {key: slice(None)}, {key}
      raise ValueError(f'{type(self).__name__} does not support `sel`.')
    return {}, set()

  def _isel(self, indexers: dict[str | Coordinate, Any]) -> Coordinate:
    key = self.name if self.name in indexers else self
    indexer = indexers[key]
    if isinstance(indexer, int):
      return Scalar()

    if isinstance(indexer, slice):
      start, stop, step = indexer.indices(self.size)
      new_size = len(range(start, stop, step))
      return DummyAxis(self.name, new_size)

    if hasattr(indexer, '__len__'):
      return DummyAxis(self.name, len(indexer))

    raise ValueError(f'Unsupported indexer type {type(indexer)}')

  def __repr__(self):
    return f'coordax.DummyAxis({self.name!r}, size={self.size})'

  @classmethod
  def from_xarray(
      cls, dims: tuple[str, ...], coords: xarray.Coordinates
  ) -> Self | NoCoordinateMatch:
    dim = dims[0]
    for name, coord in coords.variables.items():
      if dim in coord.dims:
        return NoCoordinateMatch(
            f'cannot omit a Coordinate object for dimension {dim!r}'
            f' because it is used by at least one coordinate variable: {name!r}'
        )
    return cls(name=dim, size=coords.sizes[dim])


@functools.partial(utils.export, module='coordax.experimental')
def map_indexers_using_ticks(
    axis: Coordinate,
    indexers: dict[str | Coordinate, Any],
    ticks: np.ndarray | None = None,
    ticks_are_sorted: bool | None = None,
    method: Literal['nearest'] | None = None,
) -> tuple[dict[str | Coordinate, Any], set[str | Coordinate]]:
  """Maps indexers to ticks using either fancy indexing or slices.

  Args:
    axis: The coordinate for which to map indexers.
    indexers: A mapping from dimensions to slices.
    ticks: The tick values to use for mapping indexers. If None, ticks are taken
      from the `axis`. Default is None. Can be used to map custom ticks.
    ticks_are_sorted: Whether the `ticks` are sorted.
    method: Method to use for inexact matches. Either None (default) for exact
      matches or 'nearest'. Cannot be used when indexers contain slices.

  Returns:
    A tuple of (mapped_indexers, consumed_keys).

  Raises:
    KeyError: If any values in the indexer are not found in the ticks.
    ValueError: If an unsupported method is provided or if ticks are not unique.
  """
  if axis.ndim != 1:
    raise ValueError(
        f'Mapping indexers using ticks requires 1D axis, got {axis.ndim=}'
    )
  [dim] = axis.dims
  if axis not in indexers and dim not in indexers:
    return {}, set()

  ticks = ticks if ticks is not None else axis.fields[dim].data
  if np.unique(ticks).size != ticks.size:
    raise ValueError(f'Ticks must be unique, got {ticks}')
  if ticks_are_sorted is None:
    ticks_are_sorted = np.all(ticks[:-1] < ticks[1:])

  key = axis if axis in indexers else dim
  assert isinstance(key, (str, Coordinate))  # make pytype happy
  value = indexers[key]
  if is_coord(value) and value == axis:
    return {key: slice(None)}, {key}
  if is_coord(value):
    if not isinstance(value, type(axis)) or value.dims != axis.dims:
      raise ValueError(
          'Indexing with axis requires same type and dims, got index axis'
          f' {value} for slicing {axis=}.'
      )
    value = value.fields[dim].data

  if isinstance(value, slice):
    if method is not None:
      raise NotImplementedError('Method argument not supported for slices')

    start, stop = value.start, value.stop
    if ticks_are_sorted:
      start_idx, stop_idx = 0, ticks.size
      if start is not None:
        start_idx = np.searchsorted(ticks, start, side='left')
      if stop is not None:
        stop_idx = np.searchsorted(ticks, stop, side='right')

      return {key: slice(start_idx, stop_idx)}, {key}
    else:
      mask = np.ones(ticks.size, dtype=bool)
      if start is not None:
        mask &= ticks >= start
      if stop is not None:
        mask &= ticks <= stop
      return {key: np.where(mask)[0]}, {key}

  if method == 'nearest':
    if ticks_are_sorted:
      candidates = np.searchsorted(ticks, value, side='left')
      left = np.maximum(candidates - 1, 0)
      right = np.minimum(candidates, len(ticks) - 1)
      d_left = np.abs(value - ticks[left])
      d_right = np.abs(value - ticks[right])
      # In case of ties, prefer the left (smaller) index.
      idx = np.where(d_left <= d_right, left, right)
    elif np.ndim(value) == 0:
      idx = np.abs(ticks - value).argmin()
    else:
      ticks_view = ticks.reshape((-1,) + (1,) * np.ndim(value))
      idx = np.abs(ticks_view - value).argmin(axis=0)

    if np.ndim(idx) == 0:
      idx = int(idx)
    return {key: idx}, {key}

  if method is None:
    sort_indices = None if ticks_are_sorted else np.argsort(ticks)
    sorted_ticks = ticks if ticks_are_sorted else ticks[sort_indices]
    indices = np.searchsorted(sorted_ticks, value)
    if sort_indices is not None:
      indices = sort_indices[indices]
    unique_retrieved = np.sort(np.unique(ticks[indices]))
    unique_value = np.sort(np.unique(value))
    if unique_retrieved.size != unique_value.size or np.any(
        unique_retrieved != unique_value
    ):
      raise KeyError(f'Not all values in {value} were found in {ticks}')
    if np.ndim(value) == 0:  # if value is not an array, index must be an int.
      indices = indices.item()
    return {key: indices}, {key}

  raise ValueError(f'Unknown method {method}')


@utils.export
@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class LabeledAxis(Coordinate):
  """One dimensional coordinate with custom coordinate values."""

  name: str
  ticks: np.ndarray

  def __post_init__(self):
    object.__setattr__(self, 'ticks', np.asarray(self.ticks))
    if self.ticks.ndim != 1:
      raise ValueError(f'ticks must be a 1D array, got {self.ticks.shape=}')

  @property
  def dims(self) -> tuple[str, ...]:
    return (self.name,)

  @property
  def shape(self) -> tuple[int, ...]:
    return self.ticks.shape

  @property
  def fields(self) -> dict[str, 'fields.Field']:
    # needs local import to avoid circular dependency
    from coordax import fields  # pylint: disable=g-import-not-at-top

    return {self.name: fields.field(self.ticks, self)}

  @functools.cached_property
  def _sorted_ticks(self) -> bool:
    return np.all(self.ticks[:-1] <= self.ticks[1:])

  def map_indexers(
      self,
      indexers: dict[str | Coordinate, Any],
      method: Literal['nearest'] | None = None,
  ) -> tuple[dict[str | Coordinate, Any], set[str | Coordinate]]:
    return map_indexers_using_ticks(
        axis=self,
        indexers=indexers,
        ticks=self.ticks,
        ticks_are_sorted=self._sorted_ticks,
        method=method,
    )

  def _isel(self, indexers: dict[str | Coordinate, Any]) -> Coordinate:
    key = self.name if self.name in indexers else self
    indexer = indexers[key]
    if isinstance(indexer, int):
      return Scalar()

    return LabeledAxis(self.name, self.ticks[indexer])

  def _components(self):
    return (self.name, ArrayKey(self.ticks))

  def __eq__(self, other):
    return (
        isinstance(other, LabeledAxis)
        and self._components() == other._components()
    )

  def __hash__(self) -> int:
    return hash(self._components())

  def __repr__(self):
    return f'coordax.LabeledAxis({self.name!r}, ticks={self.ticks!r})'

  @classmethod
  def from_xarray(
      cls, dims: tuple[str, ...], coords: xarray.Coordinates
  ) -> Self | NoCoordinateMatch:
    dim = dims[0]
    if dim not in coords:
      return NoCoordinateMatch(
          f'no associated coordinate for dimension {dim!r}'
      )
    if coords[dim].ndim != 1:
      return NoCoordinateMatch(f'coordinate for dimension {dim!r} is not 1D')
    return cls(dim, coords[dim].data)


@functools.partial(utils.export, module='coordax.coords')
def compose(*coordinates: Coordinate) -> Coordinate:
  # pylint: disable=line-too-long
  # fmt: off
  """Compose coordinates into a unified coordinate system.

  Args:
    *coordinates: The coordinates to compose.

  Returns:
    A single coordinate object representing the Cartesian product of the inputs.

  Examples:
    >>> import coordax as cx
    >>> x = cx.SizedAxis('x', 2)
    >>> y = cx.SizedAxis('y', 3)
    >>> cx.coords.compose(x, y)
    CartesianProduct(coordinates=(coordax.SizedAxis('x', size=2), coordax.SizedAxis('y', size=3)))
  """
  # fmt: on
  product = CartesianProduct(coordinates)
  match len(product.coordinates):
    case 0:
      return Scalar()
    case 1:
      return product.coordinates[0]
    case _:
      return product


@functools.partial(utils.export, module='coordax.coords')
def insert_axes(
    coordinate: Coordinate,
    indices_to_axes: dict[int, Coordinate],
) -> Coordinate:
  # pylint: disable=line-too-long
  # fmt: off
  """Returns ``coordinate`` with extra axes inserted at specified positions.

  Args:
    coordinate: The coordinate system to modify.
    indices_to_axes: A mapping from insertion index to the new coordinate to
      insert. Indices are relative to the *output* coordinate system.

  Returns:
    A new coordinate object with the axes inserted.

  Examples:
    >>> import coordax as cx
    >>> x = cx.SizedAxis('x', 2)
    >>> z = cx.SizedAxis('z', 4)
    >>> cx.coords.insert_axes(x, {1: z})
    CartesianProduct(coordinates=(coordax.SizedAxis('x', size=2), coordax.SizedAxis('z', size=4)))
  """
  # fmt: on
  indices_to_axes = indices_to_axes.copy()
  ndim = coordinate.ndim + len(indices_to_axes)
  normalize_idx = lambda i: i + ndim if i < 0 else i
  indices_to_axes = {normalize_idx(i): ax for i, ax in indices_to_axes.items()}
  if any(i < 0 or i >= ndim for i in indices_to_axes.keys()):
    raise ValueError(f'invalid index in {indices_to_axes.keys()=}')

  axes = list(coordinate.axes)
  for index_to_insert_at, axis in sorted(indices_to_axes.items()):
    axes.insert(index_to_insert_at, axis)
  return compose(*axes)


@functools.partial(utils.export, module='coordax.coords')
def replace_axes(
    coordinate: Coordinate,
    to_replace: Coordinate,
    replace_with: Coordinate,
) -> Coordinate:
  # pylint: disable=line-too-long
  # fmt: off
  """Returns ``coordinate`` with ``to_replace`` replaced by ``replace_with``.

  Args:
    coordinate: The coordinate system to modify.
    to_replace: A contiguous part of the coordinate system to be replaced.
    replace_with: The new coordinate to insert.

  Returns:
    A new coordinate object with ``to_replace`` substituted by ``replace_with``.

  Raises:
    ValueError: If ``to_replace`` is not a contiguous part of ``coordinate``.

  Examples:
    >>> import coordax as cx
    >>> x = cx.SizedAxis('x', 2)
    >>> y = cx.SizedAxis('y', 3)
    >>> xy = cx.coords.compose(x, y)
    >>> xy
    CartesianProduct(coordinates=(coordax.SizedAxis('x', size=2), coordax.SizedAxis('y', size=3)))
    >>> z = cx.SizedAxis('z', 4)
    >>> cx.coords.replace_axes(xy, x, z)
    CartesianProduct(coordinates=(coordax.SizedAxis('z', size=4), coordax.SizedAxis('y', size=3)))
  """
  # fmt: on
  if not to_replace.dims:
    raise ValueError(f'`to_replace` must have dimensions, got {to_replace}')

  # find where the sub-tuple `to_replace.axes` starts in `coordinate.axes`.
  start_index = -1
  for i in range(len(coordinate.axes) - len(to_replace.axes) + 1):
    if coordinate.axes[i : i + len(to_replace.axes)] == to_replace.axes:
      start_index = i
      break

  if start_index == -1:
    raise ValueError(
        f'coordinate {coordinate=} does not contiguously contain {to_replace=}'
    )

  end_index = start_index + len(to_replace.axes)
  axes = list(coordinate.axes)
  axes[start_index:end_index] = replace_with.axes
  return compose(*axes)


@functools.partial(utils.export, module='coordax.coords')
def extract(
    coord: Coordinate,
    component_type: Type[Coordinate] | tuple[Type[Coordinate], ...],
) -> Coordinate:
  """Extracts component of type `component_type` from the `coord`.

  Args:
    coord: The coordinate system to search.
    component_type: The type(s) of coordinate to extract.

  Returns:
    The single coordinate component matching the given type(s).

  Raises:
    ValueError: If there is not exactly one component of the given type(s).

  Examples:
    >>> import coordax as cx
    >>> import numpy as np
    >>> x = cx.LabeledAxis('x', np.linspace(0, np.pi, 4))
    >>> y = cx.SizedAxis('y', 3)
    >>> cx.coords.extract(cx.coords.compose(x, y), cx.SizedAxis)
    coordax.SizedAxis('y', size=3)
  """
  components = canonicalize(coord)
  of_type = [c for c in components if isinstance(c, component_type)]
  if len(of_type) != 1:
    raise ValueError(
        f'Expected exactly one instance of {component_type}, found {of_type}'
    )
  [result] = of_type
  return result


@functools.partial(utils.export, module='coordax.coords')
def from_xarray(
    data_array: xarray.DataArray,
    coord_types: Sequence[type[Coordinate]] = (LabeledAxis, DummyAxis),
) -> Coordinate:
  # pylint: disable=line-too-long
  # fmt: off
  """Convert coordinates of an ``xarray.DataArray`` into ``coordax.Coordinate``.

  Args:
    data_array: ``xarray.DataArray`` whose coordinates should be converted.
    coord_types: sequence of ``coordax.Coordinate`` subclasses with
      ``from_xarray`` methods defined. The first coordinate class that returns a
      coordinate object (indicating a match) will be used. By default,
      coordinates will use only generic ``LabeledAxis`` and ``DummyAxis``
      objects.

  Returns:
    A coordax.Coordinate object, constructed from any of the indicated types
    (plus ``CartesianProduct`` and ``Scalar``), representing the coordinates of
    the input DataArray.

  Raises:
    ValueError: if no matching coordinate is found.

  Examples:
    >>> import coordax as cx
    >>> import xarray as xr
    >>> import numpy as np
    >>> da = xr.DataArray(np.zeros((2, 3)), dims=('x', 'y'), coords={'x': [1, 2]})
    >>> cx.coords.from_xarray(da)
    CartesianProduct(coordinates=(coordax.LabeledAxis('x', ticks=array([1, 2])), coordax.DummyAxis('y', size=3)))

  See also:
    :func:`coordax.from_xarray`
  """
  # fmt: on
  dims = data_array.dims
  coords = []

  if not all(isinstance(dim, str) for dim in dims):
    raise TypeError(
        'can only convert DataArray objects with string dimensions to Field'
    )
  dims = typing.cast(tuple[str, ...], dims)

  if not coord_types:
    raise ValueError('coord_types must be non-empty')

  def get_next_match():
    reasons = []
    for coord_type in coord_types:
      if coord_type == CartesianProduct or coord_type == Scalar:
        continue
      result = coord_type.from_xarray(dims, data_array.coords)
      if isinstance(result, Coordinate):
        return result
      assert isinstance(result, NoCoordinateMatch)
      coord_name = coord_type.__module__ + '.' + coord_type.__name__
      reasons.append(f'{coord_name}: {result.reason}')

    reasons_str = '\n'.join(reasons)
    raise ValueError(
        'failed to convert xarray.DataArray to coordax.Field, because no '
        f'coordinate type matched the dimensions starting with {dims}:\n'
        f'{data_array}\n\n'
        f'Reasons why coordinate matching failed:\n{reasons_str}'
    )

  while dims:
    coord = get_next_match()
    coords.append(coord)
    assert coord.ndim > 0  # dimensions will shrink by at least one
    dims = dims[coord.ndim :]

  return compose(*coords)


_T = TypeVar('_T', bound=Callable)


def _deprecated_alias(func: _T, old_name: str, new_name: str) -> _T:
  """Returns a deprecated alias for a function."""

  @functools.wraps(func)
  def wrapper(*args, **kwargs):
    warnings.warn(
        f'{old_name} is deprecated. Please use {new_name} instead.',
        DeprecationWarning,
        stacklevel=2,
    )
    return func(*args, **kwargs)

  return typing.cast(_T, wrapper)


canonicalize_coordinates = _deprecated_alias(
    canonicalize,
    'coordax.canonicalize_coordinates',
    'coordax.coords.canonicalize',
)
compose_coordinates = _deprecated_alias(
    compose,
    'coordax.compose_coordinates',
    'coordax.coords.compose',
)
insert_axes_to_coordinate = _deprecated_alias(
    insert_axes,
    'coordax.insert_axes_to_coordinate',
    'coordax.coords.insert_axes',
)
replace_axes_in_coordinate = _deprecated_alias(
    replace_axes,
    'coordax.replace_axes_in_coordinate',
    'coordax.coords.replace_axes',
)
coordinates_from_xarray = _deprecated_alias(
    from_xarray,
    'coordax.coordinates_from_xarray',
    'coordax.coords.from_xarray',
)
