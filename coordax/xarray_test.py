# Copyright 2025 Google LLC
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
import textwrap
from typing import Callable, Self

from absl.testing import absltest
import coordax
from coordax import testing
import jax
import numpy as np
import pytest

xarray = pytest.importorskip('xarray')


# TODO(shoyer): consider moving this into coordax's public API
@jax.tree_util.register_static
class AdhocCoordinate(coordax.Coordinate):
  """Adhoc coordinate class for testing purposes.

  For simplicity, equality and hashing of adhoc coordinates is by identity.
  """

  def __init__(
      self,
      dims: tuple[str, ...],
      shape: tuple[int, ...],
      fields: Callable[[Self], dict[str, coordax.Field]] | None = None,
  ):
    self._dims = dims
    self._shape = shape
    self._fields = fields

  @property
  def dims(self) -> tuple[str, ...]:
    return self._dims

  @property
  def shape(self) -> tuple[int, ...]:
    return self._shape

  @property
  def fields(self) -> dict[str, coordax.Field]:
    return {} if self._fields is None else self._fields(self)

  @classmethod
  def from_xarray(cls, dims, coords):
    return coordax.NoCoordinateMatch('not a match')


AdhocCoordinate.__module__ = 'adhoc'


class XarrayTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    pytest.importorskip('xarray')

  def test_field_to_data_array_without_axes(self):
    data = np.arange(2 * 3).reshape((2, 3))
    field = coordax.field(data, 'x', 'y')
    actual = field.to_xarray()
    expected = xarray.DataArray(data, dims=['x', 'y'])
    xarray.testing.assert_identical(actual, expected)

  def test_field_to_data_array_with_sized_axis(self):
    data = np.arange(3)
    axis = coordax.SizedAxis('x', 3)
    field = coordax.field(data, axis)
    actual = field.to_xarray()
    expected = xarray.DataArray(data, dims=['x'])
    xarray.testing.assert_identical(actual, expected)

  def test_field_to_data_array_with_labeled_axis(self):
    data = np.arange(3)
    ticks = np.array([1, 2, 3])
    axis = coordax.LabeledAxis('x', ticks)
    field = coordax.field(data, axis)
    actual = field.to_xarray()
    expected = xarray.DataArray(data, dims=['x'], coords={'x': ticks})
    xarray.testing.assert_identical(actual, expected)

  def test_field_to_data_array_custom_coord(self):
    data_2d = np.arange(2 * 3).reshape((2, 3))
    custom_coord = AdhocCoordinate(
        dims=('x', 'y'),
        shape=(2, 3),
        fields=lambda c: {'custom': coordax.field(data_2d, c)},
    )
    field = coordax.field(10 * data_2d, custom_coord)
    actual = field.to_xarray()
    expected = xarray.DataArray(
        data=10 * data_2d,
        dims=['x', 'y'],
        coords={'custom': (('x', 'y'), data_2d)},
    )
    xarray.testing.assert_identical(actual, expected)

  def test_field_to_data_array_missing_dimension_names(self):
    data = np.arange(2 * 3).reshape((2, 3))
    field = coordax.field(data)
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'can only convert Field objects with fully named dimensions to'
        ' xarray.DataArray objects, got dimensions (None, None)',
    ):
      field.to_xarray()

  def test_field_to_data_array_inconsistent_coordinates(self):

    x = AdhocCoordinate(
        dims=('x',),
        shape=(3,),
        fields=lambda c: {'z': coordax.field(0)},
    )
    y = AdhocCoordinate(
        dims=('y',),
        shape=(2,),
        fields=lambda c: {'z': coordax.field(1)},
    )
    field = coordax.field(np.zeros((3, 2)), x, y)
    with self.assertRaisesRegex(
        ValueError, "inconsistent coordinate fields for 'z'"
    ):
      field.to_xarray()

  def test_data_array_to_field_default_success(self):
    data = np.arange(2 * 3).reshape((2, 3))
    data_array = xarray.DataArray(
        data=data, dims=['x', 'y'], coords={'y': [1, 2, 3]}
    )
    actual = coordax.Field.from_xarray(data_array)
    expected = coordax.field(data, 'x', coordax.LabeledAxis('y', [1, 2, 3]))
    testing.assert_fields_equal(actual, expected)

  def test_data_array_to_field_with_sized_and_labeled_axis(self):
    data = np.arange(2 * 3).reshape((2, 3))
    data_array = xarray.DataArray(
        data=data, dims=['x', 'y'], coords={'y': [1, 2, 3]}
    )
    actual = coordax.Field.from_xarray(
        data_array, coord_types=[coordax.SizedAxis, coordax.LabeledAxis]
    )
    expected = coordax.field(
        data, coordax.SizedAxis('x', 2), coordax.LabeledAxis('y', [1, 2, 3])
    )
    testing.assert_fields_equal(actual, expected)

  def test_data_array_to_field_no_match(self):
    # pylint: disable=line-too-long
    data = np.arange(2 * 3).reshape((2, 3))
    data_array = xarray.DataArray(
        data=data, dims=['x', 'y'], coords={'y': [1, 2, 3]}
    )

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        textwrap.dedent(
            """\
            failed to convert xarray.DataArray to coordax.Field, because no coordinate type matched the dimensions starting with ('y',):
            <xarray.DataArray (x: 2, y: 3)> Size: 48B
            array([[0, 1, 2],
                   [3, 4, 5]])
            Coordinates:
              * y        (y) int64 24B 1 2 3
            Dimensions without coordinates: x

            Reasons why coordinate matching failed:
            coordax.SizedAxis: can only reconstruct SizedAxis objects from xarray dimensions without associated coordinate variables, but found a coordinate variable for dimension 'y'"""
        ),
    ):
      coordax.Field.from_xarray(data_array, coord_types=[coordax.SizedAxis])

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        textwrap.dedent("""\
            failed to convert xarray.DataArray to coordax.Field, because no coordinate type matched the dimensions starting with ('x', 'y'):
            <xarray.DataArray (x: 2, y: 3)> Size: 48B
            array([[0, 1, 2],
                   [3, 4, 5]])
            Coordinates:
              * y        (y) int64 24B 1 2 3
            Dimensions without coordinates: x

            Reasons why coordinate matching failed:
            coordax.LabeledAxis: no associated coordinate for dimension 'x'"""),
    ):
      coordax.Field.from_xarray(data_array, coord_types=[coordax.LabeledAxis])

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        textwrap.dedent("""\
            failed to convert xarray.DataArray to coordax.Field, because no coordinate type matched the dimensions starting with ('x', 'y'):
            <xarray.DataArray (x: 2, y: 3)> Size: 48B
            array([[0, 1, 2],
                   [3, 4, 5]])
            Coordinates:
              * y        (y) int64 24B 1 2 3
            Dimensions without coordinates: x

            Reasons why coordinate matching failed:
            adhoc.AdhocCoordinate: not a match"""),
    ):
      coordax.Field.from_xarray(data_array, coord_types=[AdhocCoordinate])

  def test_data_array_to_field_custom_coord_types(self):
    data = np.arange(2 * 3).reshape((2, 3))
    data_array = xarray.DataArray(data=data, dims=['x', 'y'])
    custom_coord = AdhocCoordinate(dims=('x', 'y'), shape=(2, 3))
    custom_coord.from_xarray = lambda dims, _: custom_coord
    actual = coordax.Field.from_xarray(data_array, [custom_coord])
    expected = coordax.field(data, custom_coord)
    testing.assert_fields_equal(actual, expected)

  def test_default_coord_types_roundtrip(self):
    data = np.arange(2 * 3).reshape((2, 3))
    data_array = xarray.DataArray(
        data=data, dims=['x', 'y'], coords={'y': [1, 2, 3]}
    )
    field = coordax.Field.from_xarray(data_array)
    actual = field.to_xarray()
    xarray.testing.assert_identical(actual, data_array)


if __name__ == '__main__':
  absltest.main()
