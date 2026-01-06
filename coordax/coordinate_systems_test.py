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
from absl.testing import absltest
from absl.testing import parameterized
import coordax as cx
import numpy as np
import pytest


class CoordinateSystemsTest(parameterized.TestCase):

  PRODUCT_XY = cx.CartesianProduct((cx.SizedAxis('x', 2), cx.SizedAxis('y', 3)))

  @parameterized.named_parameters(
      dict(
          testcase_name='empty',
          coordinates=(),
          expected=(),
      ),
      dict(
          testcase_name='single_other_axis',
          coordinates=(cx.SizedAxis('x', 2),),
          expected=(cx.SizedAxis('x', 2),),
      ),
      dict(
          testcase_name='single_selected_axis',
          coordinates=(cx.SelectedAxis(cx.SizedAxis('x', 2), axis=0),),
          expected=(cx.SizedAxis('x', 2),),
      ),
      dict(
          testcase_name='pair_of_other_axes',
          coordinates=(
              cx.SizedAxis('x', 2),
              cx.LabeledAxis('y', np.arange(3)),
          ),
          expected=(
              cx.SizedAxis('x', 2),
              cx.LabeledAxis('y', np.arange(3)),
          ),
      ),
      dict(
          testcase_name='pair_of_selections_correct',
          coordinates=(
              cx.SelectedAxis(PRODUCT_XY, axis=0),
              cx.SelectedAxis(PRODUCT_XY, axis=1),
          ),
          expected=(PRODUCT_XY,),
      ),
      dict(
          testcase_name='pair_of_selections_wrong_order',
          coordinates=(
              cx.SelectedAxis(PRODUCT_XY, axis=1),
              cx.SelectedAxis(PRODUCT_XY, axis=0),
          ),
          expected=(
              cx.SelectedAxis(PRODUCT_XY, axis=1),
              cx.SelectedAxis(PRODUCT_XY, axis=0),
          ),
      ),
      dict(
          testcase_name='selection_incomplete',
          coordinates=(cx.SelectedAxis(PRODUCT_XY, axis=0),),
          expected=(cx.SelectedAxis(PRODUCT_XY, axis=0),),
      ),
      dict(
          testcase_name='selections_with_following',
          coordinates=(
              cx.SelectedAxis(PRODUCT_XY, axis=0),
              cx.SelectedAxis(PRODUCT_XY, axis=1),
              cx.SizedAxis('z', 4),
          ),
          expected=(
              PRODUCT_XY,
              cx.SizedAxis('z', 4),
          ),
      ),
      dict(
          testcase_name='selections_with_preceeding',
          coordinates=(
              cx.SizedAxis('z', 4),
              cx.SelectedAxis(PRODUCT_XY, axis=0),
              cx.SelectedAxis(PRODUCT_XY, axis=1),
          ),
          expected=(
              cx.SizedAxis('z', 4),
              PRODUCT_XY,
          ),
      ),
      dict(
          testcase_name='selections_split',
          coordinates=(
              cx.SelectedAxis(PRODUCT_XY, axis=0),
              cx.SizedAxis('z', 4),
              cx.SelectedAxis(PRODUCT_XY, axis=1),
          ),
          expected=(
              cx.SelectedAxis(PRODUCT_XY, axis=0),
              cx.SizedAxis('z', 4),
              cx.SelectedAxis(PRODUCT_XY, axis=1),
          ),
      ),
      dict(
          testcase_name='two_selected_axes_consolidate_after',
          coordinates=(
              cx.SelectedAxis(PRODUCT_XY, axis=0),
              cx.SelectedAxis(cx.SizedAxis('z', 4), axis=0),
          ),
          expected=(
              cx.SelectedAxis(PRODUCT_XY, axis=0),
              cx.SizedAxis('z', 4),
          ),
      ),
      dict(
          testcase_name='two_selected_axes_consolidate_before',
          coordinates=(
              cx.SelectedAxis(cx.SizedAxis('z', 4), axis=0),
              cx.SelectedAxis(PRODUCT_XY, axis=0),
          ),
          expected=(
              cx.SizedAxis('z', 4),
              cx.SelectedAxis(PRODUCT_XY, axis=0),
          ),
      ),
      dict(
          testcase_name='skip_scalars',
          coordinates=(
              cx.SizedAxis('w', 4),
              cx.Scalar(),
              cx.Scalar(),
              cx.SelectedAxis(PRODUCT_XY, axis=0),
              cx.coords.compose(cx.SizedAxis('z', 3), cx.Scalar()),
          ),
          expected=(
              cx.SizedAxis('w', 4),
              cx.SelectedAxis(PRODUCT_XY, axis=0),
              cx.SizedAxis('z', 3),
          ),
      ),
      dict(
          testcase_name='multiple_dummy_supported',
          coordinates=(
              cx.DummyAxis(None, 4),
              cx.SizedAxis('w', 4),
              cx.DummyAxis(None, 5),
          ),
          expected=(
              cx.DummyAxis(None, 4),
              cx.SizedAxis('w', 4),
              cx.DummyAxis(None, 5),
          ),
      ),
  )
  def test_canonicalize_coordinates(self, coordinates, expected):
    actual = cx.coords.canonicalize(*coordinates)
    self.assertEqual(actual, expected)

  @parameterized.named_parameters(
      dict(
          testcase_name='empty',
          coordinates=(),
          expected=cx.Scalar(),
      ),
      dict(
          testcase_name='single_coordinate',
          coordinates=(cx.SizedAxis('x', 2),),
          expected=cx.SizedAxis('x', 2),
      ),
      dict(
          testcase_name='selected_axes_compoents_merge',
          coordinates=(
              cx.SelectedAxis(PRODUCT_XY, axis=0),
              cx.SelectedAxis(PRODUCT_XY, axis=1),
          ),
          expected=PRODUCT_XY,
      ),
      dict(
          testcase_name='selected_axis_simplified',
          coordinates=(
              cx.SelectedAxis(cx.SizedAxis('x', 4), axis=0),
              cx.SizedAxis('z', 7),
          ),
          expected=cx.CartesianProduct(
              (cx.SizedAxis('x', 4), cx.SizedAxis('z', 7))
          ),
      ),
      dict(
          testcase_name='cartesian_product_unraveled',
          coordinates=(
              cx.SizedAxis('x', 7),
              cx.CartesianProduct((cx.SizedAxis('y', 7), cx.SizedAxis('z', 4))),
          ),
          expected=cx.CartesianProduct((
              cx.SizedAxis('x', 7),
              cx.SizedAxis('y', 7),
              cx.SizedAxis('z', 4),
          )),
      ),
      dict(
          testcase_name='consolidate_over_parts',
          coordinates=(
              cx.SelectedAxis(PRODUCT_XY, axis=0),
              cx.CartesianProduct((
                  cx.SelectedAxis(PRODUCT_XY, axis=1),
                  cx.SizedAxis('z', 4),
              )),
          ),
          expected=cx.CartesianProduct((
              cx.SizedAxis('x', 2),
              cx.SizedAxis('y', 3),
              cx.SizedAxis('z', 4),
          )),
      ),
      dict(
          testcase_name='consolidate_over_parts_skip_scalar',
          coordinates=(
              cx.SelectedAxis(PRODUCT_XY, axis=0),
              cx.CartesianProduct((
                  cx.SelectedAxis(PRODUCT_XY, axis=1),
                  cx.Scalar(),
                  cx.SizedAxis('z', 4),
              )),
          ),
          expected=cx.CartesianProduct((
              cx.SizedAxis('x', 2),
              cx.SizedAxis('y', 3),
              cx.SizedAxis('z', 4),
          )),
      ),
  )
  def test_compose(self, coordinates, expected):
    actual = cx.coords.compose(*coordinates)
    self.assertEqual(actual, expected)

  @parameterized.named_parameters(
      dict(
          testcase_name='insert_at_start',
          indices_to_axes={0: cx.SizedAxis('w', 4)},
          coordinate=PRODUCT_XY,
          expected=cx.CartesianProduct((
              cx.SizedAxis('w', 4),
              cx.SizedAxis('x', 2),
              cx.SizedAxis('y', 3),
          )),
      ),
      dict(
          testcase_name='insert_at_end',
          indices_to_axes={2: cx.SizedAxis('w', 4)},
          coordinate=PRODUCT_XY,
          expected=cx.CartesianProduct((
              cx.SizedAxis('x', 2),
              cx.SizedAxis('y', 3),
              cx.SizedAxis('w', 4),
          )),
      ),
      dict(
          testcase_name='insert_in_middle',
          indices_to_axes={1: cx.SizedAxis('w', 4)},
          coordinate=PRODUCT_XY,
          expected=cx.CartesianProduct((
              cx.SizedAxis('x', 2),
              cx.SizedAxis('w', 4),
              cx.SizedAxis('y', 3),
          )),
      ),
      dict(
          testcase_name='insert_multiple',
          indices_to_axes={
              0: cx.SizedAxis('w', 4),
              2: cx.SizedAxis('z', 5),
          },
          coordinate=PRODUCT_XY,
          expected=cx.CartesianProduct((
              cx.SizedAxis('w', 4),
              cx.SizedAxis('x', 2),
              cx.SizedAxis('z', 5),
              cx.SizedAxis('y', 3),
          )),
      ),
      dict(
          testcase_name='insert_with_negative_index',
          indices_to_axes={-1: cx.SizedAxis('w', 4)},
          coordinate=PRODUCT_XY,
          expected=cx.CartesianProduct((
              cx.SizedAxis('x', 2),
              cx.SizedAxis('y', 3),
              cx.SizedAxis('w', 4),
          )),
      ),
      dict(
          testcase_name='insert_into_scalar',
          indices_to_axes={0: cx.SizedAxis('x', 2)},
          coordinate=cx.Scalar(),
          expected=cx.SizedAxis('x', 2),
      ),
  )
  def test_insert_axes(self, indices_to_axes, coordinate, expected):
    actual = cx.coords.insert_axes(coordinate, indices_to_axes)
    self.assertEqual(actual, expected)

  def test_insert_axes_raises_out_of_range(self):
    with self.assertRaises(ValueError):
      cx.coords.insert_axes(self.PRODUCT_XY, {3: cx.SizedAxis('w', 4)})
    with self.assertRaises(ValueError):
      cx.coords.insert_axes(self.PRODUCT_XY, {-4: cx.SizedAxis('w', 4)})

  @parameterized.named_parameters(
      dict(
          testcase_name='replace_at_start',
          to_replace=cx.SizedAxis('x', 2),
          replace_with=cx.SizedAxis('w', 4),
          coordinate=PRODUCT_XY,
          expected=cx.CartesianProduct(
              (cx.SizedAxis('w', 4), cx.SizedAxis('y', 3))
          ),
      ),
      dict(
          testcase_name='replace_at_end',
          to_replace=cx.SizedAxis('y', 3),
          replace_with=cx.SizedAxis('w', 4),
          coordinate=PRODUCT_XY,
          expected=cx.CartesianProduct(
              (cx.SizedAxis('x', 2), cx.SizedAxis('w', 4))
          ),
      ),
      dict(
          testcase_name='replace_in_middle',
          to_replace=cx.SizedAxis('y', 3),
          replace_with=cx.SizedAxis('w', 4),
          coordinate=cx.coords.compose(
              cx.SizedAxis('x', 2),
              cx.SizedAxis('y', 3),
              cx.SizedAxis('z', 5),
          ),
          expected=cx.coords.compose(
              cx.SizedAxis('x', 2),
              cx.SizedAxis('w', 4),
              cx.SizedAxis('z', 5),
          ),
      ),
      dict(
          testcase_name='replace_multiple_with_single',
          to_replace=PRODUCT_XY,
          replace_with=cx.SizedAxis('w', 4),
          coordinate=cx.coords.compose(PRODUCT_XY, cx.SizedAxis('z', 5)),
          expected=cx.coords.compose(
              cx.SizedAxis('w', 4), cx.SizedAxis('z', 5)
          ),
      ),
      dict(
          testcase_name='replace_single_with_multiple',
          to_replace=cx.SizedAxis('y', 3),
          replace_with=cx.coords.compose(
              cx.SizedAxis('w', 4), cx.SizedAxis('z', 5)
          ),
          coordinate=PRODUCT_XY,
          expected=cx.coords.compose(
              cx.SizedAxis('x', 2),
              cx.SizedAxis('w', 4),
              cx.SizedAxis('z', 5),
          ),
      ),
      dict(
          testcase_name='replace_all',
          to_replace=PRODUCT_XY,
          replace_with=cx.SizedAxis('w', 4),
          coordinate=PRODUCT_XY,
          expected=cx.SizedAxis('w', 4),
      ),
  )
  def test_replace_axes(self, to_replace, replace_with, coordinate, expected):
    actual = cx.coords.replace_axes(coordinate, to_replace, replace_with)
    self.assertEqual(actual, expected)

  def test_replace_axes_raises_when_to_replace_not_found(self):
    to_replace = cx.SizedAxis('z', 4)
    with self.assertRaisesRegex(ValueError, 'does not contiguously contain'):
      cx.coords.replace_axes(
          self.PRODUCT_XY,
          to_replace,
          cx.SizedAxis('w', 5),
      )

  def test_replace_axes_raises_when_to_replace_not_contiguous(self):
    coordinate = cx.coords.compose(
        cx.SizedAxis('x', 2),
        cx.SizedAxis('z', 4),
        cx.SizedAxis('y', 3),
    )
    to_replace = self.PRODUCT_XY  # contains x and y
    with self.assertRaisesRegex(ValueError, 'does not contiguously contain'):
      cx.coords.replace_axes(coordinate, to_replace, cx.SizedAxis('w', 5))

  def test_replace_axes_raises_when_to_replace_is_empty(self):
    to_replace = cx.Scalar()
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        f'`to_replace` must have dimensions, got {to_replace!r}',
    ):
      cx.coords.replace_axes(
          self.PRODUCT_XY,
          to_replace,
          cx.SizedAxis('w', 4),
      )

  def test_dummy_axis(self):
    pytest.importorskip('xarray')

    axis = cx.DummyAxis(name='x', size=0)
    self.assertEqual(axis.dims, ('x',))
    self.assertEqual(axis.shape, (0,))
    self.assertEqual(axis.sizes, {'x': 0})
    self.assertEqual(axis.fields, {})
    self.assertEqual(repr(axis), "coordax.DummyAxis('x', size=0)")
    self.assertEqual(axis.to_xarray(), {})

    axis = cx.DummyAxis(name=None, size=10)
    self.assertEqual(axis.dims, (None,))
    self.assertEqual(axis.shape, (10,))
    self.assertEqual(axis.sizes, {})
    self.assertEqual(axis.fields, {})
    self.assertEqual(repr(axis), 'coordax.DummyAxis(None, size=10)')
    self.assertEqual(axis.to_xarray(), {})

  def test_dummy_axis_cartesian_product(self):
    x = cx.DummyAxis(name='x', size=2)
    y = cx.DummyAxis(name=None, size=3)
    z = cx.SizedAxis('z', 4)
    product = cx.CartesianProduct((x, y, z))
    self.assertEqual(product.dims, ('x', None, 'z'))
    self.assertEqual(product.shape, (2, 3, 4))
    self.assertEqual(product.sizes, {'x': 2, 'z': 4})

  def test_multiple_unnamed_dummy_axes_cartesian_product(self):
    x = cx.DummyAxis(name='x', size=2)
    y = cx.DummyAxis(name=None, size=3)
    z = cx.DummyAxis(name=None, size=4)
    product = cx.CartesianProduct((x, y, z))
    self.assertEqual(product.dims, ('x', None, None))
    self.assertEqual(product.shape, (2, 3, 4))
    self.assertEqual(product.sizes, {'x': 2})

  def test_dummy_axes_with_same_names_in_cartesian_product_raises(self):
    x = cx.DummyAxis(name='x', size=2)
    y = cx.DummyAxis(name='x', size=3)
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "coordinates contain repeated_dims=['x']",
    ):
      cx.CartesianProduct((x, y))

  def test_extract(self):
    x = cx.SizedAxis('x', 2)
    y = cx.SizedAxis('y', 3)
    z = cx.LabeledAxis('z', np.arange(4))
    xy = cx.coords.compose(x, y)
    xz = cx.coords.compose(x, z)

    self.assertEqual(cx.coords.extract(x, cx.SizedAxis), x)
    self.assertEqual(cx.coords.extract(z, cx.LabeledAxis), z)
    self.assertEqual(cx.coords.extract(xz, cx.LabeledAxis), z)
    self.assertEqual(cx.coords.extract(z, (cx.SizedAxis, cx.LabeledAxis)), z)

    with self.assertRaisesRegex(ValueError, 'Expected exactly one instance'):
      cx.coords.extract(xy, cx.SizedAxis)

    with self.assertRaisesRegex(ValueError, 'Expected exactly one instance'):
      cx.coords.extract(xz, (cx.SizedAxis, cx.LabeledAxis))

    with self.assertRaisesRegex(ValueError, 'Expected exactly one instance'):
      cx.coords.extract(x, cx.LabeledAxis)

  def test_deprecated_aliases(self):
    with self.assertWarnsRegex(
        DeprecationWarning,
        'coordax.canonicalize_coordinates is deprecated. Please use'
        ' coordax.coords.canonicalize instead.',
    ):
      self.assertEqual(cx.canonicalize_coordinates(), ())

    with self.assertWarnsRegex(
        DeprecationWarning,
        'coordax.compose_coordinates is deprecated. Please use'
        ' coordax.coords.compose instead.',
    ):
      self.assertEqual(cx.compose_coordinates(), cx.Scalar())

    with self.assertWarnsRegex(
        DeprecationWarning,
        'coordax.insert_axes_to_coordinate is deprecated. Please use'
        ' coordax.coords.insert_axes instead.',
    ):
      self.assertEqual(
          cx.insert_axes_to_coordinate(cx.Scalar(), {0: cx.SizedAxis('x', 2)}),
          cx.SizedAxis('x', 2),
      )

    with self.assertWarnsRegex(
        DeprecationWarning,
        'coordax.replace_axes_in_coordinate is deprecated. Please use'
        ' coordax.coords.replace_axes instead.',
    ):
      self.assertEqual(
          cx.replace_axes_in_coordinate(
              cx.SizedAxis('x', 2), cx.SizedAxis('x', 2), cx.SizedAxis('y', 2)
          ),
          cx.SizedAxis('y', 2),
      )

  def test_deprecated_coordinates_from_xarray(self):
    xarray = pytest.importorskip('xarray')

    with self.assertWarnsRegex(
        DeprecationWarning,
        'coordax.coordinates_from_xarray is deprecated. Please use'
        ' coordax.coords.from_xarray instead.',
    ):
      data_array = xarray.DataArray(np.zeros((2,)), dims='x')
      self.assertEqual(
          cx.coordinates_from_xarray(data_array), cx.DummyAxis('x', size=2)
      )


if __name__ == '__main__':
  absltest.main()
