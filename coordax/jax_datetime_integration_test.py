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
import datetime

from absl.testing import absltest
import coordax as cx
from coordax import testing
import jax.numpy as jnp
import numpy as np
import pytest

jdt = pytest.importorskip("jax_datetime")


class IntegrationTest(absltest.TestCase):

  def test_timedelta(self):
    dt = jdt.to_timedelta(jnp.array([1, 2, 3]), unit='D')
    field = cx.Field(dt).tag('time')
    expected = cx.Field(jnp.array([24, 48, 72])).tag('time')
    actual = field // datetime.timedelta(seconds=60 * 60)
    testing.assert_fields_equal(actual, expected)

  def test_datetime(self):
    time = np.array('2000-01-01', dtype='datetime64')
    dt = np.array([1, 2, 3], dtype='timedelta64[D]')
    expected = cx.Field(
        jdt.to_datetime(
            np.array(
                ['2000-01-02', '2000-01-03', '2000-01-04'], dtype='datetime64'
            )
        )
    )
    actual = cx.cmap(lambda x: x + dt)(cx.Field(time))
    testing.assert_fields_equal(actual, expected)

  def test_to_and_from_xarray(self):
    xarray = pytest.importorskip("xarray")
    field = cx.Field(
        jdt.to_timedelta(np.array([1, 2]), unit='D'), dims=('time',)
    )
    data_array = xarray.DataArray(
        data=np.array([1, 2], dtype='timedelta64[D]'),
        dims='time',
    )

    actual_field = cx.Field.from_xarray(data_array)
    testing.assert_fields_equal(actual_field, field)

    actual_data_array = field.to_xarray()
    xarray.testing.assert_equal(actual_data_array, data_array)


if __name__ == '__main__':
  absltest.main()
