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

# Note: import <name> as <name> is required for names to be exported.
# See PEP 484 & https://github.com/jax-ml/jax/issues/7570
# pylint: disable=g-multiple-import,useless-import-alias,g-importing-member
from coordax import coords
from coordax.coordinate_systems import (
    CartesianProduct as CartesianProduct,
    Coordinate as Coordinate,
    NoCoordinateMatch as NoCoordinateMatch,  # deprecated
    DummyAxis as DummyAxis,
    LabeledAxis as LabeledAxis,
    Scalar as Scalar,
    SizedAxis as SizedAxis,
    SelectedAxis as SelectedAxis,
    canonicalize_coordinates as canonicalize_coordinates,  # deprecated
    compose_coordinates as compose_coordinates,  # deprecated
    insert_axes_to_coordinate as insert_axes_to_coordinate,  # deprecated
    replace_axes_in_coordinate as replace_axes_in_coordinate,  # deprecated
    coordinates_from_xarray as coordinates_from_xarray,  # deprecated
)
from coordax.fields import (
    Field as Field,
    field as field,
    is_field as is_field,
    new_axis_name as new_axis_name,
    tmp_axis_name as tmp_axis_name,  # deprecated
    shape_struct_field as shape_struct_field,
    cmap as cmap,
    cpmap as cpmap,
    get_coordinate as get_coordinate,
    from_xarray as from_xarray,
    wrap_like as wrap_like,  # deprecated
    wrap as wrap,  # deprecated
    tag as tag,
    untag as untag,
)
from coordax.ndarrays import (
    NDArray as NDArray,
    register_ndarray as register_ndarray,
)
import coordax.testing  # pylint: disable=unused-import

__version__ = '0.2.3'  # keep sync with pyproject.toml
