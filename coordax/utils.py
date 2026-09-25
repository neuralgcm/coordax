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
"""Internal Coordax utilities."""
from collections.abc import Callable
from typing import TypeVar, overload


T = TypeVar('T')


@overload
def export(obj: T, module: str = 'coordax') -> T:
  ...


@overload
def export(
    obj: None = None, *, module: str = 'coordax'
) -> Callable[[T], T]:
  ...


def export(
    obj: T | None = None, module: str = 'coordax'
) -> T | Callable[[T], T]:
  if obj is None:
    return lambda x: export(x, module=module)
  obj.__module__ = module
  return obj
