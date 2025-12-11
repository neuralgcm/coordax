# API docs

```{eval-rst}
.. currentmodule:: coordax
```

## Fields

```{eval-rst}
.. autosummary::
    :toctree: _autosummary

    Field
    Field.broadcast_like
    Field.order_as
    Field.tag
    Field.untag
    Field.unwrap
    field
    cmap
    cpmap
    tag
    untag
    is_field
    get_coordinate
    new_axis_name
    shape_struct_field
```

## Coordinates

```{eval-rst}
.. autosummary::
    :toctree: _autosummary

    Coordinate
    CartesianProduct
    DummyAxis
    LabeledAxis
    Scalar
    SizedAxis
    SelectedAxis
    coords.canonicalize
    coords.compose
    coords.insert_axes
    coords.replace_axes
```

## Xarray compatibility

```{eval-rst}
.. autosummary::
    :toctree: _autosummary

    Field.to_xarray
    Field.from_xarray
    Coordinate.to_xarray
    Coordinate.from_xarray
    coords.from_xarray
    coords.NoCoordinateMatch
```

## Testing

```{eval-rst}
.. autosummary::
    :toctree: _autosummary

    testing.assert_fields_allclose
    testing.assert_fields_equal
    testing.assert_field_properties
```

## Custom array types

```{warning}
This is _not_ a stable API.
```

```{eval-rst}
.. autosummary::
    :toctree: _autosummary

    NDArray
    register_ndarray
```
