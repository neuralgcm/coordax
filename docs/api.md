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
    Field.isel
    Field.order_as
    Field.sel
    Field.tag
    Field.untag
    Field.unwrap
    field
    cmap
    cpmap
    tag
    untag
    is_field
    contains_dims
    get_coordinate
    get_coordinate_part
    new_axis_name
    shape_struct_field
```

## Coordinates

```{eval-rst}
.. autosummary::
    :toctree: _autosummary

    Coordinate
    Coordinate.isel
    Coordinate.sel
    CartesianProduct
    DummyAxis
    LabeledAxis
    Scalar
    SizedAxis
    SelectedAxis
    is_coord
    coords.ArrayKey
    coords.canonicalize
    coords.compose
    coords.extract
    coords.insert_axes
    coords.replace_axes
```

## Xarray compatibility

```{eval-rst}
.. autosummary::
    :toctree: _autosummary

    from_xarray
    Field.to_xarray
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

## Experimental

```{eval-rst}
.. autosummary::
    :toctree: _autosummary

    NDArray
    register_ndarray
```
