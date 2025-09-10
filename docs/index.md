# Coordax documentation

## Overview

Coordax makes it easy to associate array dimensions with coordinates in the
context of scientific simulation codes written in JAX. This allows for
efficient and expressive manipulation of data defined on structured grids,
enabling operations like differentiation and interpolation with respect to
physical coordinates.

Coordax was designed to meet the needs of
[NeuralGCM](https://github.com/neuralgcm/neuralgcm), but we hope it will be
useful more broadly!

## Key features

1. Compute on locally-positional axes via coordinate map (`cmap`)
2. Coordinate objects that carry discretization details and custom methods
3. Lossless conversion to and from [Xarray](https://github.com/pydata/xarray)
   data structures (e.g., for serialization)

## Questions?

The best place to ask for help or report bugs is
[on GitHub](https://github.com/neuralgcm/coordax/issues).

## Contents

```{toctree}
:maxdepth: 1
installation.md
quickstart.ipynb
api.md
```