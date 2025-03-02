
from __future__ import annotations

## Absolute bounds, ie. plus or minus
type AbsoluteBounds = tuple[int, int]

## Relative bounds, ie. plus or minus some percent
type RelativeBounds = tuple[float, float]

## Tolerance afforded to some check
type Tolerance = int | float | AbsoluteBounds | RelativeBounds
