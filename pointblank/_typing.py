from __future__ import annotations

from typing import TypeAlias

## Absolute bounds, ie. plus or minus
AbsoluteBounds: TypeAlias = tuple[int, int]

## Relative bounds, ie. plus or minus some percent
RelativeBounds: TypeAlias = tuple[float, float]

## Tolerance afforded to some check
Tolerance: TypeAlias = int | float | AbsoluteBounds | RelativeBounds

## Types for data segmentation

## Value(s) that can be used in a segment tuple
SegmentValue: TypeAlias = str | list[str]

## (column, value(s)) format for segments
SegmentTuple: TypeAlias = tuple[str, SegmentValue]

## Individual segment item (string or tuple)
SegmentItem: TypeAlias = str | SegmentTuple

## Full segment specification options
SegmentSpec: TypeAlias = str | SegmentTuple | list[SegmentItem]
