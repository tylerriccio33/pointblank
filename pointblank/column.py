from __future__ import annotations

from dataclasses import dataclass

__all__ = ["col"]
@dataclass
class ColumnSelector:
    def resolve(self, columns: list[str]) -> list[str]:
        raise NotImplementedError

    def __and__(self, other: ColumnSelector) -> ColumnSelector:
        return AndSelector(self, other)

    def __or__(self, other: ColumnSelector) -> ColumnSelector:
        return OrSelector(self, other)

    def __sub__(self, other: ColumnSelector) -> ColumnSelector:
        return SubSelector(self, other)

    def __invert__(self) -> ColumnSelector:
        return NotSelector(self)
@dataclass
class AndSelector(ColumnSelector):
    left: ColumnSelector
    right: ColumnSelector

    def resolve(self, columns: list[str]) -> list[str]:
        left_columns = self.left.resolve(columns)
        right_columns = self.right.resolve(columns)
        return [col for col in left_columns if col in right_columns]


@dataclass
class OrSelector(ColumnSelector):
    left: ColumnSelector
    right: ColumnSelector

    def resolve(self, columns: list[str]) -> list[str]:
        left_columns = self.left.resolve(columns)
        right_columns = self.right.resolve(columns)
        return list(set(left_columns + right_columns))


@dataclass
class SubSelector(ColumnSelector):
    left: ColumnSelector
    right: ColumnSelector

    def resolve(self, columns: list[str]) -> list[str]:
        left_columns = self.left.resolve(columns)
        right_columns = self.right.resolve(columns)
        return [col for col in left_columns if col not in right_columns]


@dataclass
class NotSelector(ColumnSelector):
    selector: ColumnSelector

    def resolve(self, columns: list[str]) -> list[str]:
        selected_columns = self.selector.resolve(columns)
        return [col for col in columns if col not in selected_columns]


@dataclass
class Column:
    """
    A class to represent a column in a table.
    """

    name: str

    def __repr__(self):
        return self.name


def col(name: str) -> Column:
    """
    Helper function for referencing a column in the input table.

    Many of the validation methods (i.e., `col_vals_*()` methods) in pointblank have a `value=`
    argument. These validations are comparisons between column values and a literal value, or,
    between column values and adjacent values in another column. The `col()` helper function is used
    to specify that it is a column being referenced, not a literal value.

    The `col()` doesn't check that the column exists in the input table. It acts to signal that the
    value being compared is a column value. During validation (i.e., when `interrogate()` is
    called), pointblank will then check that the column exists in the input table.

    This function can be used in the `value=` argument of the following validation methods:

    - `col_vals_gt()`
    - `col_vals_lt()`
    - `col_vals_ge()`
    - `col_vals_le()`
    - `col_vals_eq()`
    - `col_vals_ne()`
    - `col_vals_between()`
    - `col_vals_outside()`

    For the last two methods cited, `col()` can be used with either of the `left=` and `right=`
    arguments, or both.

    Parameters
    ----------
    name
        The name of the column in the input table.

    Returns
    -------
    Column
        A `Column` object representing the column.

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_header=False, report_incl_footer=False)
    ```

    Suppose we have a table with columns `a` and `b` and we'd like to validate that the values in
    column `a` are greater than the values in column `b`. We can use the `col()` helper function to
    reference the comparison column when creating the validation step.

    ```{python}
    import polars as pl
    import pointblank as pb

    tbl = pl.DataFrame(
        {
            "a": [5, 6, 5, 7, 6, 5],
            "b": [4, 2, 3, 3, 4, 3],
        }
    )

    validation = (
        pb.Validate(data=tbl)
        .col_vals_gt(columns="a", value=pb.col("b"))
        .interrogate()
    )

    validation
    ```

    From the excerpt of the validation table, values in `a` were greater than values in `b` for
    every row (or test unit). Using `value=pb.col("b")` specified that the greater-than comparison
    is across columns, not with a fixed literal value.
    """
    return Column(name=name)
