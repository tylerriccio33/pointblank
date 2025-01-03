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
class StartsWith(ColumnSelector):
    text: str
    case_sensitive: bool = False

    def resolve(self, columns: list[str]) -> list[str]:
        if self.case_sensitive:
            return [col for col in columns if col.startswith(self.text)]
        return [col for col in columns if col.lower().startswith(self.text.lower())]


@dataclass
class EndsWith(ColumnSelector):
    text: str
    case_sensitive: bool = False

    def resolve(self, columns: list[str]) -> list[str]:
        if self.case_sensitive:
            return [col for col in columns if col.endswith(self.text)]
        return [col for col in columns if col.lower().endswith(self.text.lower())]


@dataclass
class Contains(ColumnSelector):
    text: str
    case_sensitive: bool = False

    def resolve(self, columns: list[str]) -> list[str]:
        if self.case_sensitive:
            return [col for col in columns if self.text in col]
        return [col for col in columns if self.text.lower() in col.lower()]


@dataclass
class Matches(ColumnSelector):
    pattern: str
    case_sensitive: bool = False

    def resolve(self, columns: list[str]) -> list[str]:
        matches = (
            [col for col in columns if re.search(self.pattern, col)]
            if self.case_sensitive
            else [col for col in columns if re.search(self.pattern, col, re.IGNORECASE)]
        )
        return matches if matches else []


@dataclass
class Everything(ColumnSelector):
    def resolve(self, columns: list[str]) -> list[str]:
        return columns


@dataclass
class FirstN(ColumnSelector):
    n: int
    offset: int = 0

    def resolve(self, columns: list[str]) -> list[str]:
        return columns[self.offset : self.offset + self.n]


@dataclass
class LastN(ColumnSelector):
    n: int
    offset: int = 0

    def resolve(self, columns: list[str]) -> list[str]:
        reversed_columns = columns[::-1]
        selected_columns = reversed_columns[self.offset : self.offset + self.n]
        return selected_columns[::-1]


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

    exprs: str | ColumnSelector
    name: str = field(init=False)

    def __post_init__(self):
        if isinstance(self.exprs, str):
            self.name = self.exprs
        else:
            self.name = ""

    def __repr__(self):
        return self.exprs if isinstance(self.exprs, str) else repr(self.exprs)

    def resolve(self, columns: list[str]) -> list[str]:
        if self.name:
            return [self.name]
        if isinstance(self.exprs, str):
            return [self.exprs] if self.exprs in columns else []
        resolved_columns = self.exprs.resolve(columns)
        return [col for col in columns if col in resolved_columns]


def col(exprs: str | ColumnSelector) -> Column:
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

    From results of the validation table it can be seen that values in `a` were greater than values
    in `b` for every row (or test unit). Using `value=pb.col("b")` specified that the greater-than
    comparison is across columns, not with a fixed literal value.
    """
    return Column(exprs=exprs)


def starts_with(text: str, case_sensitive: bool = False) -> StartsWith:
    """
    Select columns that start with specified text.

    Parameters
    ----------
    text
        The text that the column name should start with.
    case_sensitive
        Whether column names should be treated as case-sensitive. The default is `False`.

    Returns
    -------
    StartsWith
        A `StartsWith` object, which can be used to select columns that start with the specified
        text.
    """
    return StartsWith(text=text, case_sensitive=case_sensitive)


def ends_with(text: str, case_sensitive: bool = False) -> EndsWith:
    """
    Select columns that end with specified text.

    Parameters
    ----------
    text
        The text that the column name should end with.
    case_sensitive
        Whether column names should be treated as case-sensitive. The default is `False`.

    Returns
    -------
    EndsWith
        An `EndsWith` object, which can be used to select columns that end with the specified text.
    """
    return EndsWith(text=text, case_sensitive=case_sensitive)


def contains(text: str, case_sensitive: bool = False) -> Contains:
    """
    Select columns that contain specified text.

    Parameters
    ----------
    text
        The text that the column name should contain.
    case_sensitive
        Whether column names should be treated as case-sensitive. The default is `False`.

    Returns
    -------
    Contains
        A `Contains` object, which can be used to select columns that contain the specified text.
    """
    return Contains(text=text, case_sensitive=case_sensitive)


def matches(pattern: str, case_sensitive: bool = False) -> Matches:
    """
    Select columns that match a specified regular expression pattern.

    Parameters
    ----------
    pattern
        The regular expression pattern that the column name should match.
    case_sensitive
        Whether column names should be treated as case-sensitive. The default is `False`.

    Returns
    -------
    Matches
        A `Matches` object, which can be used to select columns that match the specified pattern.
    """
    return Matches(pattern=pattern, case_sensitive=case_sensitive)


def everything() -> Everything:
    """
    Select all columns.

    Returns
    -------
    Everything
        An `Everything` object, which can be used to select all columns.
    """
    return Everything()


def first_n(n: int, offset: int = 0) -> FirstN:
    """
    Select the first `n` columns in the column list.

    Parameters
    ----------
    n
        The number of columns to select.
    offset
        The offset from the start of the column list. The default is `0`.

    Returns
    -------
    FirstN
        A `FirstN` object, which can be used to select the first `n` columns.
    """
    return FirstN(n=n, offset=offset)


def last_n(n: int, offset: int = 0) -> LastN:
    """
    Select the last `n` columns in the column list.

    Parameters
    ----------
    n
        The number of columns to select.
    offset
        The offset from the end of the column list. The default is `0`.

    Returns
    -------
    LastN
        A `LastN` object, which can be used to select the last `n` columns.
    """
    return LastN(n=n, offset=offset)
