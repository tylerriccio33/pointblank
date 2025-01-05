from __future__ import annotations

import re

from dataclasses import dataclass, field

__all__ = [
    "col",
    "starts_with",
    "ends_with",
    "contains",
    "matches",
    "everything",
    "first_n",
    "last_n",
]


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

    Parameters
    ----------
    exprs
        Either the name of a single column in the target table, provided as a string, or, an
        expression involving column selector functions (e.g., `starts_with("a")`,
        `ends_with("e") | starts_with("a")`, etc.). Please read the documentation for further
        details on which input forms are valid depending on the context.

    Returns
    -------
    Column
        A `Column` object representing the column.

    Usage with the `columns=` Argument
    -----------------------------------
    The `col()` function can be used in the `columns=` argument of the following validation methods:

    - `col_vals_gt()`
    - `col_vals_lt()`
    - `col_vals_ge()`
    - `col_vals_le()`
    - `col_vals_eq()`
    - `col_vals_ne()`
    - `col_vals_between()`
    - `col_vals_outside()`
    - `col_vals_in_set()`
    - `col_vals_not_in_set()`
    - `col_vals_null()`
    - `col_vals_not_null()`
    - `col_vals_regex()`
    - `col_exists()`

    If specifying a single column with certainty (you have the exact name), `col()` is not necessary
    since you can just pass the column name as a string (though it is still valid to use
    `col("column_name")`, if preferred). However, if you want to select columns based on complex
    logic involving multiple column selector functions (e.g., columns that start with `"a"` but
    don't end with `"e"`), you need to use `col()` to wrap expressions involving column selector
    functions and logical operators such as `&`, `|`, `-`, and `~`.

    Here is an example of such usage with the `col_vals_gt()` validation method:

    ```python
    col_vals_gt(columns=col(starts_with("a") & ~ends_with("e")), value=10)
    ```

    If using only a single column selector function, you can pass the function directly to the
    `columns=` argument of the validation method, or, you can use `col()` to wrap the function
    (either is valid though the first is more concise). Here is an example of that simpler usage:

    ```python
    col_vals_gt(columns=starts_with("a"), value=10)
    ```

    Usage with the `value=`, `left=`, and `right=` Arguments
    --------------------------------------------------------
    The `col()` function can be used in the `value=` argument of the following validation methods

    - `col_vals_gt()`
    - `col_vals_lt()`
    - `col_vals_ge()`
    - `col_vals_le()`
    - `col_vals_eq()`
    - `col_vals_ne()`

    and in the `left=` and `right=` arguments (either or both) of these two validation methods

    - `col_vals_between()`
    - `col_vals_outside()`

    You cannot use column selector functions such as `starts_with()` in either of the `value=`,
    `left=`, or `right=` arguments since there would be no guarantee that a single column will be
    resolved from the target table with this approach. The `col()` function is used to signal that
    the value being compared is a column value and not a literal value.

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

    Many validation methods have a `columns=` argument that can be used to specify the columns for
    validation (e.g., `col_vals_gt()`, `col_vals_regex()`, etc.). The `starts_with()` selector
    function can be used to select columns that start with some specified text. So if a set of table
    columns consists of `name`, `age`, and `address`, and you want to validate columns that start
    with `"a"`, you can use `columns=starts_with("a")`. This will select `age` and `address`
    columns.

    There will be a validation step created for every resolved column. Note that if there aren't any
    columns resolved from using `starts_with()` (or any other expression using selector functions),
    the validation step will fail to be evaluated during the interrogation process. Such a failure
    to evaluate will be reported in the validation results but it won't affect the interrogation
    process overall (i.e., the process won't be halted).

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

    Relevant Validation Methods where `starts_with()` can be Used
    -------------------------------------------------------------
    This selector function can be used in the `columns=` argument of the following validation
    methods:

    - `col_vals_gt()`
    - `col_vals_lt()`
    - `col_vals_ge()`
    - `col_vals_le()`
    - `col_vals_eq()`
    - `col_vals_ne()`
    - `col_vals_between()`
    - `col_vals_outside()`
    - `col_vals_in_set()`
    - `col_vals_not_in_set()`
    - `col_vals_null()`
    - `col_vals_not_null()`
    - `col_vals_regex()`
    - `col_exists()`

    The `starts_with()` selector function doesn't need to be used in isolation. Read the next
    section for information on how to compose it with other column selectors for more refined ways
    to select columns.

    Additional Flexibilty through Composition with Other Column Selectors
    ---------------------------------------------------------------------
    The `starts_with()` function can be composed with other column selectors to create fine-grained
    column selections. For example, to select columns that start with `a` and end with `e`, you can
    use the `starts_with()` and `ends_with()` functions together. The only condition is that the
    expressions are wrapped in the `col()` function, like this:

    ```python
    col(starts_with("a") & ends_with("e"))
    ```

    There are four operators that can be used to compose column selectors:

    - `&` (*and*)
    - `|` (*or*)
    - `-` (*difference*)
    - `~` (*not*)

    The `&` operator is used to select columns that satisfy both conditions. The `|` operator is
    used to select columns that satisfy either condition. The `-` operator is used to select columns
    that satisfy the first condition but not the second. The `~` operator is used to select columns
    that don't satisfy the condition. As many selector functions can be used as needed and the
    operators can be combined to create complex column selection criteria (parentheses can be used
    to group conditions and control the order of evaluation).

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_header=False, report_incl_footer=False)
    ```

    Suppose we have a table with columns `name`, `paid_2021`, `paid_2022`, and `days_worked` and
    we'd like to validate that the values in columns that start with `paid` are greater than `10`.
    We can use the `starts_with()` column selector function to specify the columns that start with
    `paid` as the columns to validate.

    ```{python}
    import polars as pl
    import pointblank as pb

    tbl = pl.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie"],
            "paid_2021": [16.32, 16.25, 15.75],
            "paid_2022": [18.62, 16.95, 18.25],
            "person_id": ["A123", "B456", "C789"],
        }
    )

    validation = (
        pb.Validate(data=tbl)
        .col_vals_gt(columns=pb.starts_with("paid"), value=10)
        .interrogate()
    )

    validation
    ```

    From the results of the validation table we get two validation steps, one for `paid_2021` and
    one for `paid_2022`. The values in both columns were greater than `10`.

    We can also use the `starts_with()` function in combination with other column selectors (within
    `col()`) to create more complex column selection criteria (i.e., to select columns that satisfy
    multiple conditions). For example, to select columns that start with `paid` and end with `2023`
    or `2024`, we can use the `&` and `|` operators to combine column selectors.

    ```{python}
    tbl = pl.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie"],
            "hours_2022": [160, 180, 160],
            "hours_2023": [182, 168, 175],
            "hours_2024": [200, 165, 190],
            "paid_2022": [18.62, 16.95, 18.25],
            "paid_2023": [19.29, 17.75, 18.35],
            "paid_2024": [20.73, 18.35, 20.10],
        }
    )

    validation = (
        pb.Validate(data=tbl)
        .col_vals_gt(
            columns=pb.col(pb.starts_with("paid") & pb.matches("23|24")),
            value=10
        )
        .interrogate()
    )

    validation
    ```

    From the results of the validation table we get two validation steps, one for `paid_2023` and
    one for `paid_2024`.
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
