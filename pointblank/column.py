from __future__ import annotations

import re

import narwhals as nw
from narwhals.typing import IntoDataFrame

from dataclasses import dataclass

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
        raise NotImplementedError  # pragma: no cover

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

    exprs: ColumnSelector | ColumnSelectorNarwhals

    def resolve(self, columns: list[str], table: IntoDataFrame | None = None) -> list[str]:

        if isinstance(self.exprs, ColumnSelector):
            resolved_columns = self.exprs.resolve(columns)
            return [col for col in columns if col in resolved_columns]

        raise TypeError(f"Unsupported type: {type(self.exprs)}")  # pragma: no cover

    def __repr__(self):
        return self.exprs if isinstance(self.exprs, str) else repr(self.exprs)


@dataclass
class ColumnLiteral(Column):
    """
    A class to represent a literal value for a column parameter (the column name).
    """

    exprs: str

    def resolve(self, columns: list[str], table: IntoDataFrame | None = None) -> list[str]:
        if isinstance(self.exprs, str):
            return [self.exprs]
        raise TypeError(f"Unsupported type: {type(self.exprs)}")  # pragma: no cover

    @property
    def name(self) -> str:
        return self.exprs

    def __repr__(self):
        return self.exprs


@dataclass
class ColumnSelectorNarwhals(Column):
    """
    A class for using Narwhals selectors

    The Narwhals selectors are available in `narwhals.selectors` and they include:
    - `boolean()`
    - `by_dtype()`
    - `categorical()`
    - `numeric()`
    - `string()`

    The `ColumnNarwhals` class is used to access these selectors for column selection. The selectors
    can be used in the `columns=` argument of a wide range of validation methods to select columns
    based on their data type or other properties.
    """

    exprs: nw.selectors.Selector

    def resolve(self, table) -> list[str]:
        # Convert the native table to a Narwhals DataFrame
        dfn = nw.from_native(table)
        # Use the selector to select columns and return their names
        columns = dfn.select(self.exprs.exprs).columns
        return columns


def col(
    exprs: str | ColumnSelector | ColumnSelectorNarwhals,
) -> Column | ColumnLiteral | ColumnSelectorNarwhals:
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

    Available Selectors
    -------------------
    There is a collection of selectors available in pointblank, allowing you to select columns based
    on attributes of column names and positions. The selectors are:

    - `starts_with()`
    - `ends_with()`
    - `contains()`
    - `matches()`
    - `everything()`
    - `first_n()`
    - `last_n()`

    Alternatively, we support selectors from the Narwhals library! Those selectors can additionally
    take advantage of the data types of the columns. The selectors are:

    - `boolean()`
    - `by_dtype()`
    - `categorical()`
    - `matches()`
    - `numeric()`
    - `string()`

    Have a look at the [Narwhals API documentation on selectors](https://narwhals-dev.github.io/narwhals/api-reference/selectors/)
    for more information.

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_header=False, report_incl_footer=False, preview_incl_header=False)
    ```

    Suppose we have a table with columns `a` and `b` and we'd like to validate that the values in
    column `a` are greater than the values in column `b`. We can use the `col()` helper function to
    reference the comparison column when creating the validation step.

    ```{python}
    import pointblank as pb
    import polars as pl

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

    If you want to select an arbitrary set of columns upon which to base a validation, you can use
    column selector functions (e.g., `starts_with()`, `ends_with()`, etc.) to specify columns in the
    `columns=` argument of a validation method. Let's use the `starts_with()` column selector
    function to select columns that start with `"paid"` and validate that the values in those
    columns are greater than `10`.

    ```{python}
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
        .col_vals_gt(columns=pb.col(pb.starts_with("paid")), value=10)
        .interrogate()
    )

    validation
    ```

    In the above example the `col()` function contains the invocation of the `starts_with()` column
    selector function. This is not strictly necessary when using a single column selector function,
    so `columns=pb.starts_with("paid")` would be equivalent usage here. However, the use of `col()`
    is required when using multiple column selector functions with logical operators. Here is an
    example of that more complex usage:

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
            columns=pb.col(pb.starts_with("paid") & pb.matches("2023|2024")),
            value=10
        )
        .interrogate()
    )

    validation
    ```

    In the above example the `col()` function contains the invocation of the `starts_with()` and
    `matches()` column selector functions, combined with the `&` operator. This is necessary to
    specify the set of columns that start with `"paid"` *and* match the text `"2023"` or `"2024"`.

    If you'd like to take advantage of Narwhals selectors, that's also possible. Here is an example
    of using the `numeric()` column selector function to select all numeric columns for validation,
    checking that their values are greater than `0`.

    ```{python}
    import narwhals.selectors as ncs

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
        .col_vals_ge(columns=pb.col(ncs.numeric()), value=0)
        .interrogate()
    )

    validation
    ```

    In the above example the `col()` function contains the invocation of the `numeric()` column
    selector function from Narwhals. As with the other selectors, this is not strictly necessary
    when using a single column selector, so `columns=ncs.numeric()` would also be fine here.

    Narwhals selectors can also use operators to combine multiple selectors. Here is an example of
    using the `numeric()` and `matches()` selectors together to select all numeric columns that fit
    a specific pattern.

    ```{python}
    tbl = pl.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie"],
            "2022_status": ["ft", "ft", "pt"],
            "2023_status": ["ft", "pt", "ft"],
            "2024_status": ["ft", "pt", "ft"],
            "2022_pay_total": [18.62, 16.95, 18.25],
            "2023_pay_total": [19.29, 17.75, 18.35],
            "2024_pay_total": [20.73, 18.35, 20.10],
        }
    )

    validation = (
        pb.Validate(data=tbl)
        .col_vals_lt(columns=pb.col(ncs.numeric() & ncs.matches("2023|2024")), value=30)
        .interrogate()
    )

    validation
    ```

    In the above example the `col()` function contains the invocation of the `numeric()` and
    `matches()` column selector functions from Narwhals, combined with the `&` operator. This is
    necessary to specify the set of columns that are numeric *and* match the text `"2023"` or
    `"2024"`.
    """
    if isinstance(exprs, str):
        return ColumnLiteral(exprs=exprs)
    elif isinstance(exprs, ColumnSelector):
        return Column(exprs=exprs)
    elif isinstance(exprs, nw.selectors.Selector):
        return ColumnSelectorNarwhals(exprs=exprs)

    raise TypeError(f"Unsupported type: {type(exprs)}")  # pragma: no cover


def starts_with(text: str, case_sensitive: bool = False) -> StartsWith:
    """
    Select columns that start with specified text.

    Many validation methods have a `columns=` argument that can be used to specify the columns for
    validation (e.g., `col_vals_gt()`, `col_vals_regex()`, etc.). The `starts_with()` selector
    function can be used to select one or more columns that start with some specified text. So if
    the set of table columns consists of

    `[name_first, name_last, age, address]`

    and you want to validate columns that start with `"name"`, you can use
    `columns=starts_with("name")`. This will select the `name_first` and `name_last` columns.

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
    column selections. For example, to select columns that start with `"a"` and end with `"e"`, you
    can use the `starts_with()` and `ends_with()` functions together. The only condition is that the
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
    pb.config(report_incl_header=False, report_incl_footer=False, preview_incl_header=False)
    ```

    Suppose we have a table with columns `name`, `paid_2021`, `paid_2022`, and `person_id` and
    we'd like to validate that the values in columns that start with `"paid"` are greater than `10`.
    We can use the `starts_with()` column selector function to specify the columns that start with
    `"paid"` as the columns to validate.

    ```{python}
    import pointblank as pb
    import polars as pl

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
    one for `paid_2022`. The values in both columns were all greater than `10`.

    We can also use the `starts_with()` function in combination with other column selectors (within
    `col()`) to create more complex column selection criteria (i.e., to select columns that satisfy
    multiple conditions). For example, to select columns that start with `"paid"` and match the text
    `"2023"` or `"2024"`, we can use the `&` operator to combine column selectors.

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

    Many validation methods have a `columns=` argument that can be used to specify the columns for
    validation (e.g., `col_vals_gt()`, `col_vals_regex()`, etc.). The `ends_with()` selector
    function can be used to select one or more columns that end with some specified text. So if the
    set of table columns consists of

    `[first_name, last_name, age, address]`

    and you want to validate columns that end with `"name"`, you can use
    `columns=ends_with("name")`. This will select the `first_name` and `last_name` columns.

    There will be a validation step created for every resolved column. Note that if there aren't any
    columns resolved from using `ends_with()` (or any other expression using selector functions),
    the validation step will fail to be evaluated during the interrogation process. Such a failure
    to evaluate will be reported in the validation results but it won't affect the interrogation
    process overall (i.e., the process won't be halted).

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

    Relevant Validation Methods where `ends_with()` can be Used
    -----------------------------------------------------------
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

    The `ends_with()` selector function doesn't need to be used in isolation. Read the next section
    for information on how to compose it with other column selectors for more refined ways to select
    columns.

    Additional Flexibilty through Composition with Other Column Selectors
    ---------------------------------------------------------------------
    The `ends_with()` function can be composed with other column selectors to create fine-grained
    column selections. For example, to select columns that end with `"e"` and start with `"a"`, you
    can use the `ends_with()` and `starts_with()` functions together. The only condition is that the
    expressions are wrapped in the `col()` function, like this:

    ```python
    col(ends_with("e") & starts_with("a"))
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
    pb.config(report_incl_header=False, report_incl_footer=False, preview_incl_header=False)
    ```

    Suppose we have a table with columns `name`, `2021_pay`, `2022_pay`, and `person_id` and
    we'd like to validate that the values in columns that end with `"pay"` are greater than `10`.
    We can use the `ends_with()` column selector function to specify the columns that end with
    `"pay"` as the columns to validate.

    ```{python}
    import pointblank as pb
    import polars as pl

    tbl = pl.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie"],
            "2021_pay": [16.32, 16.25, 15.75],
            "2022_pay": [18.62, 16.95, 18.25],
            "person_id": ["A123", "B456", "C789"],
        }
    )

    validation = (
        pb.Validate(data=tbl)
        .col_vals_gt(columns=pb.ends_with("pay"), value=10)
        .interrogate()
    )

    validation
    ```

    From the results of the validation table we get two validation steps, one for `2021_pay` and one
    for `2022_pay`. The values in both columns were all greater than `10`.

    We can also use the `ends_with()` function in combination with other column selectors (within
    `col()`) to create more complex column selection criteria (i.e., to select columns that satisfy
    multiple conditions). For example, to select columns that end with `"pay"` and match the text
    `"2023"` or `"2024"`, we can use the `&` operator to combine column selectors.

    ```{python}
    tbl = pl.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie"],
            "2022_hours": [160, 180, 160],
            "2023_hours": [182, 168, 175],
            "2024_hours": [200, 165, 190],
            "2022_pay": [18.62, 16.95, 18.25],
            "2023_pay": [19.29, 17.75, 18.35],
            "2024_pay": [20.73, 18.35, 20.10],
        }
    )

    validation = (
        pb.Validate(data=tbl)
        .col_vals_gt(
            columns=pb.col(pb.ends_with("pay") & pb.matches("2023|2024")),
            value=10
        )
        .interrogate()
    )

    validation
    ```

    From the results of the validation table we get two validation steps, one for `2023_pay` and one
    for `2024_pay`.
    """
    return EndsWith(text=text, case_sensitive=case_sensitive)


def contains(text: str, case_sensitive: bool = False) -> Contains:
    """
    Select columns that contain specified text.

    Many validation methods have a `columns=` argument that can be used to specify the columns for
    validation (e.g., `col_vals_gt()`, `col_vals_regex()`, etc.). The `contains()` selector function
    can be used to select one or more columns that contain some specified text. So if the set of
    table columns consists of

    `[profit, conv_first, conv_last, highest_conv, age]`

    and you want to validate columns that have `"conv"` in the name, you can use
    `columns=contains("conv")`. This will select the `conv_first`, `conv_last`, and `highest_conv`
    columns.

    There will be a validation step created for every resolved column. Note that if there aren't any
    columns resolved from using `contains()` (or any other expression using selector functions), the
    validation step will fail to be evaluated during the interrogation process. Such a failure to
    evaluate will be reported in the validation results but it won't affect the interrogation
    process overall (i.e., the process won't be halted).

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

    Relevant Validation Methods where `contains()` can be Used
    ----------------------------------------------------------
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

    The `contains()` selector function doesn't need to be used in isolation. Read the next section
    for information on how to compose it with other column selectors for more refined ways to select
    columns.

    Additional Flexibilty through Composition with Other Column Selectors
    ---------------------------------------------------------------------
    The `contains()` function can be composed with other column selectors to create fine-grained
    column selections. For example, to select columns that have the text `"_n"` and start with
    `"item"`, you can use the `contains()` and `starts_with()` functions together. The only
    condition is that the expressions are wrapped in the `col()` function, like this:

    ```python
    col(contains("_n") & starts_with("item"))
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
    pb.config(report_incl_header=False, report_incl_footer=False, preview_incl_header=False)
    ```

    Suppose we have a table with columns `name`, `2021_pay_total`, `2022_pay_total`, and `person_id`
    and we'd like to validate that the values in columns having `"pay"` in the name are greater than
    `10`. We can use the `contains()` column selector function to specify the column names that
    contain `"pay"` as the columns to validate.

    ```{python}
    import pointblank as pb
    import polars as pl

    tbl = pl.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie"],
            "2021_pay_total": [16.32, 16.25, 15.75],
            "2022_pay_total": [18.62, 16.95, 18.25],
            "person_id": ["A123", "B456", "C789"],
        }
    )

    validation = (
        pb.Validate(data=tbl)
        .col_vals_gt(columns=pb.contains("pay"), value=10)
        .interrogate()
    )

    validation
    ```

    From the results of the validation table we get two validation steps, one for `2021_pay_total`
    and one for `2022_pay_total`. The values in both columns were all greater than `10`.

    We can also use the `contains()` function in combination with other column selectors (within
    `col()`) to create more complex column selection criteria (i.e., to select columns that satisfy
    multiple conditions). For example, to select columns that contain `"pay"` and match the text
    `"2023"` or `"2024"`, we can use the `&` operator to combine column selectors.

    ```{python}
    tbl = pl.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie"],
            "2022_hours": [160, 180, 160],
            "2023_hours": [182, 168, 175],
            "2024_hours": [200, 165, 190],
            "2022_pay_total": [18.62, 16.95, 18.25],
            "2023_pay_total": [19.29, 17.75, 18.35],
            "2024_pay_total": [20.73, 18.35, 20.10],
        }
    )

    validation = (
        pb.Validate(data=tbl)
        .col_vals_gt(
            columns=pb.col(pb.contains("pay") & pb.matches("2023|2024")),
            value=10
        )
        .interrogate()
    )

    validation
    ```

    From the results of the validation table we get two validation steps, one for `2023_pay_total`
    and one for `2024_pay_total`.
    """
    return Contains(text=text, case_sensitive=case_sensitive)


def matches(pattern: str, case_sensitive: bool = False) -> Matches:
    """
    Select columns that match a specified regular expression pattern.

    Many validation methods have a `columns=` argument that can be used to specify the columns for
    validation (e.g., `col_vals_gt()`, `col_vals_regex()`, etc.). The `matches()` selector function
    can be used to select one or more columns matching a provided regular expression pattern. So if
    the set of table columns consists of

    `[rev_01, rev_02, profit_01, profit_02, age]`

    and you want to validate columns that have two digits at the end of the name, you can use
    `columns=matches(r"\d{2}$")`. This will select the `rev_01`, `rev_02`, `profit_01`, and
    `profit_02` columns.

    There will be a validation step created for every resolved column. Note that if there aren't any
    columns resolved from using `matches()` (or any other expression using selector functions), the
    validation step will fail to be evaluated during the interrogation process. Such a failure to
    evaluate will be reported in the validation results but it won't affect the interrogation
    process overall (i.e., the process won't be halted).

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

    Relevant Validation Methods where `matches()` can be Used
    ---------------------------------------------------------
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

    The `matches()` selector function doesn't need to be used in isolation. Read the next section
    for information on how to compose it with other column selectors for more refined ways to select
    columns.

    Additional Flexibilty through Composition with Other Column Selectors
    ---------------------------------------------------------------------
    The `matches()` function can be composed with other column selectors to create fine-grained
    column selections. For example, to select columns that have the text starting with five digits
    and end with `"_id"`, you can use the `matches()` and `ends_with()` functions together. The only
    condition is that the expressions are wrapped in the `col()` function, like this:

    ```python
    col(matches(r"^\d{5}") & ends_with("_id"))
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
    pb.config(report_incl_header=False, report_incl_footer=False, preview_incl_header=False)
    ```

    Suppose we have a table with columns `name`, `id_old`, `new_identifier`, and `pay_2021` and we'd
    like to validate that text values in columns having `"id"` or `"identifier"` in the name have a
    specific syntax. We can use the `matches()` column selector function to specify the columns that
    match the pattern.

    ```{python}
    import pointblank as pb
    import polars as pl

    tbl = pl.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie"],
            "id_old": ["ID0021", "ID0032", "ID0043"],
            "new_identifier": ["ID9054", "ID9065", "ID9076"],
            "pay_2021": [16.32, 16.25, 15.75],
        }
    )

    validation = (
        pb.Validate(data=tbl)
        .col_vals_regex(columns=pb.matches("id|identifier"), pattern=r"ID\d{4}")
        .interrogate()
    )

    validation
    ```

    From the results of the validation table we get two validation steps, one for `id_old` and one
    for `new_identifier`. The values in both columns all match the pattern `"ID\d{4}"`.

    We can also use the `matches()` function in combination with other column selectors (within
    `col()`) to create more complex column selection criteria (i.e., to select columns that satisfy
    multiple conditions). For example, to select columns that contain `"pay"` and match the text
    `"2023"` or `"2024"`, we can use the `&` operator to combine column selectors.

    ```{python}
    tbl = pl.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie"],
            "2022_hours": [160, 180, 160],
            "2023_hours": [182, 168, 175],
            "2024_hours": [200, 165, 190],
            "2022_pay_total": [18.62, 16.95, 18.25],
            "2023_pay_total": [19.29, 17.75, 18.35],
            "2024_pay_total": [20.73, 18.35, 20.10],
        }
    )

    validation = (
        pb.Validate(data=tbl)
        .col_vals_gt(
            columns=pb.col(pb.contains("pay") & pb.matches("2023|2024")),
            value=10
        )
        .interrogate()
    )

    validation
    ```

    From the results of the validation table we get two validation steps, one for `2023_pay_total`
    and one for `2024_pay_total`.
    """
    return Matches(pattern=pattern, case_sensitive=case_sensitive)


def everything() -> Everything:
    """
    Select all columns.

    Many validation methods have a `columns=` argument that can be used to specify the columns for
    validation (e.g., `col_vals_gt()`, `col_vals_regex()`, etc.). The `everything()` selector
    function can be used to select every column in the table. If you have a table with six columns
    and they're all suitable for a specific type of validation, you can use `columns=everything())`
    and all six columns will be selected for validation.

    Returns
    -------
    Everything
        An `Everything` object, which can be used to select all columns.

    Relevant Validation Methods where `everything()` can be Used
    ------------------------------------------------------------
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

    The `everything()` selector function doesn't need to be used in isolation. Read the next section
    for information on how to compose it with other column selectors for more refined ways to select
    columns.

    Additional Flexibilty through Composition with Other Column Selectors
    ---------------------------------------------------------------------
    The `everything()` function can be composed with other column selectors to create fine-grained
    column selections. For example, to select all column names except those having starting with
    "id_", you can use the `everything()` and `starts_with()` functions together. The only condition
    is that the expressions are wrapped in the `col()` function, like this:

    ```python
    col(everything() - starts_with("id_"))
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
    pb.config(report_incl_header=False, report_incl_footer=False, preview_incl_header=False)
    ```

    Suppose we have a table with several numeric columns and we'd like to validate that all these
    columns have less than `1000`. We can use the `everything()` column selector function to select
    all columns for validation.

    ```{python}
    import pointblank as pb
    import polars as pl

    tbl = pl.DataFrame(
        {
            "2023_hours": [182, 168, 175],
            "2024_hours": [200, 165, 190],
            "2023_pay_total": [19.29, 17.75, 18.35],
            "2024_pay_total": [20.73, 18.35, 20.10],
        }
    )

    validation = (
        pb.Validate(data=tbl)
        .col_vals_lt(columns=pb.everything(), value=1000)
        .interrogate()
    )

    validation
    ```

    From the results of the validation table we get four validation steps, one each column in the
    table. The values in every column were all lower than `1000`.

    We can also use the `everything()` function in combination with other column selectors (within
    `col()`) to create more complex column selection criteria (i.e., to select columns that satisfy
    multiple conditions). For example, to select every column except those that begin with `"2023"`
    we can use the `-` operator to combine column selectors.

    ```{python}
    tbl = pl.DataFrame(
        {
            "2023_hours": [182, 168, 175],
            "2024_hours": [200, 165, 190],
            "2023_pay_total": [19.29, 17.75, 18.35],
            "2024_pay_total": [20.73, 18.35, 20.10],
        }
    )

    validation = (
        pb.Validate(data=tbl)
        .col_vals_lt(columns=pb.col(pb.everything() - pb.starts_with("2023")), value=1000)
        .interrogate()
    )

    validation
    ```

    From the results of the validation table we get two validation steps, one for `2024_hours` and
    one for `2024_pay_total`.
    """
    return Everything()


def first_n(n: int, offset: int = 0) -> FirstN:
    """
    Select the first `n` columns in the column list.

    Many validation methods have a `columns=` argument that can be used to specify the columns for
    validation (e.g., `col_vals_gt()`, `col_vals_regex()`, etc.). The `first_n()` selector function
    can be used to select *n* columns positioned at the start of the column list. So if the set of
    table columns consists of

    `[rev_01, rev_02, profit_01, profit_02, age]`

    and you want to validate the first two columns, you can use `columns=first_n(2)`. This will
    select the `rev_01` and `rev_02` columns and a validation step will be created for each.

    The `offset=` parameter can be used to skip a certain number of columns from the start of the
    column list. So if you want to select the third and fourth columns, you can use
    `columns=first_n(2, offset=2)`.

    Parameters
    ----------
    n
        The number of columns to select from the start of the column list. Should be a positive
        integer value. If `n` is greater than the number of columns in the table, all columns will
        be selected.
    offset
        The offset from the start of the column list. The default is `0`. If `offset` is greater
        than the number of columns in the table, no columns will be selected.

    Returns
    -------
    FirstN
        A `FirstN` object, which can be used to select the first `n` columns.

    Relevant Validation Methods where `first_n()` can be Used
    ---------------------------------------------------------
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

    The `first_n()` selector function doesn't need to be used in isolation. Read the next section
    for information on how to compose it with other column selectors for more refined ways to select
    columns.

    Additional Flexibilty through Composition with Other Column Selectors
    ---------------------------------------------------------------------
    The `first_n()` function can be composed with other column selectors to create fine-grained
    column selections. For example, to select all column names starting with "rev" along with the
    first two columns, you can use the `first_n()` and `starts_with()` functions together. The only
    condition is that the expressions are wrapped in the `col()` function, like this:

    ```python
    col(first_n(2) | starts_with("rev"))
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
    pb.config(report_incl_header=False, report_incl_footer=False, preview_incl_header=False)
    ```

    Suppose we have a table with columns `paid_2021`, `paid_2022`, `paid_2023`, `paid_2024`, and
    `name` and we'd like to validate that the values in the first four columns are greater than
    `10`. We can use the `first_n()` column selector function to specify that the first four columns
    in the table are the columns to validate.

    ```{python}
    import pointblank as pb
    import polars as pl

    tbl = pl.DataFrame(
        {
            "paid_2021": [17.94, 16.55, 17.85],
            "paid_2022": [18.62, 16.95, 18.25],
            "paid_2023": [19.29, 17.75, 18.35],
            "paid_2024": [20.73, 18.35, 20.10],
            "name": ["Alice", "Bob", "Charlie"],
        }
    )

    validation = (
        pb.Validate(data=tbl)
        .col_vals_gt(columns=pb.first_n(4), value=10)
        .interrogate()
    )

    validation
    ```

    From the results of the validation table we get four validation steps. The values in all those
    columns were all greater than `10`.

    We can also use the `first_n()` function in combination with other column selectors (within
    `col()`) to create more complex column selection criteria (i.e., to select columns that satisfy
    multiple conditions). For example, to select the first four columns but also omit those columns
    that end with `"2023"`, we can use the `-` operator to combine column selectors.

    ```{python}
    tbl = pl.DataFrame(
        {
            "paid_2021": [17.94, 16.55, 17.85],
            "paid_2022": [18.62, 16.95, 18.25],
            "paid_2023": [19.29, 17.75, 18.35],
            "paid_2024": [20.73, 18.35, 20.10],
            "name": ["Alice", "Bob", "Charlie"],
        }
    )

    validation = (
        pb.Validate(data=tbl)
        .col_vals_gt(columns=pb.col(pb.first_n(4) - pb.ends_with("2023")), value=10)
        .interrogate()
    )

    validation
    ```

    From the results of the validation table we get three validation steps, one for `paid_2021`,
    `paid_2022`, and `paid_2024`.
    """
    return FirstN(n=n, offset=offset)


def last_n(n: int, offset: int = 0) -> LastN:
    """
    Select the last `n` columns in the column list.

    Many validation methods have a `columns=` argument that can be used to specify the columns for
    validation (e.g., `col_vals_gt()`, `col_vals_regex()`, etc.). The `last_n()` selector function
    can be used to select *n* columns positioned at the end of the column list. So if the set of
    table columns consists of

    `[age, rev_01, rev_02, profit_01, profit_02]`

    and you want to validate the last two columns, you can use `columns=last_n(2)`. This will select
    the `profit_01` and `profit_02` columns and a validation step will be created for each.

    The `offset=` parameter can be used to skip a certain number of columns from the end of the
    column list. So if you want to select the third and fourth columns from the end, you can use
    `columns=last_n(2, offset=2)`.

    Parameters
    ----------
    n
        The number of columns to select from the end of the column list. Should be a positive
        integer value. If `n` is greater than the number of columns in the table, all columns will
        be selected.
    offset
        The offset from the end of the column list. The default is `0`. If `offset` is greater than
        the number of columns in the table, no columns will be selected.

    Returns
    -------
    LastN
        A `LastN` object, which can be used to select the last `n` columns.

    Relevant Validation Methods where `last_n()` can be Used
    --------------------------------------------------------
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

    The `last_n()` selector function doesn't need to be used in isolation. Read the next section for
    information on how to compose it with other column selectors for more refined ways to select
    columns.

    Additional Flexibilty through Composition with Other Column Selectors
    ---------------------------------------------------------------------
    The `last_n()` function can be composed with other column selectors to create fine-grained
    column selections. For example, to select all column names starting with "rev" along with the
    last two columns, you can use the `last_n()` and `starts_with()` functions together. The only
    condition is that the expressions are wrapped in the `col()` function, like this:

    ```python
    col(last_n(2) | starts_with("rev"))
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
    pb.config(report_incl_header=False, report_incl_footer=False, preview_incl_header=False)
    ```

    Suppose we have a table with columns `name`, `paid_2021`, `paid_2022`, `paid_2023`, and
    `paid_2024` and we'd like to validate that the values in the last four columns are greater than
    `10`. We can use the `last_n()` column selector function to specify that the last four columns
    in the table are the columns to validate.

    ```{python}
    import pointblank as pb
    import polars as pl

    tbl = pl.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie"],
            "paid_2021": [17.94, 16.55, 17.85],
            "paid_2022": [18.62, 16.95, 18.25],
            "paid_2023": [19.29, 17.75, 18.35],
            "paid_2024": [20.73, 18.35, 20.10],
        }
    )

    validation = (
        pb.Validate(data=tbl)
        .col_vals_gt(columns=pb.last_n(4), value=10)
        .interrogate()
    )

    validation
    ```

    From the results of the validation table we get four validation steps. The values in all those
    columns were all greater than `10`.

    We can also use the `last_n()` function in combination with other column selectors (within
    `col()`) to create more complex column selection criteria (i.e., to select columns that satisfy
    multiple conditions). For example, to select the last four columns but also omit those columns
    that end with `"2023"`, we can use the `-` operator to combine column selectors.

    ```{python}
    tbl = pl.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie"],
            "paid_2021": [17.94, 16.55, 17.85],
            "paid_2022": [18.62, 16.95, 18.25],
            "paid_2023": [19.29, 17.75, 18.35],
            "paid_2024": [20.73, 18.35, 20.10],
        }
    )

    validation = (
        pb.Validate(data=tbl)
        .col_vals_gt(columns=pb.col(pb.last_n(4) - pb.ends_with("2023")), value=10)
        .interrogate()
    )

    validation
    ```

    From the results of the validation table we get three validation steps, one for `paid_2021`,
    `paid_2022`, and `paid_2024`.
    """
    return LastN(n=n, offset=offset)
