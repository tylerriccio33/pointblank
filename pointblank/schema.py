from __future__ import annotations

import copy

from dataclasses import dataclass

import narwhals as nw

from pointblank._utils import _get_tbl_type, _is_lib_present
from pointblank._constants import IBIS_BACKENDS

__all__ = ["Schema"]


@dataclass
class Schema:
    """Definition of a schema object.

    The schema object defines the structure of a table. Once it is defined, the object can be used
    in a validation workflow, using `Validate` and its methods, to ensure that the structure of a
    table matches the expected schema. The validation method that works with the schema object is
    called `col_schema_match()`.

    A schema for a table can be constructed with `Schema` in a number of ways:

    1. providing a list of column names to `columns=` (to check only the column names)
    2. using a list of two-element tuples in `columns=` (to check both column names and dtypes)
    3. providing a dictionary to `columns=`, where the keys are column names and the values are
    dtypes
    4. providing individual column arguments in the form of keyword arguments (in the form of
    `column=dtype`)

    The schema object can also be constructed by providing a DataFrame or Ibis table object (using
    the `tbl=` parameter) and the schema will be collected from either type of object. The schema
    object can be printed to display the column names and dtypes. Note that if `tbl=` is provided
    then there shouldn't be any other inputs provided through either `columns=` or `**kwargs`.

    Parameters
    ----------
    columns
        A list of strings (representing column names), a list of tuples (for column names and column
        dtypes), or a dictionary containing column and dtype information. If any of these inputs are
        provided here, it will take precedence over any column arguments provided via `**kwargs`.
    tbl
        A DataFrame (Polars or Pandas) or an Ibis table object from which the schema will be
        collected.
    **kwargs
        Individual column arguments that are in the form of `[column]=[dtype]`. These will be
        ignored if the `columns=` parameter is not `None`.

    Returns
    -------
    Schema
        A schema object.

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_header=False, report_incl_footer=False)
    ```
    A schema can be constructed via the `Schema` class in multiple ways. Let's use the following
    Polars DataFrame as a basis for constructing a schema:

    ```{python}
    import polars as pl

    df = pl.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
        "height": [5.6, 6.0, 5.8]
    })
    ```

    You could provide `Schema(columns=)` a list of tuples containing column names and data types:

    ```{python}
    schema = pb.Schema(columns=[("name", "String"), ("age", "Int64"), ("height", "Float64")])
    ```

    Alternatively, a dictionary containing column names and dtypes also works:

    ```{python}
    schema = pb.Schema(columns={"name": "String", "age": "Int64", "height": "Float64"})
    ```

    Another input method involves using individual column arguments in the form of keyword
    arguments:

    ```{python}
    schema = pb.Schema(name="String", age="Int64", height="Float64")
    ```

    Finally, could also provide a DataFrame (Polars and Pandas) or an Ibis table object to `tbl=`
    and the schema will be collected:

    ```python
    schema = pb.Schema(tbl=df)
    ```

    Whichever method you choose, you can verify the schema inputs by printing the `schema` object:

    ```{python}
    print(schema)
    ```

    The `Schema` object can be used to validate the structure of a table against the schema. The
    relevant `Validate` method for this is `col_schema_match()`. In a validation workflow, you'll
    have a target table (defined at the beginning of the workflow) and you might want to ensure that
    your expectations of the table structure are met. The `col_schema_match()` method works with a
    `Schema` object to validate the structure of the table. Here's an example of how you could use
    the `col_schema_match()` method in a validation workflow:

    ```{python}
    import pointblank as pb

    # Define the schema
    schema = pb.Schema(name="String", age="Int64", height="Float64")

    # Define a validation that checks the schema against the table (`df`)
    validation = (
        pb.Validate(data=df)
        .col_schema_match(schema=schema)
        .interrogate()
    )

    # Display the validation results
    validation
    ```

    The `col_schema_match()` validation method will validate the structure of the table against the
    schema during interrogation. If the structure of the table does not match the schema, the single
    test unit will fail. In this case, the defined schema matched the structure of the table, so the
    validation passed.

    We can also choose to check only the column names of the target table. This can be done by
    providing a simplified `Schema` object, which is given a list of column names:

    ```{python}
    schema = pb.Schema(columns=["name", "age", "height"])

    validation = (
        pb.Validate(data=df)
        .col_schema_match(schema=schema)
        .interrogate()
    )

    validation
    ```

    In this case, the schema only checks the column names of the table against the schema during
    interrogation. If the column names of the table do not match the schema, the single test unit
    will fail. In this case, the defined schema matched the column names of the table, so the
    validation passed.
    """

    columns: str | list[str] | list[tuple[str, str]] | list[tuple[str]] | dict[str, str] | None = (
        None
    )
    tbl: any | None = None

    def __init__(
        self,
        columns: (
            str | list[str] | list[tuple[str, str]] | list[tuple[str]] | dict[str, str] | None
        ) = None,
        tbl: any | None = None,
        **kwargs,
    ):
        if tbl is None and columns is None and not kwargs:
            raise ValueError(
                "Either `columns`, `tbl`, or individual column arguments must be provided."
            )

        if tbl is not None and (columns is not None or kwargs):
            raise ValueError(
                "Only one of `columns`, `tbl`, or individual column arguments can be provided."
            )

        self.tbl = tbl
        if columns is not None or kwargs:
            self.columns = _process_columns(columns=columns, **kwargs)
        else:
            self.columns = None

        self.__post_init__()

    def __post_init__(self):
        if self.columns is not None:
            self._validate_schema_inputs()
        if self.tbl is not None:
            self._collect_schema_from_table()

        # Get the table type and store as an attribute (only if a table is provided)
        if self.tbl is not None:
            self.tbl_type = _get_tbl_type(self.tbl)

    def _validate_schema_inputs(self):
        if not isinstance(self.columns, list):
            raise ValueError("`columns` must be a list.")

        if not all(isinstance(col, tuple) for col in self.columns):
            raise ValueError("All elements of `columns` must be tuples.")

    def _collect_schema_from_table(self):

        # Determine if this table can be converted to a Narwhals DataFrame
        table_type = _get_tbl_type(self.tbl)

        if table_type == "pandas" or table_type == "polars":

            tbl_nw = nw.from_native(self.tbl)

            schema_dict = dict(tbl_nw.schema.items())

            schema_dict = {k: str(v) for k, v in schema_dict.items()}

            self.columns = list(schema_dict.items())

        elif table_type in IBIS_BACKENDS:

            schema_dict = dict(self.tbl.schema().items())

            schema_dict = {k: str(v) for k, v in schema_dict.items()}

            self.columns = list(schema_dict.items())

        else:
            raise ValueError(  # pragma: no cover
                "The provided table object cannot be converted to a Narwhals DataFrame."
            )

    def _compare_schema_columns_complete_in_order(
        self,
        other: Schema,
        case_sensitive_colnames: bool,
        case_sensitive_dtypes: bool,
        full_match_dtypes: bool,
    ) -> bool:
        """
        Compare the columns of the schema with another schema. Ensure that all column names are the
        same and that they are in the same order. This method is performed when:

        - `complete`: True
        - `in_order`: True

        Parameters
        ----------
        other
            The other schema to compare against.

        Returns
        -------
        bool
            True if the columns are the same, False otherwise.
        """

        if not case_sensitive_colnames:
            this_column_list = [col.lower() for col in self.get_column_list()]
            other_column_list = [col.lower() for col in other.get_column_list()]
        else:
            this_column_list = self.get_column_list()
            other_column_list = other.get_column_list()

        # Check if the column lists are the same length, this is a quick check to determine
        # if the schemas are different
        if len(this_column_list) != len(other_column_list):
            return False

        # Check that the column names are the same in both schemas and in the same order
        if this_column_list != other_column_list:
            return False

        # Iteratively move through the columns in `this_column_list` and determine if the dtype of
        # the column in `this_column_list` is the same as the dtype in `other_column_list`
        for col in this_column_list:

            # Skip dtype checks if the tuple value of `col` only contains the column name
            # (i.e., is a single element tuple)
            if len(self.columns[this_column_list.index(col)]) == 1:
                continue

            this_dtype = self.columns[this_column_list.index(col)][1]
            other_dtype = other.columns[other_column_list.index(col)][1]

            # There may be multiple dtypes for a column, so we need to promote scalar dtypes to
            # lists before iterating through them
            if isinstance(this_dtype, str):
                this_dtype = [this_dtype]

            dtype_matches = []

            for i in range(len(this_dtype)):

                if not case_sensitive_dtypes:
                    this_dtype[i] = this_dtype[i].lower()
                    other_dtype = other_dtype.lower()

                if full_match_dtypes and this_dtype[i] == other_dtype:
                    dtype_matches.append(True)

                if not full_match_dtypes and this_dtype[i] in other_dtype:
                    dtype_matches.append(True)

            # If there are no matches for any of the dtypes provided, return False
            if not any(dtype_matches):
                return False

        return True

    def _compare_schema_columns_complete_any_order(
        self,
        other: Schema,
        case_sensitive_colnames: bool,
        case_sensitive_dtypes: bool,
        full_match_dtypes: bool,
    ) -> bool:
        """
        Compare the columns of the schema with another schema to ensure that all column names are
        available in both schemas. Column order is not considered here. This method is performed
        when:
        - `complete`: True
        - `in_order`: False

        Parameters
        ----------
        other
            The other schema to compare against.

        Returns
        -------
        bool
            True if the columns are the same, False otherwise.
        """

        if not case_sensitive_colnames:
            this_column_list = [col.lower() for col in self.get_column_list()]
            other_column_list = [col.lower() for col in other.get_column_list()]
        else:
            this_column_list = self.get_column_list()
            other_column_list = other.get_column_list()

        # Check if the column names lists are the same length, this is a quick check to determine
        # if the schemas are different
        if len(this_column_list) != len(other_column_list):
            return False

        # Iteratively move through the columns in `this_column_list` and determine if:
        # - the column is present in `other_column_list` at the same index
        # - the dtype of the column in `this_column_list` is the same as the dtype in
        #   `other_column_list`
        for col in this_column_list:
            if col not in other_column_list:
                return False
            else:

                # Skip dtype checks if the tuple value of `col` only contains the column name
                # (i.e., is a single element tuple)
                if len(self.columns[this_column_list.index(col)]) == 1:
                    continue

                this_dtype = self.columns[this_column_list.index(col)][1]

                # The corresponding column in the other schema is present but not necessarily at
                # the same index; get the index of the column in the other schema
                other_col_index = other_column_list.index(col)

                # Get the dtype of the column in the other schema
                other_dtype = other.columns[other_col_index][1]

                # There may be multiple dtypes for a column, so we need to promote scalar dtypes to
                # lists before iterating through them
                if isinstance(this_dtype, str):
                    this_dtype = [this_dtype]

                dtype_matches = []

                for i in range(len(this_dtype)):

                    if not case_sensitive_dtypes:
                        this_dtype[i] = this_dtype[i].lower()
                        other_dtype = other_dtype.lower()

                    if full_match_dtypes and this_dtype[i] == other_dtype:
                        dtype_matches.append(True)

                    if not full_match_dtypes and this_dtype[i] in other_dtype:
                        dtype_matches.append(True)

                # If there are no matches for any of the dtypes provided, return False
                if not any(dtype_matches):
                    return False

        return True

    def _compare_schema_columns_subset_in_order(
        self,
        other: Schema,
        case_sensitive_colnames: bool,
        case_sensitive_dtypes: bool,
        full_match_dtypes: bool,
    ) -> bool:
        """
        Compare the columns of the schema with another schema. Ensure that all column names in the
        schema are available in the other schema and that they are in the same order. This method is
        performed when:

        - `complete`: False
        - `in_order`: True

        Parameters
        ----------
        other
            The other schema to compare against.

        Returns
        -------
        bool
            True if the columns are the same, False otherwise.
        """

        if not case_sensitive_colnames:
            this_column_list = [col.lower() for col in self.get_column_list()]
            other_column_list = [col.lower() for col in other.get_column_list()]
        else:
            this_column_list = self.get_column_list()
            other_column_list = other.get_column_list()

        # Iteratively move through the columns in `this_column_list` and determine if:
        # - the column is present in `other_column_list`
        # - the dtype of the column in `this_column_list` is the same as the dtype in
        #   `other_column_list`
        for col in this_column_list:
            if col not in other_column_list:
                return False
            else:

                # Skip dtype checks if the tuple value of `col` only contains the column name
                # (i.e., is a single element tuple)
                if len(self.columns[this_column_list.index(col)]) == 1:
                    continue

                this_dtype = self.columns[this_column_list.index(col)][1]

                # The corresponding column in the other schema is present but not necessarily at
                # the same index; get the index of the column in the other schema
                other_col_index = other_column_list.index(col)

                # Get the dtype of the column in the other schema
                other_dtype = other.columns[other_col_index][1]

                # There may be multiple dtypes for a column, so we need to promote scalar dtypes to
                # lists before iterating through them
                if isinstance(this_dtype, str):
                    this_dtype = [this_dtype]

                dtype_matches = []

                for i in range(len(this_dtype)):

                    if not case_sensitive_dtypes:
                        this_dtype[i] = this_dtype[i].lower()
                        other_dtype = other_dtype.lower()

                    if full_match_dtypes and this_dtype[i] == other_dtype:
                        dtype_matches.append(True)

                    if not full_match_dtypes and this_dtype[i] in other_dtype:
                        dtype_matches.append(True)

                # If there are no matches for any of the dtypes provided, return False
                if not any(dtype_matches):
                    return False

        # With the subset of columns in `this_column_list`, ensure that the columns are in the same
        # order in `other_column_list`
        other_column_list_subset = [col for col in other_column_list if col in this_column_list]

        if this_column_list != other_column_list_subset:
            return False

        return True

    def _compare_schema_columns_subset_any_order(
        self,
        other: Schema,
        case_sensitive_colnames: bool,
        case_sensitive_dtypes: bool,
        full_match_dtypes: bool,
    ) -> bool:
        """
        Compare the columns of the schema with another schema to ensure that all column names are
        at least in the other schema. Column order is not considered here. This method is performed
        when:

        - `complete`: False
        - `in_order`: False

        Parameters
        ----------
        other
            The other schema to compare against.

        Returns
        -------
        bool
            True if the columns are the same, False otherwise.
        """

        if not case_sensitive_colnames:
            this_column_list = [col.lower() for col in self.get_column_list()]
            other_column_list = [col.lower() for col in other.get_column_list()]
        else:
            this_column_list = self.get_column_list()
            other_column_list = other.get_column_list()

        # Iteratively move through the columns in `this_column_list` and determine if:
        # - the column is present in `other_column_list`
        # - the dtype of the column in `this_column_list` is the same as the dtype in
        #   `other_column_list`

        for col in this_column_list:
            if col not in other_column_list:
                return False
            else:

                # Skip dtype checks if the tuple value of `col` only contains the column name
                # (i.e., is a single element tuple)
                if len(self.columns[this_column_list.index(col)]) == 1:
                    continue

                this_dtype = self.columns[this_column_list.index(col)][1]
                other_dtype = other.columns[other_column_list.index(col)][1]

                # There may be multiple dtypes for a column, so we need to promote scalar dtypes to
                # lists before iterating through them
                if isinstance(this_dtype, str):
                    this_dtype = [this_dtype]

                dtype_matches = []

                for i in range(len(this_dtype)):
                    if not case_sensitive_dtypes:
                        this_dtype[i] = this_dtype[i].lower()
                        other_dtype = other_dtype.lower()

                    if full_match_dtypes and this_dtype[i] == other_dtype:
                        dtype_matches.append(True)

                    if not full_match_dtypes and this_dtype[i] in other_dtype:
                        dtype_matches.append(True)

                # If there are no matches for any of the dtypes provided, return False
                if not any(dtype_matches):
                    return False

        return True

    def get_tbl_type(self) -> str:
        """
        Get the type of table from which the schema was collected.

        Returns
        -------
        str
            The type of table.
        """
        return self.tbl_type

    def get_column_list(self) -> list[str]:
        """
        Return a list of column names from the schema.

        Returns
        -------
        list[str]
            A list of column names.
        """
        return [col[0] for col in self.columns]

    def get_dtype_list(self) -> list[str]:
        """
        Return a list of data types from the schema.

        Returns
        -------
        list[str]
            A list of data types.
        """
        return [col[1] for col in self.columns]

    def get_schema_coerced(self, to: str | None = None) -> dict[str, str]:

        # If a table isn't provided, we cannot use this method
        if self.tbl is None:
            raise ValueError(
                "A table object must be provided to use the `get_schema_coerced()` method."
            )

        # Pandas coercions
        if self.tbl_type == "pandas":

            if to == "polars":

                # Check if Polars is available
                if not _is_lib_present("polars"):
                    raise ImportError(
                        "Performing Pandas -> Polars schema conversion requires the Polars library."
                    )

                import polars as pl

                # Convert the DataFrame to a Polars DataFrame
                new_schema = copy.deepcopy(Schema(tbl=pl.from_pandas(self.tbl)))
                return new_schema

        if self.tbl_type == "polars":

            if to == "pandas":

                # Check if Pandas is available
                if not _is_lib_present("pandas"):
                    raise ImportError(
                        "Performing Polars -> Pandas schema conversion requires the Pandas library."
                    )

                # Check if Arrow is available
                if not _is_lib_present("pyarrow"):
                    raise ImportError(
                        "Performing Polars -> Pandas schema conversion requires the pyarrow library."
                    )

                # Convert the DataFrame to a Pandas DataFrame
                new_schema = copy.deepcopy(Schema(tbl=(self.tbl.to_pandas())))
                return new_schema

    def __str__(self):
        return "Pointblank Schema\n" + "\n".join([f"  {col[0]}: {col[1]}" for col in self.columns])

    def __repr__(self):
        return f"Schema(columns={self.columns})"


def _process_columns(
    *, columns: str | list[str] | list[tuple[str, str]] | dict[str, str] | None = None, **kwargs
) -> list[tuple[str, str]]:
    """
    Process column information provided as individual arguments or as a list of
    tuples/dictionary.

    Parameters
    ----------
    columns
        A list of tuples or a dictionary containing column information.
    **kwargs
        Individual column arguments.

    Returns
    -------
    list[tuple[str, str]]
        A list of tuples containing column information.
    """
    if columns is not None:

        if isinstance(columns, list):

            if all(isinstance(col, str) for col in columns):
                return [(col,) for col in columns]
            else:
                return columns

        if isinstance(columns, str):
            return [(columns,)]

        if isinstance(columns, dict):
            return list(columns.items())
        return columns

    return list(kwargs.items())
