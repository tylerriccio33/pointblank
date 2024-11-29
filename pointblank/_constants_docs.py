ARG_DOCSTRINGS = {
    "column": """column
        The column to validate.""",
    "columns": """columns
        A single column or a list of columns to validate. If multiple columns are supplied, there
        will be a separate validation step generated for each column.""",
    "value": """value
        The value to compare against.""",
    "left": """left
        The lower bound of the range.""",
    "right": """right
        The upper bound of the range.""",
    "set": """set
        A list of values to compare against.""",
    "pattern": """pattern
        A regular expression pattern to compare against.""",
    "inclusive": """inclusive
        A tuple of two boolean values indicating whether the comparison should be inclusive. The
        position of the boolean values correspond to the `left=` and `right=` values, respectively.
        By default, both values are `True`.""",
    "na_pass": """na_pass
        Should any encountered None, NA, or Null values be considered as passing test units? By
        default, this is `False`. Set to `True` to pass test units with missing values.""",
    "pre": """pre
        A pre-processing function or lambda to apply to the data table for the validation step.""",
    "thresholds": """thresholds
        Failure threshold levels so that the validation step can react accordingly when exceeding
        the set levels for different states (`warn`, `stop`, and `notify`). This can be created
        simply as an integer or float denoting the absolute number or fraction of failing test units
        for the 'warn' level. Otherwise, you can use a tuple of 1-3 values, a dictionary of 1-3
        entries, or a Thresholds object.""",
    "active": """active
        A boolean value indicating whether the validation step should be active. Using `False` will
        make the validation step inactive (still reporting its presence and keeping indexes for the
        steps unchanged).""",
    "data": """data
        A data table.""",
    "threshold": """threshold
        The maximum number of failing test units to allow.""",
}
