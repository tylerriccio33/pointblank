ARG_DOCSTRINGS = {
    "column": """column : str
        The column to validate.""",
    "value": """value : int | float
        The value to compare against.""",
    "left": """left : int | float
        The lower bound of the range.""",
    "right": """right : int | float
        The upper bound of the range.""",
    "set": """set : list[int | float]
        A list of values to compare against.""",
    "inclusive": """inclusive : tuple[bool, bool], optional
        A tuple of two boolean values indicating whether the comparison should be inclusive. The
        position of the boolean values correspond to the `left=` and `right=` values, respectively.
        By default, both values are `True`.""",
    "na_pass": """na_pass : bool
        Should any encountered None, NA, or Null values be considered as passing test units? By
        default, this is `False`. Set to `True` to pass test units with missing values.""",
    "pre": """pre : Callable | None
        A pre-processing function or lambda to apply to the data table for the validation step.""",
    "thresholds": """thresholds : int | float | tuple | dict| Thresholds, optional
        Failure threshold levels so that the validation step can react accordingly when exceeding
        the set levels for different states (`warn`, `stop`, and `notify`). This can be created
        simply as an integer or float denoting the absolute number or fraction of failing test units
        for the 'warn' level. Otherwise, you can use a tuple of 1-3 values, a dictionary of 1-3
        entries, or a Thresholds object.""",
    "active": """active : bool, optional
        A boolean value indicating whether the validation step should be active. Using `False` will
        make the validation step inactive (still reporting its presence and keeping indexes for the
        steps unchanged).""",
    "df": """df : df : FrameT
        a DataFrame.""",
    "threshold": """threshold : int
        The maximum number of failing test units to allow.""",
}
