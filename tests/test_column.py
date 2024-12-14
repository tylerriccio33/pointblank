from pointblank.column import Column, col


def test_column_class():
    col1 = Column(name="col1")
    assert col1.name == "col1"
    assert str(col1) == "col1"


def test_col_function():
    col1 = col("col1")
    assert col1.name == "col1"
    assert str(col1) == "col1"
