import pytest
from pointblank import yaml_interrogate, validate_yaml, yaml_to_python
from pointblank.yaml import load_yaml_config, YAMLValidationError, YAMLValidator


def test_yaml_interrogate_basic_workflow():
    yaml_content = """
    tbl: small_table
    tbl_name: small_table
    label: A simple test
    steps:
    - rows_distinct
    - col_exists:
        columns: [date, date_time]
    """

    result = yaml_interrogate(yaml_content)
    assert result is not None
    # rows_distinct (1) + col_exists with 2 columns (2) = 3 total steps
    assert len(result.validation_info) == 3
    # Check that it's been interrogated (should have results)
    assert hasattr(result, "validation_info")


def test_yaml_interrogate_with_thresholds():
    yaml_content = """
    tbl: small_table
    tbl_name: small_table
    thresholds:
      warning: 0.1
      error: 0.25
      critical: 0.35
    steps:
    - rows_distinct
    """

    result = yaml_interrogate(yaml_content)
    assert result is not None
    assert len(result.validation_info) == 1


def test_yaml_interrogate_complex_example():
    yaml_content = """
    tbl: small_table
    tbl_name: small_table
    label: A simple example with the `small_table`.
    lang: en
    locale: en
    thresholds:
      warning: 0.1
      error: 0.25
      critical: 0.35
    steps:
    - col_exists:
        columns: [date, date_time]
    - col_vals_regex:
        columns: [b]
        pattern: '[0-9]-[a-z]{3}-[0-9]{3}'
    - rows_distinct
    - col_vals_gt:
        columns: [d]
        value: 100.0
    - col_vals_le:
        columns: [c]
        value: 5.0
    """

    result = yaml_interrogate(yaml_content)
    assert result is not None
    # col_exists with 2 columns (2) + col_vals_regex (1) + rows_distinct (1) + col_vals_gt (1) + col_vals_le (1) = 6
    assert len(result.validation_info) == 6


def test_yaml_column_specifications():
    # Test YAML list syntax
    yaml_content1 = """
    tbl: small_table
    steps:
    - col_exists:
        columns: [date, date_time]
    """
    result1 = yaml_interrogate(yaml_content1)
    assert result1 is not None
    assert len(result1.validation_info) == 2  # One step per column

    # Test single column as string
    yaml_content2 = """
    tbl: small_table
    steps:
    - col_exists:
        columns: date
    """
    result2 = yaml_interrogate(yaml_content2)
    assert result2 is not None
    assert len(result2.validation_info) == 1

    # Test single column as list
    yaml_content3 = """
    tbl: small_table
    steps:
    - col_exists:
        columns: [date]
    """
    result3 = yaml_interrogate(yaml_content3)
    assert result3 is not None
    assert len(result3.validation_info) == 1


def test_validation_methods_coverage():
    yaml_content = """
    tbl: small_table
    steps:
    - rows_distinct
    - rows_complete
    - col_vals_not_null:
        columns: [date]
    - col_vals_gt:
        columns: [d]
        value: 0
    - col_vals_ge:
        columns: [d]
        value: 0
    - col_vals_lt:
        columns: [c]
        value: 10
    - col_vals_le:
        columns: [c]
        value: 10
    """

    result = yaml_interrogate(yaml_content)
    assert result is not None
    assert len(result.validation_info) == 7


def test_load_yaml_config():
    yaml_content = """
    tbl: small_table
    steps:
    - rows_distinct
    """

    config = load_yaml_config(yaml_content)
    assert config["tbl"] == "small_table"
    assert len(config["steps"]) == 1


def test_validate_yaml():
    # Valid configuration
    valid_yaml = """
    tbl: small_table
    steps:
    - rows_distinct
    """
    validate_yaml(valid_yaml)  # Should not raise

    # Invalid configuration: missing tbl
    invalid_yaml1 = """
    steps:
    - rows_distinct
    """
    with pytest.raises(YAMLValidationError, match="YAML must contain 'tbl' field"):
        validate_yaml(invalid_yaml1)

    # Invalid configuration: missing steps
    invalid_yaml2 = """
    tbl: small_table
    """
    with pytest.raises(YAMLValidationError, match="YAML must contain 'steps' field"):
        validate_yaml(invalid_yaml2)

    # Invalid configuration: empty steps
    invalid_yaml3 = """
    tbl: small_table
    steps: []
    """
    with pytest.raises(YAMLValidationError, match="'steps' cannot be empty"):
        validate_yaml(invalid_yaml3)


def test_yaml_error_handling():
    # Invalid YAML syntax
    with pytest.raises(YAMLValidationError, match="Invalid YAML syntax"):
        yaml_interrogate("invalid: yaml: content: [")

    # Unknown validation method
    yaml_content = """
    tbl: small_table
    steps:
    - unknown_method
    """
    with pytest.raises(YAMLValidationError, match="Unknown validation method 'unknown_method'"):
        yaml_interrogate(yaml_content)

    # Invalid data source
    yaml_content = """
    tbl: nonexistent_dataset
    steps:
    - rows_distinct
    """
    with pytest.raises(YAMLValidationError, match="Failed to load data source"):
        yaml_interrogate(yaml_content)

    # Invalid threshold values (negative values are not allowed)
    yaml_content = """
    tbl: small_table
    thresholds:
      warning: -1
    steps:
    - rows_distinct
    """
    with pytest.raises(YAMLValidationError, match="Threshold 'warning' must be non-negative"):
        yaml_interrogate(yaml_content)


def test_step_parameter_validation():
    # Step with invalid parameter structure
    yaml_content = """
    tbl: small_table
    steps:
    - col_exists: not_a_dict
    """
    with pytest.raises(
        YAMLValidationError, match="Parameters for 'col_exists' must be a dictionary"
    ):
        yaml_interrogate(yaml_content)

    # Step with multiple methods (should fail)
    yaml_content = """
    tbl: small_table
    steps:
    - col_exists:
        columns: [date]
      rows_distinct: {}
    """
    with pytest.raises(
        YAMLValidationError,
        match="Step configuration must contain exactly one validation method",
    ):
        yaml_interrogate(yaml_content)


def test_yaml_column_parsing():
    validator = YAMLValidator()

    # Test various YAML list formats
    assert validator._parse_column_spec(["date", "date_time"]) == ["date", "date_time"]
    assert validator._parse_column_spec(["a", "b", "c"]) == ["a", "b", "c"]
    assert validator._parse_column_spec([]) == []

    # Test string formats
    assert validator._parse_column_spec("single_column") == ["single_column"]

    # Test other types
    assert validator._parse_column_spec(123) == ["123"]


def test_yaml_interrogate_rejects_dict():
    config = {"tbl": "small_table", "steps": ["rows_distinct"]}

    with pytest.raises(YAMLValidationError, match="Only YAML strings and file paths supported"):
        yaml_interrogate(config)


def test_yaml_file_integration(tmp_path):
    yaml_content = """
    tbl: small_table
    tbl_name: test_table
    label: File-based test
    steps:
    - rows_distinct
    - col_exists:
        columns: [date, date_time]
    """

    yaml_file = tmp_path / "test_validation.yaml"
    yaml_file.write_text(yaml_content)

    # Test loading config from file
    config = load_yaml_config(yaml_file)
    assert config["tbl"] == "small_table"
    assert config["tbl_name"] == "test_table"
    assert len(config["steps"]) == 2

    # Test executing workflow from file
    result = yaml_interrogate(yaml_file)
    assert result is not None
    assert (
        len(result.validation_info) == 3
    )  # rows_distinct (1) + col_exists for date (1) + col_exists for date_time (1)


def test_yaml_workflow_results_consistency():
    # This test ensures our YAML implementation produces the same results
    # as the equivalent programmatic validation

    yaml_content = """
    tbl: small_table
    steps:
    - rows_distinct
    """

    # YAML-based validation
    yaml_result = yaml_interrogate(yaml_content)

    # Equivalent programmatic validation
    from pointblank import Validate, load_dataset

    programmatic_result = Validate(load_dataset("small_table")).rows_distinct().interrogate()

    # Compare results (both should have same structure)
    assert len(yaml_result.validation_info) == len(programmatic_result.validation_info)
    assert (
        yaml_result.validation_info[0].assertion_type
        == programmatic_result.validation_info[0].assertion_type
    )


def test_comprehensive_yaml_validation():
    yaml_content = """
    tbl: small_table
    tbl_name: Comprehensive Test
    thresholds:
      warning: 1
      error: 2
      critical: 0.15
    steps:
    - col_vals_lt:
        columns: [c]
        value: 0
    - col_vals_eq:
        columns: [a]
        value: 3
    - col_vals_ne:
        columns: [c]
        value: 10
    - col_vals_le:
        columns: [a]
        value: 7
    - col_vals_ge:
        columns: [d]
        value: 500
        na_pass: true
    - col_vals_between:
        columns: [c]
        left: 0
        right: 5
        na_pass: true
    - col_vals_outside:
        columns: [a]
        left: 0
        right: 9
        inclusive: [false, true]
    - col_vals_eq:
        columns: [a]
        value: 1
    - col_vals_in_set:
        columns: [f]
        set: [lows, mids, highs]
    - col_vals_not_in_set:
        columns: [f]
        set: [low, mid, high]
    - col_vals_null:
        columns: [c]
    - col_vals_not_null:
        columns: [c]
    - col_vals_regex:
        columns: [f]
        pattern: '[0-9]-[a-z]{3}-[0-9]{3}'
    - col_exists:
        columns: [z]
    - rows_distinct
    - rows_distinct:
        columns_subset: [a, b, c]
    - col_count_match:
        count: 14
    - row_count_match:
        count: 20
    """

    try:
        result = yaml_interrogate(yaml_content)
        assert result is not None
        # This should create 18 validation steps
        assert len(result.validation_info) == 18
        # The validation should execute without errors
        assert hasattr(result, "validation_info")
        # Verify that the highest severity level is 'critical'
        assert result._get_highest_severity_level() == "critical"

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise


def test_yaml_to_python_comprehensive():
    yaml_content = """
    tbl: small_table
    tbl_name: Test Table
    thresholds:
      warning: 0.1
      error: 0.25
    steps:
    - col_vals_eq:
        columns: [a]
        value: 3
    - col_vals_outside:
        columns: [a]
        left: 0
        right: 9
        inclusive: [false, true]
    - col_vals_in_set:
        columns: [f]
        set: [low, mid, high]
    - rows_distinct:
        columns_subset: [a, b, c]
    - col_count_match:
        count: 6
    """

    try:
        python_code = yaml_to_python(yaml_content)

        # Check that the generated code contains expected elements
        assert "import pointblank as pb" in python_code
        assert "pb.Validate(" in python_code
        assert 'data=pb.load_dataset("small_table")' in python_code
        assert 'tbl_name="Test Table"' in python_code
        assert "pb.Thresholds(warning=0.1, error=0.25)" in python_code
        assert '.col_vals_eq(columns="a", value=3)' in python_code
        assert (
            '.col_vals_outside(columns="a", left=0, right=9, inclusive=(False, True))'
            in python_code
        )
        assert '.col_vals_in_set(columns="f", set=["low", "mid", "high"])' in python_code
        assert '.rows_distinct(columns_subset=["a", "b", "c"])' in python_code
        assert ".col_count_match(count=6)" in python_code
        assert ".interrogate()" in python_code

        # Check that it starts and ends with the right markers
        assert python_code.startswith("```python\n")
        assert python_code.endswith("\n```")

    except Exception as e:
        raise


def test_yaml_briefs():
    yaml_content = """
    tbl: small_table
    tbl_name: Brief Test
    brief: "**Global Brief**: {auto}"
    lang: en
    steps:
    - col_vals_eq:
        columns: [a]
        value: 3
    - col_vals_lt:
        columns: [c]
        value: 5
        brief: false
    - col_vals_gt:
        columns: [d]
        value: 100
        brief: true
    - col_vals_le:
        columns: [a]
        value: 7
        brief: "This is a custom local brief for the assertion"
    - col_vals_ge:
        columns: [d]
        value: 500
        na_pass: true
        brief: "**Step** {step}: {auto}"
    """

    try:
        result = yaml_interrogate(yaml_content)
        assert result is not None
        assert len(result.validation_info) == 5
        assert result.tbl_name == "Brief Test"

        # Check brief values at every step
        assert result.validation_info[0].brief == "**Global Brief**: {auto}"
        assert result.validation_info[1].brief is None
        assert result.validation_info[2].brief == "{auto}"
        assert result.validation_info[3].brief == "This is a custom local brief for the assertion"
        assert result.validation_info[4].brief == "**Step** 5: {auto}"

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise


def test_yaml_to_python_with_briefs():
    yaml_content = """
    tbl: small_table
    tbl_name: Brief Example
    brief: "**Global Brief**: {auto}"
    lang: el
    thresholds:
      warning: 0.236
      error: 0.6
    steps:
    - col_vals_eq:
        columns: [a]
        value: 3
    - col_vals_lt:
        columns: [c]
        value: 5
        brief: false
    - col_vals_gt:
        columns: [d]
        value: 100
        brief: true
    - col_vals_le:
        columns: [a]
        value: 7
        brief: "This is a custom local brief for the assertion"
    """

    try:
        python_code = yaml_to_python(yaml_content)

        # Check that the generated code contains expected elements
        assert "import pointblank as pb" in python_code
        assert 'brief="**Global Brief**: {auto}"' in python_code
        assert 'lang="el"' in python_code
        assert '.col_vals_lt(columns="c", value=5, brief=False)' in python_code
        assert '.col_vals_gt(columns="d", value=100, brief=True)' in python_code
        assert (
            '.col_vals_le(columns="a", value=7, brief="This is a custom local brief for the assertion")'
            in python_code
        )

    except Exception as e:
        raise


def test_python_expressions():
    # Test python: block syntax with simple expression
    yaml_content = """
tbl:
  python: |
    "worldcities.csv"
steps:
  - col_vals_not_null:
      columns: country
    """

    try:
        validator = YAMLValidator()
        config = validator.load_config(yaml_content)
        validation_result = validator.execute_workflow(config)
    except Exception as e:
        raise

    # Test python: block syntax with complex polars operations
    yaml_content = """
tbl:
  python: |
    pl.scan_csv("worldcities.csv").head(5)
steps:
  - row_count_match:
      count: 5
    """

    try:
        validator = YAMLValidator()
        config = validator.load_config(yaml_content)
        validation_result = validator.execute_workflow(config)
    except Exception as e:
        raise

    # Test security restrictions
    yaml_content = """
tbl:
  python: |
    import os
    os.system("echo test")
steps:
  - col_vals_not_null:
      columns: country
    """

    try:
        validator = YAMLValidator()
        config = validator.load_config(yaml_content)
        validation_result = validator.execute_workflow(config)
        raise AssertionError("Security restrictions not working")
    except Exception as e:
        from pointblank.yaml import YAMLValidationError

        if isinstance(e, YAMLValidationError) and ("not allowed" in str(e) or "unsafe" in str(e)):
            pass  # Expected - security restrictions work
        else:
            raise


def test_python_expressions_advanced():
    # Test python: block in validation step parameters
    yaml_content = """
tbl: worldcities.csv
steps:
  - col_vals_in_set:
      columns: country
      set:
        python: |
          ['USA', 'Canada', 'Mexico']
    """

    try:
        validator = YAMLValidator()
        config = validator.load_config(yaml_content)
        validation_result = validator.execute_workflow(config)
    except Exception as e:
        raise


def test_complex_expression_validation_yaml():
    # YAML configuration with complex Polars expression
    yaml_content = """
tbl:
  python: |
    pl.DataFrame({
        "a": [1, 2, 1, 7, 8, 6],
        "b": [0, 0, 0, 1, 1, 1],
        "c": [0.5, 0.3, 0.8, 1.4, 1.9, 1.2],
    })
steps:
  - col_vals_expr:
      expr:
        python: |
          pl.when(pl.col("b") == 0).then(pl.col("a").is_between(0, 5)).when(pl.col("b") == 1).then(pl.col("a") > 5).otherwise(pl.lit(True))
"""

    try:
        validator = YAMLValidator()
        config = validator.load_config(yaml_content)
        validation_result = validator.execute_workflow(config)

        # Verify the validation ran successfully
        assert validation_result is not None
        assert len(validation_result.validation_info) == 1

        validation_info = validation_result.validation_info[0]
        assert validation_info.n_passed == 6
        assert validation_info.n_failed == 0

    except Exception as e:
        raise


def test_complex_expression_multiline_validation_yaml():
    # YAML configuration with complex polars expression
    yaml_content = """
tbl:
  python: |
    pl.DataFrame({
        "a": [1, 2, 1, 7, 8, 6],
        "b": [0, 0, 0, 1, 1, 1],
        "c": [0.5, 0.3, 0.8, 1.4, 1.9, 1.2],
    })
steps:
  - col_vals_expr:
      expr:
        python: |
          (
            pl.when(pl.col("b") == 0)
            .then(pl.col("a")
            .is_between(0, 5))
            .when(pl.col("b") == 1)
            .then(pl.col("a") > 5)
            .otherwise(pl.lit(True))
          )
"""

    try:
        validator = YAMLValidator()
        config = validator.load_config(yaml_content)
        validation_result = validator.execute_workflow(config)

        # Verify the validation ran successfully
        assert validation_result is not None
        assert len(validation_result.validation_info) == 1

        validation_info = validation_result.validation_info[0]
        assert validation_info.n_passed == 6
        assert validation_info.n_failed == 0

    except Exception as e:
        raise


def test_yaml_to_python_with_expressions():
    # YAML configuration with complex expressions
    yaml_content = """
tbl:
  python: |
    pl.DataFrame({
        "a": [1, 2, 1, 7, 8, 6],
        "b": [0, 0, 0, 1, 1, 1],
        "c": [0.5, 0.3, 0.8, 1.4, 1.9, 1.2],
    })
tbl_name: "Expression Test Dataset"
label: "Complex polars expression validation"
steps:
  - col_vals_expr:
      expr:
        python: |
          pl.when(pl.col("b") == 0).then(pl.col("a").is_between(0, 5)).when(pl.col("b") == 1).then(pl.col("a") > 5).otherwise(pl.lit(True))
"""

    try:
        # Test YAML to Python conversion
        python_code = yaml_to_python(yaml_content)

        # Verify the generated Python code contains expected elements
        assert "pb.Validate(" in python_code
        assert "col_vals_expr(" in python_code
        assert "tbl_name=" in python_code
        assert "label=" in python_code
        assert "interrogate()" in python_code

        # Test that the original YAML validation works
        original_result = yaml_interrogate(yaml_content)
        assert original_result is not None
        assert len(original_result.validation_info) == 1
        assert original_result.tbl_name == "Expression Test Dataset"
        assert original_result.label == "Complex polars expression validation"

        # Verify validation logic works correctly
        validation_info = original_result.validation_info[0]
        assert validation_info.n_passed == 6  # All 6 rows should pass the expression
        assert validation_info.n_failed == 0  # No failures expected

    except Exception as e:
        raise


def test_pandas_df_with_pandas_expressions():
    yaml_content = """
tbl:
  python: |
    pd.DataFrame({
        "nums": [1, 2, 3, 4, 5, 6],
        "category": ["A", "B", "A", "B", "A", "B"],
        "values": [10, 20, 30, 40, 50, 60]
    })
steps:
  - col_vals_expr:
      expr: |
        lambda df: df["nums"] > 2
"""

    try:
        validator = YAMLValidator()
        config = validator.load_config(yaml_content)
        result = validator.execute_workflow(config)
        assert result is not None
        validation_info = result.validation_info[0]
        assert validation_info.n_passed == 4  # Values 3, 4, 5, 6 should pass
        assert validation_info.n_failed == 2  # Values 1, 2 should fail
    except Exception as e:
        raise


def test_yaml_to_python_polars_complex_scenarios():
    # Test with multiple polars expressions in different parameters
    yaml_content = """
tbl:
  python: |
    pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
steps:
  - col_vals_expr:
      expr:
        python: |
          pl.col("a") > 1
  - col_vals_in_set:
      columns: a
      set:
        python: |
          [1, 2, 3]
"""

    try:
        python_code = yaml_to_python(yaml_content)
        assert "import pointblank as pb" in python_code, (
            "Missing 'import pointblank as pb' statement"
        )
        assert "import polars as pl" in python_code, "Missing 'import polars as pl' statement"

    except Exception as e:
        raise


def test_yaml_to_python_no_unnecessary_imports():
    # Test 1: Simple YAML without polars should only import pointblank
    yaml_content_simple = """
tbl: small_table
steps:
  - col_vals_gt:
      columns: [a]
      value: 0
"""

    try:
        python_code = yaml_to_python(yaml_content_simple)
        assert "import pointblank as pb" in python_code, (
            "Missing 'import pointblank as pb' statement"
        )
        assert "import polars as pl" not in python_code, (
            "Unnecessary 'import polars as pl' statement"
        )
    except Exception as e:
        raise

    # Test 2: YAML with pandas expressions should only import pointblank
    yaml_content_pandas = """
tbl:
  python: |
    pd.DataFrame({"a": [1, 2, 3]})
steps:
  - col_vals_expr:
      expr: |
        lambda df: df["a"] > 1
"""

    try:
        python_code = yaml_to_python(yaml_content_pandas)
        assert "import pointblank as pb" in python_code, (
            "Missing 'import pointblank as pb' statement"
        )
        assert "import polars as pl" not in python_code, (
            "Unnecessary 'import polars as pl' statement for pandas expressions"
        )
    except Exception as e:
        raise


def test_yaml_to_python_includes_polars_import():
    yaml_content = """
tbl:
  python: |
    pl.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [0, 0, 0, 1, 1]
    })
steps:
  - col_vals_expr:
      expr:
        python: |
          pl.col("a") > 2
"""

    try:
        python_code = yaml_to_python(yaml_content)

        # Check if polars import is included
        assert "import polars as pl" in python_code, "Missing 'import polars as pl' statement"

        # Also check for pointblank import
        assert "import pointblank as pb" in python_code, (
            "Missing 'import pointblank as pb' statement"
        )

    except Exception as e:
        raise


def test_yaml_to_python_includes_pandas_import():
    yaml_content = """
tbl:
  python: |
    pd.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [0, 0, 0, 1, 1]
    })
steps:
  - col_vals_expr:
      expr: |
        lambda df: df["a"] > 2
"""

    try:
        python_code = yaml_to_python(yaml_content)

        # Check if pandas import is included
        assert "import pandas as pd" in python_code, "Missing 'import pandas as pd' statement"

        # Also check for pointblank import
        assert "import pointblank as pb" in python_code, (
            "Missing 'import pointblank as pb' statement"
        )

        # Should not include polars import
        assert "import polars as pl" not in python_code, (
            "Unnecessary 'import polars as pl' statement"
        )

    except Exception as e:
        raise


def test_yaml_to_python_full_functionality_demo():
    yaml_content = """
tbl:
  python: |
    pl.DataFrame({
        "age": [25, 30, 15, 40, 35],
        "income": [50000, 75000, 0, 100000, 60000],
        "department": ["IT", "Sales", "Intern", "Management", "IT"]
    })
tbl_name: "Employee Dataset"
label: "Comprehensive employee validation"
thresholds:
  warning: 0.1
  error: 0.25
steps:
  - col_vals_gt:
      columns: age
      value: 18
      brief: "Employees must be adults"
  - col_vals_expr:
      expr:
        python: |
          pl.col("income") > 0
      brief: "Income must be positive"
  - col_vals_in_set:
      columns: department
      set: ["IT", "Sales", "Management", "Intern"]
"""

    try:
        python_code = yaml_to_python(yaml_content)

        # Verify all expected components are present
        assert "import pointblank as pb" in python_code
        assert "import polars as pl" in python_code
        assert "pb.Validate(" in python_code
        assert "pb.Thresholds(" in python_code
        assert "tbl_name=" in python_code
        assert "label=" in python_code
        assert "col_vals_gt(" in python_code
        assert "col_vals_expr(" in python_code
        assert "col_vals_in_set(" in python_code
        assert "interrogate()" in python_code

        # Verify that the Python expressions are preserved in the generated code
        assert "pl.DataFrame({" in python_code
        assert '"age": [25, 30, 15, 40, 35]' in python_code
        assert '"income": [50000, 75000, 0, 100000, 60000]' in python_code
        assert '"department": ["IT", "Sales", "Intern", "Management", "IT"]' in python_code
        assert 'pl.col("income") > 0' in python_code

    except Exception as e:
        raise


def test_yaml_with_polars_pre_expressions():
    yaml_content = """
tbl: nycflights
label: NYC Flights Freshness Validation
steps:
  - col_vals_eq:
      columns: year
      value: 2013
      brief: "Check year is current"
  - col_vals_lt:
      columns: day
      value: 32
      pre:
        python: |
          lambda df: (
              df.filter(
                  (pl.col("year") == 2013) &
                  (pl.col("month") <= 12)
              )
          )
      brief: "Check no invalid future dates"
  - col_vals_not_null:
      columns: [dep_time, arr_time]
      pre:
        python: |
          lambda df: (
              df.filter(
                  (pl.col("year") == 2013) &
                  (pl.col("month") == 12) &
                  (pl.col("day") >= 25)
              )
          )
      brief: "Check recent flights have complete timing data"
"""

    try:
        # Test 1: YAML parsing and interrogation
        result = yaml_interrogate(yaml_content)
        assert result is not None
        assert len(result.validation_info) >= 3
        assert result.label == "NYC Flights Freshness Validation"

        # Test 2: YAML to Python conversion
        python_code = yaml_to_python(yaml_content)

        # Test 3: Verify round-trip preservation

        # Check that necessary imports are included
        assert "import pointblank as pb" in python_code
        assert "import polars as pl" in python_code

        # Check that the pre= expressions are preserved
        assert "lambda df:" in python_code
        assert 'pl.col("year") == 2013' in python_code
        assert 'pl.col("month") <= 12' in python_code
        assert 'pl.col("month") == 12' in python_code
        assert 'pl.col("day") >= 25' in python_code
        assert "df.filter(" in python_code

        # Check that other elements are preserved
        assert 'label="NYC Flights Freshness Validation"' in python_code
        assert ".col_vals_eq(" in python_code
        assert ".col_vals_lt(" in python_code
        assert ".col_vals_not_null(" in python_code
        assert "value=2013" in python_code
        assert "value=32" in python_code

    except Exception as e:
        raise


def test_yaml_pre_parameter_shortcut_syntax():
    yaml_content = """
tbl: nycflights
label: NYC Flights Freshness Validation
steps:
  - col_vals_eq:
      columns: year
      value: 2013
      brief: "Check year is current"
  - col_vals_lt:
      columns: day
      value: 32
      pre: |
        lambda df: (
            df.filter(
                (pl.col("year") == 2013) &
                (pl.col("month") <= 12)
            )
        )
      brief: "Check no invalid future dates"
  - col_vals_not_null:
      columns: [dep_time, arr_time]
      pre: |
        lambda df: (
            df.filter(
                (pl.col("year") == 2013) &
                (pl.col("month") == 12) &
                (pl.col("day") >= 25)
            )
        )
      brief: "Check recent flights have complete timing data"
"""

    try:
        # Test 1: YAML parsing and interrogation
        result = yaml_interrogate(yaml_content)
        assert result is not None
        assert len(result.validation_info) >= 3
        assert result.label == "NYC Flights Freshness Validation"

        # Test 2: YAML to Python conversion
        python_code = yaml_to_python(yaml_content)

        # Test 3: Verify round-trip preservation

        # Check that necessary imports are included
        assert "import pointblank as pb" in python_code
        assert "import polars as pl" in python_code

        # Check that the pre= expressions are preserved
        assert "lambda df:" in python_code
        assert 'pl.col("year") == 2013' in python_code
        assert 'pl.col("month") <= 12' in python_code
        assert 'pl.col("month") == 12' in python_code
        assert 'pl.col("day") >= 25' in python_code
        assert "df.filter(" in python_code

        # Check that other elements are preserved
        assert 'label="NYC Flights Freshness Validation"' in python_code
        assert ".col_vals_eq(" in python_code
        assert ".col_vals_lt(" in python_code
        assert ".col_vals_not_null(" in python_code
        assert "value=2013" in python_code
        assert "value=32" in python_code

    except Exception as e:
        raise


def test_yaml_actions_support():
    # Test actions at global level
    yaml_content = """
tbl: small_table
label: Actions Test
thresholds:
  warning: 1
actions:
  warning: "Warning: Step {step} failed with value {val}"
  error: "Error occurred in column {col}"
  highest_only: true
steps:
  - col_vals_gt:
      columns: a
      value: 1000
"""

    try:
        # Test 1: YAML parsing and interrogation should work
        result = yaml_interrogate(yaml_content)
        assert result is not None
        assert len(result.validation_info) >= 1

        # Test 2: YAML to Python conversion should preserve actions
        python_code = yaml_to_python(yaml_content)

        # Check that actions are included in generated code
        assert "pb.Actions(" in python_code
        assert "warning=" in python_code
        assert "highest_only=" in python_code

    except Exception as e:
        raise

    # Test actions at step level
    yaml_step_actions = """
tbl: small_table
steps:
  - col_vals_gt:
      columns: a
      value: 1000
      thresholds:
        warning: 1
      actions:
        warning: "Step-level warning for {col}"
        error: "Step-level error"
"""

    try:
        # Test step-level actions
        result2 = yaml_interrogate(yaml_step_actions)
        assert result2 is not None

        python_code2 = yaml_to_python(yaml_step_actions)
        assert "actions=pb.Actions(" in python_code2

    except Exception as e:
        raise


def test_yaml_actions_with_callables():
    import io
    import contextlib

    yaml_content = """
tbl: small_table
thresholds:
  warning: 1
actions:
  warning:
    python: |
      lambda: print("Custom warning action executed")
steps:
  - col_vals_gt:
      columns: a
      value: 1000
"""

    try:
        # Capture stdout to verify print statements
        captured_output = io.StringIO()

        with contextlib.redirect_stdout(captured_output):
            # Test callable actions
            result = yaml_interrogate(yaml_content)
            assert result is not None

        # Verify the action was executed and printed the expected message
        output_text = captured_output.getvalue()
        assert "Custom warning action executed" in output_text

        python_code = yaml_to_python(yaml_content)
        assert "pb.Actions(" in python_code

    except Exception as e:
        raise


def test_yaml_actions_comprehensive_demo():
    import io
    import contextlib

    yaml_content = """
tbl: small_table
label: Comprehensive Actions Demo
thresholds:
  warning: 0.1
  error: 0.2
  critical: 0.3
actions:
  warning: "Global warning: {LEVEL} threshold exceeded in step {step}"
  critical: "Global critical alert for {type} validation"
  highest_only: false
steps:
  - col_vals_gt:
      columns: a
      value: 1000
      brief: "Check that a > 1000"
      thresholds:
        warning: 1
      actions:
        warning: "[{LEVEL}: {TYPE}]: Step {step} has a problem with the value {val} in column {col} ({time})"
  - col_vals_in_set:
      columns: a
      set: [1, 2, 3]
      brief: "Check that a is in valid set"
"""

    try:
        # Capture stdout to verify action outputs
        captured_output = io.StringIO()

        with contextlib.redirect_stdout(captured_output):
            # Test execution
            result = yaml_interrogate(yaml_content)

        # Verify actions were executed and templated correctly
        output_text = captured_output.getvalue()

        # Check for step-level action with templating
        assert "[WARNING: COL_VALS_GT]" in output_text
        assert "Step 1 has a problem" in output_text
        assert "column a" in output_text

        # Check for global action
        assert "Global critical alert for col_vals_in_set validation" in output_text

        # Test code generation
        python_code = yaml_to_python(yaml_content)

        # Verify actions are preserved
        assert "pb.Actions(" in python_code
        assert "warning=" in python_code
        assert "highest_only=False" in python_code

    except Exception as e:
        raise


def test_yaml_actions_output_verification():
    import io
    import contextlib

    # Test 1: String template actions
    yaml_content_templates = """
tbl: small_table
thresholds:
  warning: 1
actions:
  warning: "Template test: Step {step} failed on column {col} with value {val}"
steps:
  - col_vals_gt:
      columns: a
      value: 1000
"""

    captured_output = io.StringIO()
    with contextlib.redirect_stdout(captured_output):
        result = yaml_interrogate(yaml_content_templates)

    output_text = captured_output.getvalue()
    assert "Template test: Step 1 failed on column a" in output_text
    assert "with value" in output_text  # Value will be dynamic but should contain this phrase

    # Test 2: Callable actions
    yaml_content_callable = """
tbl: small_table
thresholds:
  error: 1
actions:
  error:
    python: |
      lambda: print("ERROR: Callable action triggered!")
steps:
  - col_vals_gt:
      columns: a
      value: 1000
"""

    captured_output2 = io.StringIO()
    with contextlib.redirect_stdout(captured_output2):
        result2 = yaml_interrogate(yaml_content_callable)

    output_text2 = captured_output2.getvalue()
    assert "ERROR: Callable action triggered!" in output_text2

    # Test 3: Multiple action levels
    yaml_content_multi = """
tbl: small_table
thresholds:
  warning: 0.5
  error: 1
actions:
  warning: "WARN: {LEVEL} - {TYPE}"
  error: "ERR: {LEVEL} - {TYPE}"
  highest_only: false
steps:
  - col_vals_gt:
      columns: a
      value: 1000
"""

    captured_output3 = io.StringIO()
    with contextlib.redirect_stdout(captured_output3):
        result3 = yaml_interrogate(yaml_content_multi)

    output_text3 = captured_output3.getvalue()
    # Should see both warning and error since highest_only is false
    assert "WARN: WARNING - COL_VALS_GT" in output_text3
    assert "ERR: ERROR - COL_VALS_GT" in output_text3


def test_yaml_actions_print_capture_demo():
    import io
    import contextlib

    yaml_content = """
tbl: small_table
thresholds:
  warning: 1
actions:
  warning: "[CAPTURED]: This is a warning from step {step} on column {col}"
steps:
  - col_vals_gt:
      columns: a
      value: 1000
"""

    # Capture the output
    captured_output = io.StringIO()
    with contextlib.redirect_stdout(captured_output):
        result = yaml_interrogate(yaml_content)

    # Get the captured text
    output_text = captured_output.getvalue()

    # Verify specific content
    expected_text = "[CAPTURED]: This is a warning from step 1 on column a"
    assert expected_text in output_text

    print("✅ Actions print output successfully captured and verified!")

    # Also verify the validation executed correctly
    assert result is not None
    assert len(result.validation_info) == 1
    validation_info = result.validation_info[0]
    assert validation_info.assertion_type == "col_vals_gt"
    assert validation_info.n_failed > 0  # Should fail since a values are not > 1000
