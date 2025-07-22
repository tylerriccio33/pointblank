import pytest
from pointblank import yaml_interrogate, validate_yaml
from pointblank.yaml import YAMLValidationError, load_yaml_config


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

    # Invalid threshold values
    yaml_content = """
    tbl: small_table
    thresholds:
      warning: 1.5
    steps:
    - rows_distinct
    """
    with pytest.raises(
        YAMLValidationError, match="Threshold 'warning' must be a number between 0 and 1"
    ):
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
    from pointblank.yaml import YAMLValidator

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
      error: 0.10
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
        print(f"Error in comprehensive test: {e}")
        import traceback

        traceback.print_exc()
        raise


def test_yaml_to_python_comprehensive():
    from pointblank import yaml_to_python

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
        print(f"Error in yaml_to_python test: {e}")
        raise
