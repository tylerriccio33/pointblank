from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import yaml

from pointblank.validate import Validate, load_dataset


class YAMLValidationError(Exception):
    """Exception raised for YAML validation errors."""

    pass


class YAMLValidator:
    """Validates YAML configuration and converts to Validate objects."""

    # Map YAML method names to Python method names
    validation_method_map = {
        "col_exists": "col_exists",
        "col_vals_gt": "col_vals_gt",
        "col_vals_ge": "col_vals_ge",
        "col_vals_lt": "col_vals_lt",
        "col_vals_le": "col_vals_le",
        "col_vals_eq": "col_vals_eq",
        "col_vals_ne": "col_vals_ne",
        "col_vals_between": "col_vals_between",
        "col_vals_outside": "col_vals_outside",
        "col_vals_regex": "col_vals_regex",
        "col_vals_in_set": "col_vals_in_set",
        "col_vals_not_in_set": "col_vals_not_in_set",
        "col_vals_not_null": "col_vals_not_null",
        "col_vals_null": "col_vals_null",
        "rows_distinct": "rows_distinct",
        "rows_complete": "rows_complete",
        "col_count_match": "col_count_match",
        "row_count_match": "row_count_match",
    }

    def __init__(self):
        """Initialize the YAML validator."""
        pass

    def load_config(self, source: Union[str, Path]) -> dict:
        """Load and validate YAML configuration.

        Args:
            source: YAML string or Path to YAML file

        Returns:
            Parsed and validated configuration dictionary

        Raises:
            YAMLValidationError: If the YAML is invalid or malformed
        """
        try:
            if isinstance(source, (str, Path)):
                if isinstance(source, Path):
                    # It's definitely a file path
                    with open(source, "r", encoding="utf-8") as f:
                        config = yaml.safe_load(f)
                elif isinstance(source, str):
                    # Check if it looks like YAML content
                    stripped = source.strip()
                    if (
                        stripped.startswith(("tbl:", "steps:"))
                        or "\n" in stripped
                        or ":" in stripped
                    ):
                        # Looks like YAML content
                        config = yaml.safe_load(source)
                    else:
                        # Assume it's a file path
                        with open(source, "r", encoding="utf-8") as f:
                            config = yaml.safe_load(f)
            else:
                raise YAMLValidationError(
                    f"Invalid source type: {type(source)}. Only YAML strings and file paths supported."
                )

            if not isinstance(config, dict):
                raise YAMLValidationError("YAML must contain a dictionary at the root level")

            self._validate_schema(config)
            return config

        except yaml.YAMLError as e:
            raise YAMLValidationError(f"Invalid YAML syntax: {e}")
        except Exception as e:
            raise YAMLValidationError(f"Error loading YAML configuration: {e}")

    def _validate_schema(self, config: dict) -> None:
        """Validate the YAML configuration schema.

        Args:
            config: Configuration dictionary to validate

        Raises:
            YAMLValidationError: If the schema is invalid
        """
        # Check required fields
        if "tbl" not in config:
            raise YAMLValidationError("YAML must contain 'tbl' field")

        if "steps" not in config:
            raise YAMLValidationError("YAML must contain 'steps' field")

        if not isinstance(config["steps"], list):
            raise YAMLValidationError("'steps' must be a list")

        if len(config["steps"]) == 0:
            raise YAMLValidationError("'steps' cannot be empty")

        # Validate thresholds if present
        if "thresholds" in config:
            thresholds = config["thresholds"]
            if not isinstance(thresholds, dict):
                raise YAMLValidationError("'thresholds' must be a dictionary")

            for key, value in thresholds.items():
                if key not in ["warning", "error", "critical"]:
                    raise YAMLValidationError(
                        f"Invalid threshold key: {key}. Must be 'warning', 'error', or 'critical'"
                    )

                if not isinstance(value, (int, float)):
                    raise YAMLValidationError(f"Threshold '{key}' must be a number")

                if value < 0:
                    raise YAMLValidationError(f"Threshold '{key}' must be non-negative")

    def _load_data_source(self, tbl_spec: str) -> Any:
        """Load data source based on table specification.

        Parameters
        ----------
        tbl_spec
            Data source specification. Can be (1) a dataset name for `load_dataset()`, (2) a CSV file
            path (relative or absolute), or (3) a Parquet file path (relative or absolute).

        Returns
        -------
            Loaded data object.

        Raises
        ------
        YAMLValidationError
            If data source cannot be loaded.
        """
        from pointblank.validate import _process_data

        try:
            # Use the centralized data processing pipeline from validate.py
            # This handles CSV files, Parquet files, and other data sources
            processed_data = _process_data(tbl_spec)

            # If _process_data returns the original string unchanged,
            # then it's not a file path, so try load_dataset
            if processed_data is tbl_spec and isinstance(tbl_spec, str):
                return load_dataset(tbl_spec)
            else:
                return processed_data

        except Exception as e:
            raise YAMLValidationError(f"Failed to load data source '{tbl_spec}': {e}")

    def _parse_column_spec(self, columns_expr: Any) -> list[str]:
        """Parse column specification from YAML.

        Handles standard YAML syntax for columns.

        Args:
            columns_expr: Column specification (list, or string)

        Returns:
            List of column names
        """
        if isinstance(columns_expr, list):
            return [str(col) for col in columns_expr]

        if isinstance(columns_expr, str):
            # Single column name
            return [columns_expr]

        # Fallback: convert to string
        return [str(columns_expr)]

    def _parse_validation_step(self, step_config: Union[str, dict]) -> tuple[str, dict]:
        """Parse a single validation step from YAML configuration.

        Args:
            step_config: Step configuration (string for parameterless steps, dict for others)

        Returns:
            Tuple of (method_name, parameters)

        Raises:
            YAMLValidationError: If step configuration is invalid
        """
        if isinstance(step_config, str):
            # Simple step with no parameters (e.g., "rows_distinct")
            method_name = step_config
            parameters = {}
        elif isinstance(step_config, dict):
            # Step with parameters
            if len(step_config) != 1:
                raise YAMLValidationError(
                    "Step configuration must contain exactly one validation method, "
                    f"got: {list(step_config.keys())}"
                )

            method_name = list(step_config.keys())[0]
            parameters = step_config[method_name] or {}

            if not isinstance(parameters, dict):
                raise YAMLValidationError(f"Parameters for '{method_name}' must be a dictionary")
        else:
            raise YAMLValidationError(f"Invalid step configuration type: {type(step_config)}")

        # Validate that we know this method
        if method_name not in self.validation_method_map:
            available_methods = list(self.validation_method_map.keys())
            raise YAMLValidationError(
                f"Unknown validation method '{method_name}'. Available methods: {available_methods}"
            )

        # Convert `columns=` specification
        if "columns" in parameters:
            parameters["columns"] = self._parse_column_spec(parameters["columns"])

        #
        # Convert special parameter formats
        #

        # Convert `columns_subset=` if present (from `rows_[distinct|complete]()`)
        if "columns_subset" in parameters:
            parameters["columns_subset"] = self._parse_column_spec(parameters["columns_subset"])

        # Handle `inclusive=` parameter for `col_vals_[inside|outside]()` (convert list to tuple)
        if "inclusive" in parameters and isinstance(parameters["inclusive"], list):
            parameters["inclusive"] = tuple(parameters["inclusive"])

        return self.validation_method_map[method_name], parameters

    def build_validation(self, config: dict) -> Validate:
        """Convert YAML config to Validate object.

        Args:
            config: Validated configuration dictionary

        Returns:
            Validate object with configured validation steps
        """
        # Load data source
        data = self._load_data_source(config["tbl"])

        # Create Validate object
        validate_kwargs = {}

        # Set table name if provided
        if "tbl_name" in config:
            validate_kwargs["tbl_name"] = config["tbl_name"]

        # Set label if provided
        if "label" in config:
            validate_kwargs["label"] = config["label"]

        # Set thresholds if provided
        if "thresholds" in config:
            validate_kwargs["thresholds"] = config["thresholds"]

        # Set language if provided
        if "lang" in config:
            validate_kwargs["lang"] = config["lang"]

        # Set locale if provided
        if "locale" in config:
            validate_kwargs["locale"] = config["locale"]

        # Set global brief if provided
        if "brief" in config:
            validate_kwargs["brief"] = config["brief"]

        validation = Validate(data, **validate_kwargs)

        # Add validation steps
        for step_config in config["steps"]:
            method_name, parameters = self._parse_validation_step(step_config)

            # Get the method from the validation object
            method = getattr(validation, method_name)

            # Call the method with parameters
            validation = method(**parameters)

        return validation

    def execute_workflow(self, config: dict) -> Validate:
        """Execute a complete YAML validation workflow.

        Args:
            config: Validated configuration dictionary

        Returns:
            Interrogated Validate object with results
        """
        # Build the validation plan
        validation = self.build_validation(config)

        # Execute interrogation to get results
        validation = validation.interrogate()

        return validation


def yaml_interrogate(yaml: Union[str, Path]) -> Validate:
    """Execute a YAML-based validation workflow.

    This is the main entry point for YAML-based validation workflows. It takes YAML configuration
    (as a string or file path) and returns a validated `Validate` object with interrogation results.

    The YAML configuration defines the data source, validation steps, and optional settings like
    thresholds and labels. This function automatically loads the data, builds the validation plan,
    executes all validation steps, and returns the interrogated results.

    Parameters
    ----------
    yaml
        YAML configuration as string or file path. Can be: (1) a YAML string containing the
        validation configuration, or (2) a Path object or string path to a YAML file.

    Returns
    -------
    Validate
        An instance of the `Validate` class that has been configured based on the YAML input.
        This object contains the results of the validation steps defined in the YAML configuration.
        It includes metadata like table name, label, language, and thresholds if specified.

    Raises
    ------
    YAMLValidationError
        If the YAML is invalid, malformed, or execution fails. This includes syntax errors, missing
        required fields, unknown validation methods, or data loading failures.

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_header=False, report_incl_footer=False, preview_incl_header=False)
    ```
    For the examples here, we'll use YAML configurations to define validation workflows. Let's start
    with a basic YAML workflow that validates the built-in `small_table` dataset.

    ```{python}
    import pointblank as pb

    # Define a basic YAML validation workflow
    yaml_config = '''
    tbl: small_table
    steps:
    - rows_distinct
    - col_exists:
        columns: [date, a, b]
    '''

    # Execute the validation workflow
    result = pb.yaml_interrogate(yaml_config)
    result
    ```

    The validation table shows the results of our YAML-defined workflow. We can see that the
    `rows_distinct()` validation failed (because there are duplicate rows in the table), while the
    column existence checks passed.

    Now let's create a more comprehensive validation workflow with thresholds and metadata:

    ```{python}
    # Advanced YAML configuration with thresholds and metadata
    yaml_config = '''
    tbl: small_table
    tbl_name: `small_table_demo`
    label: Comprehensive data validation
    thresholds:
      warning: 0.1
      error: 0.25
      critical: 0.35
    steps:
    - col_vals_gt:
        columns: [d]
        value: 100
    - col_vals_regex:
        columns: [b]
        pattern: '[0-9]-[a-z]{3}-[0-9]{3}'
    - col_vals_not_null:
        columns: [date, a]
    '''

    # Execute the validation workflow
    result = pb.yaml_interrogate(yaml_config)
    print(f"Table name: {result.tbl_name}")
    print(f"Label: {result.label}")
    print(f"Total validation steps: {len(result.validation_info)}")
    ```

    The validation results now include our custom table name and label. The thresholds we defined
    will determine when validation steps are marked as warnings, errors, or critical failures.

    You can also load YAML configurations from files. Here's how you would work with a YAML file:

    ```{python}
    from pathlib import Path
    import tempfile

    # Create a temporary YAML file for demonstration
    yaml_content = '''
    tbl: small_table
    tbl_name: File-based Validation
    steps:
    - col_vals_between:
        columns: [c]
        left: 1
        right: 10
    - col_vals_in_set:
        columns: [f]
        set: [low, mid, high]
    '''

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        yaml_file_path = Path(f.name)

    # Load and execute validation from file
    result = pb.yaml_interrogate(yaml_file_path)
    result
    ```

    This approach is particularly useful for storing validation configurations as part of your data
    pipeline or version control system, allowing you to maintain validation rules alongside your
    code.
    """
    validator = YAMLValidator()
    config = validator.load_config(yaml)
    return validator.execute_workflow(config)


def load_yaml_config(file_path: Union[str, Path]) -> dict:
    """Load YAML configuration from file or string.

    Args:
        file_path: Path to YAML file or YAML content string

    Returns:
        Parsed configuration dictionary

    Raises:
        YAMLValidationError: If the file cannot be loaded or is invalid
    """
    validator = YAMLValidator()
    return validator.load_config(file_path)


def validate_yaml(yaml: Union[str, Path]) -> None:
    """Validate YAML configuration against the expected structure.

    This function validates that a YAML configuration conforms to the expected structure for
    validation workflows. It checks for required fields, proper data types, and valid validation
    method names. This is useful for validating configurations before execution or for building
    configuration editors and validators.

    The function performs comprehensive validation including:

    - required fields ('tbl' and 'steps')
    - proper data types for all fields
    - valid threshold configurations
    - known validation method names
    - proper step configuration structure

    Parameters
    ----------
    yaml
        YAML configuration as string or file path. Can be: (1) a YAML string containing the
        validation configuration, or (2) a Path object or string path to a YAML file.

    Raises:
    YAMLValidationError
        If the YAML is invalid, malformed, or execution fails. This includes syntax errors, missing
        required fields, unknown validation methods, or data loading failures.

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_header=False, report_incl_footer=False, preview_incl_header=False)
    ```
    For the examples here, we'll demonstrate how to validate YAML configurations before using them
    with validation workflows. This is particularly useful for building robust data validation
    systems where you want to catch configuration errors early.

    Let's start with validating a basic configuration:

    ```{python}
    import pointblank as pb

    # Define a basic YAML validation configuration
    yaml_config = '''
    tbl: small_table
    steps:
    - rows_distinct
    - col_exists:
        columns: [a, b]
    '''

    # Validate the configuration: no exception means it's valid
    pb.validate_yaml(yaml_config)
    print("Basic YAML configuration is valid")
    ```

    The function completed without raising an exception, which means our configuration is valid and
    follows the expected structure.

    Now let's validate a more complex configuration with thresholds and metadata:

    ```{python}
    # Complex YAML configuration with all optional fields
    yaml_config = '''
    tbl: small_table
    tbl_name: My Dataset
    label: Quality check
    lang: en
    locale: en
    thresholds:
      warning: 0.1
      error: 0.25
      critical: 0.35
    steps:
    - rows_distinct
    - col_vals_gt:
        columns: [d]
        value: 100
    - col_vals_regex:
        columns: [b]
        pattern: '[0-9]-[a-z]{3}-[0-9]{3}'
    '''

    # Validate the configuration
    pb.validate_yaml(yaml_config)
    print("Complex YAML configuration is valid")

    # Count the validation steps
    import pointblank.yaml as pby
    config = pby.load_yaml_config(yaml_config)
    print(f"Configuration has {len(config['steps'])} validation steps")
    ```

    This configuration includes all the optional metadata fields and complex validation steps,
    demonstrating that the validation handles the full range of supported options.

    Let's see what happens when we try to validate an invalid configuration:

    ```{python}
    # Invalid YAML configuration: missing required 'tbl' field
    invalid_yaml = '''
    steps:
    - rows_distinct
    '''

    try:
        pb.validate_yaml(invalid_yaml)
    except pb.yaml.YAMLValidationError as e:
        print(f"Validation failed: {e}")
    ```

    The validation correctly identifies that our configuration is missing the required `'tbl'`
    field.

    Here's a practical example of using validation in a workflow builder:

    ```{python}
    def safe_yaml_interrogate(yaml_config):
        \"\"\"Safely execute a YAML configuration after validation.\"\"\"
        try:
            # Validate the YAML configuration first
            pb.validate_yaml(yaml_config)
            print("✓ YAML configuration is valid")

            # Then execute the workflow
            result = pb.yaml_interrogate(yaml_config)
            print(f"Validation completed with {len(result.validation_info)} steps")
            return result

        except pb.yaml.YAMLValidationError as e:
            print(f"Configuration error: {e}")
            return None

    # Test with a valid YAML configuration
    test_yaml = '''
    tbl: small_table
    steps:
    - col_vals_between:
        columns: [c]
        left: 1
        right: 10
    '''

    result = safe_yaml_interrogate(test_yaml)
    ```

    This pattern of validating before executing helps build more reliable data validation pipelines
    by catching configuration errors early in the process.

    Note that this function only validates the structure and does not check if the specified data
    source ('tbl') exists or is accessible. Data source validation occurs during execution with
    `yaml_interrogate()`.

    See Also:
        - `yaml_interrogate()`: execute YAML-based validation workflows
    """
    validator = YAMLValidator()
    config = validator.load_config(yaml)
    return validator.execute_workflow(config)


def yaml_to_python(yaml: Union[str, Path]) -> str:
    """Convert YAML validation configuration to equivalent Python code.

    This function takes a YAML validation configuration and generates the equivalent Python code
    that would produce the same validation workflow. This is useful for documentation, code
    generation, or learning how to translate YAML workflows into programmatic workflows.

    The generated Python code includes all necessary imports, data loading, validation steps,
    and interrogation execution, formatted as executable Python code.

    Parameters
    ----------
    yaml
        YAML configuration as string or file path. Can be: (1) a YAML string containing the
        validation configuration, or (2) a Path object or string path to a YAML file.

    Returns
    -------
    str
        A formatted Python code string enclosed in markdown code blocks that replicates the YAML
        workflow. The code includes import statements, data loading, validation method calls, and
        interrogation execution.

    Raises
    ------
    YAMLValidationError
        If the YAML is invalid, malformed, or contains unknown validation methods.

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_header=False, report_incl_footer=False, preview_incl_header=False)
    ```

    Convert a basic YAML configuration to Python code:

    ```{python}
    import pointblank as pb

    # Define a YAML validation workflow
    yaml_config = '''
    tbl: small_table
    tbl_name: Data Quality Check
    steps:
    - col_vals_not_null:
        columns: [a, b]
    - col_vals_gt:
        columns: [c]
        value: 0
    '''

    # Generate equivalent Python code
    python_code = pb.yaml_to_python(yaml_config)
    print(python_code)
    ```

    The generated Python code shows exactly how to replicate the YAML workflow programmatically.
    This is particularly useful when transitioning from YAML-based workflows to code-based
    workflows, or when generating documentation that shows both YAML and Python approaches.

    For more complex workflows with thresholds and metadata:

    ```{python}
    # Advanced YAML configuration
    yaml_config = '''
    tbl: small_table
    tbl_name: Advanced Validation
    label: Production data check
    thresholds:
      warning: 0.1
      error: 0.2
    steps:
    - col_vals_between:
        columns: [c]
        left: 1
        right: 10
    - col_vals_regex:
        columns: [b]
        pattern: '[0-9]-[a-z]{3}-[0-9]{3}'
    '''

    # Generate the equivalent Python code
    python_code = pb.yaml_to_python(yaml_config)
    print(python_code)
    ```

    The generated code includes all configuration parameters, thresholds, and maintains the exact
    same validation logic as the original YAML workflow.

    This function is also useful for educational purposes, helping users understand how YAML
    configurations map to the underlying Python API calls.
    """
    # Load and validate the YAML configuration
    validator = YAMLValidator()
    config = validator.load_config(yaml)

    # Start building the Python code
    code_lines = []

    # Add imports
    code_lines.append("import pointblank as pb")
    code_lines.append("")

    # Build the chained validation call
    code_lines.append("(")

    # Build validation initialization arguments
    validate_args = []

    # Add data loading as first argument
    tbl_spec = config["tbl"]
    if tbl_spec.endswith((".csv", ".parquet")):
        # File loading
        validate_args.append(f'data=pb.load_dataset("{tbl_spec}")')
    else:
        # Dataset loading
        validate_args.append(f'data=pb.load_dataset("{tbl_spec}")')

    # Add table name if present
    if "tbl_name" in config:
        validate_args.append(f'tbl_name="{config["tbl_name"]}"')

    # Add `label=` if present
    if "label" in config:
        validate_args.append(f'label="{config["label"]}"')

    # Add `thresholds=` if present: format as `pb.Thresholds()` for an idiomatic style
    if "thresholds" in config:
        thresholds_dict = config["thresholds"]
        threshold_params = []
        for key, value in thresholds_dict.items():
            threshold_params.append(f"{key}={value}")
        thresholds_str = "pb.Thresholds(" + ", ".join(threshold_params) + ")"
        validate_args.append(f"thresholds={thresholds_str}")

    # Add language if present
    if "lang" in config:
        validate_args.append(f'lang="{config["lang"]}"')

    # Add locale if present
    if "locale" in config:
        validate_args.append(f'locale="{config["locale"]}"')

    # Add global brief if present
    if "brief" in config:
        if isinstance(config["brief"], bool):
            validate_args.append(f"brief={str(config['brief'])}")
        else:
            validate_args.append(f'brief="{config["brief"]}"')

    # Create the `pb.Validate()` call
    if len(validate_args) == 1:
        # Single argument fits on one line
        code_lines.append(f"    pb.Validate({validate_args[0]})")
    else:
        # Multiple arguments: format each on its own line
        code_lines.append("    pb.Validate(")
        for i, arg in enumerate(validate_args):
            if i == len(validate_args) - 1:
                code_lines.append(f"        {arg},")
            else:
                code_lines.append(f"        {arg},")
        code_lines.append("    )")

    # Add validation steps as chained method calls
    for step_config in config["steps"]:
        method_name, parameters = validator._parse_validation_step(step_config)

        # Format parameters
        param_parts = []
        for key, value in parameters.items():
            if key in ["columns", "columns_subset"]:
                if isinstance(value, list):
                    if len(value) == 1:
                        # Single column as string
                        param_parts.append(f'{key}="{value[0]}"')
                    else:
                        # Multiple columns as list
                        columns_str = "[" + ", ".join([f'"{col}"' for col in value]) + "]"
                        param_parts.append(f"{key}={columns_str}")
                else:
                    param_parts.append(f'{key}="{value}"')
            elif key == "brief":
                # Handle brief parameter: can be a boolean or a string
                if isinstance(value, bool):
                    param_parts.append(f"brief={str(value)}")
                else:
                    param_parts.append(f'brief="{value}"')
            elif isinstance(value, str):
                param_parts.append(f'{key}="{value}"')
            elif isinstance(value, bool):
                param_parts.append(f"{key}={str(value)}")
            elif isinstance(value, tuple):
                # Handle tuples like `inclusive=(False, True)`
                tuple_str = "(" + ", ".join([str(item) for item in value]) + ")"
                param_parts.append(f"{key}={tuple_str}")
            elif isinstance(value, list):
                # Handle lists/tuples (like `set=` parameter)
                if all(isinstance(item, str) for item in value):
                    list_str = "[" + ", ".join([f'"{item}"' for item in value]) + "]"
                else:
                    list_str = str(list(value))
                param_parts.append(f"{key}={list_str}")
            else:
                param_parts.append(f"{key}={value}")

        if param_parts:
            params_str = ", ".join(param_parts)
            code_lines.append(f"    .{method_name}({params_str})")
        else:
            code_lines.append(f"    .{method_name}()")

    # Add interrogation method call
    code_lines.append("    .interrogate()")
    code_lines.append(")")

    # Join all code lines and wrap in single markdown code block
    python_code = "\n".join(code_lines)
    return f"```python\n{python_code}\n```"
