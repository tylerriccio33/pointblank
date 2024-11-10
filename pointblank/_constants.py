GENERAL_COLUMN_TYPES = [
    "numeric",  # Numeric data types (`int`, `float`)
    "str",  # String data type (`string`)
    "bool",  # Boolean data type (`boolean`)
    "datetime",  # Date or Datetime data type (`datetime`)
    "duration",  # Duration data type (`duration`)
]

COMPATIBLE_TYPES = {
    "gt": ["numeric"],
    "lt": ["numeric"],
    "eq": ["numeric"],
    "ne": ["numeric"],
    "ge": ["numeric"],
    "le": ["numeric"],
    "between": ["numeric"],
    "outside": ["numeric"],
    "in_set": ["numeric", "str"],
    "not_in_set": ["numeric", "str"],
}

TYPE_METHOD_MAP = {
    "col_vals_gt": "gt",
    "col_vals_lt": "lt",
    "col_vals_eq": "eq",
    "col_vals_ne": "ne",
    "col_vals_ge": "ge",
    "col_vals_le": "le",
    "col_vals_between": "between",
    "col_vals_outside": "outside",
    "col_vals_in_set": "in_set",
    "col_vals_not_in_set": "not_in_set",
}

COMPARE_TYPE_MAP = {
    "gt": "COMPARE_ONE",
    "lt": "COMPARE_ONE",
    "eq": "COMPARE_ONE",
    "ne": "COMPARE_ONE",
    "ge": "COMPARE_ONE",
    "le": "COMPARE_ONE",
    "between": "COMPARE_TWO",
    "outside": "COMPARE_TWO",
    "in_set": "COMPARE_SET",
    "not_in_set": "COMPARE_SET",
}
