<div align="center">

<img src="images/pointblank_hex_logo.png" alt="Pointblank logo" width="350px"/>

_Find out if your data is what you think it is._

[![License](https://img.shields.io/github/license/rich-iannone/pointblank)](https://img.shields.io/github/license/rich-iannone/great-tables)

[![CI Build](https://github.com/rich-iannone/pointblank/workflows/CI%20Tests/badge.svg?branch=main)](https://github.com/rich-iannone/pointblank/actions?query=workflow%3A%22CI+Tests%22+branch%3Amain)
[![Repo Status](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)

[![Contributors](https://img.shields.io/github/contributors/rich-iannone/pointblank)](https://github.com/rich-iannone/pointblank/graphs/contributors)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v2.1%20adopted-ff69b4.svg)](https://www.contributor-covenant.org/version/2/1/code_of_conduct.html)

</div>

## Introduction

`pointblank` is a table validation and testing library for Python. It helps you ensure that your tabular data meets certain expectations and constraints and it can present the results in a beautiful and useful tabular reporting framework. There are two main workflows that are supported:

1. **Data Validation and Reporting**: Use the `Validate` class to validate your data against a collection of rules, generate a tabular report of the results, and extract key data to solve DQ issues.
2. **Data Testing**: Use the `Test` class to write tests for your data and use them in notebook code or in a testing framework like `pytest`.

These workflows make it possible to catch data quality issues early, and the tooling enables you to get at the root causes of DQ issues quickly.

## Features

- **Declarative Syntax**: Define your data validation rules using a declarative syntax.
- **Flexible**: We support tables from Polars, Pandas, Duckdb, MySQL, PostgreSQL, SQLite, and Parquet without any changes to your code.
- **Beautiful Reports**: Generate beautiful HTML reports of your data validation results.
- **Functional Output**: Get JSON output of your data validation results for further processing.
- **Data Testing**: Write tests for your data and use them in your notebooks or testing framework.
- **Easy to Use**: Get started quickly with a simple API and clear documentation.
- **Powerful**: You can develop complex data validation rules with fleixble options for customization.

## Example

Let's say you have a Polars DataFrame and you want to validate it against a set of constraints. Here's how you can do that using the `pb.Validate` class and its library of validation methods:

```python
import pointblank as pb
import polars as pl

# Create a Polars DataFrame
tbl_pl = pl.DataFrame({"x": [1, 2, 3, 4], "y": [4, 5, 6, 7]})

# Validate data using Polars DataFrame
v = (
    pb.Validate(data=tbl_pl) # Add data to be validated
    .col_vals_gt(columns="x", value=0) # STEP 1 |
    .col_vals_lt(columns="x", value=3) # STEP 2 | <-- The validation plan
    .col_vals_le(columns="y", value=7) # STEP 3 |
    .interrogate() # This will execute all validation steps
)

# Get an HTML report of the validation
v.get_json_report()
```

<img src="images/pointblank-validation-html-report.png" alt="Validation Report">

## Installation

You can install `pointblank` using pip:

```bash
pip install pointblank
```
