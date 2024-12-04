<div align="center">

<img src="images/pointblank_hex_logo.png" alt="Pointblank logo" width="350px"/>

_Find out if your data is what you think it is._

[![License](https://img.shields.io/github/license/rich-iannone/pointblank)](https://img.shields.io/github/license/rich-iannone/great-tables)

[![CI Build](https://github.com/rich-iannone/pointblank/actions/workflows/ci-tests.yaml/badge.svg)](https://github.com/rich-iannone/pointblank/actions/workflows/ci-tests.yaml)
[![Repo Status](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)

[![Contributors](https://img.shields.io/github/contributors/rich-iannone/pointblank)](https://github.com/rich-iannone/pointblank/graphs/contributors)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v2.1%20adopted-ff69b4.svg)](https://www.contributor-covenant.org/version/2/1/code_of_conduct.html)

</div>

## Introduction

`pointblank` is a table validation and testing library for Python. It helps you ensure that your tabular data meets certain expectations and constraints and it can present the results in a beautiful and useful tabular reporting framework.

Let's take a Polars DataFrame and validate it against a set of constraints. We do that using the `pb.Validate` class and its collection of validation methods:

```python
import pointblank as pb

v = (
    pb.Validate(data=pb.load_dataset(dataset="small_table")) # Use pb.Validate to start
    .col_vals_gt(columns="d", value=100)       # STEP 1 |
    .col_vals_le(columns="c", value=5)         # STEP 2 | <-- Use methods to build a validation plan
    .col_exists(columns=["date", "date_time"]) # STEP 3 |
    .interrogate() # This will query the data by running all validation steps
)

v.get_tabular_report()
```

<img src="images/pointblank-tabular-report.png" alt="Validation Report">

## Features

- **Declarative Syntax**: Define your data validation rules using a declarative syntax.
- **Flexible**: We support tables from Polars, Pandas, Duckdb, MySQL, PostgreSQL, SQLite, and Parquet without any changes to your code.
- **Beautiful Reports**: Generate beautiful HTML reports of your data validation results.
- **Functional Output**: Get JSON output of your data validation results for further processing.
- **Data Testing**: Write tests for your data and use them in your notebooks or testing framework.
- **Easy to Use**: Get started quickly with a simple API and clear documentation.
- **Powerful**: You can develop complex data validation rules with fleixble options for customization.

## Installation

You can install `pointblank` using pip:

```bash
pip install pointblank
```
