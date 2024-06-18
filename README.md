# Stockstat

Stockstat is an application designed to simplify company analysis and investing for beginners. By providing user-friendly tools and insights, Stockstat aims to empower those new to the investing world to make informed decisions and build their investment portfolios with confidence.

## Usage

This project is a hybrid. It is centered around cleaning, parsing, and filling in missing entries in the EDGAR dataset. In one part, the data is used to train a prediction model, and in the second part, it helps with backtesting investment strategies.

## Files Description

### `extract.ipynb`
- **Purpose**: Transforms raw data into a clean dataset ready for analysis.
- **Processes**: Includes data extraction, cleaning, and preprocessing.
- **Dependencies**: Utilizes transformation functions from `extract_functions.py`, cleaning function from `cleanup.py` and reshapes data using `reshape.py`

### `infer.ipynb`
- **Purpose**: To train the prediction model
- **Processes**: Performs feature selection, parameter optimization, and model training.
- **Dependencies**: Employs functions from `infer_functions.py`.

### `cleanup.py`
- **Purpose**: Provides data cleaning functions.
- **Functionality**: Selects, cleans, and labels data in CSV format for analysis readiness.

### `conversions.py`
- **Purpose**: Handles conversion of features between each other.
- **Functionality**: Includes conversion charts to estimate unreported data based on available features, crucial for machine learning applications.

### `net_stuff.py`
- **Purpose**: Manages network operations.
- **Functionality**: Implements timeout and retry mechanisms fo existing functions from API libraries.

### `reshape.py`
- **Purpose**: Reshapes the data from json format to a parsed dictionary readable by the dataloader in `extract.ipynb`
- **Functionality**: Employs a greedy search algorithm to pick the data we need.

### `utils.ipynb`
- **Purpose**: Aids in miscellaneous testing and utilities.
- **Functionality**: Provides various utility functions and testing tools for development use.

## Active Development Phase

These notebooks are not meant to be used by end-users. An interface will be created for simplified data extraction, aimed at enhancing user experience and accessibility.
