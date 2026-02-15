# PotSim Data Processing API Documentation

## Dependencies
- polars >= 1.10
- pandas >= 2.0
- numpy >= 2.0

<br/>

## Usage
Copy the *potsimprocessor.py* and *metadata.json* on the project folder.
```python
import potsimloader as psl
```
The pip library is going to come soon allowing easy installation.

<br/>

## Type Definitions

```python
DataFrameType = Union[pl.DataFrame, pl.LazyFrame, pd.DataFrame]
PathType = Union[str, Path]
```

<br/>

## Core Functions

### read_data

```python
def read_data(
    dataset_path: PathType,
    weather_path: Optional[PathType] = None,
    usecols: Optional[List[str]] = None,
    scenarios: Optional[Dict[str, Any]] = None,
    lazy: bool = False,
    as_pandas: bool = False
) -> DataFrameType
```

Reads and processes data from files with optional filtering and column selection.

**Parameters:**
- `dataset_path`: Path to the main dataset file (`.csv` or `.parquet`)
- `weather_path`: Optional path to weather data file
- `usecols`: List of column names to select
- `scenarios`: Dictionary of scenario filters to apply
- `lazy`: If True, returns a LazyFrame for memory-efficient processing
- `as_pandas`: If True, converts output to Pandas DataFrame

**Returns:**
- Processed data in the specified format

**Raises:**
- `FileNotFoundError`: If input files don't exist
- `ValueError`: If specified columns are not found
- `ValueError`: If weather data doesn't contain required joining columns

---

### apply_filter

```python
def apply_filter(
    data: DataFrameType,
    filters: Dict[str, Any],
    lazy: bool = False,
    as_pandas: bool = False
) -> DataFrameType
```

Filters the dataset based on column values.

**Parameters:**
- `data`: Input dataset
- `filters`: Dictionary mapping column names to filter values
- `lazy`: If True, returns a LazyFrame
- `as_pandas`: If True, converts output to Pandas DataFrame

**Returns:**
- Filtered dataset in the specified format

**Raises:**
- `ValueError`: If filter columns are not found in data

---

### add_scenarios

```python
def add_scenarios(
    to_data: DataFrameType,
    scenarios: Dict[str, Any],
    from_data: Optional[Union[DataFrameType, PathType]] = None,
    lazy: bool = False,
    as_pandas: bool = False
) -> DataFrameType
```

Adds new scenarios to an existing dataset.

**Parameters:**
- `to_data`: Target dataset
- `scenarios`: Dictionary of new scenarios to add
- `from_data`: Source data for new scenarios
- `lazy`: If True, returns a LazyFrame
- `as_pandas`: If True, converts output to Pandas DataFrame

**Returns:**
- Updated dataset with new scenarios

**Raises:**
- `ValueError`: If scenario columns are invalid

---

### add_features

```python
def add_features(
    to_data: DataFrameType,
    features: List[str],
    from_data: Optional[Union[DataFrameType, PathType]] = None,
    lazy: bool = False,
    as_pandas: bool = False
) -> DataFrameType
```

Adds new features to an existing dataset.

**Parameters:**
- `to_data`: Target dataset
- `features`: List of feature names to add
- `from_data`: Source data for new features
- `lazy`: If True, returns a LazyFrame
- `as_pandas`: If True, converts output to Pandas DataFrame

**Returns:**
- Updated dataset with new features

**Raises:**
- `ValueError`: If requested features are not found in source data

---

### add_data

```python
def add_data(
    to_data: DataFrameType,
    from_data: Optional[Union[DataFrameType, PathType]] = None,
    features: Optional[List[str]] = None,
    scenarios: Optional[Dict[str, Any]] = None,
    lazy: bool = False,
    as_pandas: bool = False
) -> DataFrameType
```

Combines functionality of `add_features` and `add_scenarios` to add both to an existing dataset.

**Parameters:**
- `to_data`: Target dataset
- `from_data`: Source data
- `features`: List of features to add
- `scenarios`: Dictionary of scenarios to add
- `lazy`: If True, returns a LazyFrame
- `as_pandas`: If True, converts output to Pandas DataFrame

**Returns:**
- Updated dataset with new features and scenarios

**Raises:**
- `ValueError`: If neither features nor scenarios are provided

<br/>

## Utility Functions

### get_schema

```python
def get_schema(data: DataFrameType) -> Dict[str, str]
```

Returns the schema of the dataset.

**Parameters:**
- `data`: Input dataset

**Returns:**
- Dictionary mapping column names to their data types

---

### get_metadata

```python
def get_metadata(data: DataFrameType, all: bool = False) -> pd.DataFrame
```

Retrieves metadata information for columns in the dataset.

**Parameters:**
- `data`: Input dataset
- `all`: If True, returns metadata for all columns; if False, only for existing columns

**Returns:**
- Pandas DataFrame containing column metadata (name, type, description)

---

### get_current_scenarios

```python
def get_current_scenarios(data: DataFrameType) -> Dict[str, List]
```

Returns unique values for each scenario column in the dataset.

**Parameters:**
- `data`: Input dataset

**Returns:**
- Dictionary mapping scenario column names to their unique values

---

### get_memory_usage

```python
def get_memory_usage(
    data: Union[pl.DataFrame, pd.DataFrame],
    unit: str = 'b'
) -> Union[int, float]
```

Calculates memory usage of the dataset.

**Parameters:**
- `data`: Input DataFrame (Polars or Pandas)
- `unit`: Memory unit ('b', 'kb', 'mb', 'gb', 'tb')

**Returns:**
- Memory usage in specified unit

**Raises:**
- `ValueError`: If invalid unit is specified
- `TypeError`: If input is not a DataFrame

<br/>

## Scenario Columns

The following columns are recognized as scenario columns:
- Year
- PlantingDay
- Treatment
- NFirstApp
- IrrgDep
- IrrgThresh

<br/>

## Supported File Formats

- CSV (.csv)
- Parquet (.parquet)

<br/>

## Global Cache Variables

```python
_POTSIM_PATH: Optional[Path] = None
_POTSIM_FRAME: Optional[pl.LazyFrame] = None
```

These variables store the path and lazy object of the last loaded dataset for reuse in subsequent operations.
