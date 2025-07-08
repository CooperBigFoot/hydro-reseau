# CAMELS-CH Hive Partitioned Data

## Structure

The CAMELS-CH time series data is organized using hive partitioning:

```bash
data/CAMELS_CH/time_series/
├── type=simulation/
│   ├── catchment_id=2004/
│   │   └── ts_data.parquet
│   ├── catchment_id=2007/
│   │   └── ts_data.parquet
│   └── ... (331 catchments total)
└── type=observation/
    ├── catchment_id=2004/
    │   └── ts_data.parquet
    ├── catchment_id=2007/
    │   └── ts_data.parquet
    └── ... (331 catchments total)
```

## Usage

### Read specific catchment

```python
import polars as pl

df = pl.read_parquet("data/CAMELS_CH/time_series/type=simulation/catchment_id=2125/ts_data.parquet")
```

### Scan all data efficiently

```python
# Scan only simulation data
lazy_df = pl.scan_parquet("data/CAMELS_CH/time_series/type=simulation/**/ts_data.parquet")

# Filter and collect
result = lazy_df.filter(pl.col("date") >= "2020-01-01").collect()
```

## Benefits

- **Fast filtering**: Query by type or catchment without reading all data
- **Storage efficient**: ~41% compression vs CSV
- **Type safe**: Parquet preserves data types
- **Memory efficient**: Use lazy evaluation with `scan_parquet()`
