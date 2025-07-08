# CAMELS-CH Static Attributes Merged Dataset

## Overview

The static attributes for all 331 CAMELS-CH catchments have been merged into a single parquet file located at:

```bash
data/camels_ch/static_attributes_merged.parquet
```

## Structure

- **331 rows**: One per catchment (gauge_id)
- **331 columns**: Including gauge_id and 330 attribute columns
- **File size**: 0.44 MB (compressed parquet)

## Column Naming Conventions

### Suffixes

- **_sim**: Simulation-based attributes (from `simulation_based/` folder)
  - Example: `p_mean_sim`, `pet_mean_sim` (simulated climate data)
- **_supplement**: Supplementary topographic attributes
  - Example: `elevation_mean_supplement`

### Prefixes

- **sg_**: SoilGrid supplement attributes (already present in source)
  - Example: `sg_sand_perc`, `sg_silt_perc`

### No suffix/prefix

- Original observational attributes from the main files
- Example: `p_mean`, `geo_porosity`, `glac_area`

## Data Sources

The merged dataset combines 14 CSV files:

### Main Attributes (9 files)

1. Climate (observed)
2. Geology
3. Glacier
4. Human influence
5. Hydrogeology
6. Hydrology (observed)
7. Landcover
8. Soil
9. Topographic

### Simulation-based (2 files)

1. Climate (simulated)
2. Hydrology (simulated)

### Supplements (3 files)

1. Geology supplement
2. Soil supplement (SoilGrid data)
3. Topographic supplement

## Usage Example

```python
import polars as pl

# Load the merged dataset
df = pl.read_parquet("data/camels_ch/static_attributes_merged.parquet")

# Example: Get all attributes for a specific catchment
catchment_2125 = df.filter(pl.col("gauge_id") == 2125)

# Example: Compare observed vs simulated precipitation
precip_comparison = df.select([
    "gauge_id",
    "p_mean",      # Observed
    "p_mean_sim"   # Simulated
])

# Example: Get all soil attributes (main + supplement)
soil_cols = [col for col in df.columns if "soil" in col.lower() or col.startswith("sg_")]
soil_data = df.select(["gauge_id"] + soil_cols)
```

## Benefits

1. **Single source of truth**: All static attributes in one file
2. **Efficient access**: No need to join multiple files
3. **Clear naming**: Suffixes/prefixes indicate data source
4. **Type-safe**: Parquet preserves data types
5. **Compressed**: ~44% of the size of combined CSVs
