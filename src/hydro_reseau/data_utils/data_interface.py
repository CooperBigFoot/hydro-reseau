"""CAMELS-CH data interface for efficient access to time series and static attributes.

This module provides a unified interface to access CAMELS-CH hydrological data,
including both time series (observations and simulations) and static catchment
attributes. All data is returned as Polars DataFrames for efficient processing.
"""

import logging
from pathlib import Path
from typing import Literal

import polars as pl

logger = logging.getLogger(__name__)


class CAMELSCHDataset:
    """Interface for accessing CAMELS-CH hydrological data.

    This class provides methods to efficiently query and retrieve time series
    data (observations and simulations) and static catchment attributes from
    the CAMELS-CH dataset stored in Parquet format with Hive partitioning.

    Attributes:
        data_dir: Path to the CAMELS-CH data directory.

    Example:
        >>> dataset = CAMELSCHDataset("data/camels_ch")
        >>> ts = dataset.get_timeseries([2125], start_date="2000-01-01")
        >>> attrs = dataset.get_static_attributes([2125])
    """

    def __init__(self, data_dir: str | Path | None = None) -> None:
        """Initialize the CAMELS-CH dataset interface.

        Args:
            data_dir: Path to the CAMELS-CH data directory. If None,
                     defaults to "data/camels_ch".

        Raises:
            FileNotFoundError: If the specified data directory doesn't exist.
            ValueError: If the data directory structure is invalid.
        """
        self._data_dir = Path(data_dir or "data/camels_ch")
        self._validate_data_directory()

        # Cache for metadata
        self._catchment_ids: list[int] | None = None
        self._static_attrs_columns: list[str] | None = None

        logger.info(f"Initialized CAMELSCHDataset with data_dir: {self._data_dir}")

    def _validate_data_directory(self) -> None:
        """Validate that the data directory exists and has the expected structure.

        Raises:
            FileNotFoundError: If the data directory doesn't exist.
            ValueError: If the expected subdirectories are missing.
        """
        if not self._data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self._data_dir}")

        # Check for expected subdirectories
        time_series_dir = self._data_dir / "time_series"
        static_attrs_file = self._data_dir / "static_attributes_merged.parquet"

        if not time_series_dir.exists():
            raise ValueError(f"Time series directory not found: {time_series_dir}")

        if not static_attrs_file.exists():
            raise ValueError(f"Static attributes file not found: {static_attrs_file}")

        # Check for observation and simulation subdirectories
        obs_dir = time_series_dir / "type=observation"
        sim_dir = time_series_dir / "type=simulation"

        if not obs_dir.exists() or not sim_dir.exists():
            raise ValueError("Missing observation or simulation data directories")

    def get_timeseries(
        self,
        catchment_ids: int | list[int],
        data_type: Literal["observation", "simulation"] = "observation",
        start_date: str | None = None,
        end_date: str | None = None,
        variables: list[str] | None = None,
        lazy: bool = False,
    ) -> pl.DataFrame | pl.LazyFrame:
        """Retrieve time series data for specified catchments.

        Fetches time series data from the Hive-partitioned Parquet files,
        with optional filtering by date range and variables.

        Args:
            catchment_ids: Single catchment ID or list of catchment IDs.
            data_type: Type of data to retrieve ("observation" or "simulation").
            start_date: Start date in ISO format (YYYY-MM-DD). If None, includes
                       all available data from the beginning.
            end_date: End date in ISO format (YYYY-MM-DD). If None, includes
                     all available data until the end.
            variables: List of variable names to retrieve. If None, returns all
                      available variables for the specified data type.
            lazy: If True, returns a LazyFrame for deferred execution.

        Returns:
            Polars DataFrame or LazyFrame containing the requested time series
            data with columns: date, catchment_id, and requested variables.

        Raises:
            ValueError: If catchment IDs are invalid, date format is incorrect,
                       or requested variables don't exist.
            FileNotFoundError: If data files for specified catchments are missing.

        Example:
            >>> # Get discharge and precipitation for two catchments
            >>> ts = dataset.get_timeseries(
            ...     catchment_ids=[2125, 2126],
            ...     data_type="observation",
            ...     start_date="2000-01-01",
            ...     end_date="2010-12-31",
            ...     variables=["discharge_vol", "precipitation"]
            ... )
        """
        # Normalize catchment_ids to list
        if isinstance(catchment_ids, int):
            catchment_ids = [catchment_ids]

        # Validate catchment IDs
        valid_ids = self.validate_catchments(catchment_ids)
        if len(valid_ids) != len(catchment_ids):
            invalid_ids = set(catchment_ids) - set(valid_ids)
            raise ValueError(f"Invalid catchment IDs: {invalid_ids}")

        # Build paths for requested catchments
        paths = []
        for catchment_id in catchment_ids:
            path = (
                self._data_dir
                / "time_series"
                / f"type={data_type}"
                / f"catchment_id={catchment_id}"
                / "ts_data.parquet"
            )
            if not path.exists():
                raise FileNotFoundError(f"Data file not found for catchment {catchment_id}: {path}")
            paths.append(str(path))

        logger.debug(f"Reading time series from {len(paths)} files")

        # Read data and add catchment_id column
        if lazy:
            # For lazy evaluation, read each file separately and add catchment_id
            lazy_frames = []
            for catchment_id, path in zip(catchment_ids, paths, strict=False):
                lf = pl.scan_parquet(path).with_columns(
                    pl.lit(catchment_id).alias("catchment_id")
                )
                lazy_frames.append(lf)
            df = pl.concat(lazy_frames)
            # Convert date column after concatenation
            df = df.with_columns(pl.col("date").str.to_date())
        else:
            # For eager evaluation, read and concatenate
            dfs = []
            for catchment_id, path in zip(catchment_ids, paths, strict=False):
                catchment_df = pl.read_parquet(path)
                # Add catchment_id only, don't convert date yet
                catchment_df = catchment_df.with_columns(
                    pl.lit(catchment_id).alias("catchment_id")
                )
                dfs.append(catchment_df)
            df = pl.concat(dfs)
            # Convert date column after concatenation
            df = df.with_columns(pl.col("date").str.to_date())

        # Filter by date range if specified
        if start_date or end_date:
            conditions = []
            if start_date:
                conditions.append(pl.col("date") >= pl.lit(start_date).str.to_date())
            if end_date:
                conditions.append(pl.col("date") <= pl.lit(end_date).str.to_date())

            if conditions:
                df = df.filter(pl.all_horizontal(conditions))

        # Select specific variables if requested
        if variables is not None:
            # Always include date and catchment_id
            columns_to_select = ["date", "catchment_id"]

            # Validate and add requested variables
            available_vars = self.list_timeseries_variables(data_type)
            for var in variables:
                # Handle variables with units in parentheses
                matching_cols = [col for col in available_vars if col.startswith(var)]
                if not matching_cols:
                    raise ValueError(
                        f"Variable '{var}' not found in {data_type} data. Available variables: {available_vars}"
                    )
                columns_to_select.extend(matching_cols)

            df = df.select(columns_to_select)

        # Sort by catchment and date for consistent output
        df = df.sort(["catchment_id", "date"])

        logger.info(f"Retrieved time series data for {len(catchment_ids)} catchments")

        return df

    def get_static_attributes(
        self,
        catchment_ids: int | list[int] | None = None,
        attribute_patterns: list[str] | None = None,
        include_simulated: bool = True,
    ) -> pl.DataFrame:
        """Retrieve static attributes for specified catchments.

        Fetches static catchment attributes from the merged Parquet file,
        with optional filtering by catchment IDs and attribute patterns.

        Args:
            catchment_ids: Single ID, list of IDs, or None for all catchments.
            attribute_patterns: List of column patterns to match (e.g., ["p_*", "geo_*"]).
                              Patterns support wildcards (*). If None, returns all attributes.
            include_simulated: Whether to include simulated attributes (columns ending
                             with '_sim'). Default is True.

        Returns:
            Polars DataFrame with requested static attributes.

        Raises:
            ValueError: If catchment IDs are invalid or patterns match no columns.

        Example:
            >>> # Get precipitation and geology attributes for specific catchments
            >>> attrs = dataset.get_static_attributes(
            ...     catchment_ids=[2125, 2126],
            ...     attribute_patterns=["p_*", "geo_*"],
            ...     include_simulated=False
            ... )
        """
        # Read the static attributes file
        static_attrs_path = self._data_dir / "static_attributes_merged.parquet"
        df = pl.read_parquet(static_attrs_path)

        # Filter by catchment IDs if specified
        if catchment_ids is not None:
            if isinstance(catchment_ids, int):
                catchment_ids = [catchment_ids]

            # Validate catchment IDs
            valid_ids = self.validate_catchments(catchment_ids)
            if len(valid_ids) != len(catchment_ids):
                invalid_ids = set(catchment_ids) - set(valid_ids)
                raise ValueError(f"Invalid catchment IDs: {invalid_ids}")

            df = df.filter(pl.col("gauge_id").is_in(catchment_ids))

        # Select columns based on patterns
        if attribute_patterns is not None or not include_simulated:
            columns_to_select = ["gauge_id"]  # Always include gauge_id
            all_columns = df.columns

            if attribute_patterns is not None:
                # Match patterns
                for pattern in attribute_patterns:
                    if "*" in pattern:
                        # Convert wildcard pattern to regex
                        import re

                        regex_pattern = pattern.replace("*", ".*")
                        regex_pattern = f"^{regex_pattern}$"
                        matching_cols = [
                            col for col in all_columns if re.match(regex_pattern, col) and col != "gauge_id"
                        ]
                    else:
                        # Exact match
                        matching_cols = [col for col in all_columns if col == pattern and col != "gauge_id"]

                    columns_to_select.extend(matching_cols)
            else:
                # If no patterns specified, include all columns
                columns_to_select = all_columns

            # Filter out simulated columns if requested
            if not include_simulated:
                columns_to_select = [col for col in columns_to_select if not col.endswith("_sim")]

            # Ensure we have at least one attribute column
            if len(columns_to_select) == 1:  # Only gauge_id
                raise ValueError("No columns matched the specified patterns and filters")

            # Remove duplicates while preserving order
            seen = set()
            columns_to_select = [col for col in columns_to_select if not (col in seen or seen.add(col))]

            df = df.select(columns_to_select)

        logger.info(f"Retrieved {df.width - 1} attributes for {df.height} catchments")

        return df

    def list_catchments(self) -> list[int]:
        """List all available catchment IDs in the dataset.

        Returns:
            Sorted list of catchment IDs.

        Example:
            >>> catchments = dataset.list_catchments()
            >>> print(f"Dataset contains {len(catchments)} catchments")
        """
        if self._catchment_ids is None:
            # Read from static attributes file for complete list
            static_attrs_path = self._data_dir / "static_attributes_merged.parquet"
            df = pl.read_parquet(static_attrs_path, columns=["gauge_id"])
            self._catchment_ids = sorted(df["gauge_id"].to_list())
            logger.debug(f"Loaded {len(self._catchment_ids)} catchment IDs")

        return self._catchment_ids

    def list_timeseries_variables(self, data_type: Literal["observation", "simulation"]) -> list[str]:
        """List available time series variables for the specified data type.

        Args:
            data_type: Type of data ("observation" or "simulation").

        Returns:
            List of variable names available in the time series data.

        Example:
            >>> obs_vars = dataset.list_timeseries_variables("observation")
            >>> print(f"Observation variables: {obs_vars}")
        """
        # Read a sample file to get column names
        catchments = self.list_catchments()
        if not catchments:
            return []

        sample_path = (
            self._data_dir / "time_series" / f"type={data_type}" / f"catchment_id={catchments[0]}" / "ts_data.parquet"
        )

        if not sample_path.exists():
            logger.warning(f"Sample file not found: {sample_path}")
            return []

        # Read just the schema
        df = pl.read_parquet(sample_path, n_rows=1)
        variables = [col for col in df.columns if col != "date"]

        logger.debug(f"Found {len(variables)} variables for {data_type} data")
        return variables

    def list_static_attributes(self) -> list[str]:
        """List all available static attribute columns.

        Returns:
            List of attribute column names (excluding gauge_id).

        Example:
            >>> attrs = dataset.list_static_attributes()
            >>> climate_attrs = [a for a in attrs if a.startswith("p_")]
        """
        if self._static_attrs_columns is None:
            static_attrs_path = self._data_dir / "static_attributes_merged.parquet"
            df = pl.read_parquet(static_attrs_path, n_rows=1)
            self._static_attrs_columns = [col for col in df.columns if col != "gauge_id"]
            logger.debug(f"Found {len(self._static_attrs_columns)} static attributes")

        return self._static_attrs_columns

    def get_date_range(
        self, catchment_id: int, data_type: Literal["observation", "simulation"] = "observation"
    ) -> tuple[str, str]:
        """Get the available date range for a specific catchment.

        Args:
            catchment_id: Catchment ID to check.
            data_type: Type of data to check.

        Returns:
            Tuple of (start_date, end_date) in ISO format.

        Raises:
            ValueError: If the catchment ID is invalid.
            FileNotFoundError: If data file is missing.

        Example:
            >>> start, end = dataset.get_date_range(2125)
            >>> print(f"Data available from {start} to {end}")
        """
        # Validate catchment ID
        if catchment_id not in self.list_catchments():
            raise ValueError(f"Invalid catchment ID: {catchment_id}")

        path = self._data_dir / "time_series" / f"type={data_type}" / f"catchment_id={catchment_id}" / "ts_data.parquet"

        if not path.exists():
            raise FileNotFoundError(f"Data file not found for catchment {catchment_id}")

        # Read only the date column
        df = pl.scan_parquet(path).select("date")
        stats = df.select(pl.col("date").min().alias("start"), pl.col("date").max().alias("end")).collect()

        start_date = stats["start"][0]
        end_date = stats["end"][0]

        return str(start_date), str(end_date)

    def validate_catchments(self, catchment_ids: int | list[int]) -> list[int]:
        """Validate catchment IDs against available data.

        Args:
            catchment_ids: Single ID or list of IDs to validate.

        Returns:
            List of valid catchment IDs from the input.

        Example:
            >>> valid_ids = dataset.validate_catchments([2125, 9999, 2126])
            >>> print(f"Valid IDs: {valid_ids}")  # [2125, 2126]
        """
        if isinstance(catchment_ids, int):
            catchment_ids = [catchment_ids]

        available = set(self.list_catchments())
        valid_ids = [cid for cid in catchment_ids if cid in available]

        if len(valid_ids) < len(catchment_ids):
            invalid_ids = set(catchment_ids) - available
            logger.warning(f"Invalid catchment IDs found: {invalid_ids}")

        return valid_ids
