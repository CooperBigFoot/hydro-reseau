"""Test suite for CAMELS-CH data interface module.

Tests cover the main functionality of the CAMELSCHDataset class including
data retrieval, error handling, and edge cases.
"""

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from hydro_reseau.data_utils.data_interface import CAMELSCHDataset


class TestCAMELSCHDataset:
    """Test suite for CAMELSCHDataset class."""

    @pytest.fixture
    def mock_data_dir(self, tmp_path: Path) -> Path:
        """Create a mock data directory structure for testing."""
        # Create directory structure
        data_dir = tmp_path / "camels_ch"
        time_series_dir = data_dir / "time_series"
        obs_dir = time_series_dir / "type=observation"
        sim_dir = time_series_dir / "type=simulation"

        # Create directories
        obs_dir.mkdir(parents=True)
        sim_dir.mkdir(parents=True)

        # Create sample static attributes file
        static_df = pl.DataFrame(
            {
                "gauge_id": [2125, 2126, 2127],
                "area": [100.5, 200.3, 150.7],
                "p_mean": [1200.0, 1100.0, 1300.0],
                "p_mean_sim": [1210.0, 1090.0, 1310.0],
                "geo_porosity": [0.15, 0.20, 0.18],
                "soil_depth": [1.5, 2.0, 1.8],
            }
        )
        static_df.write_parquet(data_dir / "static_attributes_merged.parquet")

        # Create sample time series files
        dates = pl.date_range(date(2000, 1, 1), date(2000, 12, 31), interval="1d", eager=True).alias("date")

        # Observation data
        for catchment_id in [2125, 2126, 2127]:
            obs_df = pl.DataFrame(
                {
                    "date": dates,
                    "discharge_vol(m3/s)": [10.0 + i * 0.1 for i in range(len(dates))],
                    "precipitation(mm/d)": [5.0 + i * 0.05 for i in range(len(dates))],
                    "temperature_mean(degC)": [15.0 + i * 0.01 for i in range(len(dates))],
                }
            )
            catchment_dir = obs_dir / f"catchment_id={catchment_id}"
            catchment_dir.mkdir()
            obs_df.write_parquet(catchment_dir / "ts_data.parquet")

        # Simulation data
        for catchment_id in [2125, 2126, 2127]:
            sim_df = pl.DataFrame(
                {
                    "date": dates,
                    "discharge_vol_sim(m3/s)": [10.5 + i * 0.1 for i in range(len(dates))],
                    "precipitation_sim(mm/d)": [5.2 + i * 0.05 for i in range(len(dates))],
                    "temperature_sim(degC)": [15.2 + i * 0.01 for i in range(len(dates))],
                }
            )
            catchment_dir = sim_dir / f"catchment_id={catchment_id}"
            catchment_dir.mkdir()
            sim_df.write_parquet(catchment_dir / "ts_data.parquet")

        return data_dir

    @pytest.fixture
    def dataset(self, mock_data_dir: Path) -> CAMELSCHDataset:
        """Create a CAMELSCHDataset instance with mock data."""
        return CAMELSCHDataset(data_dir=mock_data_dir)

    def test_initialization_with_valid_directory(self, mock_data_dir: Path) -> None:
        """Test that dataset initializes correctly with valid directory."""
        dataset = CAMELSCHDataset(data_dir=mock_data_dir)
        assert dataset._data_dir == mock_data_dir

    def test_initialization_with_missing_directory(self, tmp_path: Path) -> None:
        """Test that initialization fails with missing directory."""
        missing_dir = tmp_path / "nonexistent"
        with pytest.raises(FileNotFoundError, match="Data directory not found"):
            CAMELSCHDataset(data_dir=missing_dir)

    def test_initialization_with_invalid_structure(self, tmp_path: Path) -> None:
        """Test that initialization fails with invalid directory structure."""
        # Create directory but without expected structure
        invalid_dir = tmp_path / "invalid"
        invalid_dir.mkdir()

        with pytest.raises(ValueError, match="Time series directory not found"):
            CAMELSCHDataset(data_dir=invalid_dir)

    def test_list_catchments(self, dataset: CAMELSCHDataset) -> None:
        """Test listing available catchments."""
        catchments = dataset.list_catchments()
        assert catchments == [2125, 2126, 2127]
        assert all(isinstance(c, int) for c in catchments)

    def test_list_timeseries_variables(self, dataset: CAMELSCHDataset) -> None:
        """Test listing time series variables."""
        # Test observation variables
        obs_vars = dataset.list_timeseries_variables("observation")
        assert "discharge_vol(m3/s)" in obs_vars
        assert "precipitation(mm/d)" in obs_vars
        assert "temperature_mean(degC)" in obs_vars

        # Test simulation variables
        sim_vars = dataset.list_timeseries_variables("simulation")
        assert "discharge_vol_sim(m3/s)" in sim_vars
        assert "precipitation_sim(mm/d)" in sim_vars
        assert "temperature_sim(degC)" in sim_vars

    def test_list_static_attributes(self, dataset: CAMELSCHDataset) -> None:
        """Test listing static attributes."""
        attrs = dataset.list_static_attributes()
        assert "area" in attrs
        assert "p_mean" in attrs
        assert "p_mean_sim" in attrs
        assert "geo_porosity" in attrs
        assert "gauge_id" not in attrs  # Should be excluded

    def test_validate_catchments(self, dataset: CAMELSCHDataset) -> None:
        """Test catchment validation."""
        # Test with valid catchments
        valid = dataset.validate_catchments([2125, 2126])
        assert valid == [2125, 2126]

        # Test with mix of valid and invalid
        mixed = dataset.validate_catchments([2125, 9999, 2126])
        assert mixed == [2125, 2126]

        # Test with single catchment
        single = dataset.validate_catchments(2125)
        assert single == [2125]

    def test_get_date_range(self, dataset: CAMELSCHDataset) -> None:
        """Test getting date range for a catchment."""
        start, end = dataset.get_date_range(2125)
        assert start == "2000-01-01"
        assert end == "2000-12-31"

        # Test with invalid catchment
        with pytest.raises(ValueError, match="Invalid catchment ID"):
            dataset.get_date_range(9999)

    def test_get_timeseries_single_catchment(self, dataset: CAMELSCHDataset) -> None:
        """Test retrieving time series for a single catchment."""
        ts = dataset.get_timeseries(2125, data_type="observation")

        assert isinstance(ts, pl.DataFrame)
        assert "date" in ts.columns
        assert "catchment_id" in ts.columns
        assert ts["catchment_id"].unique().to_list() == [2125]
        assert len(ts) == 366  # Leap year 2000

    def test_get_timeseries_multiple_catchments(self, dataset: CAMELSCHDataset) -> None:
        """Test retrieving time series for multiple catchments."""
        ts = dataset.get_timeseries([2125, 2126], data_type="observation")

        assert isinstance(ts, pl.DataFrame)
        assert sorted(ts["catchment_id"].unique().to_list()) == [2125, 2126]
        assert len(ts) == 366 * 2  # Two catchments

    def test_get_timeseries_with_date_filter(self, dataset: CAMELSCHDataset) -> None:
        """Test retrieving time series with date filtering."""
        ts = dataset.get_timeseries(2125, start_date="2000-06-01", end_date="2000-06-30")

        assert len(ts) == 30  # June has 30 days
        assert ts["date"].min() == date(2000, 6, 1)
        assert ts["date"].max() == date(2000, 6, 30)

    def test_get_timeseries_with_variable_selection(self, dataset: CAMELSCHDataset) -> None:
        """Test retrieving specific variables from time series."""
        ts = dataset.get_timeseries(2125, variables=["discharge_vol", "precipitation"])

        # Should have date, catchment_id, and the requested variables
        assert len(ts.columns) == 4
        assert "date" in ts.columns
        assert "catchment_id" in ts.columns
        assert any("discharge_vol" in col for col in ts.columns)
        assert any("precipitation" in col for col in ts.columns)

    def test_get_timeseries_lazy(self, dataset: CAMELSCHDataset) -> None:
        """Test retrieving time series as LazyFrame."""
        lazy_ts = dataset.get_timeseries(2125, lazy=True)

        assert isinstance(lazy_ts, pl.LazyFrame)
        # Collect to verify it works
        ts = lazy_ts.collect()
        assert isinstance(ts, pl.DataFrame)
        assert len(ts) > 0

    def test_get_timeseries_invalid_catchment(self, dataset: CAMELSCHDataset) -> None:
        """Test error handling for invalid catchment IDs."""
        with pytest.raises(ValueError, match="Invalid catchment IDs"):
            dataset.get_timeseries(9999)

    def test_get_timeseries_invalid_variable(self, dataset: CAMELSCHDataset) -> None:
        """Test error handling for invalid variable names."""
        with pytest.raises(ValueError, match="Variable .* not found"):
            dataset.get_timeseries(2125, variables=["invalid_var"])

    def test_get_static_attributes_all(self, dataset: CAMELSCHDataset) -> None:
        """Test retrieving all static attributes."""
        attrs = dataset.get_static_attributes()

        assert isinstance(attrs, pl.DataFrame)
        assert len(attrs) == 3  # Three catchments
        assert "gauge_id" in attrs.columns
        assert "area" in attrs.columns
        assert "p_mean" in attrs.columns
        assert "p_mean_sim" in attrs.columns

    def test_get_static_attributes_specific_catchments(self, dataset: CAMELSCHDataset) -> None:
        """Test retrieving static attributes for specific catchments."""
        attrs = dataset.get_static_attributes(catchment_ids=[2125, 2127])

        assert len(attrs) == 2
        assert sorted(attrs["gauge_id"].to_list()) == [2125, 2127]

    def test_get_static_attributes_with_patterns(self, dataset: CAMELSCHDataset) -> None:
        """Test retrieving static attributes with column patterns."""
        attrs = dataset.get_static_attributes(attribute_patterns=["p_*", "geo_*"])

        columns = attrs.columns
        assert "gauge_id" in columns
        assert "p_mean" in columns
        assert "p_mean_sim" in columns
        assert "geo_porosity" in columns
        assert "area" not in columns  # Doesn't match patterns
        assert "soil_depth" not in columns

    def test_get_static_attributes_exclude_simulated(self, dataset: CAMELSCHDataset) -> None:
        """Test excluding simulated attributes."""
        attrs = dataset.get_static_attributes(include_simulated=False)

        columns = attrs.columns
        assert "p_mean" in columns
        assert "p_mean_sim" not in columns  # Should be excluded

    def test_get_static_attributes_invalid_catchment(self, dataset: CAMELSCHDataset) -> None:
        """Test error handling for invalid catchment IDs in static attributes."""
        with pytest.raises(ValueError, match="Invalid catchment IDs"):
            dataset.get_static_attributes(catchment_ids=[9999])

    def test_get_static_attributes_no_matching_patterns(self, dataset: CAMELSCHDataset) -> None:
        """Test error handling when no columns match patterns."""
        with pytest.raises(ValueError, match="No columns matched"):
            dataset.get_static_attributes(
                attribute_patterns=["xyz_*"],  # Non-existent pattern
                include_simulated=False,
            )

    def test_simulation_data_retrieval(self, dataset: CAMELSCHDataset) -> None:
        """Test retrieving simulation data."""
        sim_ts = dataset.get_timeseries(2125, data_type="simulation")

        assert isinstance(sim_ts, pl.DataFrame)
        assert any("_sim" in col for col in sim_ts.columns)
        assert len(sim_ts) == 366

    def test_caching_behavior(self, dataset: CAMELSCHDataset) -> None:
        """Test that metadata is cached after first access."""
        # First call should populate cache
        catchments1 = dataset.list_catchments()
        assert dataset._catchment_ids is not None

        # Second call should use cache
        catchments2 = dataset.list_catchments()
        assert catchments1 == catchments2

        # Same for static attributes
        attrs1 = dataset.list_static_attributes()
        assert dataset._static_attrs_columns is not None
        attrs2 = dataset.list_static_attributes()
        assert attrs1 == attrs2
