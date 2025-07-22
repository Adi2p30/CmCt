import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import timedelta
from multiprocessing import Pool, cpu_count

import geopandas as gpd
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import xarray as xr
from matplotlib import rc
from numba import jit, prange
from scipy import stats

from cmct.calving_modules import shapefile_utils

# from cmct.shapefile_utils import *

# from .shapefile_utils import (
#     get_nonzero_indices,
#     scaling_shape_to_target,
#     shapefile_to_xy,
# )


def safe_float_conversion(data, default_value=np.nan):
    """
    Safely convert data to float64, handling non-numeric values.

    Parameters
    ----------
    data : array-like
        Data to convert
    default_value : float, optional
        Value to use for non-numeric entries (default: np.nan)

    Returns
    -------
    numpy.ndarray
        Array converted to float64 with non-numeric values replaced
    """
    try:
        return data.astype(np.float64)
    except (ValueError, TypeError):
        result = np.full(data.shape, default_value, dtype=np.float64)
        # Try to convert element by element for mixed types
        flat_data = data.flatten()
        flat_result = result.flatten()

        for i in range(len(flat_data)):
            try:
                val = float(flat_data[i])
                if np.isfinite(val):
                    flat_result[i] = val
                else:
                    flat_result[i] = default_value
            except (ValueError, TypeError, OverflowError):
                flat_result[i] = default_value

        return flat_result.reshape(data.shape)


logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_basins(basin_filename, basins):
    basin_data = shapefile_utils.load_basin_polygons(basin_filename)
    if basins == "all":
        selected_basins = ["CW", "NE", "SE", "SW", "NO", "NW", "unassigned"]
    else:
        selected_basins = basins

    # Filter basin_data to only include selected basins
    basin_polygons = {}

    for basin_name in selected_basins:
        if basin_name in basin_data:
            basin_polygons[basin_name] = basin_data[basin_name]

    return basin_polygons, selected_basins


def load_gsfc_calving(filepath, basins=None):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    try:
        gsfc = GSFCcalving(filepath, basins)

    except Exception as error:
        print("Error: Failed to load GSFC dataset.")
        print(error)
        gsfc = None
        raise ValueError("Failed to load GSFC calving data.")

    return gsfc


def load_model_calving(filepath):
    try:
        model_res = Modelcalving(filepath)
    except Exception as error:
        logging.error("Error: Failed to load Model dataset.")
        logging.error(error)
        model_res = None
    return model_res


def load_residuals(residuals):
    if isinstance(residuals, str):
        if not os.path.exists(residuals):
            raise FileNotFoundError(f"File not found: {residuals}")
        try:
            residuals = Residual(
                xr.open_dataset(residuals, autoclose=True, engine="netcdf4")
            )
        except Exception as error:
            logging.error("Error: Failed to load residuals dataset.")
            logging.error(error)
            residuals = None

    elif isinstance(residuals, xr.Dataset):
        return Residual(residuals)


class GSFCcalving:
    def __init__(self, nc_path, basins=None):
        # Open as xarray Dataset

        self.ds = xr.open_dataset(
            nc_path, autoclose=True, engine="netcdf4", use_cftime=True
        )
        # self.ds =
        self.ds["ice_mask"] = self.ds["ice_mask"] / 100
        # Safely convert ice_mask to float64 for np.isnan compatibility
        ice_mask_data = safe_float_conversion(self.ds["ice_mask"].values)
        self.ds["ice_mask"] = (self.ds["ice_mask"].dims, ice_mask_data)
        # self.basins = basins

    # Direct access to variables as attributes
    @property
    def time(self):
        return self.ds["year"]

    @property
    def ice_mask(self):
        return self.ds["ice_mask"]

    @property
    def x(self):
        return self.ds["x"]

    @property
    def y(self):
        return self.ds["y"]

    @property
    def basins(self):
        return self.basins

    # def _set_times_as_datetimes(self, days):
    #     return np.datetime64('2002-01-01T00:00:00') + np.array([int(d*24) for d in days], dtype='timedelta64[h]')


class Modelcalving:
    def __init__(self, nc_path):
        # Open as xarray Dataset
        self.ds = xr.open_dataset(
            nc_path, autoclose=True, engine="netcdf4", use_cftime=True
        )
        self.ds["ice_mask"] = self.ds["sftgif"]
        # Safely convert ice_mask to float64 for np.isnan compatibility
        ice_mask_data = safe_float_conversion(self.ds["ice_mask"].values)
        self.ds["ice_mask"] = (self.ds["ice_mask"].dims, ice_mask_data)

        # Making the variables consistent REMOVE IF NECESSARY
        self.ds = self.ds.drop("sftgif")
        # Direct access to variables as attributes

    @property
    def x(self):
        return self.ds["x"]

    @property
    def y(self):
        return self.ds["y"]

    @property
    def lat(self):
        return self.ds["lat"]

    @property
    def lon(self):
        return self.ds["lon"]

    @property
    def ice_mask(self):
        return self.ds["ice_mask"]

    @property
    def time(self):
        return self.ds["time"]

    # def _set_times_as_datetimes(self, days):
    #     return np.datetime64('2002-01-01T00:00:00') + np.array([int(d*24) for d in days], dtype='timedelta64[h]')

    def close(self):
        self.ds.close()

    def print_info(self):
        print("GSFC Calving Data:")
        print(f"Latitude: {self.lat.values}")
        print(f"Longitude: {self.lon.values}")
        print(f"Time: {self.time.values}")
        print(f"ice_mask: {self.ice_mask.values}")
        print(f"X coordinates: {self.x.values}")
        print(f"Y coordinates: {self.y.values}")


class Residual:
    def __init__(self, residuals):
        self.ds = residuals

    @property
    def x(self):
        return self.ds["x"]

    @property
    def y(self):
        return self.ds["y"]

    @property
    def lat(self):
        return self.ds["lat"]

    @property
    def lon(self):
        return self.ds["lon"]

    @property
    def ice_mask(self):
        return self.ds["ice_mask"]

    @property
    def time(self):
        return self.ds["time"]

    def get_basin_data(self, year, basin_id=None):
        if basin_id is None:
            # Return all data for the year
            data = self.ds.sel(time=year).residual.values
        else:
            # Return data only for the specified basin
            basin_mask = self.ds.basin.sel(time=year) == basin_id
            data = self.ds.residual.sel(time=year).where(basin_mask).values

        return data

    def to_netCDF(self, output_path):
        """
        Save the residuals dataset to a NetCDF file.

        Parameters
        ----------
        output_path : str
            Path to save the NetCDF file.
        """
        self.ds.to_netcdf(output_path, mode="w", format="NETCDF4")
        logging.info(f"Residuals saved to {output_path}")

    def to_json(self, output_path):
        """
        Save the residuals dataset to a JSON file.

        Parameters
        ----------
        output_path : str
            Path to save the JSON file.
        """
        self.ds.to_dataframe().to_json(output_path)
        logging.info(f"Residuals saved to {output_path}")


def calculate_basin_statistics(residuals):
    """
    Calculate comprehensive statistics for each basin across all years.

    Parameters
    ----------
    residuals : Residual
        Residual object containing the dataset with basin assignments and residual values

    Returns
    -------
    dict
        Dictionary with structure: {year: {basin_name: {stat_name: value}}}
        Statistics include: count, mean, std, min, max, rms, rss, sum,
        winsorized_mean, outlier_weighted_mean

    """
    logger = logging.getLogger(__name__)
    logger.info("Starting basin statistics calculation")

    basin_stats = {}

    basin_names = residuals.ds.basin_names.values
    times = residuals.ds.time.values

    logger.info(f"Processing {len(times)} time steps and {len(basin_names)} basins")

    for time_idx, year in enumerate(times):
        if year not in basin_stats:
            basin_stats[year] = {}

        residual_data = residuals.ds.residual.isel(time=time_idx)
        basin_data = residuals.ds.basin.isel(time=time_idx)

        logger.debug(f"Processing year {year} (time index {time_idx})")

        for basin_idx, basin_name in enumerate(basin_names):
            basin_mask = basin_data == basin_idx

            basin_residuals = residual_data.where(basin_mask).values

            valid_data = basin_residuals[~np.isnan(basin_residuals)]

            if len(valid_data) > 0:
                logger.debug(
                    f"  Basin {basin_name}: {len(valid_data)} valid data points"
                )

                # Calculate basic statistics
                mean_val = np.mean(valid_data)
                std_val = np.std(valid_data)

                # Calculate winsorized mean (trimmed mean with 5% limits)
                zero_fraction = np.sum(valid_data == 0) / len(valid_data)

                if zero_fraction > 0.9:  # If more than 90% are zeros
                    non_zero_data = valid_data[valid_data != 0]
                    if len(non_zero_data) > 0:
                        if len(non_zero_data) > 10:
                            winsorized_nonzero = stats.mstats.winsorize(
                                non_zero_data, limits=0.05
                            )
                            winsorized_mean_nonzero = float(np.mean(winsorized_nonzero))
                        else:
                            winsorized_mean_nonzero = np.mean(non_zero_data)
                        winsorized_mean = winsorized_mean_nonzero * (1 - zero_fraction)
                    else:
                        winsorized_mean = 0.0
                else:
                    winsorized_data = stats.mstats.winsorize(valid_data, limits=0.05)
                    winsorized_mean = float(np.mean(winsorized_data))

                median_val = np.median(valid_data)

                if zero_fraction > 0.8:  # For zero-heavy data
                    # Give more reasonable weights to avoid extreme values
                    weights = 1.0 / (np.abs(valid_data) + 0.01)  # Larger epsilon
                    weights = np.minimum(weights, 100.0)
                    outlier_weighted_mean = np.average(valid_data, weights=weights)
                else:
                    weights = 1.0 / (np.abs(valid_data - median_val) + 1e-6)
                    weights = np.minimum(weights, 1000.0)
                    outlier_weighted_mean = np.average(valid_data, weights=weights)

                basin_stats[year][basin_name] = {
                    "count": len(valid_data),
                    "mean": mean_val,
                    "std": std_val,
                    "min": np.min(valid_data),
                    "max": np.max(valid_data),
                    "rms": np.sqrt(np.mean(np.square(valid_data))),
                    "rss": np.sum(np.square(valid_data)),
                    "sum": np.sum(valid_data),
                    "winsorized_mean": winsorized_mean,
                    "outlier_weighted_mean": outlier_weighted_mean,
                }
            else:
                logger.warning(f"  Basin {basin_name}: No valid data for year {year}")
                basin_stats[year][basin_name] = {
                    "count": 0,
                    "mean": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "rms": np.nan,
                    "rss": np.nan,
                    "sum": np.nan,
                    "winsorized_mean": np.nan,
                    "outlier_weighted_mean": np.nan,
                }

    logger.info("Basin statistics calculation completed")
    return basin_stats


def format_basin_stats(basin_stats):
    """
    Format basin statistics in a readable format.

    Parameters
    ----------
    basin_stats : dict
        Dictionary containing basin statistics.

    Returns
    -------
    str
        Formatted string with basin statistics.
    """
    output_lines = []

    for i in basin_stats.keys():
        output_lines.append(f"=== Statistics for Year {i} ===")
        output_lines.append(
            "Basin | Sum       | Count    | Mean        | Winsorized  | Outlier Wgt | Std        | RMS"
        )
        output_lines.append("-" * 85)

        for basin_name, basin_stat in basin_stats[i].items():
            if basin_stat["count"] > 0:
                output_lines.append(
                    f"{basin_name:5} | {basin_stat['sum']:8} | {basin_stat['count']:8} | {basin_stat['mean']:11.8f} | {basin_stat['winsorized_mean']:11.8f} | {basin_stat['outlier_weighted_mean']:11.8f} | {basin_stat['std']:10.6f} | {basin_stat['rms']:10.6f}"
                )
        output_lines.append("-" * 85)
        output_lines.append("\n")

    return "\n".join(output_lines)

    # for basin_name, stats in basin_stats[2007].items():
    #     if stats["count"] > 0:
    #         print(
    #             f"{basin_name:5} | {stats['count']:8} | {stats['mean']:11.8f} | {stats['winsorized_mean']:11.8f} | {stats['outlier_weighted_mean']:11.8f} | {stats['std']:10.6f} | {stats['rms']:10.6f}"
    #         )


def calculate_gsfc_statistics(gsfc, basin_polygons_dict):
    """
    Calculate comprehensive statistics for each basin across all years for GSFC data.

    Parameters
    ----------
    gsfc : GSFCcalving
        GSFC calving object containing the dataset with ice mask values
    basin_polygons_dict : dict
        Dictionary containing basin polygons for spatial assignment

    Returns
    -------
    dict
        Dictionary with structure: {basin_name: {year: {stat_name: value}}}
        Statistics include: count, mean, std, min, max, rms, rss, sum,
        winsorized_mean, outlier_weighted_mean
    """

    logger = logging.getLogger(__name__)
    logger.info("Starting GSFC basin statistics calculation")

    # Import the required modules for basin assignment
    from cmct.calving_modules.residual_calculation import (
        # create_basin_mask_debug,
        create_basin_mask_optimized,
        prepare_basin_polygons,
    )

    # Get times from GSFC data
    times = gsfc.time.values
    logger.info(f"Processing {len(times)} time steps")

    # Get coordinate arrays from GSFC data
    x_coords = gsfc.x.values.astype(np.float32)
    y_coords = gsfc.y.values.astype(np.float32)

    logger.info(f"GSFC grid dimensions: {len(y_coords)} x {len(x_coords)}")

    # Prepare basin data for assignment
    basin_names, basin_polygons_x, basin_polygons_y, basin_lengths = (
        prepare_basin_polygons(basin_polygons_dict)
    )

    logger.info(f"Found {len(basin_names)} basins: {basin_names}")

    # Create basin mask for GSFC coordinates
    logger.info("Creating basin assignments for GSFC coordinates...")

    basin_mask = create_basin_mask_optimized(
        x_coords, y_coords, basin_polygons_x, basin_polygons_y, basin_lengths
    )

    # Debug: Check basin assignment results
    unique_basins = np.unique(basin_mask)
    logger.info(f"Basin assignment complete. Unique basin IDs: {unique_basins}")

    for basin_id in unique_basins:
        if basin_id >= 0:
            count = np.sum(basin_mask == basin_id)
            basin_name = (
                basin_names[basin_id] if basin_id < len(basin_names) else "Unknown"
            )
            logger.info(f"  Basin {basin_id} ({basin_name}): {count} points")
        else:
            count = np.sum(basin_mask == basin_id)
            logger.info(f"  Unassigned points: {count}")

    # Initialize nested dictionary structure: {basin_name: {year: {stats}}}
    basin_stats = {}
    for basin_name in basin_names:
        basin_stats[basin_name] = {}

    # Process each time step
    for time_idx, year in enumerate(times):
        logger.debug(f"Processing year {year} (time index {time_idx})")
        ice_mask_data = gsfc.ds.ice_mask.isel(year=time_idx).values

        # Process each basin
        for basin_idx, basin_name in enumerate(basin_names):
            basin_mask_current = basin_mask == basin_idx

            basin_ice_mask = ice_mask_data[basin_mask_current]

            valid_data = basin_ice_mask[~np.isnan(basin_ice_mask)]

            if len(valid_data) > 0:
                logger.debug(
                    f"  Basin {basin_name}: {len(valid_data)} valid data points"
                )

                mean_val = np.mean(valid_data)
                std_val = np.std(valid_data)

                zero_fraction = np.sum(valid_data == 0) / len(valid_data)

                if zero_fraction > 0.9:  # If more than 90% are zeros
                    non_zero_data = valid_data[valid_data != 0]
                    if len(non_zero_data) > 0:
                        if len(non_zero_data) > 10:
                            winsorized_nonzero = stats.mstats.winsorize(
                                non_zero_data, limits=0.05
                            )
                            winsorized_mean_nonzero = float(np.mean(winsorized_nonzero))
                        else:
                            winsorized_mean_nonzero = np.mean(non_zero_data)
                        winsorized_mean = winsorized_mean_nonzero * (1 - zero_fraction)
                    else:
                        winsorized_mean = 0.0
                else:
                    winsorized_data = stats.mstats.winsorize(valid_data, limits=0.05)
                    winsorized_mean = float(np.mean(winsorized_data))

                # Calculate outlier-weighted mean
                median_val = np.median(valid_data)

                if zero_fraction > 0.8:  # For zero-heavy data
                    # Use squared distance to reduce extreme weighting

                    weights = 1.0 / (np.abs(valid_data) + 0.01)  # Larger epsilon
                    weights = np.minimum(weights, 100.0)  # Cap at 100x weight
                    outlier_weighted_mean = np.average(valid_data, weights=weights)
                else:
                    # Standard outlier weighting for more balanced data
                    weights = 1.0 / (np.abs(valid_data - median_val) + 1e-6)
                    weights = np.minimum(weights, 1000.0)
                    outlier_weighted_mean = np.average(valid_data, weights=weights)

                basin_stats[basin_name][year] = {
                    "count": len(valid_data),
                    "mean": mean_val,
                    "std": std_val,
                    "min": np.min(valid_data),
                    "max": np.max(valid_data),
                    "rms": np.sqrt(np.mean(np.square(valid_data))),
                    "rss": np.sum(np.square(valid_data)),
                    "sum": np.sum(valid_data),
                    "winsorized_mean": winsorized_mean,
                    "outlier_weighted_mean": outlier_weighted_mean,
                }
            else:
                # No valid data for this basin/year combination
                logger.warning(f"  Basin {basin_name}: No valid data for year {year}")
                basin_stats[basin_name][year] = {
                    "count": 0,
                    "mean": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "rms": np.nan,
                    "rss": np.nan,
                    "sum": np.nan,
                    "winsorized_mean": np.nan,
                    "outlier_weighted_mean": np.nan,
                }

    logger.info("GSFC basin statistics calculation completed")
    return basin_stats


def format_gsfc_basin_stats(basin_stats):
    """
    Format GSFC basin statistics in a readable format.

    Parameters
    ----------
    basin_stats : dict
        Dictionary containing GSFC basin statistics with structure:
        {basin_name: {year: {stat_name: value}}}

    Returns
    -------
    str
        Formatted string with basin statistics organized by basin and year.
    """
    output_lines = []
    output_lines.append("=== GSFC Basin Statistics ===\n")

    for basin_name, yearly_stats in basin_stats.items():
        if not yearly_stats:  # Skip basins with no data
            continue

        output_lines.append(f"Basin: {basin_name}")
        output_lines.append("-" * 100)
        output_lines.append(
            "Year | Count    | Mean        | Winsorized  | Outlier Wgt | Std        | RMS        | Sum      | Min      | Max"
        )
        output_lines.append("-" * 100)

        # Sort years for consistent output
        sorted_years = sorted(yearly_stats.keys())

        for year in sorted_years:
            stats = yearly_stats[year]
            if stats["count"] > 0:
                output_lines.append(
                    f"{year:4} | {stats['count']:8} | {stats['mean']:11.8f} | {stats['winsorized_mean']:11.8f} | {stats['outlier_weighted_mean']:11.8f} | {stats['std']:10.6f} | {stats['rms']:10.6f} | {stats['sum']:8.4f} | {stats['min']:8.4f} | {stats['max']:8.4f}"
                )
            else:
                output_lines.append(
                    f"{year:4} | {stats['count']:8} | {'N/A':>11} | {'N/A':>11} | {'N/A':>11} | {'N/A':>10} | {'N/A':>10} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8}"
                )

        output_lines.append("-" * 100)
        output_lines.append("")  # Add spacing between basins

    return "\n".join(output_lines)
