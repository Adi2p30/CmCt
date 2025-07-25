import gc
import logging
import time

import numpy as np
import pandas as pd
import xarray as xr
from numba import jit, prange


@jit(nopython=True, parallel=False, cache=True)
def compute_residuals_and_stats(gsfc_values, model_values, x_indices, y_indices):
    n_points = len(x_indices)
    residuals = np.full(n_points, np.nan, dtype=np.float32)

    valid_count = 0
    abs_sum = 0.0
    sq_sum = 0.0
    residual_sum = 0.0

    for i in prange(n_points):
        gsfc_val = gsfc_values[y_indices[i], x_indices[i]]
        model_val = model_values[y_indices[i], x_indices[i]]

        if not (np.isnan(gsfc_val) or np.isnan(model_val)):
            residual = gsfc_val - model_val
            residuals[i] = residual
            valid_count += 1
            residual_sum += residual
            abs_sum += abs(residual)
            sq_sum += residual * residual

    return residuals, valid_count, abs_sum, sq_sum, residual_sum


@jit(nopython=True, inline="always", fastmath=False)
def point_in_polygon(x, y, polygon_x, polygon_y):
    """Optimized point-in-polygon using winding number algorithm."""
    n = len(polygon_x)
    winding_number = 0

    for i in range(n):
        j = (i + 1) % n
        if polygon_y[i] <= y:
            if polygon_y[j] > y:  # Upward crossing
                cross_product = (polygon_x[j] - polygon_x[i]) * (y - polygon_y[i]) - (
                    x - polygon_x[i]
                ) * (polygon_y[j] - polygon_y[i])
                if cross_product > 0:
                    winding_number += 1
        else:
            if polygon_y[j] <= y:  # Downward crossing
                cross_product = (polygon_x[j] - polygon_x[i]) * (y - polygon_y[i]) - (
                    x - polygon_x[i]
                ) * (polygon_y[j] - polygon_y[i])
                if cross_product < 0:
                    winding_number -= 1

    return winding_number != 0


def prepare_basin_polygons(basin_polygons_dict, target_crs="EPSG:3413"):
    """
    Convert basin polygons to numba-friendly arrays with proper coordinate transformation.

    Parameters:
    -----------
    basin_polygons_dict : dict
        Dictionary of basin polygons in geographic coordinates
    target_crs : str
        Target coordinate reference system (default: EPSG:3413 for polar stereographic)

    Returns:
    --------
    tuple
        (basin_names, basin_polygons_x, basin_polygons_y, basin_lengths)
    """
    from pyproj import Transformer
    from shapely.ops import transform

    basin_names = []
    all_x = []
    all_y = []
    basin_lengths = []

    # Create transformer from WGS84 (geographic) to polar stereographic
    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)

    for name, polygon in basin_polygons_dict.items():
        if name == "unassigned":
            continue

        # Transform polygon from geographic to projected coordinates
        try:
            projected_polygon = transform(transformer.transform, polygon)
            coords = np.array(projected_polygon.exterior.coords)

            basin_names.append(name)
            all_x.extend(coords[:, 0])
            all_y.extend(coords[:, 1])
            basin_lengths.append(len(coords))

            logging.info(f"Transformed basin {name}: {len(coords)} points")

        except Exception as e:
            logging.error(f"Failed to transform basin {name}: {e}")
            continue

    return (
        basin_names,
        np.array(all_x, dtype=np.float32),
        np.array(all_y, dtype=np.float32),
        np.array(basin_lengths, dtype=np.int32),
    )


@jit(nopython=True, parallel=False, cache=True)
def create_basin_mask_optimized(
    x_coords, y_coords, basin_polygons_x, basin_polygons_y, basin_lengths
):
    """Create basin mask using vectorized operations with proper coordinate handling."""
    n_y, n_x = len(y_coords), len(x_coords)
    n_basins = len(basin_lengths)
    basin_mask = np.full((n_y, n_x), -1, dtype=np.int32)

    # Create coordinate meshgrid
    for i in prange(n_y):
        for j in prange(n_x):
            x = x_coords[j]
            y = y_coords[i]

            polygon_start = 0
            for basin_idx in range(n_basins):
                polygon_end = polygon_start + basin_lengths[basin_idx]
                poly_x = basin_polygons_x[polygon_start:polygon_end]
                poly_y = basin_polygons_y[polygon_start:polygon_end]

                if point_in_polygon(x, y, poly_x, poly_y):
                    basin_mask[i, j] = basin_idx
                    break

                polygon_start = polygon_end

    return basin_mask


def create_basin_mask_debug(
    x_coords, y_coords, basin_polygons_x, basin_polygons_y, basin_lengths
):
    """
    Create basin mask with debugging information.

    This is a non-numba version for debugging purposes that provides detailed
    information about the basin assignment process.
    """
    n_y, n_x = len(y_coords), len(x_coords)
    n_basins = len(basin_lengths)
    basin_mask = np.full((n_y, n_x), -1, dtype=np.int32)

    logging.info(f"Creating basin mask for {n_x} x {n_y} grid with {n_basins} basins")

    # Debug: Print coordinate ranges
    logging.info(
        f"Data coordinates: X=[{x_coords.min():.1f}, {x_coords.max():.1f}], Y=[{y_coords.min():.1f}, {y_coords.max():.1f}]"
    )
    logging.info(
        f"Basin polygon coordinates: X=[{basin_polygons_x.min():.1f}, {basin_polygons_x.max():.1f}], Y=[{basin_polygons_y.min():.1f}, {basin_polygons_y.max():.1f}]"
    )

    points_assigned = 0
    total_points = n_x * n_y

    for i in range(n_y):
        if i % 500 == 0:  # Progress indicator
            logging.info(f"Processing row {i}/{n_y}")

        for j in range(n_x):
            x = x_coords[j]
            y = y_coords[i]

            polygon_start = 0
            for basin_idx in range(n_basins):
                polygon_end = polygon_start + basin_lengths[basin_idx]
                poly_x = basin_polygons_x[polygon_start:polygon_end]
                poly_y = basin_polygons_y[polygon_start:polygon_end]

                if point_in_polygon(x, y, poly_x, poly_y):
                    basin_mask[i, j] = basin_idx
                    points_assigned += 1
                    break

                polygon_start = polygon_end

    logging.info(
        f"Basin assignment complete: {points_assigned}/{total_points} points assigned to basins ({100 * points_assigned / total_points:.1f}%)"
    )

    return basin_mask


@jit(nopython=True, parallel=False, cache=True, fastmath=False)
def compute_residuals_vectorized(gsfc_data, model_data):
    """Compute residuals in a vectorized manner."""
    n_time, n_y, n_x = gsfc_data.shape
    residuals = np.full((n_time, n_y, n_x), np.nan, dtype=np.float32)

    for t in prange(n_time):
        for i in prange(n_y):
            for j in prange(n_x):
                gsfc_val = gsfc_data[t, i, j]
                model_val = model_data[t, i, j]

                if not (np.isnan(gsfc_val) or np.isnan(model_val)):
                    residuals[t, i, j] = gsfc_val - model_val

    return residuals


@jit(nopython=True, cache=True, fastmath=False)
def compute_stats_vectorized(residuals):
    """Compute statistics for residuals in a vectorized manner."""
    n_time, n_y, n_x = residuals.shape
    stats = np.full(
        (n_time, 5), np.nan, dtype=np.float32
    )  # [avg_abs, rms, sum, valid_count, total_points]

    for t in range(n_time):
        abs_sum = 0.0
        sq_sum = 0.0
        residual_sum = 0.0
        valid_count = 0
        total_points = n_y * n_x

        for i in range(n_y):
            for j in range(n_x):
                val = residuals[t, i, j]
                if not np.isnan(val):
                    valid_count += 1
                    residual_sum += val
                    abs_sum += abs(val)
                    sq_sum += val * val

        if valid_count > 0:
            avg_abs = abs_sum / valid_count
            rms = np.sqrt(sq_sum / valid_count)

            stats[t, 0] = avg_abs
            stats[t, 1] = rms
            stats[t, 2] = residual_sum
            stats[t, 3] = valid_count
            stats[t, 4] = total_points
        else:
            stats[t, 3] = 0  # valid_count
            stats[t, 4] = total_points  # total_points

    return stats


def compute_basin_mask_once(gsfc, model, basin_polygons_dict, year=None):
    """
    Compute basin mask once using the first available year of data.

    Parameters
    ----------
    gsfc : GSFCcalving
        GSFC calving data object
    model : ModelCalving
        Model calving data object (first ensemble member)
    basin_polygons_dict : dict
        Dictionary of basin polygons
    year : int, optional
        Specific year to use for mask creation. If None, uses first available year.

    Returns
    -------
    tuple
        (basin_mask, basin_names, x_coords, y_coords)
    """
    logging.info("Computing basin mask once for all ensemble members...")

    # Prepare basin data
    basin_names, basin_polygons_x, basin_polygons_y, basin_lengths = (
        prepare_basin_polygons(basin_polygons_dict)
    )

    # Get coordinate arrays from model - use first available year if not specified
    if year is None:
        year = model.ds.time.values[0]

    model_sample = model.ds.sel(time=year)
    x_coords = model_sample.x.values.astype(np.float32)
    y_coords = model_sample.y.values.astype(np.float32)

    logging.info(f"Grid dimensions: {len(y_coords)} x {len(x_coords)}")
    logging.info(f"Using year {year} for basin mask creation")

    # Create basin mask
    logging.info("Creating basin mask...")
    try:
        basin_mask = create_basin_mask_optimized(
            x_coords, y_coords, basin_polygons_x, basin_polygons_y, basin_lengths
        )
    except Exception as e:
        logging.error(f"Error in optimized basin mask creation: {e}")
        logging.info("Falling back to debug version...")
        basin_mask = create_basin_mask_debug(
            x_coords, y_coords, basin_polygons_x, basin_polygons_y, basin_lengths
        )

    # Log basin assignment statistics
    unique_basins = np.unique(basin_mask)
    logging.info(f"Basin assignment complete. Unique basin IDs: {unique_basins}")

    for basin_id in unique_basins:
        if basin_id >= 0:
            count = np.sum(basin_mask == basin_id)
            basin_name = (
                basin_names[basin_id] if basin_id < len(basin_names) else "Unknown"
            )
            logging.info(f"  Basin {basin_id} ({basin_name}): {count} points")
        else:
            count = np.sum(basin_mask == basin_id)
            logging.info(f"  Unassigned points: {count}")

    assigned_points = np.sum(basin_mask >= 0)
    total_points = basin_mask.size
    assignment_rate = 100 * assigned_points / total_points
    logging.info(
        f"Basin assignment rate: {assigned_points}/{total_points} ({assignment_rate:.1f}%)"
    )

    return basin_mask, basin_names, x_coords, y_coords


def create_calving_dataset_with_precomputed_mask(
    gsfc, model, years, basin_mask, basin_names, x_coords, y_coords
):
    """
    Create a comprehensive dataset with calving data using a precomputed basin mask.

    Parameters
    ----------
    gsfc : GSFCcalving
        GSFC calving data object
    model : ModelCalving
        Model calving data object
    years : list
        List of years to process
    basin_mask : np.ndarray
        Precomputed basin mask (y, x)
    basin_names : list
        List of basin names
    x_coords : np.ndarray
        X coordinates
    y_coords : np.ndarray
        Y coordinates

    Returns
    -------
    xarray.Dataset
        Dataset with dimensions (time, y, x) and variables for residuals,
        basin assignments, and ice masks
    """
    logging.info("Creating calving dataset with precomputed basin mask...")
    start_time = time.time()

    # Prepare GSFC data for all years
    gsfc_data_list = []
    for year in years:
        gsfc_year = gsfc.ds.sel(year=year)

        # Use xarray's interpolation to ensure exact coordinate matching
        # Interpolate GSFC data to the exact coordinates used for the basin mask
        gsfc_interp = gsfc_year.ice_mask.interp(x=x_coords, y=y_coords, method="linear")
        gsfc_aligned = gsfc_interp.values.astype(np.float32)

        # Handle any NaN values that might result from interpolation
        gsfc_aligned = np.nan_to_num(gsfc_aligned, nan=0.0)

        gsfc_data_list.append(gsfc_aligned)

    gsfc_data_all = np.stack(gsfc_data_list, axis=0)
    logging.info(f"GSFC data shape: {gsfc_data_all.shape}")

    # Prepare model data for all years
    model_data_list = []
    for year in years:
        model_year = model.ds.sel(time=year)

        # Instead of using searchsorted, use xarray's interpolation to ensure exact coordinate matching
        # Interpolate model data to the exact coordinates used for the basin mask
        model_interp = model_year.ice_mask.interp(
            x=x_coords, y=y_coords, method="linear"
        )
        model_aligned = model_interp.values.astype(np.float32)

        # Handle any NaN values that might result from interpolation
        model_aligned = np.nan_to_num(model_aligned, nan=0.0)

        model_data_list.append(model_aligned)

    model_data_all = np.stack(model_data_list, axis=0)

    # Check if model data needs to be transposed to match GSFC data dimension order
    if model_data_all.shape != gsfc_data_all.shape:
        logging.info(
            f"Transposing model data from {model_data_all.shape} to match GSFC shape {gsfc_data_all.shape}"
        )
        # Transpose the spatial dimensions (keep time dimension as first)
        model_data_all = model_data_all.transpose(0, 2, 1)

    logging.info(f"Model data shape after alignment: {model_data_all.shape}")
    logging.info(f"Basin mask shape: {basin_mask.shape}")

    # Ensure all data arrays have consistent shapes
    if gsfc_data_all.shape != model_data_all.shape:
        logging.error(
            f"Shape mismatch after transpose: GSFC {gsfc_data_all.shape} vs Model {model_data_all.shape}"
        )
        raise ValueError(
            f"Data shape mismatch after transpose: GSFC {gsfc_data_all.shape} vs Model {model_data_all.shape}"
        )

    if gsfc_data_all.shape[1:] != basin_mask.shape:
        logging.error(
            f"Basin mask shape mismatch: Data {gsfc_data_all.shape[1:]} vs Mask {basin_mask.shape}"
        )
        raise ValueError(
            f"Basin mask shape mismatch: Data {gsfc_data_all.shape[1:]} vs Mask {basin_mask.shape}"
        )

    logging.info("Computing residuals...")
    residuals_all = compute_residuals_vectorized(gsfc_data_all, model_data_all)

    logging.info("Computing statistics...")
    stats_array = compute_stats_vectorized(residuals_all)

    # Broadcast basin mask to all years
    n_years = len(years)
    basins_all = np.broadcast_to(
        basin_mask[None, :, :], (n_years, basin_mask.shape[0], basin_mask.shape[1])
    ).copy()

    # Create statistics list
    stats_list = []
    for i, year in enumerate(years):
        stats = {
            "year": year,
            "avg_abs_residual": float(stats_array[i, 0])
            if not np.isnan(stats_array[i, 0])
            else 0.0,
            "rms_residual": float(stats_array[i, 1])
            if not np.isnan(stats_array[i, 1])
            else 0.0,
            "sum_residual": float(stats_array[i, 2])
            if not np.isnan(stats_array[i, 2])
            else 0.0,
            "valid_points": int(stats_array[i, 3])
            if not np.isnan(stats_array[i, 3])
            else 0,
            "total_points": int(stats_array[i, 4])
            if not np.isnan(stats_array[i, 4])
            else 0,
        }
        stats_list.append(stats)

    logging.info("Creating xarray dataset...")

    # Create dataset
    ds = xr.Dataset(
        {
            "residual": (["time", "y", "x"], residuals_all),
            "basin": (["time", "y", "x"], basins_all),
            "gsfc_ice_mask": (["time", "y", "x"], gsfc_data_all),
            "model_ice_mask": (["time", "y", "x"], model_data_all),
        },
        coords={
            "time": years,
            "x": x_coords,
            "y": y_coords,
            "basin_names": ("basin_id", basin_names),
        },
        attrs={
            "title": "Calving comparison analysis",
            "projection": "EPSG:3413",
            "units": "ice_mask units",
            "creation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )

    # Add statistics as variables
    stats_df = pd.DataFrame(stats_list)
    for col in stats_df.columns:
        if col != "year":
            ds[f"stats_{col}"] = ("time", stats_df[col].values)

    end_time = time.time()
    logging.info(f"Dataset creation completed in {end_time - start_time:.2f} seconds")

    return ds


# Backward compatibility function - kept for legacy code
def create_calving_dataset(gsfc, model, years, basin_polygons_dict):
    """
    Legacy function that computes basin mask and creates dataset.
    For new code, use compute_basin_mask_once() followed by
    create_calving_dataset_with_precomputed_mask().
    """
    # Compute basin mask
    basin_mask, basin_names, x_coords, y_coords = compute_basin_mask_once(
        gsfc, model, basin_polygons_dict, years[0]
    )

    # Create dataset with precomputed mask
    return create_calving_dataset_with_precomputed_mask(
        gsfc, model, years, basin_mask, basin_names, x_coords, y_coords
    )
