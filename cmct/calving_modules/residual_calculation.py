import logging
import time

import numpy as np
import pandas as pd
import xarray as xr
from numba import jit, prange


@jit(nopython=True, parallel=True, cache=True)
def compute_residuals_and_stats(gsfc_values, model_values, x_indices, y_indices):
    """Compute residuals and statistics in a single pass."""
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


@jit(nopython=True)
def point_in_polygon(x, y, polygon_x, polygon_y):
    """Check if a single point is inside a polygon using ray casting."""
    n = len(polygon_x)
    inside = False

    p1x, p1y = polygon_x[0], polygon_y[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon_x[i % n], polygon_y[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


@jit(nopython=True, parallel=True)
def assign_basins_batch(
    x_coords,
    y_coords,
    basin_polygons_x,
    basin_polygons_y,
    basin_lengths,
    batch_start,
    batch_end,
):
    """Assign basin IDs to a batch of points."""
    batch_size = batch_end - batch_start
    basin_ids = np.full(batch_size, -1, dtype=np.int32)
    n_basins = len(basin_lengths)

    for i in prange(batch_size):
        x = x_coords[batch_start + i]
        y = y_coords[batch_start + i]

        polygon_start = 0
        for basin_idx in range(n_basins):
            polygon_end = polygon_start + basin_lengths[basin_idx]
            poly_x = basin_polygons_x[polygon_start:polygon_end]
            poly_y = basin_polygons_y[polygon_start:polygon_end]

            if point_in_polygon(x, y, poly_x, poly_y):
                basin_ids[i] = basin_idx
                break

            polygon_start = polygon_end

    return basin_ids


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


def process_year_data(
    gsfc_data,
    model_data,
    x_coords,
    y_coords,
    basin_polygons_x,
    basin_polygons_y,
    basin_lengths,
    basin_names,
    year,
):
    """Process a single year of data with basin assignments."""

    # Get coordinate indices
    gsfc_x = gsfc_data.x.values.astype(np.float32)
    gsfc_y = gsfc_data.y.values.astype(np.float32)

    x_indices = np.searchsorted(gsfc_x, x_coords)
    x_indices = np.clip(x_indices, 0, len(gsfc_x) - 1)

    y_indices = np.searchsorted(gsfc_y, y_coords)
    y_indices = np.clip(y_indices, 0, len(gsfc_y) - 1)

    # Convert ice mask data
    gsfc_values = gsfc_data.ice_mask.values.astype(np.float32)
    model_values = model_data.ice_mask.values.astype(np.float32)

    # Flatten coordinates for processing
    x_flat = np.repeat(x_coords, len(y_coords))
    y_flat = np.tile(y_coords, len(x_coords))
    x_idx_flat = np.repeat(x_indices, len(y_indices))
    y_idx_flat = np.tile(y_indices, len(x_indices))

    # Compute residuals
    residuals, valid_count, abs_sum, sq_sum, residual_sum = compute_residuals_and_stats(
        gsfc_values, model_values, x_idx_flat, y_idx_flat
    )

    # Assign basins in sequential batches (removed concurrency)
    n_points = len(x_flat)
    batch_size = 50000
    basin_assignments = np.full(n_points, -1, dtype=np.int32)

    # Process batches sequentially instead of in parallel
    for start in range(0, n_points, batch_size):
        end = min(start + batch_size, n_points)
        batch_result = assign_basins_batch(
            x_flat,
            y_flat,
            basin_polygons_x,
            basin_polygons_y,
            basin_lengths,
            start,
            end,
        )
        basin_assignments[start:end] = batch_result

    # Reshape to 2D grids
    n_x, n_y = len(x_coords), len(y_coords)
    residual_grid = residuals.reshape(n_x, n_y).T
    basin_grid = basin_assignments.reshape(n_x, n_y).T

    # Get aligned ice masks
    gsfc_aligned = gsfc_values[np.ix_(y_indices, x_indices)]
    model_aligned = model_values[np.ix_(y_indices, x_indices)]

    # Calculate statistics
    avg_residual = abs_sum / valid_count if valid_count > 0 else 0.0
    rms_residual = np.sqrt(sq_sum / valid_count) if valid_count > 0 else 0.0

    stats = {
        "avg_abs_residual": round(avg_residual, 3),
        "rms_residual": round(rms_residual, 3),
        "sum_residual": round(residual_sum, 3),
        "valid_points": int(valid_count),
        "total_points": n_points,
    }

    return residual_grid, basin_grid, gsfc_aligned, model_aligned, stats


@jit(nopython=True, parallel=True, cache=True)
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


@jit(nopython=True, parallel=True, cache=True)
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


@jit(nopython=True, cache=True)
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


def create_calving_dataset(gsfc, model, years, basin_polygons_dict):
    """
    Create a comprehensive dataset with calving data across multiple years and basins.
    OPTIMIZED VERSION using pure numpy arrays for maximum speed.

    Parameters
    ----------
    gsfc : GSFCcalving
        GSFC calving data object
    model : ModelCalving
        Model calving data object
    years : list
        List of years to process
    basin_polygons_dict : dict
        Dictionary of basin polygons

    Returns
    -------
    xarray.Dataset
        Dataset with dimensions (time, y, x) and variables for residuals,
        basin assignments, and ice masks
    """

    logging.info("Starting optimized calving dataset creation...")
    start_time = time.time()

    # Prepare basin data once
    basin_names, basin_polygons_x, basin_polygons_y, basin_lengths = (
        prepare_basin_polygons(basin_polygons_dict)
    )

    # Get coordinate arrays - use model coordinates as reference
    model_sample = model.ds.sel(time=years[0])
    x_coords = model_sample.x.values.astype(np.float32)
    y_coords = model_sample.y.values.astype(np.float32)

    logging.info(f"Grid dimensions: {len(y_coords)} x {len(x_coords)}")

    # Extract all data at once using numpy indexing

    # Get all GSFC data for the years (vectorized)
    gsfc_data_list = []
    for year in years:
        gsfc_year = gsfc.ds.sel(year=year)
        # Find corresponding indices in model grid
        gsfc_x = gsfc_year.x.values
        gsfc_y = gsfc_year.y.values

        # Use numpy's searchsorted for fast index finding
        x_indices = np.searchsorted(gsfc_x, x_coords)
        x_indices = np.clip(x_indices, 0, len(gsfc_x) - 1)
        y_indices = np.searchsorted(gsfc_y, y_coords)
        y_indices = np.clip(y_indices, 0, len(gsfc_y) - 1)

        # Extract aligned data
        gsfc_aligned = gsfc_year.ice_mask.values[np.ix_(y_indices, x_indices)].astype(
            np.float32
        )
        gsfc_data_list.append(gsfc_aligned)

    # Stack all GSFC data
    gsfc_data_all = np.stack(gsfc_data_list, axis=0)

    # Get all model data for the years (vectorized)
    model_data_list = []
    for year in years:
        model_year = model.ds.sel(time=year)
        model_data_list.append(model_year.ice_mask.values.astype(np.float32))

    # Stack all model data
    model_data_all = np.stack(model_data_list, axis=0)

    logging.info("Computing residuals...")
    # Compute residuals vectorized across all years
    residuals_all = compute_residuals_vectorized(gsfc_data_all, model_data_all)

    logging.info("Computing statistics...")
    # Compute statistics vectorized
    stats_array = compute_stats_vectorized(residuals_all)

    logging.info("Creating basin assignments...")

    # Debug: Print coordinate ranges before basin assignment
    logging.info(
        f"Data coordinates: X=[{x_coords.min():.1f}, {x_coords.max():.1f}], Y=[{y_coords.min():.1f}, {y_coords.max():.1f}]"
    )
    logging.info(
        f"Basin polygon coordinates: X=[{basin_polygons_x.min():.1f}, {basin_polygons_x.max():.1f}], Y=[{basin_polygons_y.min():.1f}, {basin_polygons_y.max():.1f}]"
    )

    # Create basin mask once (it's the same for all years)
    # Use debug version for better monitoring, switch to optimized version in production
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

    # Debug: Check basin assignment results
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

    # Verify basin assignment success
    assigned_points = np.sum(basin_mask >= 0)
    total_points = basin_mask.size
    assignment_rate = 100 * assigned_points / total_points
    logging.info(
        f"Basin assignment rate: {assigned_points}/{total_points} ({assignment_rate:.1f}%)"
    )

    # Replicate basin mask for all years
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

    # Add statistics as data variables
    stats_df = pd.DataFrame(stats_list)
    for col in stats_df.columns:
        if col != "year":
            ds[f"stats_{col}"] = ("time", stats_df[col].values)

    end_time = time.time()
    logging.info(
        f"Optimized dataset creation completed in {end_time - start_time:.2f} seconds"
    )

    return ds
