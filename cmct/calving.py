import datetime
import os
from datetime import timedelta
import cartopy
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import geopandas as gpd
import h5py
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import xarray as xr
import logging
from numba import jit, prange
from concurrent.futures import ProcessPoolExecutor, as_completed
from matplotlib import rc
from shapely.geometry import Point
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# from .time_utils import check_datarange
rc("mathtext", default="regular")


def load_gsfc_calving(filepath):
    """
    Load GSFC calving data from an nc file.

    Parameters
    ----------
    filepath : str
        Path to the netCDF file containing the calving data.

    Returns
    -------
    Modelcalving
        An instance of the Modelcalving class with the loaded data.
        
    """
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    try:    
        gsfc = GSFCcalving(filepath)
        
    except Exception as error:
        print("Error: Failed to load GSFC dataset.")
        print(error)
        gsfc = None
        raise ValueError("Failed to load GSFC calving data.")
        
    return gsfc



def load_model_calving(filepath):
    """
    Load GSFC calving data from an nc file.

    Parameters
    ----------
    f : netCDF4.Dataset
        The netCDF4 dataset containing the calving data.

    Returns
    -------
    GSFCcalving
        An instance of the GSFCcalving class with the loaded data.
        
    """
    
    try:
        model_res = Modelcalving(filepath)
    except Exception as error:
        print("Error: Failed to load Model dataset.")
        print(error)
        model_res = None
    return model_res

class GSFCcalving:
    def __init__(self, nc_path):
        # Open as xarray Dataset
        self.ds = xr.open_dataset(nc_path, autoclose=True, engine='netcdf4',use_cftime=True)
        self.ds["ice_mask"] = self.ds["ice_mask"] / 100
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

    # def _set_times_as_datetimes(self, days):
    #     return np.datetime64('2002-01-01T00:00:00') + np.array([int(d*24) for d in days], dtype='timedelta64[h]')
    
    
class Modelcalving:
    def __init__(self, nc_path):
        # Open as xarray Dataset
        self.ds = xr.open_dataset(nc_path, autoclose=True, engine='netcdf4',use_cftime=True)
        self.ds["ice_mask"] = self.ds["sftgif"]
        
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
    
    # @property
    # def lat_bnds(self):
    #     return self.ds["lat_bnds"]
    
    # @property
    # def lon_bnds(self):
    #     return self.ds["lon_bnds"]
    
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
        
        
        
        
def convert_to_standard_datetime(time_var):
    """
    Convert a time variable to a standard datetime string format.
    Parameters
    ----------
    time_var : xarray.DataArray
        The time variable to convert.
    Returns
    -------
    str
        The time variable in 'YYYY-MM-DDTHH:MM:SS' format.
    """

    return time_var.dt.strftime('%Y-%m-%dT%H:%M:%S')


def check_data_daterange(gsfc_time: list, model_time: list, start_date: int, end_date: int):
    print(type(gsfc_time), type(model_time), type(start_date), type(end_date))
    
    gsfc_time.sort()
    gsfc_time_min = gsfc_time[0]
    gsfc_time_max = gsfc_time[-1]
    
    model_time.sort()
    model_time_min = model_time[0]
    model_time_max = model_time[-1]

    minimum_time = max(gsfc_time_min, model_time_min)
    maximum_time = min(gsfc_time_max, model_time_max)

    if not (minimum_time <= start_date <= end_date and start_date <= end_date <= maximum_time):
        raise ValueError(f"Date range {start_date} to {end_date} is outside the available data range: {minimum_time} to {maximum_time}.")
    else: 
        print(f"The selected dates {start_date} and {end_date} are within the range of the model data. These are accepted.")


# Currentlty not implemented, not required.
def match_resolution(obs, res):
    #TODO: Future Implementation
    """
    Match the resolution of the observation data to the specified resolution.

    Parameters
    ----------
    obs : xarray.Dataset
        The observation dataset.
    res : str
        The desired resolution (e.g., '1km', '5km').

    Returns
    -------
    xarray.Dataset
        The observation dataset with matched resolution.
    """
    if res == "1km":
        return obs.coarsen(lat=10, lon=10).mean()
    elif res == "5km":
        return obs.coarsen(lat=50, lon=50).mean()
    else:
        raise ValueError(f"Unsupported resolution: {res}")
    
#--------------------
# PARALLEL PROCESSING CALVING
#--------------------

"""
SIMPLE (LEGACY) -> HYPER OPTIMIZED JSON (LEGACY) -> NO JSON JIT OPTIMIZED (CURRENT)
=
"""
# def find_absolute_calving_per_year(gsfc, model, year):
#    """
#    Find absolute calving data for a specified year.

#    Parameters
#    ----------
#    gsfc : GSFCcalving
#        The GSFC calving data object.
#    model : ModelCalving
#        The model calving data object.
#    year : int
#        The year for which to find absolute calving data.

#    Returns
#    -------
#    xarray.Dataset
#        The absolute calving data for the specified year.
#    """
#    logging.info(f"Finding absolute calving data for year {year}")
   
#    gsfc_year = gsfc.ds.sel(year=year)
#    model_year = model.ds.sel(time=year)
#    logging.info(f"Selected data for year {year} from both datasets")

#    if gsfc_year is None or model_year is None:
#        logging.error(f"No data available for the year {year} in either GSFC or model datasets")
#        raise ValueError(f"No data available for the year {year} in either GSFC or model datasets.")
   
#    else:
#        logging.info("Starting to process grid points")
#        calving_data_list = []
#        processed_points = 0
#        valid_points = 0
#        agg_residual = 0
#        sq_agg_residual = 0
       
#        for x in model_year.x.values:
#            for y in model_year.y.values:
#                processed_points += 1
#                if np.isnan(gsfc_year.ice_mask.sel(x=x, y=y).values) or np.isnan(model_year.ice_mask.sel(x=x, y=y).values):
#                    continue
#                else:
#                    valid_points += 1
#                    residual = round(float(gsfc_year.ice_mask.sel(x=x, y=y).values - model_year.ice_mask.sel(x=x, y=y).values), 3)
#                    agg_residual += abs(residual)
#                    sq_agg_residual += residual ** 2
#                    calving_data = {
                       
#                        'x': round(float(x), 3),
#                        'y': round(float(y), 3),
#                        'gsfc_ice_mask': round(float(gsfc_year.ice_mask.sel(x=x, y=y).values), 3),
#                        'model_ice_mask': round(float(model_year.ice_mask.sel(x=x, y=y).values), 3),
#                        'residual':  residual
#                    }
#                    calving_data_list.append(calving_data)
        
       
#        RMS = (sq_agg_residual/valid_points)**(1/2)


#        statistical_analyses = {"AVG_RESIDUAL": round(agg_residual/valid_points, 3), "RMS_RESIDUAL": round(RMS, 3), "VALID_POINTS": valid_points, "PROCESSED_POINTS": processed_points}
       
#        logging.info(f"Processed {processed_points} grid points, found {valid_points} valid points")
#        logging.info(f"Returning {len(calving_data_list)} calving data records")
#        return calving_data_list, statistical_analyses
   
# def process_coordinate_batch(args):
#     """
#     Process a batch of coordinates and return results.
    
#     Parameters
#     ----------
#     args : tuple
#         (x_batch, y_batch, gsfc_data, model_data, batch_id)
    
#     Returns
#     -------
#     tuple
#         (calving_data_list, valid_points, sum_abs_residual, sum_sq_residual)
#     """
#     x_batch, y_batch, gsfc_data, model_data, batch_id = args
    
#     calving_data_list = []
#     valid_points = 0
#     sum_abs_residual = 0.0
#     sum_sq_residual = 0.0
    
#     for x in x_batch:
#         for y in y_batch:
#             try:
#                 gsfc_val = float(gsfc_data.sel(x=x, y=y, method='nearest').values)
#                 model_val = float(model_data.sel(x=x, y=y, method='nearest').values)
                
#                 if np.isnan(gsfc_val) or np.isnan(model_val):
#                     continue
                
#                 residual = round(gsfc_val - model_val, 3)
                
#                 calving_data = {
#                     'x': round(float(x), 3),
#                     'y': round(float(y), 3),
#                     'gsfc_ice_mask': round(gsfc_val, 3),
#                     'model_ice_mask': round(model_val, 3),
#                     'residual': residual
#                 }
#                 calving_data_list.append(calving_data)
                
#                 valid_points += 1
#                 sum_abs_residual += abs(residual)
#                 sum_sq_residual += residual ** 2
                
#             except Exception as e:
#                 continue
    
#     return calving_data_list, valid_points, sum_abs_residual, sum_sq_residual

# def create_coordinate_batches(x_coords, y_coords, batch_size=50):
#     """
#     Split coordinates into smaller batches for parallel processing.
    
#     Parameters
#     ----------
#     x_coords : array
#         X coordinates
#     y_coords : array  
#         Y coordinates
#     batch_size : int
#         Number of coordinates per batch
        
#     Returns
#     -------
#     list
#         List of (x_batch, y_batch) tuples
#     """
#     batches = []

#     for i in range(0, len(x_coords), batch_size):
#         x_batch = x_coords[i:i + batch_size]
        
#         for j in range(0, len(y_coords), batch_size):
#             y_batch = y_coords[j:j + batch_size]
#             batches.append((x_batch, y_batch))
    
#     return batches

# def find_absolute_calving_per_year_parallel(gsfc, model, year, num_workers=None, batch_size=50):
#     """
#     Find absolute calving data for a specified year using parallel processing.
    
#     This version is simpler and fixes the indexing issues by processing 
#     coordinates in small batches rather than using complex vectorization.

#     Parameters
#     ----------
#     gsfc : GSFCcalving
#         The GSFC calving data object.
#     model : ModelCalving
#         The model calving data object.
#     year : int
#         The year for which to find absolute calving data.
#     num_workers : int, optional
#         Number of worker processes. If None, uses all CPU cores.
#     batch_size : int, optional
#         Number of coordinates to process per batch. Default is 50.

#     Returns
#     -------
#     tuple
#         (calving_data_list, statistical_analyses)
#     """
#     start_time = time.time()
#     logging.info(f"Finding absolute calving data for year {year} using parallel processing")
    
#     gsfc_year = gsfc.ds.sel(year=year)
#     model_year = model.ds.sel(time=year)
#     logging.info(f"Selected data for year {year} from both datasets")

#     if gsfc_year is None or model_year is None:
#         logging.error(f"No data available for the year {year}")
#         raise ValueError(f"No data available for the year {year}")
    
#     x_coords = model_year.x.values
#     y_coords = model_year.y.values
#     total_points = len(x_coords) * len(y_coords)
    
#     logging.info(f"Total grid points to process: {total_points}")
#     logging.info(f"Batch size: {batch_size}")
    
#     coordinate_batches = create_coordinate_batches(x_coords, y_coords, batch_size)
#     logging.info(f"Created {len(coordinate_batches)} batches")
    
#     batch_args = [
#         (x_batch, y_batch, gsfc_year.ice_mask, model_year.ice_mask, i)
#         for i, (x_batch, y_batch) in enumerate(coordinate_batches)
#     ]
    
#     if num_workers is None:
#         num_workers = min(cpu_count(), len(coordinate_batches))
    
#     logging.info(f"Using {num_workers} worker processes")
    
#     all_calving_data = []
#     total_valid_points = 0
#     total_abs_residual = 0.0
#     total_sq_residual = 0.0
    
#     with ProcessPoolExecutor(max_workers=num_workers) as executor:
#         futures = [executor.submit(process_coordinate_batch, args) for args in batch_args]
    
#         completed_batches = 0
#         for future in as_completed(futures):
#             try:
#                 calving_data_list, valid_points, abs_residual, sq_residual = future.result()
                
#                 all_calving_data.extend(calving_data_list)
#                 total_valid_points += valid_points
#                 total_abs_residual += abs_residual
#                 total_sq_residual += sq_residual            
#                 completed_batches += 1
                
#                 if completed_batches % 20 == 0:
#                     logging.info(f"Completed {completed_batches}/{len(coordinate_batches)} batches")
                    
#             except Exception as exc:
#                 logging.error(f"Batch processing failed: {exc}")
#                 continue
    
#     if total_valid_points > 0:
#         avg_residual = round(total_abs_residual / total_valid_points, 3)
#         rms_residual = round((total_sq_residual / total_valid_points) ** 0.5, 3)
#     else:
#         avg_residual = 0.0
#         rms_residual = 0.0
#         logging.warning("No valid data points found")
    
#     statistical_analyses = {
#         "AVG_RESIDUAL": avg_residual,
#         "RMS_RESIDUAL": rms_residual,
#         "VALID_POINTS": total_valid_points,
#         "PROCESSED_POINTS": total_points
#     }
    
#     end_time = time.time()
#     processing_time = round(end_time - start_time, 2)
    
#     logging.info(f"Processing completed in {processing_time} seconds")
#     logging.info(f"Found {total_valid_points} valid points out of {total_points} total points")
#     logging.info(f"Returning {len(all_calving_data)} calving data records")
    
#     return all_calving_data, statistical_analyses

# def find_absolute_calving_per_year_simple(gsfc, model, year):
#     """
#     Simple non-parallel version for comparison or debugging.
#     """
#     logging.info(f"Finding absolute calving data for year {year} (simple version)")
    
#     gsfc_year = gsfc.ds.sel(year=year)
#     model_year = model.ds.sel(time=year)

#     if gsfc_year is None or model_year is None:
#         raise ValueError(f"No data available for the year {year}")
    
#     calving_data_list = []
#     valid_points = 0
#     total_abs_residual = 0.0
#     total_sq_residual = 0.0
#     total_points = 0
    
#     for x in model_year.x.values:
#         for y in model_year.y.values:
#             total_points += 1
            
#             try:
#                 gsfc_val = float(gsfc_year.ice_mask.sel(x=x, y=y).values)
#                 model_val = float(model_year.ice_mask.sel(x=x, y=y).values)

#                 if np.isnan(gsfc_val) or np.isnan(model_val):
#                     continue
                    
#                 residual = round(gsfc_val - model_val, 3)
                
#                 calving_data = {
#                     'x': round(float(x), 3),
#                     'y': round(float(y), 3),
#                     'gsfc_ice_mask': round(gsfc_val, 3),
#                     'model_ice_mask': round(model_val, 3),
#                     'residual': residual
#                 }
#                 calving_data_list.append(calving_data)
                
#                 valid_points += 1
#                 total_abs_residual += abs(residual)
#                 total_sq_residual += residual ** 2
                
#             except Exception:
#                 continue
    
#     if valid_points > 0:
#         avg_residual = round(total_abs_residual / valid_points, 3)
#         rms_residual = round((total_sq_residual / valid_points) ** 0.5, 3)
#     else:
#         avg_residual = 0.0
#         rms_residual = 0.0
    
#     statistical_analyses = {
#         "AVG_RESIDUAL": avg_residual,
#         "RMS_RESIDUAL": rms_residual,
#         "VALID_POINTS": valid_points,
#         "PROCESSED_POINTS": total_points
#     }
    
#     return calving_data_list, statistical_analyses


# """
# # Parallel version (recommended)
# calving_data, stats = find_absolute_calving_per_year_parallel(
#     gsfc, model, year, num_workers=4, batch_size=50
# )

# # Simple version (for debugging or small datasets)
# calving_data, stats = find_absolute_calving_per_year_simple(gsfc, model, year)
# """



# HYPER-OPTIMIZED VERSION
# HYPER-OPTIMIZED VERSION
# HYPER-OPTIMIZED VERSION

# @jit(nopython=True, parallel=True, cache=True)
# def compute_residuals_vectorized(gsfc_data, model_data, x_coords, y_coords, x_indices, y_indices):
#     """
#     Hyper-optimized numba JIT compiled function for residual computation.
    
#     Parameters
#     ----------
#     gsfc_data : numpy.ndarray
#         2D array of GSFC ice mask data
#     model_data : numpy.ndarray 
#         2D array of model ice mask data
#     x_coords : numpy.ndarray
#         1D array of x coordinates
#     y_coords : numpy.ndarray
#         1D array of y coordinates
#     x_indices : numpy.ndarray
#         1D array of x indices for data arrays
#     y_indices : numpy.ndarray
#         1D array of y indices for data arrays
        
#     Returns
#     -------
#     tuple
#         Arrays of results: (x_vals, y_vals, gsfc_vals, model_vals, residuals, valid_mask)
#     """
#     n_x = len(x_coords)
#     n_y = len(y_coords)
#     total_points = n_x * n_y
    
#     x_vals = np.empty(total_points, dtype=np.float32)
#     y_vals = np.empty(total_points, dtype=np.float32)
#     gsfc_vals = np.empty(total_points, dtype=np.float32)
#     model_vals = np.empty(total_points, dtype=np.float32)
#     residuals = np.empty(total_points, dtype=np.float32)
#     valid_mask = np.empty(total_points, dtype=np.bool_)
    
#     for idx in prange(total_points):
#         i = idx // n_y
#         j = idx % n_y
        
#         x_coord = x_coords[i]
#         y_coord = y_coords[j]
        
#         x_idx = x_indices[i]
#         y_idx = y_indices[j]
        
#         gsfc_val = gsfc_data[y_idx, x_idx]  # Note: y first for numpy arrays
#         model_val = model_data[y_idx, x_idx]
        
        
#         # Check for valid data
#         is_valid = not (np.isnan(gsfc_val) or np.isnan(model_val))
        
#         x_vals[idx] = x_coord
#         y_vals[idx] = y_coord
#         gsfc_vals[idx] = gsfc_val
#         model_vals[idx] = model_val
#         residuals[idx] = gsfc_val - model_val if is_valid else np.nan
#         valid_mask[idx] = is_valid
    
#     return x_vals, y_vals, gsfc_vals, model_vals, residuals, valid_mask

# NOT LEGACY CURERENTLY USED
def prepare_data_arrays(gsfc_year, model_year):
    """
    Prepare and align data arrays for optimized computation.
    
    Parameters
    ----------
    gsfc_year : xarray.Dataset
        GSFC data for specific year
    model_year : xarray.Dataset
        Model data for specific year
        
    Returns
    -------
    tuple
        Prepared arrays and coordinate information
    """
    model_x = model_year.x.values.astype(np.float32)
    model_y = model_year.y.values.astype(np.float32)
    gsfc_x = gsfc_year.x.values.astype(np.float32)
    gsfc_y = gsfc_year.y.values.astype(np.float32)
    
    x_indices = np.searchsorted(gsfc_x, model_x)
    x_indices = np.clip(x_indices, 0, len(gsfc_x) - 1)
    
    y_indices = np.searchsorted(gsfc_y, model_y)
    y_indices = np.clip(y_indices, 0, len(gsfc_y) - 1)
    
    gsfc_data = gsfc_year.ice_mask.values.astype(np.float32)
    model_data = model_year.ice_mask.values.astype(np.float32)
    
    return gsfc_data, model_data, model_x, model_y, x_indices, y_indices

# def process_chunk_optimized(args):
#     """
#     Process a chunk of data using the hyper-optimized computation function.
    
#     Parameters
#     ----------
#     args : tuple
#         (chunk_data, chunk_indices, chunk_id)
        
#     Returns
#     -------
#     tuple
#         (results_dict_list, valid_count, abs_residual_sum, sq_residual_sum)
#     """
#     gsfc_data, model_data, x_coords, y_coords, x_indices, y_indices, chunk_id = args
       
#     # Use the JIT compiled function
#     x_vals, y_vals, gsfc_vals, model_vals, residuals, valid_mask = compute_residuals_vectorized(
#         gsfc_data, model_data, x_coords, y_coords, x_indices, y_indices
#     )
    
#     valid_indices = np.where(valid_mask)[0]
    
#     # Create result dictionaries only for valid points
#     results = []
#     abs_residual_sum = 0.0
#     sq_residual_sum = 0.0
    
#     for idx in valid_indices:
#         residual = float(residuals[idx])
#         result = {
#             'x': round(float(x_vals[idx]), 3),
#             'y': round(float(y_vals[idx]), 3),
#             'gsfc_ice_mask': round(float(gsfc_vals[idx]), 3),
#             'model_ice_mask': round(float(model_vals[idx]), 3),
#             'residual': round(residual, 3)
#         }
#         results.append(result)
#         abs_residual_sum += abs(residual)
#         sq_residual_sum += residual ** 2
    
#     return results, len(valid_indices), abs_residual_sum, sq_residual_sum

# def create_optimized_chunks(x_coords, y_coords, chunk_size=2500):
#     """
#     Create optimized chunks that balance memory usage and parallelization.
    
#     Parameters
#     ----------
#     x_coords : numpy.ndarray
#         X coordinates
#     y_coords : numpy.ndarray
#         Y coordinates
#     chunk_size : int
#         Target number of points per chunk
        
#     Returns
#     -------
#     list
#         List of (x_chunk, y_chunk) coordinate pairs
#     """
#     total_points = len(x_coords) * len(y_coords)
    
#     # Calculate optimal chunk dimensions
#     if total_points <= chunk_size:
#         return [(x_coords, y_coords)]
    
#     # Try to create roughly square chunks
#     chunk_dim = int(np.sqrt(chunk_size))
    
#     chunks = []
#     for i in range(0, len(x_coords), chunk_dim):
#         x_chunk = x_coords[i:i + chunk_dim]
#         for j in range(0, len(y_coords), chunk_dim):
#             y_chunk = y_coords[j:j + chunk_dim]
#             chunks.append((x_chunk, y_chunk))
    
#     return chunks

# def find_absolute_calving_per_year_hyper_optimized(gsfc, model, year, num_workers=None, chunk_size=2500):
#     """
#     Hyper-optimized version using JIT compilation, vectorization, and smart chunking.
    
#     This version uses:
#     - Numba JIT compilation for core computation
#     - Vectorized operations instead of nested loops
#     - Optimized memory layout and data types
#     - Smart chunking for better cache utilization
#     - Reduced Python overhead
    
#     Parameters
#     ----------
#     gsfc : GSFCcalving
#         The GSFC calving data object
#     model : ModelCalving  
#         The model calving data object
#     year : int
#         The year for which to find absolute calving data
#     num_workers : int, optional
#         Number of worker processes. If None, uses optimal number
#     chunk_size : int, optional
#         Target number of points per chunk. Default is 2500
        
#     Returns
#     -------
#     tuple
#         (calving_data_list, statistical_analyses)
#     """
#     start_time = time.time()
#     logging.info(f"Starting hyper-optimized calving analysis for year {year}")
    
#     gsfc_year = gsfc.ds.sel(year=year)
#     model_year = model.ds.sel(time=year)
    
#     if gsfc_year is None or model_year is None:
#         raise ValueError(f"No data available for the year {year}")
    
#     logging.info("Preparing optimized data arrays...")
#     gsfc_data, model_data, x_coords, y_coords, x_indices, y_indices = prepare_data_arrays(
#         gsfc_year, model_year
#     )
    
#     total_points = len(x_coords) * len(y_coords)
#     logging.info(f"Total grid points: {total_points}")
    
#     #Chunking
#     coordinate_chunks = create_optimized_chunks(x_coords, y_coords, chunk_size)
#     logging.info(f"Created {len(coordinate_chunks)} optimized chunks")
    
#     chunk_args = []
#     for i, (x_chunk, y_chunk) in enumerate(coordinate_chunks):
#         x_start = np.searchsorted(x_coords, x_chunk[0])
#         x_end = x_start + len(x_chunk)
#         y_start = np.searchsorted(y_coords, y_chunk[0])
#         y_end = y_start + len(y_chunk)
        
#         x_idx_chunk = x_indices[x_start:x_end]
#         y_idx_chunk = y_indices[y_start:y_end]
        
#         chunk_args.append((
#             gsfc_data, model_data, x_chunk, y_chunk, 
#             x_idx_chunk, y_idx_chunk, i
#         ))
    
#     if num_workers is None:
#         import os
#         num_workers = min(os.cpu_count(), len(coordinate_chunks), 8)  # Cap at 8 for memory
    
#     logging.info(f"Using {num_workers} worker processes")
    
#     all_results = []
#     total_valid_points = 0
#     total_abs_residual = 0.0
#     total_sq_residual = 0.0
    
#     #PARALLEL
#     with ProcessPoolExecutor(max_workers=num_workers) as executor:
#         futures = [executor.submit(process_chunk_optimized, args) for args in chunk_args]
        
#         completed_chunks = 0
#         for future in as_completed(futures):
#             try:
#                 results, valid_count, abs_sum, sq_sum = future.result()
                
#                 all_results.extend(results)
#                 total_valid_points += valid_count
#                 total_abs_residual += abs_sum
#                 total_sq_residual += sq_sum
                
#                 completed_chunks += 1
#                 if completed_chunks % 10 == 0:
#                     logging.info(f"Completed {completed_chunks}/{len(coordinate_chunks)} chunks")
                    
#             except Exception as exc:
#                 logging.error(f"Chunk processing failed: {exc}")
#                 continue
    
#     #STATISTICS
#     if total_valid_points > 0:
#         avg_residual = round(total_abs_residual / total_valid_points, 3)
#         rms_residual = round((total_sq_residual / total_valid_points) ** 0.5, 3)
#     else:
#         avg_residual = 0.0
#         rms_residual = 0.0
#         logging.warning("No valid data points found")
    
#     statistical_analyses = {
#         "AVG_RESIDUAL": avg_residual,
#         "RMS_RESIDUAL": rms_residual,
#         "VALID_POINTS": total_valid_points,
#         "PROCESSED_POINTS": total_points
#     }
    
#     end_time = time.time()
#     processing_time = round(end_time - start_time, 2)
    
#     logging.info(f"Hyper-optimized processing completed in {processing_time} seconds")
#     logging.info(f"Found {total_valid_points} valid points out of {total_points} total")
#     logging.info(f"Performance: {total_points/processing_time:.0f} points/second")
    
#     return all_results, statistical_analyses









@jit(nopython=True, parallel=True, cache=True)
def create_residual_grid(gsfc_data, model_data, x_coords, y_coords, x_indices, y_indices):
    """
    Ultra-fast JIT compiled function to create residual grid directly.
    
    Parameters
    ----------
    gsfc_data : numpy.ndarray
        2D array of GSFC ice mask data
    model_data : numpy.ndarray 
        2D array of model ice mask data
    x_coords : numpy.ndarray
        1D array of x coordinates
    y_coords : numpy.ndarray
        1D array of y coordinates
    x_indices : numpy.ndarray
        1D array of x indices for data arrays
    y_indices : numpy.ndarray
        1D array of y indices for data arrays
        
    Returns
    -------
    tuple
        (residual_grid, valid_count, abs_sum, sq_sum)
    """
    n_x = len(x_coords)
    n_y = len(y_coords)
    
    residual_grid = np.full((n_y, n_x), np.nan, dtype=np.float32)
    valid_count = 0
    abs_sum = 0.0
    sq_sum = 0.0
    
    for i in prange(n_x):
        x_idx = x_indices[i]
        for j in prange(n_y):
            y_idx = y_indices[j]
            
            gsfc_val = gsfc_data[y_idx, x_idx]
            model_val = model_data[y_idx, x_idx]
            
            if not (np.isnan(gsfc_val) or np.isnan(model_val)):
                residual = gsfc_val - model_val
                residual_grid[j, i] = residual
                valid_count += 1
                abs_sum += abs(residual)
                sq_sum += residual * residual
    
    return residual_grid, valid_count, abs_sum, sq_sum

def find_absolute_calving_per_year_direct_ds(gsfc, model, year):
    """
    Ultra-fast version that directly creates xarray Dataset without JSON intermediate.
    
    Uses JIT-compiled numpy operations for maximum speed.
    
    Parameters
    ----------
    gsfc : GSFCcalving
        The GSFC calving data object
    model : ModelCalving  
        The model calving data object
    year : int
        The year for which to find absolute calving data
        
    Returns
    -------
    tuple
        (xarray.Dataset, statistical_analyses)
    """
    start_time = time.time()
    logging.info(f"Starting direct Dataset creation for year {year}")
    
    gsfc_year = gsfc.ds.sel(year=year)
    model_year = model.ds.sel(time=year)
    
    if gsfc_year is None or model_year is None:
        raise ValueError(f"No data available for the year {year}")
    
    logging.info("Preparing optimized data arrays...")
    gsfc_data, model_data, x_coords, y_coords, x_indices, y_indices = prepare_data_arrays(
        gsfc_year, model_year
    )
    
    total_points = len(x_coords) * len(y_coords)
    logging.info(f"Processing {total_points} grid points...")
    
    # Use JIT compiled function for ultra-fast computation
    residual_grid, valid_count, abs_sum, sq_sum = create_residual_grid(
        gsfc_data, model_data, x_coords, y_coords, x_indices, y_indices
    )
    
    # Calculate statistics
    if valid_count > 0:
        avg_residual = round(abs_sum / valid_count, 3)
        rms_residual = round((sq_sum / valid_count) ** 0.5, 3)
    else:
        avg_residual = 0.0
        rms_residual = 0.0
        logging.warning("No valid data points found")
    
    statistical_analyses = {
        "AVG_RESIDUAL": avg_residual,
        "RMS_RESIDUAL": rms_residual,
        "VALID_POINTS": int(valid_count),
        "PROCESSED_POINTS": total_points
    }
    
    # Create xarray Dataset directly
    ds = xr.Dataset(
        {
            'residual': (('y', 'x'), residual_grid),
            'gsfc_ice_mask': (('y', 'x'), gsfc_data[y_indices][:, x_indices]),
            'model_ice_mask': (('y', 'x'), model_data[y_indices][:, x_indices])
        },
        coords={
            'x': x_coords,
            'y': y_coords,
            'year': year
        },
        attrs={
            'title': f'Calving comparison residuals for {year}',
            'avg_residual': avg_residual,
            'rms_residual': rms_residual,
            'valid_points': int(valid_count),
            'processed_points': total_points
        }
    )
    
    end_time = time.time()
    processing_time = round(end_time - start_time, 2)
    
    logging.info(f"Direct Dataset creation completed in {processing_time} seconds")
    logging.info(f"Found {valid_count} valid points out of {total_points} total")
    logging.info(f"Performance: {total_points/processing_time:.0f} points/second")
    
    return ds, statistical_analyses