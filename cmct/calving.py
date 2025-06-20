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
    def sftgif(self):
        return self.ds["sftgif"]
    
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
        print(f"SFTGIF: {self.sftgif.values}")
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

def find_absolute_calving_per_year(gsfc, model, year):
   """
   Find absolute calving data for a specified year.

   Parameters
   ----------
   gsfc : GSFCcalving
       The GSFC calving data object.
   model : ModelCalving
       The model calving data object.
   year : int
       The year for which to find absolute calving data.

   Returns
   -------
   xarray.Dataset
       The absolute calving data for the specified year.
   """
   logging.info(f"Finding absolute calving data for year {year}")
   
   gsfc_year = gsfc.ds.sel(year=year)
   model_year = model.ds.sel(time=year)
   logging.info(f"Selected data for year {year} from both datasets")

   if gsfc_year is None or model_year is None:
       logging.error(f"No data available for the year {year} in either GSFC or model datasets")
       raise ValueError(f"No data available for the year {year} in either GSFC or model datasets.")
   
   else:
       logging.info("Starting to process grid points")
       calving_data_list = []
       processed_points = 0
       valid_points = 0
       agg_residual = 0
       sq_agg_residual = 0
       
       for x in model_year.x.values:
           for y in model_year.y.values:
               processed_points += 1
               if np.isnan(gsfc_year.ice_mask.sel(x=x, y=y).values) or np.isnan(model_year.sftgif.sel(x=x, y=y).values):
                   continue
               else:
                   valid_points += 1
                   residual = round(float(gsfc_year.ice_mask.sel(x=x, y=y).values - model_year.sftgif.sel(x=x, y=y).values), 3)
                   agg_residual += abs(residual)
                   sq_agg_residual += residual ** 2
                   calving_data = {
                       
                       'x': round(float(x), 3),
                       'y': round(float(y), 3),
                       'gsfc_ice_mask': round(float(gsfc_year.ice_mask.sel(x=x, y=y).values), 3),
                       'model_sftgif': round(float(model_year.sftgif.sel(x=x, y=y).values), 3),
                       'residual':  residual
                   }
                   calving_data_list.append(calving_data)
        
       
       RMS = (sq_agg_residual/valid_points)**(1/2)


       statistical_analyses = {"AVG_RESIDUAL": round(agg_residual/valid_points, 3), "RMS_RESIDUAL": round(RMS, 3), "VALID_POINTS": valid_points, "PROCESSED_POINTS": processed_points}
       
       logging.info(f"Processed {processed_points} grid points, found {valid_points} valid points")
       logging.info(f"Returning {len(calving_data_list)} calving data records")
       return calving_data_list, statistical_analyses
   
def process_coordinate_batch(args):
    """
    Process a batch of coordinates and return results.
    
    Parameters
    ----------
    args : tuple
        (x_batch, y_batch, gsfc_data, model_data, batch_id)
    
    Returns
    -------
    tuple
        (calving_data_list, valid_points, sum_abs_residual, sum_sq_residual)
    """
    x_batch, y_batch, gsfc_data, model_data, batch_id = args
    
    calving_data_list = []
    valid_points = 0
    sum_abs_residual = 0.0
    sum_sq_residual = 0.0
    
    for x in x_batch:
        for y in y_batch:
            try:
                gsfc_val = float(gsfc_data.sel(x=x, y=y, method='nearest').values)
                model_val = float(model_data.sel(x=x, y=y, method='nearest').values)
                
                if np.isnan(gsfc_val) or np.isnan(model_val):
                    continue
                
                residual = round(gsfc_val - model_val, 3)
                
                calving_data = {
                    'x': round(float(x), 3),
                    'y': round(float(y), 3),
                    'gsfc_ice_mask': round(gsfc_val, 3),
                    'model_sftgif': round(model_val, 3),
                    'residual': residual
                }
                calving_data_list.append(calving_data)
                
                valid_points += 1
                sum_abs_residual += abs(residual)
                sum_sq_residual += residual ** 2
                
            except Exception as e:
                continue
    
    return calving_data_list, valid_points, sum_abs_residual, sum_sq_residual

def create_coordinate_batches(x_coords, y_coords, batch_size=50):
    """
    Split coordinates into smaller batches for parallel processing.
    
    Parameters
    ----------
    x_coords : array
        X coordinates
    y_coords : array  
        Y coordinates
    batch_size : int
        Number of coordinates per batch
        
    Returns
    -------
    list
        List of (x_batch, y_batch) tuples
    """
    batches = []

    for i in range(0, len(x_coords), batch_size):
        x_batch = x_coords[i:i + batch_size]
        
        for j in range(0, len(y_coords), batch_size):
            y_batch = y_coords[j:j + batch_size]
            batches.append((x_batch, y_batch))
    
    return batches

def find_absolute_calving_per_year_parallel(gsfc, model, year, num_workers=None, batch_size=50):
    """
    Find absolute calving data for a specified year using parallel processing.
    
    This version is simpler and fixes the indexing issues by processing 
    coordinates in small batches rather than using complex vectorization.

    Parameters
    ----------
    gsfc : GSFCcalving
        The GSFC calving data object.
    model : ModelCalving
        The model calving data object.
    year : int
        The year for which to find absolute calving data.
    num_workers : int, optional
        Number of worker processes. If None, uses all CPU cores.
    batch_size : int, optional
        Number of coordinates to process per batch. Default is 50.

    Returns
    -------
    tuple
        (calving_data_list, statistical_analyses)
    """
    start_time = time.time()
    logging.info(f"Finding absolute calving data for year {year} using parallel processing")
    
    gsfc_year = gsfc.ds.sel(year=year)
    model_year = model.ds.sel(time=year)
    logging.info(f"Selected data for year {year} from both datasets")

    if gsfc_year is None or model_year is None:
        logging.error(f"No data available for the year {year}")
        raise ValueError(f"No data available for the year {year}")
    
    x_coords = model_year.x.values
    y_coords = model_year.y.values
    total_points = len(x_coords) * len(y_coords)
    
    logging.info(f"Total grid points to process: {total_points}")
    logging.info(f"Batch size: {batch_size}")
    
    coordinate_batches = create_coordinate_batches(x_coords, y_coords, batch_size)
    logging.info(f"Created {len(coordinate_batches)} batches")
    
    batch_args = [
        (x_batch, y_batch, gsfc_year.ice_mask, model_year.sftgif, i)
        for i, (x_batch, y_batch) in enumerate(coordinate_batches)
    ]
    
    if num_workers is None:
        num_workers = min(cpu_count(), len(coordinate_batches))
    
    logging.info(f"Using {num_workers} worker processes")
    
    all_calving_data = []
    total_valid_points = 0
    total_abs_residual = 0.0
    total_sq_residual = 0.0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_coordinate_batch, args) for args in batch_args]
    
        completed_batches = 0
        for future in as_completed(futures):
            try:
                calving_data_list, valid_points, abs_residual, sq_residual = future.result()
                
                all_calving_data.extend(calving_data_list)
                total_valid_points += valid_points
                total_abs_residual += abs_residual
                total_sq_residual += sq_residual            
                completed_batches += 1
                
                if completed_batches % 20 == 0:
                    logging.info(f"Completed {completed_batches}/{len(coordinate_batches)} batches")
                    
            except Exception as exc:
                logging.error(f"Batch processing failed: {exc}")
                continue
    
    if total_valid_points > 0:
        avg_residual = round(total_abs_residual / total_valid_points, 3)
        rms_residual = round((total_sq_residual / total_valid_points) ** 0.5, 3)
    else:
        avg_residual = 0.0
        rms_residual = 0.0
        logging.warning("No valid data points found")
    
    statistical_analyses = {
        "AVG_RESIDUAL": avg_residual,
        "RMS_RESIDUAL": rms_residual,
        "VALID_POINTS": total_valid_points,
        "PROCESSED_POINTS": total_points
    }
    
    end_time = time.time()
    processing_time = round(end_time - start_time, 2)
    
    logging.info(f"Processing completed in {processing_time} seconds")
    logging.info(f"Found {total_valid_points} valid points out of {total_points} total points")
    logging.info(f"Returning {len(all_calving_data)} calving data records")
    
    return all_calving_data, statistical_analyses

def find_absolute_calving_per_year_simple(gsfc, model, year):
    """
    Simple non-parallel version for comparison or debugging.
    """
    logging.info(f"Finding absolute calving data for year {year} (simple version)")
    
    gsfc_year = gsfc.ds.sel(year=year)
    model_year = model.ds.sel(time=year)

    if gsfc_year is None or model_year is None:
        raise ValueError(f"No data available for the year {year}")
    
    calving_data_list = []
    valid_points = 0
    total_abs_residual = 0.0
    total_sq_residual = 0.0
    total_points = 0
    
    for x in model_year.x.values:
        for y in model_year.y.values:
            total_points += 1
            
            try:
                gsfc_val = float(gsfc_year.ice_mask.sel(x=x, y=y).values)
                model_val = float(model_year.sftgif.sel(x=x, y=y).values)
                
                if np.isnan(gsfc_val) or np.isnan(model_val):
                    continue
                    
                residual = round(gsfc_val - model_val, 3)
                
                calving_data = {
                    'x': round(float(x), 3),
                    'y': round(float(y), 3),
                    'gsfc_ice_mask': round(gsfc_val, 3),
                    'model_sftgif': round(model_val, 3),
                    'residual': residual
                }
                calving_data_list.append(calving_data)
                
                valid_points += 1
                total_abs_residual += abs(residual)
                total_sq_residual += residual ** 2
                
            except Exception:
                continue
    
    if valid_points > 0:
        avg_residual = round(total_abs_residual / valid_points, 3)
        rms_residual = round((total_sq_residual / valid_points) ** 0.5, 3)
    else:
        avg_residual = 0.0
        rms_residual = 0.0
    
    statistical_analyses = {
        "AVG_RESIDUAL": avg_residual,
        "RMS_RESIDUAL": rms_residual,
        "VALID_POINTS": valid_points,
        "PROCESSED_POINTS": total_points
    }
    
    return calving_data_list, statistical_analyses


"""
# Parallel version (recommended)
calving_data, stats = find_absolute_calving_per_year_parallel(
    gsfc, model, year, num_workers=4, batch_size=50
)

# Simple version (for debugging or small datasets)
calving_data, stats = find_absolute_calving_per_year_simple(gsfc, model, year)
"""

# def json_netCDF(gsfc, output_path):
    