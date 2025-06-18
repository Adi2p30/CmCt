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