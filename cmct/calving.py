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
from matplotlib import rc
from shapely.geometry import Point
from .time_utils import check_datarange

rc("mathtext", default="regular")


ded load_gsfc_calving(filepath):



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
        gsfc = Modelcalving(filepath)
    except Exception as error:
        print("Error: Failed to load GSFC dataset.")
        print(error)
        gsfc = None
    return gsfc



class Modelcalving:
    def __init__(self, nc_path):
        # Open as xarray Dataset
        self.ds = xr.open_dataset(nc_path)
        # Direct access to variables as attributes
        self.lat = self.ds["lat"]
        self.lat_bnds = self.ds["lat_bnds"]
        self.lon = self.ds["lon"]
        self.lon_bnds = self.ds["lon_bnds"]
        self.time = self.ds["time"]
        self.sftgif = self.ds["sftgif"]
        self.x = self.ds["x"]
        self.y = self.ds["y"]
        
        
        
        self.x_min = self.x.min().item()
        self.x_max = self.x.max().item()
        
        self.y_min = self.y.min().item()
        self.y_max = self.y.max().item()
        
    def close(self):
        self.ds.close()

    def _set_times_as_datetimes(self, days):
        return np.datetime64("2002-01-01T00:00:00") + np.array(
            [int(d * 24) for d in days], dtype="timedelta64[h]"
        )

    def print_info(self):
        print("GSFC Calving Data:")
        print(f"Latitude: {self.lat.values}")
        print(f"Longitude: {self.lon.values}")
        print(f"Time: {self._set_times_as_datetimes(self.time.values)}")
        print(f"SFTGIF: {self.sftgif.values}")
        print(f"X coordinates: {self.x.values}")
        print(f"Y coordinates: {self.y.values}")

def calc_obs_delta_cmwe(obs, start_date, end_date):
    t_0 = np.datetime64(start_date)
    t_1 = np.datetime64(end_date)

    i_0 = np.abs(obs.times_start - t_0).argmin()
    i_1 = np.abs(obs.times_end - t_1).argmin()

    return obs.cmwe[:, i_1] - obs.cmwe[:, i_0]


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
    
    
def find_absolute_calving(gsfc, start_date, end_date):
    """
    Find absolute calving data within a specified date range.

    Parameters
    ----------
    gsfc : GSFCcalving
        The GSFC calving data object.
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.

    Returns
    -------
    xarray.Dataset
        The absolute calving data within the specified date range.
    """
    if not check_datarange(gsfc.time, start_date, end_date):
        raise ValueError("Date range is outside the available data range.")

    global min_lat
    global min_lon
    
    global max_lat
    global max_lon

    
    
    # Convert dates to numpy datetime64
    t_start = np.datetime64(start_date)
    t_end = np.datetime64(end_date)

    # Filter the dataset based on time
    mask = (gsfc.time >= t_start) & (gsfc.time <= t_end)
    
    return gsfc.ds.sel(time=mask)