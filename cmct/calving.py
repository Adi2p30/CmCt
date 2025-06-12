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


def load_gsfc_calving(filepath):
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
        gsfc = GSFCcalving(filepath)
    except Exception as error:
        print("Error: Failed to load GSFC dataset.")
        print(error)
        gsfc = None
    return gsfc



class GSFCcalving:
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

