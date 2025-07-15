import datetime
import gc
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import timedelta
from multiprocessing import Pool, cpu_count

import cftime
import geopandas as gpd
import h5py
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import shapely.geometry
import xarray as xr
from matplotlib import rc
from numba import jit, prange
from scipy import stats

from cmct.shapefile_utils import *

from .shapefile_utils import (
    get_nonzero_indices,
    scaling_shape_to_target,
    shapefile_to_xy,
)


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


def load_residuals(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    try:
        residuals = Residual(residuals)
    except Exception as error:
        logging.error("Error: Failed to load residuals dataset.")
        logging.error(error)
        residuals = None
    return residuals


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
        self.basins = None
        

