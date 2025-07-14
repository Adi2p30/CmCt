import datetime
import os
from datetime import timedelta
import geopandas as gpd
import h5py
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
from scipy import stats
import cftime
import xarray as xr
import logging
import gc
from numba import jit, prange
from concurrent.futures import ProcessPoolExecutor, as_completed
from matplotlib import rc
from cmct.shapefile_utils import * 
import shapely.geometry
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
from .shapefile_utils import shapefile_to_xy, get_nonzero_indices, scaling_shape_to_target
import time
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# from .time_utils import check_datarange
rc("mathtext", default="regular")

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


def load_gsfc_calving(filepath, basins=None):
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
        gsfc = GSFCcalving(filepath, basins)
        
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
        logging.error("Error: Failed to load Model dataset.")
        logging.error(error)
        model_res = None
    return model_res

def load_residuals(residuals):
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
        residuals = Residual(residuals)
    except Exception as error:
        logging.error("Error: Failed to load residuals dataset.")
        logging.error(error)
        residuals = None
    return residuals


class GSFCcalving:
    def __init__(self, nc_path, basins = None):
        # Open as xarray Dataset
        
        self.ds = xr.open_dataset(nc_path, autoclose=True, engine='netcdf4',use_cftime=True)
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
        self.ds = xr.open_dataset(nc_path, autoclose=True, engine='netcdf4',use_cftime=True)
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
        
#--------------------
# NOTE: ERROR CHECKING
#--------------------

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
    """
    Check if the requested date range is available in both datasets.
    Handles different time formats flexibly (cftime, datetime, float years, etc.)
    
    Parameters
    ----------
    gsfc_time : list or array
        Time values from GSFC dataset
    model_time : list or array
        Time values from model dataset  
    start_date : int
        Start year for comparison
    end_date : int
        End year for comparison
    """
    import cftime
    import numpy as np
    from datetime import datetime
    
    try:
        import pandas as pd
        HAS_PANDAS = True
    except ImportError:
        HAS_PANDAS = False
    
    def extract_year(time_val):
        """Extract year from various time formats"""
        if isinstance(time_val, (int, float)):
            # Assume it's already a year (possibly decimal year)
            return float(time_val)
        elif hasattr(time_val, 'year'):
            # cftime, datetime, or similar objects with year attribute
            return float(time_val.year)
        elif isinstance(time_val, np.datetime64):
            # numpy datetime64
            if HAS_PANDAS:
                return float(pd.to_datetime(time_val).year)
            else:
                # Fallback without pandas
                return float(str(time_val)[:4])
        elif isinstance(time_val, str):
            # Try to parse string dates
            if HAS_PANDAS:
                try:
                    dt = pd.to_datetime(time_val)
                    return float(dt.year)
                except:
                    pass
            # If pandas parsing fails or not available, try to extract year from string
            import re
            year_match = re.search(r'\d{4}', str(time_val))
            if year_match:
                return float(year_match.group())
        
        # Fallback: try to convert to float directly
        try:
            return float(time_val)
        except:
            raise ValueError(f"Cannot extract year from time value: {time_val} (type: {type(time_val)})")
    
    # Convert time arrays to years
    try:
        gsfc_years = [extract_year(t) for t in gsfc_time]
        model_years = [extract_year(t) for t in model_time]
    except Exception as e:
        print(f"Error processing time values: {e}")
        print(f"GSFC time sample: {gsfc_time[:3] if len(gsfc_time) > 3 else gsfc_time}")
        print(f"Model time sample: {model_time[:3] if len(model_time) > 3 else model_time}")
        raise
    
    # Sort and get min/max years
    gsfc_years.sort()
    gsfc_time_min = gsfc_years[0]
    gsfc_time_max = gsfc_years[-1]
    
    model_years.sort()
    model_time_min = model_years[0]
    model_time_max = model_years[-1]

    minimum_time = max(gsfc_time_min, model_time_min)
    maximum_time = min(gsfc_time_max, model_time_max)
    
    print(f"GSFC data range: {gsfc_time_min:.1f} to {gsfc_time_max:.1f}")
    print(f"Model data range: {model_time_min:.1f} to {model_time_max:.1f}")
    print(f"Overlapping range: {minimum_time:.1f} to {maximum_time:.1f}")
    print(f"Requested range: {start_date} to {end_date}")

    if not (minimum_time <= start_date <= end_date <= maximum_time):
        raise ValueError(f"Date range {start_date} to {end_date} is outside the available overlapping data range: {minimum_time:.1f} to {maximum_time:.1f}.")
    else: 
        print(f"✓ The selected dates {start_date} to {end_date} are within the overlapping data range.")


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
    
    
def load_basins(cmct_dir, basins):
    basin_data = load_basin_polygons(cmct_dir + "/bin/Calving/GRE_Basins_IMBIE2_v1.3/GRE_Basins_IMBIE2_v1.3.shp")
    if basins == "all":
        selected_basins = [
            "CW", "NE", "SE", "SW", "NO", "NW", "unassigned"
        ]
    else:
        selected_basins = basins 
   
    # Filter basin_data to only include selected basins
    basin_polygons = {}
    
    for basin_name in selected_basins:
        if basin_name in basin_data:
            basin_polygons[basin_name] = basin_data[basin_name]

    return basin_polygons, selected_basins



#--------------------
# NOTE: RESIDUAL CALCULATION
#--------------------



"""
SIMPLE (LEGACY) -> HYPER OPTIMIZED JSON (LEGACY) -> NO JSON JIT OPTIMIZED (CURRENT)
=
"""

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
    
    # Safe conversion of ice_mask data to float32 with NaN handling
    gsfc_ice_mask = gsfc_year.ice_mask.values
    model_ice_mask = model_year.ice_mask.values
    
    # Convert to float32 safely, handling any non-numeric values
    try:
        gsfc_data = gsfc_ice_mask.astype(np.float32)
    except (ValueError, TypeError):
        # If conversion fails, create float32 array and copy valid values
        gsfc_data = np.full(gsfc_ice_mask.shape, np.nan, dtype=np.float32)
        try:
            # Try to convert to float64 first, then check for finite values
            temp_data = gsfc_ice_mask.astype(np.float64)
            numeric_mask = np.isfinite(temp_data)
            gsfc_data[numeric_mask] = temp_data[numeric_mask].astype(np.float32)
        except (ValueError, TypeError):
            # If all else fails, leave as NaN
            pass
    
    try:
        model_data = model_ice_mask.astype(np.float32)
    except (ValueError, TypeError):
        # If conversion fails, create float32 array and copy valid values
        model_data = np.full(model_ice_mask.shape, np.nan, dtype=np.float32)
        try:
            # Try to convert to float64 first, then check for finite values
            temp_data = model_ice_mask.astype(np.float64)
            numeric_mask = np.isfinite(temp_data)
            model_data[numeric_mask] = temp_data[numeric_mask].astype(np.float32)
        except (ValueError, TypeError):
            # If all else fails, leave as NaN
            pass
    
    # Clean up temporary variables
    del model_x, model_y, gsfc_x, gsfc_y, gsfc_ice_mask, model_ice_mask
    gc.collect()
    
    return gsfc_data, model_data, model_year.x.values.astype(np.float32), model_year.y.values.astype(np.float32), x_indices, y_indices


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
    face_sum = 0.0
    
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
                face_sum += residual
                abs_sum += abs(residual)
                sq_sum += residual * residual
    
    return residual_grid, valid_count, abs_sum, sq_sum, face_sum

@jit(nopython=True, parallel=True, cache=True)
def create_aligned_masks(gsfc_data, model_data, x_coords, y_coords, x_indices, y_indices):
    """
    JIT-compiled function to create aligned ice mask arrays.
    
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
        (gsfc_mask_aligned, model_mask_aligned)
    """
    n_x = len(x_coords)
    n_y = len(y_coords)
    
    gsfc_mask_aligned = np.full((n_y, n_x), np.nan, dtype=np.float32)
    model_mask_aligned = np.full((n_y, n_x), np.nan, dtype=np.float32)
    
    for i in prange(n_x):
        x_idx = x_indices[i]
        for j in prange(n_y):
            y_idx = y_indices[j]
            gsfc_mask_aligned[j, i] = gsfc_data[y_idx, x_idx]
            model_mask_aligned[j, i] = model_data[y_idx, x_idx]
    
    return gsfc_mask_aligned, model_mask_aligned

def find_calving_per_year_direct_ds(gsfc, model, year, basin='all'):
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

    # try:    
    #     gsfc_date_type = gsfc.ds["year"].dtype
    #     model_date_type = model.ds["time"].dtype
        
    #     if gsfc_date_type != model_date_type:
    #         logging.warning(f"Date types do not match: GSFC {gsfc_date_type}, Model {model_date_type}")
        
    gsfc_year = gsfc.ds.sel(year=year)
    model_year = model.ds.sel(time=cftime.DatetimeNoLeap(year, 1, 1, 0, 0, 0, 0, has_year_zero=True))

    if gsfc_year is None or model_year is None:
        raise ValueError(f"No data available for the year {year}")
    
    gsfc_data, model_data, x_coords, y_coords, x_indices, y_indices = prepare_data_arrays(
        gsfc_year, model_year
    )
    
    # Clean up year data after extracting needed arrays
    del gsfc_year, model_year
    gc.collect()
    
    total_points = len(x_coords) * len(y_coords)
    # logging.info(f"Processing {total_points} grid points...")
    
    # Use JIT compiled function for ultra-fast computation
    residual_grid, valid_count, abs_sum, sq_sum, sum = create_residual_grid(
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
        "ABS_AVG_RESIDUAL": avg_residual,
        "RMS_RESIDUAL": rms_residual,
        "SUM_RESIDUAL": sum,
        "VALID_POINTS": int(valid_count),
        "PROCESSED_POINTS": total_points,
    }
    
    # Create aligned ice mask arrays for the target grid using JIT function
    gsfc_mask_aligned, model_mask_aligned = create_aligned_masks(
        gsfc_data, model_data, x_coords, y_coords, x_indices, y_indices
    )
    
    # Create xarray Dataset directly
    ds = xr.Dataset(
        {
            'residual': (('y', 'x'), residual_grid),
            'gsfc_ice_mask': (('y', 'x'), gsfc_mask_aligned),
            'model_ice_mask': (('y', 'x'), model_mask_aligned)
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
            'sum_residual': sum,
            
            'valid_points': int(valid_count),
            'processed_points': total_points
        }
    )
    
    # Clean up large temporary arrays
    del gsfc_data, model_data, residual_grid, x_indices, y_indices
    del gsfc_mask_aligned, model_mask_aligned
    del abs_sum, sq_sum, sum, valid_count
    gc.collect()
    
    end_time = time.time()
    processing_time = round(end_time - start_time, 2)
    
    # logging.info(f"Found {ds.attrs['valid_points']} valid points out of {total_points} total")    
    return ds, statistical_analyses


        
#--------------------
# NOTE: BASIN ASSIGNMENT
#--------------------
def standardize_basin_projection(basin_polygons_dict, gsfc_data, target_crs='EPSG:3413'):
    """
    Standardize basin projections to match data coordinate system and ensure proper alignment.
    
    Parameters:
    -----------
    basin_polygons_dict : dict
        Dictionary containing basin polygons
    gsfc_data : GSFCcalving
        The GSFC data object containing coordinate information
    target_crs : str
        Target coordinate reference system (ISMIP6 standard projection)
    
    Returns:
    --------
    dict : Reprojected and aligned basin polygons
    """
    import pyproj
    from shapely.ops import transform
    from shapely.geometry import Polygon
    
    # Get data coordinate bounds and resolution
    x_coords = gsfc_data.ds.x.values
    y_coords = gsfc_data.ds.y.values
    
    # Create coordinate bounds
    x_min, x_max = float(x_coords.min()), float(x_coords.max())
    y_min, y_max = float(y_coords.min()), float(y_coords.max())
    
    print(f"Data coordinate bounds: X=[{x_min:.0f}, {x_max:.0f}], Y=[{y_min:.0f}, {y_max:.0f}]")
    
    # Define ISMIP6 Greenland projection (EPSG:3413)
    proj_ismip6 = pyproj.CRS.from_epsg(3413)
    
    # Transform basins to ISMIP6 projection if needed
    standardized_basins = {}
    
    for basin_name, polygon in basin_polygons_dict.items():
        if basin_name == 'unassigned':
            continue
            
        # Ensure polygon is in ISMIP6 projection
        if hasattr(polygon, 'exterior'):
            # Get coordinates
            coords = list(polygon.exterior.coords)
            
            # Check if coordinates are already in reasonable range for ISMIP6
            x_poly = [coord[0] for coord in coords]
            y_poly = [coord[1] for coord in coords]
            
            # If coordinates seem to be in lat/lon (small values), transform them
            if max(abs(x) for x in x_poly) < 180 and max(abs(y) for y in y_poly) < 90:
                print(f"Converting {basin_name} from lat/lon to ISMIP6 projection...")
                
                # Transform from WGS84 to ISMIP6
                transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:3413', always_xy=True)
                
                def transform_coords(x, y, z=None):
                    return transformer.transform(x, y)
                
                polygon = transform(transform_coords, polygon)
                coords = list(polygon.exterior.coords)
                x_poly = [coord[0] for coord in coords]
                y_poly = [coord[1] for coord in coords]
            
            # Clip polygon to data bounds with buffer
            buffer_pct = 0.1  # 10% buffer
            x_buffer = (x_max - x_min) * buffer_pct
            y_buffer = (y_max - y_min) * buffer_pct
            
            # Create clipping box
            clip_box = Polygon([
                (x_min - x_buffer, y_min - y_buffer),
                (x_max + x_buffer, y_min - y_buffer),
                (x_max + x_buffer, y_max + y_buffer),
                (x_min - x_buffer, y_max + y_buffer)
            ])
            
            # Clip polygon to data extent
            try:
                clipped_polygon = polygon.intersection(clip_box)
                
                if clipped_polygon.is_empty:
                    print(f"Warning: {basin_name} basin is outside data extent")
                    continue
                    
                # Handle MultiPolygon result
                if hasattr(clipped_polygon, 'geoms'):
                    # Take the largest polygon if multiple
                    clipped_polygon = max(clipped_polygon.geoms, key=lambda p: p.area)
                
                standardized_basins[basin_name] = clipped_polygon
                
                # Print basin extent for verification
                bounds = clipped_polygon.bounds
                print(f"{basin_name}: X=[{bounds[0]:.0f}, {bounds[2]:.0f}], Y=[{bounds[1]:.0f}, {bounds[3]:.0f}]")
                
            except Exception as e:
                print(f"Error processing {basin_name}: {e}")
                standardized_basins[basin_name] = polygon
    
    print(f"Standardized {len(standardized_basins)} basins")
    return standardized_basins



# Ultra-optimized point-in-polygon using NumPy batch processing
@jit(nopython=True)
def vectorized_point_in_polygon_batch(points_x, points_y, poly_x, poly_y):
    """
    Ultra-optimized batch point-in-polygon test.
    Processes multiple points against one polygon simultaneously using ray casting.
    """
    n_points = len(points_x)
    n_poly = len(poly_x)
    inside = np.zeros(n_points, dtype=np.bool_)
    
    for i in range(n_points):
        x, y = points_x[i], points_y[i]
        count = 0
        
        j = n_poly - 1
        for k in range(n_poly):
            xi, yi = poly_x[k], poly_y[k]
            xj, yj = poly_x[j], poly_y[j]
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                count += 1
            j = k
        
        inside[i] = count % 2 == 1
    
    return inside

def basin_assignment(x_coords, y_coords, basin_polygons_arrays, batch_size=50000):
    """
    Ultra-optimized basin assignment using batch processing for the entire dataset.
    Processes points in batches to manage memory and improve cache efficiency.
    """
    n_points = len(x_coords)
    n_basins = len(basin_polygons_arrays)
    basin_assignments = np.full(n_points, -1, dtype=np.int32)
    
    total_start_time = time.time()
    
    # Process in batches
    for batch_start in range(0, n_points, batch_size):
        batch_end = min(batch_start + batch_size, n_points)
        batch_x = x_coords[batch_start:batch_end]
        batch_y = y_coords[batch_start:batch_end]
        batch_size_actual = len(batch_x)
        
        batch_num = batch_start // batch_size + 1
        total_batches = (n_points - 1) // batch_size + 1
        
        batch_start_time = time.time()
        
        # Test each basin for this batch
        for basin_idx in range(n_basins):
            poly_x, poly_y = basin_polygons_arrays[basin_idx]
            
            # Find points not yet assigned
            unassigned_mask = basin_assignments[batch_start:batch_end] == -1
            if not np.any(unassigned_mask):
                continue
                
            unassigned_indices = np.where(unassigned_mask)[0]
            unassigned_x = batch_x[unassigned_indices]
            unassigned_y = batch_y[unassigned_indices]
            
            # Test unassigned points against current basin
            inside_mask = vectorized_point_in_polygon_batch(unassigned_x, unassigned_y, poly_x, poly_y)
            
            # Assign points that are inside this basin
            global_indices = batch_start + unassigned_indices[inside_mask]
            basin_assignments[global_indices] = basin_idx
            
            # Clean up temporary arrays
            del unassigned_indices, unassigned_x, unassigned_y, inside_mask, global_indices
        
        # Clean up batch variables
        del batch_x, batch_y, unassigned_mask
        
        batch_time = time.time() - batch_start_time
        points_per_sec = batch_size_actual / batch_time if batch_time > 0 else 0
        
        # Force garbage collection every few batches
        if batch_num % 10 == 0:
            gc.collect()
    
    total_time = time.time() - total_start_time
    gc.collect()  # Final cleanup
    return basin_assignments, total_time


def find_basin_calving_per_year_direct_ds(gsfc, model, year, basin_mask):
    """
    Find calving data for a specific basin using direct Dataset creation.
    
    Parameters
    ----------
    gsfc : GSFCcalving
        The GSFC calving data object
    model : ModelCalving  
        The model calving data object
    year : int
        The year for which to find absolute calving data
    basin_mask : numpy.ndarray
        2D boolean array indicating the basin mask
        
    Returns
    -------
    tuple
        (xarray.Dataset, statistical_analyses)
    """
    ds, stats = find_calving_per_year_direct_ds(gsfc, model, year)
    
    # Apply basin mask to the residuals
    ds['residual'] = ds['residual'].where(basin_mask)
    
    return ds, stats

class Residual:
    
    def __init__(self, residuals):
        
        """
        Initialize the Residual class with GSFC and model calving data.

        Parameters
        ----------
        gsfc_calving : GSFCcalving
            The GSFC calving data object.
        model_calving : Modelcalving
            The model calving data object.
        """
        self.ds = residuals
        self.basins = None
        
def rotated_data_year(gsfc, year):
    """
    Get the residual data for a specific year.

    Parameters
    ----------
    year : int
        The year for which to retrieve the residual data.

    Returns
    -------
    xarray.DataArray
        The residual data for the specified year.
    """
    if "year" not in gsfc.ds.dims:
        raise ValueError("The dataset does not contain a 'year' dimension.")
    
    data = gsfc.ds.sel(year=year).residual
    return np.rot90(data, k=0)

def fit_basins_per_year(gsfc, basin_aggregation = False, shape_file = None, year = 2006):
    """
    Fit residuals by basins using the provided basin mask.

    Parameters
    ----------
    basin_mask : numpy.ndarray
        2D boolean array indicating the basin mask.

    Returns
    -------
    dict
        Dictionary with basin names as keys and aggregated statistics as values.
    """
    if not basin_aggregation:
        exit("Basin aggregation is not enabled. Please set basin_aggregation to True to proceed with basin-wise plotting.")
    
    shape_file_gdf = shapefile_to_xy(shape_file)
    gdf_shape = [float(shape_file_gdf.x.min()), float(shape_file_gdf.x.max()), float(shape_file_gdf.y.min()), float(shape_file_gdf.y.max())]

    rotated_data = rotated_data_year(gsfc, year)
    rotated_data_shape = get_nonzero_indices(rotated_data)
    shape_file_gdf = scaling_shape_to_target(shape_file_gdf, rotated_data_shape)
    new_gdf_shape = [float(shape_file_gdf.x.min()), float(shape_file_gdf.x.max()), float(shape_file_gdf.y.min()), float(shape_file_gdf.y.max())]
    
    # Clean up temporary variables
    del rotated_data
    gc.collect()
    
    return gdf_shape, new_gdf_shape, rotated_data_shape
    
def aggregating_basins(gsfc, basin_polygons_dict, years=None):
    """
    Aggregate residuals by basins for multiple years, creating a structured dataset.
    
    Parameters
    ----------
    basin_polygons_dict : dict
        Dictionary with basin names as keys and shapely.geometry.Polygon objects as values
    years : list, optional
        List of years to process. If None, processes all years in the dataset
        
    Returns
    -------
    dict
        Nested dictionary structure: {year: {basin: {'x': [], 'y': [], 'data_value': []}}}
    """
    if years is None:
        years = gsfc.ds.year.values

    # Convert basin polygons to numpy arrays for optimized processing
    basin_names = list(basin_polygons_dict.keys())
    basin_polygons_arrays = []
    
    for basin_name in basin_names:
        polygon = basin_polygons_dict[basin_name]
        coords = list(polygon.exterior.coords)
        x_coords = np.array([coord[0] for coord in coords], dtype=np.float64)
        y_coords = np.array([coord[1] for coord in coords], dtype=np.float64)
        basin_polygons_arrays.append((x_coords, y_coords))
        
        # Clean up temporary variables
        del polygon, coords, x_coords, y_coords
    
    # Initialize the result structure
    basin_aggregated_data = {}
    
    # Process each year
    for year in years:
        logging.info(f"Processing year {year}...")
        
        # Get data for this year
        year_data = gsfc.ds.sel(year=year)
        rotated_data = rotated_data_year(gsfc, year)

        # Get coordinate information
        if hasattr(year_data, 'x') and hasattr(year_data, 'y'):
            x_coords_1d = year_data.x.values
            y_coords_1d = year_data.y.values
            
        else:
            # Fallback: generate coordinates from data shape
            height, width = rotated_data.shape
            # You may need to adjust these bounds based on your data
            x_min, x_max = -1000000, 1000000  # Adjust as needed
            y_min, y_max = -4000000, 0        # Adjust as needed
            x_coords_1d = np.linspace(x_min, x_max, width)
            y_coords_1d = np.linspace(y_min, y_max, height)
        
        # Create coordinate meshgrid
        x_coords_2d, y_coords_2d = np.meshgrid(x_coords_1d, y_coords_1d, indexing='xy')
        x_flat = x_coords_2d.flatten().astype(np.float64)
        y_flat = y_coords_2d.flatten().astype(np.float64)
        
        # Clean up coordinate grids
        del x_coords_2d, y_coords_2d, x_coords_1d, y_coords_1d
        gc.collect()
        
        # Execute basin assignment
        basin_assignments, _ = basin_assignment(
            x_flat, y_flat, basin_polygons_arrays, batch_size=100000
        )
        
        # Initialize basin points for this year
        basin_points = {basin_name: {'x': [], 'y': [], 'data_value': []} for basin_name in basin_names}
        basin_points['unassigned'] = {'x': [], 'y': [], 'data_value': []}
        
        # Flatten the data
        data_flat = rotated_data.flatten()
        
        # Clean up rotated_data as we now have flattened version
        del rotated_data
        gc.collect()
        
        # Process only valid data points
        valid_data_mask = ~np.isnan(data_flat)
        valid_indices = np.where(valid_data_mask)[0]
        
        # Clean up mask as we have indices
        del valid_data_mask
        
        # Assign points to basins
        for i in valid_indices:
            basin_idx = basin_assignments[i]
            x, y = x_flat[i], y_flat[i]
            data_value = data_flat[i]
            
            if basin_idx >= 0:
                basin_name = basin_names[basin_idx]
                basin_points[basin_name]['x'].append(x)
                basin_points[basin_name]['y'].append(y)
                basin_points[basin_name]['data_value'].append(data_value)
            else:
                basin_points['unassigned']['x'].append(x)
                basin_points['unassigned']['y'].append(y)
                basin_points['unassigned']['data_value'].append(data_value)
        
        # Clean up large arrays after processing
        del x_flat, y_flat, data_flat, basin_assignments, valid_indices
        gc.collect()
        
        # Convert lists to numpy arrays for better performance
        for basin_name in basin_points:
            for key in basin_points[basin_name]:
                basin_points[basin_name][key] = np.array(basin_points[basin_name][key])
        
        # Store the result for this year
        basin_aggregated_data[year] = basin_points
        
        # Print summary for this year
        total_valid = sum(len(basin_points[basin]['x']) for basin in basin_points)
        total_assigned = sum(len(basin_points[basin]['x']) for basin in basin_names)
        logging.info(f"Year {year}: {total_valid:,} valid points, {total_assigned:,} assigned to basins")
        
        
        # Clean up year_data
        del year_data
        gc.collect()
    
    # Clean up basin polygons arrays
    del basin_polygons_arrays
    gc.collect()
    
    return basin_aggregated_data


def calculate_basin_statistics(basin_dataset):
    """Calculate comprehensive statistics for each basin across all years"""
    basin_stats = {}
    
    for year, year_data in basin_dataset.items():
        if year not in basin_stats:
            basin_stats[year] = {}
            
        for basin_name, basin_data in year_data.items():
            if len(basin_data['data_value']) > 0:
                basin_stats[year][basin_name] = {
                    'count': len(basin_data['data_value']),
                    'mean': np.mean(basin_data['data_value']),
                    'std': np.std(basin_data['data_value']),
                    'min': np.min(basin_data['data_value']),
                    'max': np.max(basin_data['data_value']),
                    'rms': np.sqrt(np.mean(np.square(basin_data['data_value']))),
                    'rss': np.sum(np.square(basin_data['data_value'])),
                    'sum': np.sum(basin_data['data_value']),
                    'winsorized_mean': stats.mstats.winsorize(basin_data['data_value'], limits=0.05).mean(),
                    'outlier_weighted_mean': np.average(basin_data['data_value'], weights=np.abs(basin_data['data_value'] - np.median(basin_data['data_value'])))

                }
    
    return basin_stats

def format_basin_statistics(basin_names, stats):
    print("Basin Statistics Summary:")
    print("=" * 80)

    for year in sorted(stats.keys()):
        print(f"\nYear {year}:")
        print(f"{'Basin':<12} {'Count':<8} {'Sum':<10} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'RMS':<10} {'RSS':<10} {'Winsorized Mean':<10} {'Outlier Weighted Mean':<10}")
        print("-" * 80)
        
        for basin_name in ['CW', 'NE', 'SE', 'SW', 'NO', 'NW']:
            if basin_name in stats[year]:
                s = stats[year][basin_name]
                print(f"{basin_name:<12} {s['count']:<8} {s['sum']:<10.4f} {s['mean']:<10.4f} {s['std']:<10.4f} "
                    f"{s['min']:<10.4f} {s['max']:<10.4f} {s['rms']:<10.4f} {s['rss']:<10.4f} {s['winsorized_mean']:<10.4f} {s['outlier_weighted_mean']:<10.4f}")

        if 'unassigned' in stats[year]:
            s = stats[year]['unassigned']
            print(f"{'unassigned':<12} {s['count']:<8} {s['sum']:<10.4f} {s['mean']:<10.4f} {s['std']:<10.4f} "
                f"{s['min']:<10.4f} {s['max']:<10.4f} {s['rms']:<10.4f} {s['rss']:<10.4f} {s['winsorized_mean']:<10.4f} {s['outlier_weighted_mean']:<10.4f}")

        
#--------------------
# NOTE: RESIDUAL PLOTTING
#--------------------



def make_time_residuals_plot(stats, colors, basin_names):
    """
    Create a time series plot for each basin showing sum values over the years.
    """

    years = sorted(stats.keys())

    # Create a figure with subplots for each basin
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    fig.suptitle('Basin Sum Values Over Time (Individual Scales)', fontsize=16)

    # Flatten axes array for easier indexing
    axes = axes.flatten()

    for i, basin_name in enumerate(basin_names):
        ax = axes[i]
        
        # Extract sum values for this basin across all years
        basin_sums = []
        valid_years = []
        
        for year in years:
            if basin_name in stats[year]:
                basin_sums.append(stats[year][basin_name]['sum'])
                valid_years.append(year)
        
        # Plot the data
        if basin_sums:  # Only plot if there's data
            ax.plot(valid_years, basin_sums, marker='o', linewidth=2, 
                    color=colors[basin_name], markersize=6, label=basin_name)
            ax.fill_between(valid_years, basin_sums, alpha=0.3, color=colors[basin_name])
            
            # Set individual y-axis limits to highlight the delta for this basin
            y_min_basin = min(basin_sums)
            y_max_basin = max(basin_sums)
            y_range_basin = y_max_basin - y_min_basin
            
            # Add padding to make the changes more visible
            if y_range_basin > 0:
                padding = y_range_basin * 0.1  # 10% padding
                ax.set_ylim(y_min_basin - padding, y_max_basin + padding)
            else:
                # If no variation, still show some range around the value
                padding = abs(y_min_basin) * 0.01 if y_min_basin != 0 else 1
                ax.set_ylim(y_min_basin - padding, y_max_basin + padding)
            
            # Customize the subplot
            ax.set_title(f'{basin_name} Basin Sum Over Time', fontsize=12, fontweight='bold')
            ax.set_xlabel('Year')
            ax.set_ylabel('Sum Value')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add value labels on points
            for x, y in zip(valid_years, basin_sums):
                ax.annotate(f'{y:.1f}', (x, y), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=8)
            
            # Add range information to the title
            delta = y_max_basin - y_min_basin
            ax.set_title(f'{basin_name} Basin (Δ={delta:.1f})', fontsize=12, fontweight='bold')
            
        else:
            ax.text(0.5, 0.5, f'No data for {basin_name}', 
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{basin_name} Basin Sum Over Time', fontsize=12, fontweight='bold')

    # Remove the empty subplot (we have 7 basins, 8 subplots)
    if len(basin_names) < len(axes):
        axes[-1].remove()

    plt.tight_layout()
    plt.show()

    # Also create a combined plot showing all basins on one graph
    plt.figure(figsize=(12, 8))

    for basin_name in basin_names:
        basin_sums = []
        valid_years = []
        
        for year in years:
            if basin_name in stats[year]:
                basin_sums.append(stats[year][basin_name]['sum'])
                valid_years.append(year)
        
        if basin_sums:  # Only plot if there's data
            plt.plot(valid_years, basin_sums, marker='o', linewidth=2, 
                    color=colors[basin_name], markersize=6, label=basin_name)

    plt.title('All Basin Sum Values Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Sum Value', fontsize=12)
    plt.legend(title='Basin', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Print summary statistics with delta information
    print("\nSummary Statistics for Basin Sums:")
    print("=" * 80)
    for basin_name in basin_names:
        basin_sums = []
        for year in years:
            if basin_name in stats[year]:
                basin_sums.append(stats[year][basin_name]['sum'])
        
        if basin_sums:
            delta = max(basin_sums) - min(basin_sums)
            print(f"{basin_name:>12}: Min={min(basin_sums):8.4f}, Max={max(basin_sums):8.4f}, "
                f"Mean={np.mean(basin_sums):8.4f}, Std={np.std(basin_sums):8.4f}, Delta={delta:8.4f}")
        else:
            print(f"{basin_name:>12}: No data available")

    # Clean up
    del basin_sums, valid_years
    gc.collect()
