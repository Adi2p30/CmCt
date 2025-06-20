import json
import xarray as xr
import numpy as np
import netCDF4 as nc


def json_to_netcdf(json_file):
    """
    Convert JSON calving data to NetCDF format.
    
    Parameters
    ----------
    json_file : str
        Path to input JSON file
    netcdf_file : str
        Path to output NetCDF file
    """
    if type(json_file) == str:
        with open(json_file, 'r') as f:
            data = json.load(f)
    elif type(json_file) == list:
        data = json_file
    
    else:
        raise ValueError("json_file must be a string or a list of dictionaries")
    
    all_data = []
    for year_entry in data:
        year = year_entry['time']
        for point in year_entry['data']:
            all_data.append({
                'year': year,
                'x': point['x'],
                'y': point['y'],
                'gsfc_ice_mask': point['gsfc_ice_mask'],
                'model_sftgif': point['model_sftgif'],
                'residual': point['residual']
            })
    
    years = []
    x_coords = []
    y_coords = []
    
    for point in all_data:
        if point['year'] not in years:
            years.append(point['year'])
        if point['x'] not in x_coords:
            x_coords.append(point['x'])
        if point['y'] not in y_coords:
            y_coords.append(point['y'])
    
    years.sort()
    x_coords.sort()
    y_coords.sort()
    
    # Method 1, making empty arrays and then filling them with data 
    n_years = len(years)
    n_x = len(x_coords) 
    n_y = len(y_coords)
    
    gsfc_array = np.full((n_years, n_x, n_y), np.nan)
    model_array = np.full((n_years, n_x, n_y), np.nan)
    residual_array = np.full((n_years, n_x, n_y), np.nan)
    
    for point in all_data:
        year_idx = years.index(point['year'])
        x_idx = x_coords.index(point['x'])
        y_idx = y_coords.index(point['y'])
        
        gsfc_array[year_idx, x_idx, y_idx] = point['gsfc_ice_mask']
        model_array[year_idx, x_idx, y_idx] = point['model_sftgif']
        residual_array[year_idx, x_idx, y_idx] = point['residual']
    
    # Create the dataset
    print("Creating NetCDF dataset...")
    
    dataset = xr.Dataset(
        {
            'gsfc_ice_mask': (['year', 'x', 'y'], gsfc_array),
            'model_sftgif': (['year', 'x', 'y'], model_array),
            'residual': (['year', 'x', 'y'], residual_array)
        },
        coords={
            'year': years,
            'x': x_coords,
            'y': y_coords
        }
    )
    
    # Add some basic information
    dataset.attrs['title'] = 'Calving Data'
    dataset.attrs['description'] = 'GSFC vs Model comparison data'
    return dataset

# Example usage:
# dataset = json_to_netcdf('residuals.json', 'residuals.nc')