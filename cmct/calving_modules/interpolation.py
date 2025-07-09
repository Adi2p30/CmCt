import xarray as xr
import rioxarray as rxr
import numpy as np
import sys 
import netCDF4 as nc
import os 
from PIL import Image
import numpy as np
import gc
# import xesmf as xe


class Resampling:
    def __init__(self, input_data, target_data=None, target_resolution=None, interpolation_method='linear'):
        self.input_data = input_data
        self.target_data = target_data
        self.target_resolution = target_resolution
        self.interpolation_method = interpolation_method
        
        # Debug validation - check if interpolation_method was mistakenly passed as target_resolution
        if isinstance(target_resolution, str):
            raise ValueError(f"ERROR: interpolation_method '{target_resolution}' was passed as target_resolution. "
                           f"Use keyword arguments: Resampling(input_data, target_data=target, interpolation_method='{target_resolution}')")
        
        
            
    def resample(self):
        """
        Reproject the input data to the target CRS and resolution.
        """
        input_ds = self.input_data.ds

        obs_x = input_ds.x
        obs_y = input_ds.y
        
        if self.target_data is not None:
            target_ds = self.target_data.ds
            tgt_x = target_ds.x
            tgt_y = target_ds.y
            
            # Use the specified interpolation method
            ds_resampled = self._perform_interpolation(input_ds, tgt_x, tgt_y)
            
        elif self.target_resolution is not None:
            target_x = np.linspace(obs_x.min(), obs_x.max(), self.target_resolution[0])
            target_y = np.linspace(obs_y.min(), obs_y.max(), self.target_resolution[1])
            
            ds_resampled = self._perform_interpolation(input_ds, target_x, target_y)

        else:
            raise ValueError("Either target_data or target_resolution must be provided for resampling.")
        
        ds_resampled = ds_resampled.fillna(0)
        
        # Clean up temporary variables
        del obs_x, obs_y
        if self.target_data is not None:
            del target_ds, tgt_x, tgt_y
        gc.collect()
            
        return ds_resampled
    
    def _perform_interpolation(self, input_ds, target_x, target_y):
        """
        Perform the actual interpolation based on the selected method.
        
        Parameters
        ----------
        input_ds : xarray.Dataset
            Input dataset to interpolate
        target_x : xarray.DataArray or numpy.ndarray
            Target x coordinates
        target_y : xarray.DataArray or numpy.ndarray
            Target y coordinates
            
        Returns
        -------
        xarray.Dataset
            Interpolated dataset
        """
        
        
        # Check input data characteristics for debugging
        
        for var in input_ds.data_vars:
            data_vals = input_ds[var].values.flatten()
            valid_vals = data_vals[~np.isnan(data_vals)]
                
        if self.interpolation_method == 'linear':
            """
            Use case: Fast processing needed 
            """
            
            ds_resampled = input_ds.interp(x=target_x, y=target_y, method='linear')
            
        elif self.interpolation_method == 'nearest':
            """
            Use case: Fast processing, When the target grid is sparse or irregular, nearest neighbor interpolation can be more appropriate.
            """
            
            ds_resampled = input_ds.interp(x=target_x, y=target_y, method='nearest')

        elif self.interpolation_method == 'akima':
            """
            Use case: When the target grid is dense and you want to preserve the shape of the data, Makima interpolation can be used.
            """
            
            ds_resampled = input_ds.interp(x=target_x, y=target_y, method='akima')
            
        elif self.interpolation_method == 'pchip':
            """
            Physical data, no overshoot.
            Use case: When the target grid is dense and you want to preserve the shape of the data without overshooting, PCHIP interpolation can be used.
            """
            
            ds_resampled = input_ds.interp(x=target_x, y=target_y, method='pchip')
        
        elif self.interpolation_method == 'cubic':
            """
            Use case: When the target grid is dense and you want to preserve the shape of the data, Cubic interpolation can be used.
            """
            
            ds_resampled = input_ds.interp(x=target_x, y=target_y, method='cubic')
        
        elif self.interpolation_method == 'slinear':
            """
            Use case: When the target grid is dense and you want to preserve the shape of the data, Spline interpolation can be used.
            """

            ds_resampled = input_ds.interp(x=target_x, y=target_y, method='slinear')
            
            
        elif self.interpolation_method == 'lanczos3d':
            
            # For Lanczos, we need to handle each variable separately using PIL
            ds_resampled_vars = {}
            ds_resampled_coords = {}
            
            # Handle coordinates properly
            target_x_values = target_x.values if hasattr(target_x, 'values') else target_x
            target_y_values = target_y.values if hasattr(target_y, 'values') else target_y
            
            for var_name, var_data in input_ds.data_vars.items():
                if len(var_data.dims) == 2 and ('y' in var_data.dims and 'x' in var_data.dims):
                    # Handle 2D spatial variables
                    ice_array = var_data.values
                    
                    # Ensure correct orientation (y, x)
                    if var_data.dims == ('x', 'y'):
                        ice_array = ice_array.T
                    
                    # Handle NaN values
                    ice_array_clean = np.nan_to_num(ice_array, nan=0.0)
                    
                    # Normalize to 0-255 for PIL
                    ice_min, ice_max = ice_array_clean.min(), ice_array_clean.max()
                    
                    if ice_max > ice_min:
                        ice_normalized = ((ice_array_clean - ice_min) / (ice_max - ice_min) * 255).astype(np.uint8)
                    else:
                        ice_normalized = np.zeros_like(ice_array_clean, dtype=np.uint8)
                    
                    # PIL resize expects (width, height) which corresponds to (x, y)
                    target_width = len(target_x_values)
                    target_height = len(target_y_values)
                    
                    # Resize using PIL
                    resized_normalized = np.array(Image.fromarray(ice_normalized).resize(
                        (target_width, target_height), 
                        Image.LANCZOS
                    ))
                    
                    # Denormalize back to original range
                    if ice_max > ice_min:
                        resized = (resized_normalized.astype(np.float32) / 255.0) * (ice_max - ice_min) + ice_min
                    else:
                        resized = np.full_like(resized_normalized, ice_min, dtype=np.float32)
                    
                    # Store with correct dimensions
                    ds_resampled_vars[var_name] = (['y', 'x'], resized)
                    
                    # Clean up temporary arrays
                    del ice_array, ice_array_clean, ice_normalized, resized_normalized, resized
                    
                elif len(var_data.dims) == 3 and ('y' in var_data.dims and 'x' in var_data.dims):
                    # Handle 3D variables (e.g., with time dimension)
                    spatial_dims = ['x', 'y']
                    other_dims = [dim for dim in var_data.dims if dim not in spatial_dims]
                    
                    if len(other_dims) == 1:
                        other_dim = other_dims[0]
                        other_coord = input_ds.coords[other_dim]
                        
                        # Process each slice along the non-spatial dimension
                        processed_slices = []
                        
                        for i in range(len(other_coord)):
                            slice_data = var_data.isel({other_dim: i})
                            
                            # Process this 2D slice using Lanczos
                            ice_array = slice_data.values
                            if slice_data.dims == ('x', 'y'):
                                ice_array = ice_array.T
                            
                            ice_array_clean = np.nan_to_num(ice_array, nan=0.0)
                            ice_min, ice_max = ice_array_clean.min(), ice_array_clean.max()
                            
                            if ice_max > ice_min:
                                ice_normalized = ((ice_array_clean - ice_min) / (ice_max - ice_min) * 255).astype(np.uint8)
                            else:
                                ice_normalized = np.zeros_like(ice_array_clean, dtype=np.uint8)
                            
                            resized_normalized = np.array(Image.fromarray(ice_normalized).resize(
                                (len(target_x_values), len(target_y_values)), 
                                Image.LANCZOS
                            ))
                            
                            if ice_max > ice_min:
                                resized = (resized_normalized.astype(np.float32) / 255.0) * (ice_max - ice_min) + ice_min
                            else:
                                resized = np.full_like(resized_normalized, ice_min, dtype=np.float32)
                            
                            processed_slices.append(resized)
                            
                            # Clean up slice temporaries
                            del ice_array, ice_array_clean, ice_normalized, resized_normalized, resized
                        
                        # Stack the processed slices
                        stacked_data = np.stack(processed_slices, axis=0)
                        
                        # Determine the correct dimension order
                        if var_data.dims == (other_dim, 'y', 'x'):
                            ds_resampled_vars[var_name] = ([other_dim, 'y', 'x'], stacked_data)
                        elif var_data.dims == ('y', 'x', other_dim):
                            stacked_data = np.transpose(stacked_data, (1, 2, 0))
                            ds_resampled_vars[var_name] = (['y', 'x', other_dim], stacked_data)
                        else:
                            # Default to (other_dim, y, x)
                            ds_resampled_vars[var_name] = ([other_dim, 'y', 'x'], stacked_data)
                        
                        # Store the coordinate for this dimension
                        ds_resampled_coords[other_dim] = other_coord
                        
                        del processed_slices, stacked_data
                    else:
                        # Too complex, fall back to nearest neighbor
                        var_interp = var_data.interp(x=target_x, y=target_y, method='nearest')
                        ds_resampled_vars[var_name] = var_interp
                else:
                    # For non-spatial variables, use nearest neighbor as fallback
                    try:
                        var_interp = var_data.interp(x=target_x, y=target_y, method='nearest')
                        ds_resampled_vars[var_name] = var_interp
                    except:
                        # If interpolation fails, just copy the variable as-is
                        ds_resampled_vars[var_name] = var_data
            
            # Create coordinate dictionary
            coords_dict = {'y': target_y_values, 'x': target_x_values}
            coords_dict.update(ds_resampled_coords)
            
            # Create the resampled dataset
            ds_resampled = xr.Dataset(
                ds_resampled_vars,
                coords=coords_dict
            )
            
            # Clean up temporary variables
            del ds_resampled_vars, ds_resampled_coords
            gc.collect()
            
        else:
            raise ValueError(f"Unsupported interpolation method: {self.interpolation_method}. "
                           f"Supported methods are: 'linear', 'nearest', 'cubic', 'makima', 'pchip', 'lanczos3d'")
        
        for var in ds_resampled.data_vars:
            data_vals = ds_resampled[var].values.flatten()
            valid_vals = data_vals[~np.isnan(data_vals)]
        
        return ds_resampled


#  __name__ == "__main__":
#     # Setup logging for the main script
#     logger = logging.getLogger(__name__)
    
#     obs_filename = '/Users/aditya_pachpande/Documents/GitHub/CmCt/data/calving/observed_icemask_ismip_annual.nc'
#     model_filename = '/Users/aditya_pachpande/Documents/GitHub/CmCt/test/calving/sftgif_GIS_JPL_ISSM_historical.nc'
    
#     gsfc = calving.load_gsfc_calving(obs_filename)
#     model = calving.load_model_calving(model_filename)
    
#     print(gsfc.ds)
#     print(model.ds)
    
#     resampler = Resampling(gsfc, model)
#     resampled_data = resampler.resample()


#     print(f"\nResampled data shape: {resampled_data.dims}")
#     print(f"Resampled data coordinates: {list(resampled_data.coords.keys())}")
        
#  __name__ == "__main__":
#     # Setup logging for the main script
#     logger = logging.getLogger(__name__)
    
#     obs_filename = '/Users/aditya_pachpande/Documents/GitHub/CmCt/data/calving/observed_icemask_ismip_annual.nc'
#     model_filename = '/Users/aditya_pachpande/Documents/GitHub/CmCt/test/calving/sftgif_GIS_JPL_ISSM_historical.nc'
    
#     gsfc = calving.load_gsfc_calving(obs_filename)
#     model = calving.load_model_calving(model_filename)
    
#     print(gsfc.ds)
#     print(model.ds)
    
#     resampler = Resampling(gsfc, model)
#     resampled_data = resampler.resample()


#     print(f"\nResampled data shape: {resampled_data.dims}")
#     print(f"Resampled data coordinates: {list(resampled_data.coords.keys())}")