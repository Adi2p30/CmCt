import xarray as xr
import rioxarray as rxr
import numpy as np
import sys 
import netCDF4 as nc
import os 
import logging
from PIL import Image
import numpy as np
# import xesmf as xe

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Resampling:
    def __init__(self, input_data, target_data=None, target_resolution=None, interpolation_method='linear'):
        self.input_data = input_data
        self.target_data = target_data
        self.target_resolution = target_resolution
        self.logger = logging.getLogger(self.__class__.__name__)
        self.interpolation_method = interpolation_method
            
    def resample(self):
        """
        Reproject the input data to the target CRS and resolution.
        """
        input_ds = self.input_data.ds

        obs_x = input_ds.x
        obs_y = input_ds.y
        
        self.logger.info(f"Input x range: {obs_x.min().values} to {obs_x.max().values}")
        self.logger.info(f"Input y range: {obs_y.min().values} to {obs_y.max().values}")
        
        if self.target_data is not None:
            target_ds = self.target_data.ds
            tgt_x = target_ds.x
            tgt_y = target_ds.y
            
            self.logger.info(f"Target x range: {tgt_x.min().values} to {tgt_x.max().values}, size: {len(tgt_x)}")
            self.logger.info(f"Target y range: {tgt_y.min().values} to {tgt_y.max().values}, size: {len(tgt_y)}")
            
            self.logger.info("Interpolating to target grid...")
            ds_resampled = input_ds.interp(x=tgt_x, y=tgt_y, method='linear')
            
            
        elif self.target_resolution is not None:
            self.logger.info(f"Using target resolution: {self.target_resolution}")
            
            target_x = np.linspace(obs_x.min(), obs_x.max(), self.target_resolution[0])
            target_y = np.linspace(obs_y.min(), obs_y.max(), self.target_resolution[1])
            
            if self.interpolation_method == 'linear':
                """
                Use case: Fast processing needed 
                """
                self.logger.info("Performing linear interpolation...")
                ds_resampled = input_ds.interp(x=target_x, y=target_y, method='linear')
                
            elif self.interpolation_method == 'nearest':
                """
                Use case: Fast processing, When the target grid is sparse or irregular, nearest neighbor interpolation can be more appropriate.
                """
                self.logger.info("Performing nearest neighbor interpolation...")
                ds_resampled = input_ds.interp(x=target_x, y=target_y, method='nearest')

            elif self.interpolation_method == 'makima':
                """
                Use case: When the target grid is dense and you want to preserve the shape of the data, Makima interpolation can be used.
                """
                self.logger.info("Performing Makima interpolation...")
                ds_resampled = input_ds.interp(x=target_x, y=target_y, method='makima')
                
            elif self.interpolation_method == 'pchip':
                """
                Physical data, no overshoot.
                Use case: When the target grid is dense and you want to preserve the shape of the data without overshooting, PCHIP interpolation can be used.
                """
                self.logger.info("Performing cubic interpolation...")
                ds_resampled = input_ds.interp(x=target_x, y=target_y, method='pchip')
                
            elif self.interpolation_method == 'lanczos3d':
                """
                Use case: Smoooooooooth
                """
                self.logger.info("Performing Lanczos 3D interpolation...")
                ice_array = input_ds['ice_mask'].values
                resized = np.array(Image.fromarray(ice_array).resize(
                    (target_x.size, target_y.size), 
                    Image.LANCZOS
                ))
                ds_resampled = xr.DataArray(resized, dims=['y', 'x'], coords={'y': target_y, 'x': target_x})
                
            # elif self.interpolation_method == 'conservative':
            #     """
            #     Use case: To preserve the total amount of ice, especially when downscaling.
            #     """
                
            #     self.logger.info("Performing conservative interpolation...")
            #     regridder = xe.Regridder(input_ds, target_ds, 'conservative')
            #     ds_resampled = regridder(input_ds)

        else:
            raise ValueError("Either target_data or target_resolution must be provided for resampling.")
        
        ds_resampled = ds_resampled.fillna(0)
                
        self.logger.info("Resampling completed successfully.")
            
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