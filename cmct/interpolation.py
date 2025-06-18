import xarray as xr
import rioxarray as rxr
import numpy as np
import sys 
import netCDF4 as nc
import os 
import logging
import calving

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Resampling:
    def __init__(self, input_data, target_data=None, target_resolution=None):
        self.input_data = input_data
        self.target_data = target_data
        self.target_resolution = target_resolution
        self.logger = logging.getLogger(self.__class__.__name__)
            
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
            ds_resampled = input_ds.interp(x=target_x, y=target_y, method='linear')
        
        else:
            raise ValueError("Either target_data or target_resolution must be provided for resampling.")
                
        return ds_resampled
        

if __name__ == "__main__":
    # Setup logging for the main script
    logger = logging.getLogger(__name__)
    
    obs_filename = '/Users/aditya_pachpande/Documents/GitHub/CmCt/data/calving/observed_icemask_ismip_annual.nc'
    model_filename = '/Users/aditya_pachpande/Documents/GitHub/CmCt/test/calving/sftgif_GIS_JPL_ISSM_historical.nc'
    
    gsfc = calving.load_gsfc_calving(obs_filename)
    model = calving.load_model_calving(model_filename)
    
    print(gsfc.ds)
    print(model.ds)
    
    resampler = Resampling(gsfc, model)
    resampled_data = resampler.resample()


    print(f"\nResampled data shape: {resampled_data.dims}")
    print(f"Resampled data coordinates: {list(resampled_data.coords.keys())}")