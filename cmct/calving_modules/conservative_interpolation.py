import logging
import xarray as xr

# This library has energy conserving interpolation methods
from dolfin import *

def dolfin_conservative_interpolation(gsfc, target_resolution):
    """
    Using dolfin library for conservative interpolation.
    Source: https://fenicsproject.org/
    """
    logging.info("Running dolfin conservative interpolation.")
    


def manual_conservative_interpolation(gsfc, target_resolution):
    """
    Conserving mass and energy during interpolation.
    Source: https://journals.ametsoc.org/view/journals/mwre/127/9/1520-0493_1999_127_2204_fasocr_2.0.co_2.xml
    """
    logging.info("Running conservative interpolation.")
    
    if target_resolution is None:
        raise ValueError("Target resolution must be specified for conservative interpolation.")

    # Finding volume first
    gsfc = gsfc.coarsen(dim={"x": target_resolution, "y": target_resolution}, boundary="trim").mean()
    flux = gsfc.volume.sum(dim=["x", "y"])
    
    gradient_flux = 
    
    

    