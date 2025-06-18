import numpy as np
import xarray as xr

def harmonic_mean_xarray(arr, dim=None, skipna=True):
    """Compute the harmonic mean of an xarray DataArray."""
    if skipna:
        valid = arr.notnull()
        n = valid.sum(dim=dim)
        hm = n / (1 / arr.where(valid)).sum(dim=dim)
    else:
        n = arr.count(dim=dim)
        hm = n / (1 / arr).sum(dim=dim)
    return hm

def geometric_mean_xarray(arr, dim=None, skipna=True):
    """Compute the geometric mean of an xarray DataArray."""
    if skipna:
        valid = arr.notnull()
        gm = np.exp(np.log(arr.where(valid)).sum(dim=dim) / valid.sum(dim=dim))
    else:
        gm = np.exp(np.log(arr).sum(dim=dim) / arr.count(dim=dim))
    return gm

def average_xarray(arr, dim=None, skipna=True):
    """Compute the arithmetic mean of an xarray DataArray."""
    return arr.mean(dim=dim, skipna=skipna)

def rms_xarray(arr, dim=None, skipna=True):
    """Compute the root mean square of an xarray DataArray."""
    return np.sqrt((arr**2).mean(dim=dim, skipna=skipna))

# Example usage:
# arr = xr.DataArray([1, 2, 3, 4, 5])
# print(harmonic_mean_xarray(arr).values)
# print(geometric_mean_xarray(arr).values)
# print(average_xarray(arr).values)
# print(rms_xarray(arr).values)
