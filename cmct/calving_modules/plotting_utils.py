import numpy as np


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

    data = gsfc.ds.sel(time=year).residual
    return np.rot90(data, k=0)
