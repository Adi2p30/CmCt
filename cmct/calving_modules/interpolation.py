import gc
import logging
import os
import sys

import netCDF4 as nc
import numpy as np
import rioxarray as rxr
import xarray as xr
from PIL import Image


class Interpolater:
    def __init__(
        self,
        input_data,
        target_data=None,
        target_resolution=None,
        interpolation_method="linear",
    ):
        self.input_data = input_data
        self.target_data = target_data
        self.target_resolution = target_resolution
        self.interpolation_method = interpolation_method

        if isinstance(target_resolution, str):
            raise ValueError(
                f"ERROR: interpolation_method '{target_resolution}' was passed as target_resolution. "
                f"Use keyword arguments: Interpolater(input_data, target_data=target, interpolation_method='{target_resolution}')"
            )

    def interpolate(self):
        """
        Reproject the input data to the target CRS and resolution.
        """
        input_ds = self.input_data.ds

        obs_x = input_ds.x
        obs_y = input_ds.y

        logging.info(f"Input x coordinates: {obs_x.values}")
        logging.info(f"Input y coordinates: {obs_y.values}")

        if self.target_data is not None:
            target_ds = self.target_data.ds
            tgt_x = target_ds.x
            tgt_y = target_ds.y

            x_same = np.array_equal(tgt_x.values, obs_x.values)
            y_same = np.array_equal(tgt_y.values, obs_y.values)

            logging.debug(f"Target x coordinates: {tgt_x.values}")
            logging.debug(f"Target y coordinates: {tgt_y.values}")

            if x_same and y_same:
                return input_ds

            ds_resampled = self._perform_interpolation(input_ds, tgt_x, tgt_y)

        elif self.target_resolution is not None:
            target_x = np.linspace(obs_x.min(), obs_x.max(), self.target_resolution[0])
            target_y = np.linspace(obs_y.min(), obs_y.max(), self.target_resolution[1])

            ds_resampled = self._perform_interpolation(input_ds, target_x, target_y)

        else:
            raise ValueError(
                "Either target_data or target_resolution must be provided for resampling."
            )

        ds_resampled = ds_resampled.fillna(0)

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
        for var in input_ds.data_vars:
            data_vals = input_ds[var].values.flatten()

        if self.interpolation_method == "linear":
            ds_resampled = input_ds.interp(x=target_x, y=target_y, method="linear")

        elif self.interpolation_method == "nearest":
            ds_resampled = input_ds.interp(x=target_x, y=target_y, method="nearest")

        elif self.interpolation_method == "akima":
            ds_resampled = input_ds.interp(x=target_x, y=target_y, method="akima")

        elif self.interpolation_method == "pchip":
            ds_resampled = input_ds.interp(x=target_x, y=target_y, method="pchip")

        elif self.interpolation_method == "cubic":
            ds_resampled = input_ds.interp(x=target_x, y=target_y, method="cubic")

        elif self.interpolation_method == "slinear":
            ds_resampled = input_ds.interp(x=target_x, y=target_y, method="slinear")

        elif self.interpolation_method == "lanczos3d":
            ds_resampled_vars = {}
            ds_resampled_coords = {}

            target_x_values = (
                target_x.values if hasattr(target_x, "values") else target_x
            )
            target_y_values = (
                target_y.values if hasattr(target_y, "values") else target_y
            )

            for var_name, var_data in input_ds.data_vars.items():
                if len(var_data.dims) == 2 and (
                    "y" in var_data.dims and "x" in var_data.dims
                ):
                    ice_array = var_data.values

                    if var_data.dims == ("x", "y"):
                        ice_array = ice_array.T

                    ice_array_clean = np.nan_to_num(ice_array, nan=0.0)
                    ice_min, ice_max = ice_array_clean.min(), ice_array_clean.max()

                    if ice_max > ice_min:
                        ice_normalized = (
                            (ice_array_clean - ice_min) / (ice_max - ice_min) * 255
                        ).astype(np.uint8)
                    else:
                        ice_normalized = np.zeros_like(ice_array_clean, dtype=np.uint8)

                    target_width = len(target_x_values)
                    target_height = len(target_y_values)

                    resized_normalized = np.array(
                        Image.fromarray(ice_normalized).resize(
                            (target_width, target_height), Image.LANCZOS
                        )
                    )

                    if ice_max > ice_min:
                        resized = (resized_normalized.astype(np.float32) / 255.0) * (
                            ice_max - ice_min
                        ) + ice_min
                    else:
                        resized = np.full_like(
                            resized_normalized, ice_min, dtype=np.float32
                        )

                    ds_resampled_vars[var_name] = (["y", "x"], resized)

                    del (
                        ice_array,
                        ice_array_clean,
                        ice_normalized,
                        resized_normalized,
                        resized,
                    )

                elif len(var_data.dims) == 3 and (
                    "y" in var_data.dims and "x" in var_data.dims
                ):
                    spatial_dims = ["x", "y"]
                    other_dims = [
                        dim for dim in var_data.dims if dim not in spatial_dims
                    ]

                    if len(other_dims) == 1:
                        other_dim = other_dims[0]
                        other_coord = input_ds.coords[other_dim]

                        processed_slices = []

                        for i in range(len(other_coord)):
                            slice_data = var_data.isel({other_dim: i})

                            ice_array = slice_data.values
                            if slice_data.dims == ("x", "y"):
                                ice_array = ice_array.T

                            ice_array_clean = np.nan_to_num(ice_array, nan=0.0)
                            ice_min, ice_max = (
                                ice_array_clean.min(),
                                ice_array_clean.max(),
                            )

                            if ice_max > ice_min:
                                ice_normalized = (
                                    (ice_array_clean - ice_min)
                                    / (ice_max - ice_min)
                                    * 255
                                ).astype(np.uint8)
                            else:
                                ice_normalized = np.zeros_like(
                                    ice_array_clean, dtype=np.uint8
                                )

                            resized_normalized = np.array(
                                Image.fromarray(ice_normalized).resize(
                                    (len(target_x_values), len(target_y_values)),
                                    Image.LANCZOS,
                                )
                            )

                            if ice_max > ice_min:
                                resized = (
                                    resized_normalized.astype(np.float32) / 255.0
                                ) * (ice_max - ice_min) + ice_min
                            else:
                                resized = np.full_like(
                                    resized_normalized, ice_min, dtype=np.float32
                                )

                            processed_slices.append(resized)

                            del (
                                ice_array,
                                ice_array_clean,
                                ice_normalized,
                                resized_normalized,
                                resized,
                            )

                        stacked_data = np.stack(processed_slices, axis=0)

                        if var_data.dims == (other_dim, "y", "x"):
                            ds_resampled_vars[var_name] = (
                                [other_dim, "y", "x"],
                                stacked_data,
                            )
                        elif var_data.dims == ("y", "x", other_dim):
                            stacked_data = np.transpose(stacked_data, (1, 2, 0))
                            ds_resampled_vars[var_name] = (
                                ["y", "x", other_dim],
                                stacked_data,
                            )
                        else:
                            ds_resampled_vars[var_name] = (
                                [other_dim, "y", "x"],
                                stacked_data,
                            )

                        ds_resampled_coords[other_dim] = other_coord

                        del processed_slices, stacked_data
                    else:
                        var_interp = var_data.interp(
                            x=target_x, y=target_y, method="nearest"
                        )
                        ds_resampled_vars[var_name] = var_interp
                else:
                    try:
                        var_interp = var_data.interp(
                            x=target_x, y=target_y, method="nearest"
                        )
                        ds_resampled_vars[var_name] = var_interp
                    except:
                        ds_resampled_vars[var_name] = var_data

            coords_dict = {"y": target_y_values, "x": target_x_values}
            coords_dict.update(ds_resampled_coords)

            ds_resampled = xr.Dataset(ds_resampled_vars, coords=coords_dict)

            del ds_resampled_vars, ds_resampled_coords
            gc.collect()

        else:
            raise ValueError(
                f"Unsupported interpolation method: {self.interpolation_method}. "
                f"Supported methods are: 'linear', 'nearest', 'cubic', 'akima', 'pchip', 'slinear', 'lanczos3d'"
            )

        for var in ds_resampled.data_vars:
            data_vals = ds_resampled[var].values.flatten()

        return ds_resampled
