import pandas as pd
import warnings
import xarray as xr
import os
import numpy as np


class SMAPgrid:
    """A template of the 36km resolution EASEgrid that can be use to map SMAPL3 data for the desired spatial extent"""

    def __init__(self, cfg=None):
        self.cfg = cfg
        self.verbose = cfg["MODEL"]["verbose"].lower() in ["true", "yes", "1"]

        self.data_dir = cfg["PATHS"]["data_dir"]
        self.datarods_dir = cfg["PATHS"]["datarods_dir"]

        self.get_attributes()
        self.coord_info = self.get_coordinates()
        self.coord_info_subset = self.get_subset()

        self.template_xarray = self.get_template_xarray()

    def get_attributes(self):
        """Get attributes of the 36km resolution EASEgrid used for SMAPL3 data"""
        self.epsg = "4326"
        self.min_lon = self.cfg.getfloat("EXTENT", "min_lon")
        self.min_lat = self.cfg.getfloat("EXTENT", "min_lat")
        self.max_lon = self.cfg.getfloat("EXTENT", "max_lon")
        self.max_lat = self.cfg.getfloat("EXTENT", "max_lat")

    def get_coordinates(self):
        """Get the information on the coordinate-index pair"""
        file_path = os.path.join(self.data_dir, self.datarods_dir, "coord_info.csv")
        coord_info = pd.read_csv(file_path)
        return coord_info

    def get_subset(self):
        """Get the subset of the extent specified in the config file"""
        # Get the subset of the extent
        _subset = self.crop_by_extent()
        # Mask with openwater
        subset = self.mask_by_openwater(_subset)
        return subset

    def crop_by_extent(self):
        """Crop the coordinate into subset of the extent specified in the config file"""
        subset = self.coord_info[
            (self.coord_info["latitude"] >= self.min_lat)
            & (self.coord_info["latitude"] <= self.max_lat)
            & (self.coord_info["longitude"] >= self.min_lon)
            & (self.coord_info["longitude"] <= self.max_lon)
        ].copy()
        if self.verbose:
            print(f"Number of subset pixels: {len(subset)}")

        ### Mask with openwater pixels
        return subset

    def mask_by_openwater(self, _subset):
        """Mask the coordinate if they are on the openwater"""
        file_path = os.path.join(
            self.data_dir, self.datarods_dir, "coord_open_water.csv"
        )
        coord_open_water = pd.read_csv(file_path)
        subset = (
            pd.merge(
                _subset,
                coord_open_water,
                on=["EASE_row_index", "EASE_column_index"],
                how="left",
                indicator=True,
            )
            .query('_merge == "left_only"')
            .drop(columns="_merge")
        )
        if self.verbose:
            print(f"Number of subset pixels without openwater: {len(subset)}")
        return subset

    def get_EASE_index_subset(self):
        """Get the list of EASE index of the extent"""
        return self.coord_info_subset[["EASE_row_index", "EASE_column_index"]].values

    def get_EASE_coordinate_subset(self):
        """Get the list of EASE coordinates of the extent"""
        return self.coord_info_subset[["latitude", "longitude"]].to_list()

    def get_template_xarray(self):
        """Get the template xaray with nan data with EASE coordinates of the extent"""
        # Create a 2D numpy array filled with NaNs
        y_coords = self.coord_info_subset["latitude"].to_list()
        x_coords = self.coord_info_subset["longitude"].to_list()
        _data = np.empty((len(y_coords), len(x_coords)))
        _data[:] = np.nan

        # Create an xarray DataArray with the empty data and the coordinates
        da = xr.DataArray(_data, coords=[("y", y_coords), ("x", x_coords)], name="data")

        return da

    def remap_results(self, results):
        df_results = pd.DataFrame(results)
        da = self.template_xarray.copy()
        for index, row in df_results.iterrows():
            i = row["EASE_row_index"]
            j = row["EASE_column_index"]
            avg_q = row["q"].mean()
            da.isel(y=i, x=j).values = avg_q
        return da
