import pandas as pd
import warnings
import xarray as xr
import os
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import pyproj

from .mylogger import getLogger

# Create a logger
log = getLogger(__name__)

"""

Name:           SMAPgrid.py
Compatibility:  Python 3.7.0
Description:    Description of what program does

URL:            https://

Requires:       list of libraries required

Dev ToDo:       None

AUTHOR:         Ryoko Araki (initial dev); Bryn Morgan (refactor)
ORGANIZATION:   University of California, Santa Barbara
Contact:        raraki@ucsb.edu
Copyright:      (c) Ryoko Araki & Bryn Morgan 2024


"""


class SMAPgrid:
    """A template of the 36km resolution EASEgrid that can be use to map SMAPL3 data for the desired spatial extent"""

    def __init__(self, cfg=None):
        self.cfg = cfg
        # self.verbose = cfg["MODEL"]["verbose"].lower() in ["true", "yes", "1"]
        self.verbose = cfg.get("verbose").lower() in ["true", "yes", "1"]

        # self.data_dir = cfg["PATHS"]["data_dir"]
        # self.datarods_dir = cfg["PATHS"]["datarods_dir"]
        # self.output_dir = cfg["PATHS"]["output_dir"]
        # self.data_dir = cfg.get("data_dir")
        # self.datarods_dir = cfg.get("datarods_dir")
        self.output_dir = cfg.get("output_dir")

        self.get_attributes()
        self.coord_info = self.get_coordinates()
        self.coord_info_subset = self.get_subset()

        self.template_xarray = self.get_template_xarray()

    def get_attributes(self):
        """Get attributes of the 36km resolution EASEgrid used for SMAPL3 data"""
        self.epsg = "4326"
        # self.min_lon = self.cfg.getfloat("EXTENT", "min_lon")
        # self.min_lat = self.cfg.getfloat("EXTENT", "min_lat")
        # self.max_lon = self.cfg.getfloat("EXTENT", "max_lon")
        # self.max_lat = self.cfg.getfloat("EXTENT", "max_lat")
        self.min_lon = self.cfg.getfloat("min_lon")
        self.min_lat = self.cfg.getfloat("min_lat")
        self.max_lon = self.cfg.getfloat("max_lon")
        self.max_lat = self.cfg.getfloat("max_lat")

        if not ((self.min_lon < self.max_lon) and (self.min_lat < self.max_lat)):
            # If the condition is not met, issue a warning
            log.warning(
                "min_lon should be less than max_lon, and min_lat should be less than max_lat"
            )

    def get_coordinates(self):
        """Get the information on the coordinate-index pair"""
        file_path = os.path.join(self.cfg.get("data_dir"), "coord_info.csv")
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
            log.info(f"Number of pixels in the spatial extent: {len(subset)}")

        ### Mask with openwater pixels
        return subset

    def mask_by_openwater(self, _subset):
        """Mask the coordinate if they are on the openwater"""
        file_path = os.path.join(self.cfg.get("data_dir"), "coord_open_water.csv")
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
            log.info(f"Number of pixels without openwater: {len(subset)}")
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
        y_coords = sorted(set(self.coord_info["latitude"]), reverse=True)
        x_coords = sorted(set(self.coord_info["longitude"]))
        _data = np.empty((len(y_coords), len(x_coords)))
        _data[:] = np.nan

        # Create an xarray DataArray with the empty data and the coordinates
        da = xr.DataArray(_data, coords=[("y", y_coords), ("x", x_coords)], name="data")
        da.attrs["crs"] = "EPSG:4326"  # pyproj.CRS.from_epsg(4326)
        return da

    def remap_results(self, df_results):
        # Save results in a dataarray format
        da = self.template_xarray.copy()
        for index, row in df_results.iterrows():
            i = row["EASE_row_index"]
            j = row["EASE_column_index"]
            avg_q = np.average(row["q_q"])
            da.isel(y=i, x=j).values = avg_q

        # Save the data
        filename = f"output_q.nc"
        da.to_netcdf(os.path.join(self.output_dir, filename))

        return da

    # Not working
    # def plot_remapped_results(self, da):
    #     # Plot and save the figure
    #     filename = f"output_q.png"

    #     # Create a figure and axis with a specified projection (e.g., PlateCarree)
    #     fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})

    #     # Add coastlines to the map
    #     ax.add_feature(cfeature.COASTLINE)

    #     # Customize the plot (e.g., add gridlines, set extent)
    #     ax.gridlines(draw_labels=True, linestyle="--")

    #     # Set the map extent (you can customize these coordinates)
    #     ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())

    #     da.plot(ax=ax)

    #     fig.savefig(os.path.join(self.output_dir, filename))
    #     plt.close(fig)
    #     return da
