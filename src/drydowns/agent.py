from .data import Data
from .model import DrydownModel
from .separator import EventSeparator
from .SMAPgrid import SMAPgrid
import warnings
from datetime import datetime
import os
import getpass
import pandas as pd
import logging
from .mylogger import getLogger

# Create a logger
log = getLogger(__name__)


def create_output_dir(parent_dir):
    username = getpass.getuser()
    formatted_now = datetime.now().strftime("%Y-%m-%d")
    output_dir = rf"{parent_dir}/{username}_{formatted_now}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        log.info(f"Directory '{output_dir}' created.")
    return output_dir


class Agent:
    def __init__(self, cfg=None, logger=None):
        self.cfg = cfg
        self.logger = logger
        # self.verbose = cfg["MODEL"]["verbose"].lower() in ["true", "yes", "1"]
        self.verbose = cfg.get("verbose").lower() in ["true", "yes", "1"]
        
        self._smapgrid = SMAPgrid(cfg=self.cfg)
        self.data_ids = self._smapgrid.get_EASE_index_subset()
        
        # self.output_dir = create_output_dir(parent_dir=cfg["PATHS"]["output_dir"])
        # self.output_dir = create_output_dir(parent_dir=cfg.get("output_dir"))
        self.output_dir = cfg.get("output_dir")


    def initialize(self):
        None

    def run(self, did):
        """Run the analysis for one pixel

        Args:
            did (list.shape[1,2]): Data ID; 
            a pair of EASE index, representing [0,0] the EASE row index (y, or 
            latitude) and [0,1] EASE column index (x, or longitude)
        """

        try:
            # _______________________________________________________________________________________________
            # Get the sampling point attributes (EASE pixel)
            if self.verbose:
                log.info(
                    f"Currently processing pixel {did}",
                )

            # _______________________________________________________________________________________________
            # Read dataset for a pixel
            data = Data(self.cfg, did)

            # If there is no soil moisture data available for the pixel, skip the analysis
            if data.df["soil_moisture_daily"].isna().all():
                warnings.warn(
                    f"No soil moisture data at the EASE pixel {did}"
                )
                return None

            # _______________________________________________________________________________________________
            # Run the stormevent separation
            separator = EventSeparator(self.cfg, data)
            events = separator.separate_events(output_dir=self.output_dir)

            # If there is no drydown event detected for the pixel, skip the analysis
            # Check if there is SM data
            if not events:
                log.warning(f"No event drydown was detected at {did}")
                return None

            log.info(
                f"Event separation success at {did}: {len(events)} events detected"
            )

            # _______________________________________________________________________________________________
            # Execute the main analysis --- fit drydown models
            drydown_model = DrydownModel(self.cfg, data, events)
            drydown_model.fit_models(output_dir=self.output_dir)

            results_df = drydown_model.return_result_df()

            log.info(
                f"Drydown model analysis completed at {did}: {len(results_df)}/{len(events)} events fitted"
            )

            return results_df

        except Exception as e:
            print(f"Error in thread: {did}")
            print(f"Error message: {str(e)}")

    # def finalize(self, results):
    #     """Finalize the analysis from all the pixels

    #     Args:
    #         results (list): concatenated results returned from serial/multi-threadding analysis
    #     """
    #     df_results = self.save_to_csv(results)
    #     # self.smapgrid.remap_results(df_results)
    #     # self.smapgrid.plot_remapped_results(da)

    # def save_to_csv(self, results):
    #     if len(results) > 1:
    #         try:
    #             df = pd.concat(results)
    #         except:
    #             df = results
    #     else:
    #         df = results
    #     df.to_csv(os.path.join(self.output_dir, "all_results.csv"))
    #     return df

    def finalize(self, results):
        """Finalize the analysis from all the pixels

        Args:
            results (list): concatenated results returned from serial/multi-threadding analysis
        """
        if len(results) > 1:
            try:
                df = pd.concat(results)
            except:
                df = results
        else:
            df = results

        # date = datetime.now().strftime("%Y-%m-%d")
        date = datetime.now().strftime('%d%b').lower()
        out_bn = self.cfg.get(
            'output_fid',
            'ismn_results'
        )
        fid = f"{out_bn}_{date}"

        self.save(df, fid=fid)


    def save(self, df, fid='smap_results'):
        """Save the results to a csv file"""
        filename = f"{fid}.csv"
        log.info(f"Saving {filename} to {self._output_dir}")
        df.to_csv(os.path.join(self._output_dir, filename), index=False)
        log.info(f"Saved {filename} to {self._output_dir}")

