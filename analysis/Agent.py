from Data import Data
from DrydownModel import DrydownModel
from EventSeparator import EventSeparator
from SMAPgrid import SMAPgrid
import warnings
from datetime import datetime
import os
import getpass


def create_output_dir():
    username = getpass.getuser()
    formatted_now = datetime.now().strftime("%Y-%m-%d")
    output_dir = rf"/home/waves/projects/smap-drydown/output/fit_models_py_{username}_{formatted_now}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created.")
    return output_dir


class Agent:
    def __init__(self, cfg=None):
        self.cfg = cfg
        self.smapgrid = SMAPgrid(cfg=self.cfg)
        self.target_EASE_idx = self.smapgrid.get_EASE_index_subset()
        self.verbose = cfg["MODEL"]["verbose"].lower() in ["true", "yes", "1"]
        self.output_dir = create_output_dir()

    def initialize(self):
        None

    def run(self, sample_EASE_index):
        """Run the analysis for one pixel

        Args:
            sample_EASE_index (list.shape[1,2]): a pair of EASE index, representing [0,0] the EASE row index (y, or latitude) and [0,1] EASE column index (x, or longitude)
        """

        # _______________________________________________________________________________________________
        # Get the sampling point attributes (EASE pixel)
        if self.verbose:
            print(f"Currently processing pixel {sample_EASE_index}")

        # _______________________________________________________________________________________________
        # Read dataset for a pixel
        data = Data(self.cfg, sample_EASE_index)

        # If there is no soil moisture data available for the pixel, skip the analysis
        if data.df["soil_moisture_daily"].isna().all():
            warnings.warn(
                f"No soil moisture data at the EASE pixel {sample_EASE_index}"
            )
            return None

        # _______________________________________________________________________________________________
        # Run the stormevent separation
        separator = EventSeparator(self.cfg, data)
        events = separator.separate_events(output_dir=self.output_dir)

        # If there is no drydown event detected for the pixel, skip the analysis
        # Check if there is SM data
        if not events:
            warnings.warn(
                f"No event drydown was detected at {data.EASE_row_index, data.EASE_column_index}"
            )
            return None

        print(
            f"Event separation success at {data.EASE_row_index, data.EASE_column_index}: {len(events)} events detected"
        )

        # _______________________________________________________________________________________________
        # Execute the main analysis --- fit drydown models
        drydown_model = DrydownModel(self.cfg, data, events)
        drydown_model.fit_models(output_dir=self.output_dir)

        return drydown_model.return_result_df()

    def finalize(self, results):
        """Finalize the analysis from all the pixels

        Args:
            results (list): concatinated results returned from serial/multi-threadding analysis
        """
        self.smapgrid.remap_results(results)
