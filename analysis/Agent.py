from Data import Data
from DrydownModel import DrydownModel
from EventSeparator import EventSeparator
from SMAPgrid import SMAPgrid
import warnings

class Agent:
    def __init__(self, cfg=None):    
        self.cfg = cfg
        self.smapgrid = SMAPgrid(cfg=self.cfg)
        self.target_EASE_idx = self.smapgrid.get_EASE_index_subset()
        self.verbose = cfg["MODEL"]["verbose"].lower() in ['true', 'yes', '1']

    def initialize(self):
        None

    def run(self, sample_EASE_index):
        """ Run the analysis for one pixel 
        
        Args:
            sample_EASE_index (list.shape[1,2]): a pair of EASE index, representing [0,0] the EASE row index (y, or latitude) and [0,1] EASE column index (x, or longitude)
        """

        #_______________________________________________________________________________________________
        # Get the sampling point attributes (EASE pixel)
        if self.verbose:
            print(f"Currently processing pixel {sample_EASE_index}")

        #_______________________________________________________________________________________________
        # Read dataset for a pixel
        data = Data(self.cfg, sample_EASE_index)

        # If there is no soil moisture data available for the pixel, skip the analysis
        if data.sm['soil_moisture_daily'].isna().all():
            warnings.warn(f"No soil moisture data at the EASE pixel {sample_EASE_index}")
            return None

        #_______________________________________________________________________________________________
        # Run the stormevent separation
        separator = EventSeparator(data)
        events = separator.separate_events(data)

        # If there is no drydown event detected for the pixel, skip the analysis
        if events is None:
            return None

        #_______________________________________________________________________________________________
        # Execute the main analysis --- fit drydown models
        drydown_model = DrydownModel(data, events)
        return drydown_model.fit_models(events)
    

    def finalize(self, results):
        """ Finalize the analysis from all the pixels

        Args:
            results (list): concatinated results returned from serial/multi-threadding analysis
        """
        self.smapgrid.remap_results(results)
