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
        """ Run the analysis for one pixel """

        #_______________________________________________________________________________________________
        # Get the sampling point attributes (EASE pixel)
        if self.verbose:
            print(f"Currently processing pixel {sample_EASE_index}")

        #_______________________________________________________________________________________________
        # Read dataset for the pixel
        data = Data(self.cfg, sample_EASE_index)
        if data.sm['soil_moisture_daily'].isna().all():
            warnings.warn(f"No soil moisture data at the EASE pixel {sample_EASE_index}")
            return None

        #_______________________________________________________________________________________________
        # Run the stormevent separation
        separator = EventSeparator(data)
        events = separator.separate_events(data)
        if events is None:
            return None

        #_______________________________________________________________________________________________
        # Execute the main analysis --- fit drydown models
        drydown_model = DrydownModel(data, events)
        return drydown_model.fit_models(events)
    

    def finalize(self, results):
        self.smapgrid.remap_results(results)
