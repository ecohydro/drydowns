import numpy as np
import pandas as pd

import threading

from .smapdata import SMAPData
from .sensordata import SensorData
# from .event import Event
# from .towerevent import SensorEvent

from .model import ExponentialModel, NonlinearModel, SigmoidModel
from .mylogger import getLogger

# Create a logger
log = getLogger(__name__)


class DrydownModelHandler:

    models = {
        'exponential' : ExponentialModel,
        'q' : NonlinearModel,
        'sigmoid' : SigmoidModel
    }

    def __init__(self, cfg, data, events):
            
        self.cfg = cfg
        self.data = data
        self.events = events

        # self.specs = self.get_specs()


        if self.cfg["run_mode"] == "parallel":
            current_thread = threading.current_thread()
            current_thread.name = (f"{self.data.id[0]}, {self.data.id[1]}")
            self.thread_name = current_thread.name
        else:
            self.thread_name = "main thread"

    # def get_specs(self):
    #     specs = {
    #         'force_PET' : self.cfg.getboolean('force_PET'),
    #         'fit_theta_star' : self.cfg.getboolean('fit_theta_star'),
    #         # 'run_mode' : self.cfg.get('run_mode'),
    #     }
    #     return specs

    # def get_models(self):
    #     mod_dict = {
    #         k : self.cfg.getboolean(k + '_model') for k in ['exponential', 'q', 'sigmoid']
    #     }
    #     self.cfg.getboolean('exponential_model')

    def fit_events(self):
        for event in self.events:
            self.fit_event(event)

    def fit_event(self, event):
        for k in self.models.keys():
            if self.cfg.getboolean(k + '_model'):
                # self._fit_event(event, k)
                obj = self.models[k]
                model = obj(self.cfg, self.data, event)
                model.fit_event(event)
    
    def get_results(self):
        # results = [
        #     self.get_event_results(event) for event in self.events if self.get_event_results(event)
        # ]
        results = []
        for event in self.events:
            try:
                results.append(self._get_event_results(event))
            except Exception as e:
                log.debug(f"Exception raised in the thread {self.thread_name}: {e}")
                continue
        df = pd.DataFrame(results)
        # if not results:
        #     df = pd.DataFrame()
        return df

    # def get_event_results(self, event):
    #     try:
    #         results = self._get_event_results(event)
    #     except Exception as e:
    #         log.debug(f"Exception raised in the thread {self.thread_name}: {e}")
    #         results = None
    #     return results

    def _get_event_results(self, event):
        if isinstance(self.data, SensorData):
            col_ids = ('SITE_ID', 'Sensor_ID')
        elif isinstance(self.data, SMAPData):
            col_ids = ("EASE_row_index", "EASE_column_index")

        results = {
            col_ids[0] : self.data.id[0],
            col_ids[1] : self.data.id[1],
            **event.describe(),
            'min_sm' : self.data.min_sm,
            'max_sm' : self.data.max_sm,
            'theta_fc' : self.data.theta_fc,
            'porosity' : self.data.n,
            'pet' : event.pet,
        }
        try:
            results.update({
                'et' : event.get_et(),
                'total_et' : np.sum(event.get_et()),
                'precip' : event.calc_precip(),
            })
        except:
            pass
        
        for mod,abbrev in zip(self.models.keys(), ['exp', 'q', 'sig']):
            if self.cfg.getboolean(mod + '_model') & hasattr(event, mod):
                mod_results = getattr(event, mod)
                results.update({f"{abbrev}_{k}" : v for k, v in mod_results.items()})
        
        return results

