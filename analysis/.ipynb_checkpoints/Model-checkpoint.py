import Data
import Events
import Map
import Point

class Model:
    def __init__(self, config=None):    
        smapgrid = SMAPgrid(config=config)


    def initialize(self):
        None


    def run(self, EASEindex):
        data = Data(config, EASEindex)
        events = data.separate_events()
        drydown = []
        drydowns = self.calc_drydown(events)
        return drydowns


    def finalize(self, results):
        smapgrid.remap_results(results)

    def calc_drydown(self, event):
        
        
        
        
        
        
def exponential_model()

def q_model()
