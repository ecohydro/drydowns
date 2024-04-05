#-------------------------------------------------------------------------------
# IMPORTS
#-------------------------------------------------------------------------------
from towerdata import SoilSensorData
from model import DrydownModel
from towerseparator import TowerEventSeparator
# from SMAPgrid import SMAPgrid
import warnings
from datetime import datetime
import os
import getpass
import numpy as np
import pandas as pd
import logging
from mylogger import getLogger

import fluxtower

# Create a logger
log = getLogger(__name__)


#-------------------------------------------------------------------------------
# FUNCTIONS
#-------------------------------------------------------------------------------

def create_output_dir(parent_dir):
    username = getpass.getuser()
    formatted_now = datetime.now().strftime("%Y-%m-%d")
    output_dir = rf"{parent_dir}/{username}_{formatted_now}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        log.info(f"Directory '{output_dir}' created.")
    return output_dir



def get_cols(
    tower : fluxtower.FluxTower, 
    var_cols : list = ['SWC', 'P', 'TA', 'VPD', 'LE', 'ET', 'e_a']
):
    col_list = []
    col_dict = {}
    for var in var_cols:
        col_list += tower.get_var_cols(variable=var, exclude='ERA')
        cols = tower.get_var_cols(variable=var, exclude='ERA')
        col_dict[var] = {
            'var_cols' : sorted([col for col in cols if 'QC' not in col]), 
            'qc_cols' : sorted([col for col in cols if 'QC' in col])
        }

    return col_list, col_dict

def get_grps(tower, cols):
    vi = tower._get_var_info(as_dict=False)
    grps = vi[vi.VARNAME.isin(cols)].GROUP_ID.unique()
    return grps

#-------------------------------------------------------------------------------
# VARIABLES
#-------------------------------------------------------------------------------

meta = fluxtower.flx_tower.META

#-------------------------------------------------------------------------------
# TOWERAGENT CLASS
#-------------------------------------------------------------------------------

class TowerAgent:
    def __init__(self, cfg=None, logger=None, tower_ids : list = None, output_dir=None):
        # config
        self.cfg = cfg
        # logger
        self.logger = logger

        # data_ids
        # If a list of tower ids is provided, use that.
        if tower_ids:
            self.data_ids = tower_ids
        # Otherwise, use all the towers with soil moisture data.
        else:
            self.data_ids = meta[meta.DATAVALUE.str.contains('SWC_')].SITE_ID.unique()

        # filenames
        self._filenames = sorted(os.listdir(cfg["PATHS"]["data_dir"]))

        # verbose
        self.verbose = cfg["MODEL"]["verbose"].lower() in ["true", "yes", "1"]

        # output_dir
        if output_dir:
            self._output_dir = output_dir
        else:
            # self._output_dir = create_output_dir(parent_dir=cfg["PATHS"]["output_dir"])
            self._output_dir = cfg.get('PATHS', 'output_dir')

    def initialize(self):
        None

    def run(self, did, return_data=False):
        """
        Run the analysis for one tower (which may contain multiple soil moisture
        columns)
        """
        try: 
            if self.verbose:
                log.info(f"Currently processing tower {did}")
            
            # 1. Initialize tower
            log.info(f"Initalizing Tower {did}")
            tower = self._init_tower(did)

            # 2. Get list of soil moisture columns
            cols, col_dict = get_cols(tower)
            sm_cols = col_dict['SWC']['var_cols']
            grps = get_grps(tower, sm_cols)

            log.info(f"Found {len(sm_cols)} soil moisture columns for {did} : {sm_cols}")

            # 3. Run the analysis for each soil moisture column
            data = []
            results = []
            # for col in sm_cols:
            #     output = self.run_sensor(tower, col, return_data=return_data)
            for grp in grps:
                output = self.run_sensor(tower, grp, return_data=return_data)
                if return_data:
                    data.append(output[0])
                    results.append(output[1])
                else:
                    results.append(output)
            
            # If all results are NOne, return
            if all([r is None for r in results]):
                log.warning(f"No drydown events detected for {did}")
                if return_data:
                    return data, None
                else:
                    return None
             

            log.info(
                f"Finished with tower {did}: {len(results)}/{len(sm_cols)} sensors analyzed"
            )

            results = pd.concat(results, ignore_index=True)
            results['IGBP'] = tower.metadata.get('IGBP', None)
            results['LAT'] = tower.coords[0]
            results['LON'] = tower.coords[1]
            results['MAT'] = tower.metadata.get('MAT', np.nan)
            results['MAP'] = tower.metadata.get('MAP', np.nan)


            if return_data:
                return data, results
            else:
                return results

        except Exception as e:
            print(f"Error in thread: {did}")
            print(f"Error message: {str(e)}")


    # def run_sensor(self, tower, swc_col, return_data=False):
    def run_sensor(self, tower, grp, return_data=False):
        """ Run the analysis for one soil moisture sensor"""
        try:
            
            # 1. Initialize sensor data object
            log.info(f"Initializing sensor {grp}")
            data = SoilSensorData(self.cfg, tower, grp)

            # 2. Separate events
            log.info(f"Separating events for sensor {grp}")
            data.separate_events()

            if not data.events:
                log.warning(f"No event drydown was detected for {grp}")
                if return_data:
                    return data, None
                else:
                    return None
            else:
                log.info(f"Found {len(data.events)} events for sensor {grp}")

            # 3. Fit drydown models
            log.info(f"Fitting drydown models for sensor {grp}")
            model = DrydownModel(self.cfg, data, data.events)
            model.fit_models(output_dir=self._output_dir)
            # 4. Return results
            results = model.return_result_df()

            log.info(
                f"Drydown model analysis completed for {grp}: {len(results)}/{len(data.events)} events fitted"
            )
            if return_data:
                return data, results
            else:
                return results

        except Exception as e:
            print(f"Error in thread: {grp}")
            print(f"Error message: {str(e)}")



    def _init_tower(self, tid):
        """Initialize a tower object from the Site ID (str)"""
        filename = next((t for t in self._filenames if tid in t),None)
        
        tower = fluxtower.FluxNetTower(os.path.join(self.cfg["PATHS"]["data_dir"], filename))
        tower.add_vp_cols()
        tower.add_et_cols()

        return tower

    def finalize(self, results):
        """Finalize the analysis from all the pixels

        Args:
            results (list): concatenated results returned from serial/multi-threadding analysis
        """
        try:
            df = pd.concat(results)
        except:
            df = results

        # date = datetime.now().strftime("%Y-%m-%d")
        date = datetime.now().strftime('%d%b').lower()
        fid = f"flx_results_{date}"

        self.save(df, fid=fid)


    def save(self, df, fid='flx_results'):
        """Save the results to a csv file"""
        filename = f"{fid}.csv"
        log.info(f"Saving {filename} to {self._output_dir}")
        df.to_csv(os.path.join(self._output_dir, filename), index=False)
        log.info(f"Saved {filename} to {self._output_dir}")

