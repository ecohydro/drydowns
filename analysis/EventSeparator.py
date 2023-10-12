import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from Event import Event


class EventSeparator:
    def __init__(self, data, output_dir, plot=False):
        self.ds_synced = ds_synced
        self.output_dir = output_dir
        self.plot = plot
        self.event_df = None
        self.events = self.get_event_instances()

    def get_event_instances():
        """Create a list of event instances"""

    # def calculate_thresholds(self):
    #     max_sm = self.ds_synced.soil_moisture_daily.max()
    #     min_sm = self.ds_synced.soil_moisture_daily.min()
    #     sm_range = max_sm - min_sm
    #     dSdt_thresh = sm_range * 0.05
    #     target_rmsd = 0.08
    #     return dSdt_thresh, target_rmsd

    # def identify_event_starts_and_ends(self, dSdt_thresh, target_rmsd):
    #     negative_increments = self.ds_synced.dSdt < 0
    #     positive_increments = self.ds_synced.dSdt > target_rmsd
    #     self.ds_synced['event_start'] = negative_increments.values & np.concatenate(([False], positive_increments[:-1]))

    #     event_end = np.zeros(self.ds_synced.shape[0], dtype=bool)
    #     # ... your loop here to fill event_end based on your conditions

    #     self.ds_synced['event_end'] = event_end
    #     self.ds_synced['event_end'] = self.ds_synced['event_end'].shift(-1)
    #     self.ds_synced = self.ds_synced[:-1]

    # def plot_events(self):
    #     fig, (ax11, ax12) = plt.subplots(2,1, figsize=(20, 5))
    #     self.ds_synced.soil_moisture_daily.plot(ax=ax11, alpha=0.5)
    #     # ... other plotting code

    # def create_event_dataframe(self):
    #     start_indices = self.ds_synced[self.ds_synced['event_start']].index
    #     end_indices = self.ds_synced[self.ds_synced['event_end']].index
    #     event_data = [{'event_start': start_index,
    #                 # ... other data
    #                 }
    #                 for start_index, end_index in zip(start_indices, end_indices)]
    #     self.event_df = pd.DataFrame(event_data)

    # def filter_events(self, min_consecutive_days=5):
    #     self.event_df = self.event_df[self.event_df['soil_moisture_daily'].apply(lambda x: pd.notna(x).sum()) >= min_consecutive_days].copy()
    #     self.event_df.reset_index(drop=True, inplace=True)

    # def separate_events(self):
    #     dSdt_thresh, target_rmsd = self.calculate_thresholds()
    #     self.identify_event_starts_and_ends(dSdt_thresh, target_rmsd)
    #     if self.plot:
    #         self.plot_events()
    #     self.create_event_dataframe()
    #     self.filter_events()

    # def separate_events():
    #     # Any positive increment smaller than 5% of the observed range of soil moisture at the site is excluded if it would otherwise truncate a drydown.
    #     max_sm = ds_synced.soil_moisture_daily.max()
    #     min_sm = ds_synced.soil_moisture_daily.min()
    #     sm_range = max_sm - min_sm
    #     dSdt_thresh = sm_range * 0.05

    #     # To avoid noise creating spurious drydowns, identified drydowns were excluded from the analysis when the positive increment preceding the drydown was less than two times the target unbiased root-mean-square difference for SMAP observations (0.08).
    #     target_rmsd = 0.08
    #     dSdt_thresh

    #     negative_increments = ds_synced.dSdt < 0

    #     # To avoid noise creating spurious drydowns, identified drydowns were excluded from the analysis when the positive increment preceding the drydown was less than two times the target unbiased root-mean-square difference for SMAP observations (0.08).
    #     positive_increments = ds_synced.dSdt > target_rmsd

    #     # TODO: NOT lose drydown starting after NaN

    #     # Negative dSdt preceded with positive dSdt
    #     ds_synced['event_start'] = negative_increments.values & np.concatenate(([False], positive_increments[:-1]))
    #     ds_synced['event_start'][ds_synced['event_start']].index

    #     precip_thresh = 2
    #     event_end = np.zeros(ds_synced.shape[0], dtype=bool)

    #     for i in range(1, ds_synced.shape[0]):
    #         if ds_synced['event_start'][i]:
    #             start_index = i
    #             for j in range(i+1, ds_synced.shape[0]):
    #                 if np.isnan(ds_synced['dS'][j]):
    #                     None
    #                 # TODO: putthis threshold back once Ive got precip data
    #                 #or ds_synced['precip'][j] > precip_thresh:
    #                 if ds_synced['dS'][j] >= dSdt_thresh:
    #                     # Any positive increment smaller than 5% of the observed range of soil moisture at the site is excluded (if there is not precipitation) if it would otherwise truncate a drydown.
    #                     event_end[j] = True
    #                     break

    #     # create a new column for event_end
    #     ds_synced['event_end'] = event_end
    #     ds_synced['event_end'] = ds_synced['event_end'].shift(-1)
    #     ds_synced = ds_synced[:-1]
    #     ds_synced['event_start'][ds_synced['event_end']].index
    #     ds_synced['dSdt(t-1)'] = ds_synced.dSdt.shift(+1)

    #     if plot:
    #         fig, (ax11, ax12) = plt.subplots(2,1, figsize=(20, 5))
    #         ds_synced.soil_moisture_daily.plot(ax=ax11, alpha=0.5)
    #         ax11.scatter(ds_synced.soil_moisture_daily[ds_synced['event_start']].index, ds_synced.soil_moisture_daily[ds_synced['event_start']].values, color='orange', alpha=0.5)
    #         ax11.scatter(ds_synced.soil_moisture_daily[ds_synced['event_end']].index, ds_synced.soil_moisture_daily[ds_synced['event_end']].values, color='orange', marker='x', alpha=0.5)
    #         # ds_synced.precip.plot(ax=ax12, alpha=0.5)
    #         fig.savefig(os.path.join(output_dir, f'pt_{EASE_row_index:03d}_{EASE_column_index:03d}_timeseries.png'))

    #     start_indices = ds_synced[ds_synced['event_start']].index
    # ``  end_indices = ds_synced[ds_synced['event_end']].index

    #     # Create a new DataFrame with each row containing a list of soil moisture values between each pair of event_start and event_end
    #     event_data = [{'event_start': start_index,
    #                 'event_end': end_index,
    #                 'soil_moisture_daily': list(ds_synced.loc[start_index:end_index, 'soil_moisture_daily'].values),
    #                 'normalized_S': list(ds_synced.loc[start_index:end_index, 'normalized_S'].values),
    #                 # 'precip': list(ds_synced.loc[start_index:end_index, 'precip'].values),
    #                 'PET': list(ds_synced.loc[start_index:end_index, 'pet'].values),
    #                 #'LAI': list(ds_synced.loc[start_index:end_index, 'LAI'].values),
    #                 #'NDVI': list(ds_synced.loc[start_index:end_index, 'NDVI'].values),
    #                 'delta_theta': ds_synced.loc[start_index, 'dSdt(t-1)'],
    #                     #'bulk_density': ds_synced.loc[start_index, 'bulk_density'],
    #                     #'sand_fraction': ds_synced.loc[start_index, 'sand_fraction'],
    #                     #'clay_fraction': ds_synced.loc[start_index, 'clay_fraction']
    #                 }
    #                 for start_index, end_index in zip(start_indices, end_indices)]
    #     event_df = pd.DataFrame(event_data)``

    #     min_consective_days = 5
    #     event_df_long = event_df[event_df['soil_moisture_daily'].apply(lambda x: pd.notna(x).sum()) >= min_consective_days].copy()
    #     event_df_long = event_df_long.reset_index(drop=True)
    #     event_df_long

    # def separate_events(self, data, EASEindex):
    #     """ Separate soil moisture timeseries into events """

    #     events = data.separate_events()

    #     # Check if there is SM data
    #     if events.isna().all():
    #         warnings.warn(f"No event drydown was detected at {EASE_index}")
    #         return None

    #     print(f"Event delineation success at {EASE_index}")
