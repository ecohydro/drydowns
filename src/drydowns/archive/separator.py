import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
import threading
import logging

from .event import Event

from .mylogger import getLogger, modifyLogger


# Create a logger
log = getLogger(__name__)


class ThreadNameHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            # Add thread name to the log message
            record.threadName = threading.current_thread().name
            super(ThreadNameHandler, self).emit(record)
        except Exception:
            self.handleError(record)


class EventSeparator:
    def __init__(self, cfg, data):
        self.cfg = cfg
        # self.verbose = cfg["MODEL"]["verbose"].lower() in ["true", "yes", "1"]
        self.verbose = cfg["verbose"].lower() in ["true", "yes", "1"]
        self.data = data
        self.init_params()

        current_thread = threading.current_thread()
        current_thread.name = (
            # f"[{self.data.EASE_row_index},{self.data.EASE_column_index}]"
            f"[{self.data.id[0]},{self.data.id[1]}]"

        )
        self.thread_name = current_thread.name

        # # Not working at the moment ...
        # custom_handler = ThreadNameHandler()
        # log = modifyLogger(name=__name__, custom_handler=custom_handler)

    def init_params(self):
        # self.precip_thresh = self.cfg.getfloat("EVENT_SEPARATION", "precip_thresh")
        # self.target_rmsd = self.cfg.getfloat("EVENT_SEPARATION", "target_rmsd")
        # _noise_thresh = (self.data.max_sm - self.data.min_sm) * self.cfg.getfloat(
        #     "EVENT_SEPARATION", "frac_range_thresh"
        # )
        # self.noise_thresh = np.minimum(_noise_thresh, self.target_rmsd * 2)
        # self.min_duration = self.cfg.getint(
        #     "EVENT_SEPARATION", "min_duration"
        # )
        # self.plot = self.cfg["MODEL"]["plot_results"].lower() in ["true", "yes", "1"]
        self.precip_thresh = self.cfg.getfloat("precip_thresh")
        self.target_rmsd = self.cfg.getfloat("target_rmsd")
        _noise_thresh = (self.data.max_sm - self.data.min_sm) * self.cfg.getfloat(
            "frac_range_thresh"
        )
        self.noise_thresh = np.minimum(_noise_thresh, self.target_rmsd * 2)
        self.min_duration = self.cfg.getint("min_duration")
        self.plot = self.cfg["plot_results"].lower() in ["true", "yes", "1"]



    def separate_events(self, output_dir):
        """Separate soil moisture timeseries into events"""
        self.output_dir = output_dir

        self.identify_event_starts()
        self.adjust_event_starts_1()
        self.adjust_event_starts_2()
        self.identify_event_ends()
        self.events_df = self.create_event_dataframe()
        if self.events_df.empty:
            return None

        self.filter_events(self.min_duration)
        self.events = self.create_events(self.events_df)

        # if self.plot:
        #     self.plot_events()

        return self.events

    def identify_event_starts(self):
        """Identify the start date of the event"""
        # The event starts where negative increment of soil mositure follows the positive increment of soil moisture
        negative_increments = self.data.df.dSdt < 0
        positive_increments = self.data.df.dSdt > (self.target_rmsd * 2)
        self.data.df['start_date'] = negative_increments.values & np.concatenate(
            ([False], positive_increments[:-1])
        )

    def adjust_event_starts_1(self):
        """
        In case the soil moisture data is nan on the initial event_start dates, 
        look for data in the previous timesteps
        """

        # Get the event start dates
        event_start_idx = self.data.df['start_date'][self.data.df['start_date']].index

        # Loop for event start dates
        for i, event_start_date in enumerate(event_start_idx):
            for j in range(
                0, 6
            ):  # Look back up to 6 timesteps to seek for sm value which is not nan
                current_date = event_start_date - pd.Timedelta(days=j)

                # If rainfall exceeds threshold, stop there
                if self.data.df.loc[current_date].precip > self.precip_thresh:
                    # If SM value IS NOT nap.nan, update the event start date value to this timestep
                    if not np.isnan(
                        self.data.df.loc[
                            current_date
                        ].soil_moisture_daily_before_masking
                    ):
                        update_date = current_date
                    # If SM value IS nap.nan, don't update the event start date value
                    else:
                        update_date = event_start_date
                    break

                # If dS > 0, stop there
                if self.data.df.loc[current_date].dS > 0:
                    update_date = event_start_date
                    break

                # If reached to the NON-nan SM value, update start date value to this timestep
                if ((i - j) >= 0) or (
                    not np.isnan(
                        self.data.df.loc[
                            current_date
                        ].soil_moisture_daily_before_masking
                    )
                ):
                    update_date = current_date
                    break

            # Update the startdate timestep
            self.data.df.loc[event_start_date, 'start_date'] = False
            self.data.df.loc[update_date, 'start_date'] = True

    def adjust_event_starts_2(self):
        """
        In case the soil moisture data is nan on the initial event_start dates, 
        look for data in the later timesteps
        """

        # Get the event start dates
        event_start_idx_2 = pd.isna(
            self.data.df["soil_moisture_daily_before_masking"][
                self.data.df['start_date']
            ]
        ).index

        for i, event_start_date in enumerate(event_start_idx_2):
            update_date = event_start_date
            for j in range(
                0, 6
            ):  # Look ahead up to 6 timesteps to seek for sm value which is not nan, or start of the precip event
                current_date = event_start_date + pd.Timedelta(days=j)
                # If Non-nan SM value is detected, update start date value to this timstep
                if current_date > self.data.end_date:
                    update_date = current_date
                    break

                if not pd.isna(
                    self.data.df.loc[current_date].soil_moisture_daily_before_masking
                ):
                    update_date = current_date
                    break

            self.data.df.loc[event_start_date, 'start_date'] = False
            self.data.df.loc[update_date, 'start_date'] = True

    def identify_event_ends(self):
        """Detect the end of a storm event"""

        # Initialize
        num_events = self.data.df.shape[0]
        self.data.df['end_date'] = np.zeros(len(self.data.df), dtype=bool)
        event_start_idx = self.data.df['start_date'][self.data.df['start_date']].index
        for i, event_start_date in enumerate(event_start_idx):
            for j in range(1, len(self.data.df)):
                current_date = event_start_date + pd.Timedelta(days=j)

                if current_date > self.data.end_date:
                    break

                if np.isnan(
                    self.data.df.loc[current_date].soil_moisture_daily_before_masking
                ):
                    continue

                if (
                    (self.data.df.loc[current_date].dSdt >= self.noise_thresh)
                    or (self.data.df.loc[current_date].precip > self.precip_thresh)
                ) or self.data.df.loc[current_date].event_start:
                    # Any positive increment smaller than 5% of the observed range of soil moisture at the site is excluded (if there is not precipitation) if it would otherwise truncate a drydown.
                    self.data.df.loc[current_date, 'end_date'] = True
                    break
                else:
                    continue

        # create a new column for event_end
        self.data.df["dSdt(t-1)"] = self.data.df.dSdt.shift(+1)

    def create_event_dataframe(self):
        start_indices = self.data.df[self.data.df['start_date']].index
        end_indices = self.data.df[self.data.df['end_date']].index

        # Create a new DataFrame with each row containing a list of soil moisture values between each pair of event_start and event_end
        event_data = [
            {
                'start_date': start_index,
                'end_date': end_index,
                "min_sm": self.data.min_sm,
                "max_sm": self.data.max_sm,
                'soil_moisture': list(
                    self.data.df.loc[
                        start_index:end_index, 'soil_moisture'
                    ].values
                ),
                "soil_moisture_daily_before_masking": list(
                    self.data.df.loc[
                        start_index:end_index, "soil_moisture_daily_before_masking"
                    ].values
                ),
                "normalized_sm": list(
                    self.data.df.loc[start_index:end_index, "normalized_sm"].values
                ),
                "precip": list(
                    self.data.df.loc[start_index:end_index, "precip"].values
                ),
                "PET": list(self.data.df.loc[start_index:end_index, "pet"].values),
                "delta_theta": self.data.df.loc[start_index, "dSdt(t-1)"],
            }
            for start_index, end_index in zip(start_indices, end_indices)
        ]
        return pd.DataFrame(event_data)

    def filter_events(self, min_consecutive_days=5):
        self.events_df = self.events_df[
            self.events_df['soil_moisture'].apply(lambda x: pd.notna(x).sum())
            >= min_consecutive_days
        ].copy()
        self.events_df.reset_index(drop=True, inplace=True)

    def create_events(self, events_df):
        """Create a list of Event instances for easier handling of data for DrydownModel class"""
        event_instances = [
            Event(
                index, 
                row.to_dict(),
                # **row.to_dict(),
                # theta_w = self.min_sm,
                # theta_star = self.max_sm,
                # z = self.z,


            ) for index, row in events_df.iterrows()
        ]
        return event_instances

    def plot_events(self):
        fig, (ax11, ax12) = plt.subplots(2, 1, figsize=(20, 5))

        self.data.df.soil_moisture_daily_before_masking.plot(ax=ax11, alpha=0.5)
        ax11.scatter(
            self.data.df.soil_moisture_daily_before_masking[
                self.data.df['start_date']
            ].index,
            self.data.df.soil_moisture_daily_before_masking[
                self.data.df['start_date']
            ].values,
            color="orange",
            alpha=0.5,
        )
        ax11.scatter(
            self.data.df.soil_moisture_daily_before_masking[
                self.data.df['end_date']
            ].index,
            self.data.df.soil_moisture_daily_before_masking[
                self.data.df['end_date']
            ].values,
            color="orange",
            marker="x",
            alpha=0.5,
        )
        self.data.df.precip.plot(ax=ax12, alpha=0.5)

        # Save results
        filename = f"{self.data.EASE_row_index:03d}_{self.data.EASE_column_index:03d}_eventseparation.png"
        output_dir2 = os.path.join(self.output_dir, "plots")
        if not os.path.exists(output_dir2):
            # Use a lock to ensure only one thread creates the directory
            with threading.Lock():
                # Check again if the directory was created while waiting
                if not os.path.exists(output_dir2):
                    os.makedirs(output_dir2)

        fig.savefig(os.path.join(output_dir2, filename))
        plt.close()
