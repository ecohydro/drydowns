import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from Event import Event
import warnings


class EventSeparator:
    def __init__(self, cfg, Data):
        self.cfg = cfg
        self.data = Data
        self.init_params()

    def init_params(self):
        self.precip_thresh = float(self.cfg["EVENT_SEPARATION"]["precip_thresh"])
        self.dSdt_thresh = (self.data.max_sm - self.data.min_sm) * float(
            self.cfg["EVENT_SEPARATION"]["increment_thresh_fraction"]
        )
        self.target_rmsd = float(self.cfg["EVENT_SEPARATION"]["target_rmsd"])
        self.minimium_consective_days = int(
            self.cfg["EVENT_SEPARATION"]["minimium_consective_days"]
        )
        self.plot = self.cfg["MODEL"]["plot_results"].lower() in ["true", "yes", "1"]

    def separate_events(self, output_dir):
        """Separate soil moisture timeseries into events"""
        self.output_dir = output_dir

        self.identify_event_starts()
        self.identify_event_ends()
        self.events_df = self.create_event_dataframe()
        self.filter_events(self.minimium_consective_days)
        self.events = self.create_event_instances(self.events_df)

        if self.plot:
            self.plot_events()

        return self.events

    def identify_event_starts(self):
        # The event starts where negative increament of soil mositure follows the positive increment of soil moisture
        negative_increments = self.data.dSdt < 0
        positive_increments = self.data.dSdt > self.target_rmsd
        self.data["event_start"] = negative_increments.values & np.concatenate(
            ([False], positive_increments[:-1])
        )

    def identify_event_ends(self):
        event_end = np.zeros(self.data.shape[0], dtype=bool)

        for i in range(1, self.data.shape[0]):
            if self.data["event_start"][i]:
                for j in range(i + 1, self.data.shape[0]):
                    if np.isnan(self.data["dS"][j]):
                        None
                    # TODO: putthis threshold back once Ive got precip data
                    # or data['precip'][j] > precip_thresh:
                    if self.data["dS"][j] >= self.dSdt_thresh:
                        # Any positive increment smaller than 5% of the observed range of soil moisture at the site is excluded (if there is not precipitation) if it would otherwise truncate a drydown.
                        event_end[j] = True
                        break

        # create a new column for event_end
        self.data["event_end"] = event_end
        self.data["event_end"] = self.data["event_end"].shift(-1)
        self.data = self.data[:-1]
        self.data["event_start"][self.data["event_end"]].index
        self.data["dSdt(t-1)"] = self.data.dSdt.shift(+1)

    def create_event_dataframe(self):
        start_indices = self.data[self.data["event_start"]].index
        end_indices = self.data[self.data["event_end"]].index

        # Create a new DataFrame with each row containing a list of soil moisture values between each pair of event_start and event_end
        event_data = [
            {
                "event_start": start_index,
                "event_end": end_index,
                "soil_moisture_daily": list(
                    self.data.loc[start_index:end_index, "soil_moisture_daily"].values
                ),
                "normalized_S": list(
                    self.data.loc[start_index:end_index, "normalized_S"].values
                ),
                "precip": list(self.data.loc[start_index:end_index, "precip"].values),
                "PET": list(self.data.loc[start_index:end_index, "pet"].values),
                "delta_theta": self.data.loc[start_index, "dSdt(t-1)"],
            }
            for start_index, end_index in zip(start_indices, end_indices)
        ]
        return pd.DataFrame(event_data)

    def filter_events(self, min_consecutive_days=5):
        self.events_df = self.events_df[
            self.events_df["soil_moisture_daily"].apply(lambda x: pd.notna(x).sum())
            >= min_consecutive_days
        ].copy()
        self.events_df.reset_index(drop=True, inplace=True)

    def create_event_instances(self, events_df):
        """Create a list of Event instances for easier handling of data for DrydownModel class"""
        event_instances = [Event(row._asdict()) for index, row in events_df.iterrows()]
        return event_instances

    def plot_events(self):
        fig, (ax11, ax12) = plt.subplots(2, 1, figsize=(20, 5))

        self.data.soil_moisture_daily.plot(ax=ax11, alpha=0.5)
        ax11.scatter(
            self.data.soil_moisture_daily[self.data["event_start"]].index,
            self.data.soil_moisture_daily[self.data["event_start"]].values,
            color="orange",
            alpha=0.5,
        )
        ax11.scatter(
            self.data.soil_moisture_daily[self.data["event_end"]].index,
            self.data.soil_moisture_daily[self.data["event_end"]].values,
            color="orange",
            marker="x",
            alpha=0.5,
        )
        # data.precip.plot(ax=ax12, alpha=0.5)

        # Save results
        filename = f"{self.data.EASE_row_index:03d}_{self.data.EASE_column_index:03d}_eventseparation.png"
        output_dir2 = os.path.join(self.output_dir, "plots")
        if ~os.path.exists(output_dir2):
            os.makedirs(output_dir2)

        fig.savefig(os.path.join(output_dir2, filename))
