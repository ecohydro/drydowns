from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from MyLogger import getLogger
import threading

# Create a logger
log = getLogger(__name__)


def exponential_model(t, delta_theta, theta_w, tau):
    return delta_theta * np.exp(-t / tau) + theta_w


def neg_log_likelihood(params, t, y):
    delta_theta, theta_w, tau, sigma = params
    y_hat = exponential_model(t, delta_theta, theta_w, tau)
    residuals = y - y_hat
    ssr = np.sum(residuals**2)
    n = len(y)
    sigma2 = ssr / n
    ll = -(n / 2) * np.log(2 * np.pi * sigma2) - (1 / (2 * sigma2)) * ssr
    return -ll


def q_model(t, k, q, delta_theta, theta_star=1.0, theta_w=0.0):
    s0 = (delta_theta - theta_w) ** (1 - q)

    a = (1 - q) / ((theta_star - theta_w) ** q)

    return (-k * a * t + s0) ** (1 / (1 - q)) + theta_w


class DrydownModel:
    def __init__(self, cfg, Data, Events):
        self.cfg = cfg
        self.data = Data
        self.events = Events
        self.plot_results = cfg["MODEL"]["plot_results"].lower() in ["true", "yes", "1"]
        self.force_PET = cfg["MODEL"]["force_PET"].lower() in ["true", "yes", "1"]

        if cfg["MODEL"]["run_mode"] == "parallel":
            current_thread = threading.current_thread()
            current_thread.name = (
                f"[{self.data.EASE_row_index},{self.data.EASE_column_index}]"
            )
            self.thread_name = current_thread.name
        else:
            self.thread_name = "main thread"

    def fit_models(self, output_dir):
        """Loop through the list of events, fit the drydown models, and update the Event intances' attributes"""
        self.output_dir = output_dir

        for i, event in enumerate(self.events):
            try:
                updated_event = self.fit_one_event(event)
                # Replace the old Event instance with updated one
                self.events[i] = updated_event
            except Exception as e:
                log.debug(f"Exception raised in the thread {self.thread_name}: {e}")

        if self.plot_results:
            self.plot_drydown_models_in_timesreies()

    def fit_one_event(self, event):
        """Fit multiple drydown models for one event

        Args:
            event (_type_): _description_

        Returns:
            _type_: _description_
        """
        # _____________________________________________
        # Fit exponential model
        popt, r_squared, y_opt = self.fit_exponential_model(event)
        event.add_attributes("exponential", popt, r_squared, y_opt)

        # _____________________________________________
        # Fit q model
        popt, r_squared, y_opt = self.fit_q_model(event)
        event.add_attributes("q", popt, r_squared, y_opt, self.force_PET)

        # _____________________________________________
        # Finalize results for one event
        if self.plot_results:
            self.plot_drydown_models(event)

        return event

    def fit_model(self, event, model, bounds, p0, norm=False):
        """Base function for fitting models

        Args:
            event (_type_): _description_
            model (_type_): _description_
            bounds (_type_): _description_
            p0 (_type_): _description_
            norm (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        try:
            if norm:
                y_fit = event.norm_y
            else:
                y_fit = event.y

            # Fit the model
            popt, _ = curve_fit(
                f=model, xdata=event.x, ydata=y_fit, p0=p0, bounds=bounds
            )

            # Get the optimal fit
            y_opt = model(event.x, *popt)

            if norm:
                y_opt = y_opt * self.data.range_sm + self.data.min_sm

            # Calculate the residuals
            residuals = event.y - y_opt
            ss_res = np.sum(residuals**2)
            r_squared = 1 - ss_res / np.sum((event.y - np.nanmean(event.y)) ** 2)

            return popt, r_squared, y_opt

        except Exception as e:
            log.debug(f"Exception raised in the thread {self.thread_name}: {e}")

    def fit_exponential_model(self, event):
        """Fits an exponential model to the given event data and returns the fitted parameters.

        Args:
            event (EventData): An object containing event data.

        Returns:
            dict or None: A dictionary containing the fitted parameters and statistics, or None if an error occurs.
        """

        # ___________________________________________________________________________________
        # Define the boundary condition for optimizing the exponential_model(t, delta_theta, theta_w, tau)

        ### Delta_theta ###
        min_delta_theta = 0
        max_delta_theta = 2 * event.subset_sm_range
        ini_delta_theta = 0.5 * event.subset_sm_range

        ### Theta_w ###
        min_theta_w = self.data.min_sm
        max_theta_w = event.subset_min_sm
        ini_theta_w = (min_theta_w + max_theta_w) / 2

        ### Tau ###
        min_tau = 0
        max_tau = np.inf
        ini_tau = 1

        bounds = [
            (min_delta_theta, min_theta_w, min_tau),
            (max_delta_theta, max_theta_w, max_tau),
        ]
        p0 = [ini_delta_theta, ini_theta_w, ini_tau]

        # ______________________________________________________________________________________
        # Execute the event fit
        return self.fit_model(
            event=event, model=exponential_model, bounds=bounds, p0=p0, norm=False
        )

    def fit_q_model(self, event):
        """Fits a q model to the given event data and returns the fitted parameters.

        Args:
            event (EventData): An object containing event data.

        Returns:
            dict or None: A dictionary containing the fitted parameters and statistics, or None if an error occurs.
        """

        # ___________________________________________________________________________________
        # Define the boundary condition for optimizing q_model(t, k, q, delta_theta)

        ### k (should be equal to PET to reduce dimensionality ###
        min_k = 0
        max_k = event.pet
        ini_k = event.pet

        ### q ###
        min_q = 0.0
        max_q = 10
        ini_q = 1.0 + 1.0e-03

        ### delta_theta ###
        min_delta_theta = 0.0
        max_delta_theta = 1.0
        ini_delta_theta = 0.1

        # ______________________________________________________________________________________
        # Execute the event fit for the normalized timeseries between 0 and 1

        if not self.force_PET:
            bounds = [(min_k, min_q, min_delta_theta), (max_k, max_q, max_delta_theta)]
            p0 = [ini_k, ini_q, ini_delta_theta]

            return self.fit_model(
                event=event,
                model=lambda t, k, q, delta_theta: q_model(
                    t, k, q, delta_theta, 1.0, 0.0
                ),
                bounds=bounds,
                p0=p0,
                norm=True,
            )

        if self.force_PET:
            bounds = [(min_q, min_delta_theta), (max_q, max_delta_theta)]
            p0 = [ini_q, ini_delta_theta]

            return self.fit_model(
                event=event,
                model=lambda t, q, delta_theta: q_model(
                    t, event.pet, q, delta_theta, 1.0, 0.0
                ),
                bounds=bounds,
                p0=p0,
                norm=True,
            )

    def return_result_df(self):
        """Return results in the pandas dataframe format for easier concatination"""

        results = []
        for event in self.events:
            try:
                if not self.force_PET:
                    q_k = event.q["k"]
                else:
                    q_k = event.pet

                _results = {
                    "EASE_row_index": self.data.EASE_row_index,
                    "EASE_column_index": self.data.EASE_column_index,
                    "event_start": event.start_date,
                    "event_end": event.end_date,
                    "time": event.x,
                    "sm": event.y,
                    "min_sm": event.min_sm,
                    "max_sm": event.max_sm,
                    "pet": event.pet,
                    "exp_delta_theta": event.exponential["delta_theta"],
                    "exp_theta_w": event.exponential["theta_w"],
                    "exp_tau": event.exponential["tau"],
                    "exp_r_squared": event.exponential["r_squared"],
                    "exp_y_opt": event.exponential["y_opt"],
                    "q_k": q_k,
                    "q_q": event.q["q"],
                    "q_delta_theta": event.q["delta_theta"],
                    "q_r_squared": event.q["r_squared"],
                    "q_y_opt": event.q["y_opt"],
                }
                results.append(_results)
            except Exception as e:
                log.debug(f"Exception raised in the thread {self.thread_name}: {e}")
                continue

        # Convert results into dataframe
        df_results = pd.DataFrame(results)

        # If the result is empty, return nothing
        if not results:
            return pd.DataFrame()
        else:
            return df_results

    def plot_drydown_models(self, event, ax=None, plot_mode="single"):
        # Plot exponential model
        date_range = pd.date_range(start=event.start_date, end=event.end_date, freq="D")
        x = date_range[event.x]

        # Create a figure and axes
        if ax is None:
            fig, ax = plt.subplots()

        # ______________________________________
        # Plot observed data
        if plot_mode == "single":
            ax.scatter(x, event.y)

        # ______________________________________
        try:
            ax.plot(
                x,
                event.exponential["y_opt"],
                alpha=0.7,
                linestyle="--",
                color="orange",
                label=f"expoential: R^2={event.exponential['r_squared']:.2f}; tau={event.exponential['tau']:.2f}",
            )
        except Exception as e:
            log.debug(f"Exception raised in the thread {self.thread_name}: {e}")

        # ______________________________________
        # Plot q model
        try:
            ax.plot(
                x,
                event.q["y_opt"],
                alpha=0.7,
                linestyle="--",
                color="green",
                label=f"q model: R^2={event.q['r_squared']:.2f}; q={event.q['q']:.2f}; PET={event.pet:.2f}",
            )
        except Exception as e:
            log.debug(f"Exception raised in the thread {self.thread_name}: {e}")

        # ______________________________________
        if plot_mode == "single":
            ax.set_title(f"Event {event.index}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Soil Moisture")
            ax.set_xlim([event.start_date, event.end_date])
            ax.legend()
            # Rotate the x tick labels
            ax.tick_params(axis="x", rotation=45)
        elif plot_mode == "multiple":
            try:
                exp_param = f"expoential: R^2={event.exponential['r_squared']:.2f}; tau={event.exponential['tau']:.2f}"

                ax.text(
                    x[0],
                    event.q["y_opt"][0] + 0.03,
                    f"param={exp_param}",
                    fontsize=12,
                    ha="left",
                    va="bottom",
                    color="orange",
                )
            except Exception as e:
                log.debug(f"Exception raised in the thread {self.thread_name}: {e}")

            try:
                q_param = (
                    f"q model: R^2={event.q['r_squared']:.2f}; q={event.q['q']:.2f}"
                )
                ax.text(
                    x[0],
                    event.q["y_opt"][0],
                    f"{q_param}",
                    fontsize=12,
                    ha="left",
                    va="bottom",
                    color="green",
                )
            except Exception as e:
                log.debug(f"Exception raised in the thread {self.thread_name}: {e}")

        # ___________________________________________________________________________________
        # Save results
        if plot_mode == "single":
            filename = f"{self.data.EASE_row_index:03d}_{self.data.EASE_column_index:03d}_event_{event.index}.png"
            output_dir2 = os.path.join(self.output_dir, "plots")
            if not os.path.exists(output_dir2):
                os.makedirs(output_dir2)

            plt.savefig(
                os.path.join(output_dir2, filename),
                dpi=600,
                bbox_inches="tight",
            )

        # Close the current figure to release resources
        plt.close()

    def plot_drydown_models_in_timesreies(self):
        years_of_record = max(self.data.df.index.year) - min(self.data.df.index.year)
        fig, (ax11, ax12) = plt.subplots(2, 1, figsize=(20 * years_of_record, 5))

        self.data.df.soil_moisture_daily.plot(ax=ax11, alpha=0.5)
        ax11.scatter(
            self.data.df.soil_moisture_daily[self.data.df["event_start"]].index,
            self.data.df.soil_moisture_daily[self.data.df["event_start"]].values,
            color="orange",
            alpha=0.5,
        )
        ax11.scatter(
            self.data.df.soil_moisture_daily[self.data.df["event_end"]].index,
            self.data.df.soil_moisture_daily[self.data.df["event_end"]].values,
            color="orange",
            marker="x",
            alpha=0.5,
        )
        ax11.set_ylabel("VSWC[m3/m3]")
        self.data.df.precip.plot(ax=ax12, alpha=0.5)
        ax12.set_ylabel("Precipitation[mm/d]")

        for event in self.events:
            self.plot_drydown_models(event, ax=ax11, plot_mode="multiple")

        # Save results
        filename = f"{self.data.EASE_row_index:03d}_{self.data.EASE_column_index:03d}_events_in_ts.png"
        output_dir2 = os.path.join(self.output_dir, "plots")
        if not os.path.exists(output_dir2):
            os.makedirs(output_dir2)

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir2, filename))

        plt.close()
