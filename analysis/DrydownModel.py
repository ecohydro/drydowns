from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.optimize import minimize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


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


def q_model(t, k, q, delta_theta):
    """Reduced form of q model for scipy.curvefit input. Can only be applied for normalized timeseries between 0 to 1"""
    theta_star = 1.0
    theta_w = 0.0

    s0 = (delta_theta - theta_w) ** (1 - q)

    a = (1 - q) / ((theta_star - theta_w) ** q)

    return (-k * a * t + s0) ** (1 / (1 - q)) + theta_w


def original_q_model(t, k, q, delta_theta, theta_star=1.0, theta_w=0.0):
    s0 = (delta_theta - theta_w) ** (1 - q)

    a = (1 - q) / ((theta_star - theta_w) ** q)

    return (-k * a * t + s0) ** (1 / (1 - q)) + theta_w


class DrydownModel:
    def __init__(self, cfg, Data, Events):
        self.cfg = cfg
        self.data = Data
        self.events = Events
        self.plot_results = cfg["MODEL"]["plot_results"].lower() in ["true", "yes", "1"]

    def fit_drydown_models(self, output_dir):
        """Fit the drydown models"""
        # _____________________________________________
        # Execute the analysis

        updated_events = []

        for i, event in enumerate(self.events.events()):
            try:
                popt, r_squared, y_opt = self.fit_exponential(event)
                event.add_attributes("exponential", popt, r_squared, y_opt)
                popt, r_squared, y_opt = self.fit_q(event)
                event.add_attributes("q", popt, r_squared, y_opt)

                updated_events.append(event)

                if self.plot_results:
                    self.plot_drydown_models(event, output_dir)

            except Exception as e:
                print(e)
                return None

        # Replace the old Event instance with updated one
        self.events.events = updated_events

    def return_result_df(self):
        results = []
        for event in self.events.events():
            _results = {
                "event_start": event.start_date,
                "event_end": event.end_date,
                "exp_delta_theta": event.exponential["delta_theta"],
                "exp_theta_w": event.exponential["theta_w"],
                "exp_tau": event.exponential["tau"],
                "exp_r_squared": event.exponential["r_squared"],
                "exp_y_opt": event.exponential["y_opt"],
                "q_k": event.q["k"],
                "q_q": event.q["q"],
                "q_delta_theta": event.q["delta_theta"],
                "q_r_squared": event.q["r_squared"],
                "q_y_opt": event.q["y_opt"],
            }
            results.append(_results)

        # Convert results into dataframe
        df_results = pd.DataFrame(results)

        # Merge results
        df = pd.merge(
            self.events.event_df,
            df_results,
            on=["event_start", "event_end"],
            how="outer",
        )

        return df

    def fit_exponential_model(self, event):
        """Fits an exponential model to the given event data and returns the fitted parameters.

        Args:
            event (EventData): An object containing event data.

        Returns:
            dict or None: A dictionary containing the fitted parameters and statistics, or None if an error occurs.
        """

        # ___________________________________________________________________________________
        # Define the bounds for the function exponential_model(t, delta_theta, theta_w, tau)

        # Delta_theta
        min_delta_theta = 0
        max_delta_theta = 2 * event.subset_sm_range
        ini_delta_theta = 0.5 * event.subset_sm_range

        # Theta_w
        min_theta_w = self.data.min_sm
        max_theta_w = event.subset_min_sm
        ini_delta_theta = (min_theta_w + max_theta_w) / 2

        # Tau
        min_tau = 0
        max_tau = np.inf
        ini_tau = 1

        bounds = [
            (min_delta_theta, min_theta_w, min_tau),
            (max_delta_theta, max_theta_w, max_tau),
        ]
        p0 = [ini_delta_theta, ini_delta_theta, ini_tau]

        # ___________________________________________________________________________________
        try:
            # Fit the data
            # popt: Optimal values for the parameters so that the sum of the squared residuals of f(xdata, *popt) - ydata is minimized
            # pcov: The estimated covariance of popt
            popt, _ = curve_fit(
                f=exponential_model, xdata=event.x, ydata=event.y, p0=p0, bounds=bounds
            )

            # Reroduce the analytical solution and calculate the residuals
            y_opt = exponential_model(event.x, *popt)
            residuals = event.y - y_opt
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((event.y - np.nanmean(event.y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            # Save the results
            return popt, r_squared, y_opt

        except Exception as e:
            print("An error occurred:", e)

    def fit_q_model(self, event):
        """Fits a q model to the given event data and returns the fitted parameters.

        Args:
            event (EventData): An object containing event data.

        Returns:
            dict or None: A dictionary containing the fitted parameters and statistics, or None if an error occurs.
        """

        # Define the bounds
        # q_model(t, k, q, delta_theta)
        epsilon = 1.0e-64

        min_k = event.pet - epsilon
        max_k = event.pet + epsilon
        ini_k = event.pet

        min_q = 0.0
        max_q = 10
        ini_q = 1.0 + 1.0e-03

        min_delta_theta = 0.0
        max_delta_theta = 1.0
        ini_delta_theta = 0.1

        bounds = [(min_k, min_q, min_delta_theta), (max_k, max_q, max_delta_theta)]
        p0 = [ini_k, ini_q, ini_delta_theta]

        try:
            # Fit the model
            popt, _ = curve_fit(
                f=q_model,
                xdata=event.x,
                ydata=event.norm_y,
                p0=p0,
                bounds=bounds,
            )

            # Reroduce the analytical solution and calculate the residuals
            y_opt = q_model(event.x, *popt)

            # Put back to original scale to calculate the residuals
            residuals = event.y - y_opt * self.data.soil_moisture_range
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((event.y - np.nanmean(event.y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            return popt, r_squared, y_opt * self.data.soil_moisture_range

        except Exception as e:
            print("An error occurred:", e)
            return None

    def plot_drydown_models(self, event, output_dir):
        ax = plt.plot()

        # Plot observed data
        ax.scatter(event.x, event.y)

        # Plot exponential model
        ax.plot(
            event.x[~np.isnan(event.y)],
            event.exponential["y_opt"],
            alpha=0.7,
            label=f"expoential: R^2={event.exponential['r_squared']:.2f}; tau={event.exponential['tau']:.2f}",
        )

        # Plot q model
        ax.plot(
            event.x[~np.isnan(event.y)],
            event.q["y_opt"],
            alpha=0.7,
            label=f"q model: R^2={event.q['r_squared']:.2f}; q={event.q['q']:.5f})",
        )

        # Plot PET
        ax2 = ax.twinx()
        ax2.scatter(event.x, event.pet, color="orange", alpha=0.5)
        ax.set_title(f"Event {self.index}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Soil Moisture")
        ax.set_xlim([event.start_date, event.end_date])
        ax.legend()
        ax2.set_ylim([0, 8])
        ax2.set_ylabel("PET")

        # Rotate the x tick labels
        ax.tick_params(axis="x", rotation=45)

        filename = f"{self.data.EASE_row_index:03d}_{self.data.EASE_column_index:03d}_event_{self.index}.png"
        output_subdir = os.path.join(output_dir, "plots")
        if ~os.path.exists(output_subdir):
            os.makedirs(output_subdir)

        plt.savefig(
            os.path.join(output_subdir, filename),
            dpi=600,
            bbox_inches="tight",
        )
        plt.close()  # Close the current figure to release resources
