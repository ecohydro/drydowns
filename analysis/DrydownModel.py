from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from MyLogger import getLogger
import threading
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from utils import is_true

# Create a logger
log = getLogger(__name__)


def exponential_model(t, delta_theta, theta_w, tau):
    """
    Calculate the drydown curve for soil moisture over time using non-linear plant stress model.

    Parameters:
        t (int): Timestep, in day.
        delta_theta (float): Shift/increment in soil moisture after precipitation, in m3/m3.
        theta_w (float, optional): Wilting point soil moisture content, equal to s_star * porosity, in m3/m3. Default is 0.0.
        tau (float): decay rate, in 1/day.

    Returns:
        float: Rate of change in soil moisture (dtheta/dt) for the given timestep, in m3/m3/day.

    Reference:
        McColl, K.A., W. Wang, B. Peng, R. Akbar, D.J. Short Gianotti, et al. 2017.
        Global characterization of surface soil moisture drydowns.
        Geophys. Res. Lett. 44(8): 3682–3690. doi: 10.1002/2017GL072819.
    """
    return delta_theta * np.exp(-t / tau) + theta_w


def q_model(t, k, q, delta_theta, theta_star=1.0, theta_w=0.0):
    """
    Calculate the drydown curve for soil moisture over time using non-linear plant stress model.

    Parameters:
        t (int): Timestep, in day.
        k (float): Product of soil thickness (z) and maximum rate of change in normalized soil moisture (k), equivalent to maximum ET rate (ETmax), in m3/m3/day.
        q (float): Degree of non-linearity in the soil moisture response.
        delta_theta (float): Shift/increment in soil moisture after precipitation, in m3/m3. It is equal to theta_0 - theta_w.
        theta_star (float, optional): Critical soil moisture content, equal to s_star * porosity, in m3/m3. Default is 1.0.
        theta_w (float, optional): Wilting point soil moisture content, equal to s_star * porosity, in m3/m3. Default is 0.0.

    Returns:
        float: Rate of change in soil moisture (dtheta/dt) for the given timestep, in m3/m3/day.
    """

    b = delta_theta ** (1 - q)

    a = (1 - q) / ((theta_star - theta_w) ** q)

    return (-k * a * t + b) ** (1 / (1 - q)) + theta_w


def loss_sigmoid(t, theta, theta50, k, a):
    """
    Calculate the loss function (dtheta/dt vs theta relationship) using sigmoid model

    Parameters:
    t (int): Timestep, in day.
    theta (float): Volumetric soil moisture content, in m3/m3.
    theta50 (float, optional): 50 percentile soil moisture content, equal to s50 * porosity, in m3/m3
    k (float): Degree of non-linearity in the soil moisture response. k = k0 (original coefficient of sigmoid) / n (porosity), in m3/m3
    a (float): The spremum of dtheta/dt, a [-/day] = ETmax [mm/day] / z [mm]

    Returns:
    float: Rate of change in soil moisture (dtheta/dt) for the given soil mositure content, in m3/m3/day.
    """
    exp_arg = np.clip(
        -k * (theta - theta50), -np.inf, 10000
    )  # Clip exponent item to avoid failure
    d_theta = -1 * a / (1 + np.exp(exp_arg))
    return d_theta


# Function to solve the DE with given parameters and return y at the time points
def solve_de(t_obs, y_init, parameters):
    """
    The sigmoid loss function is a differential equation of dy/dt = f(y, a, b), which cannot be analytically solved,
    so the fitting of this model to drydown is numerically impelmented.
    solve_ivp finds y(t) approximately satisfying the differential equations, given an initial value y(t0)=y0.

    Parameters:
    t_obs (int): Timestep, in day.
    y_init (float): Observed volumetric soil moisture content, in m3/m3.
    parameters: a list of the follwing parameters
        theta50 (float, optional): 50 percentile soil moisture content, equal to s50 * porosity, in m3/m3
        k (float): Degree of non-linearity in the soil moisture response. k = k0 (original coefficient of sigmoid) / n (porosity), in m3/m3
        a (float): The spremum of dtheta/dt, a [-/day] = ETmax [mm/day] / z [mm]
    """
    theta50, k, a = parameters
    sol = solve_ivp(
        lambda t, theta: loss_sigmoid(t, theta, theta50, k, a),
        [t_obs[0], t_obs[-1]],
        [y_init],
        t_eval=t_obs,
        vectorized=True,
    )
    return sol.y.ravel()


# The objective function to minimize (sum of squared errors)
def objective_function(parameters, y_obs, y_init, t_obs):
    y_model = solve_de(t_obs, y_init, parameters)
    error = y_obs - y_model
    return np.sum(error**2)


class DrydownModel:
    def __init__(self, cfg, Data, Events):
        self.cfg = cfg
        self.data = Data
        self.events = Events
        self.plot_results = is_true(cfg["MODEL"]["plot_results"])
        self.force_PET = is_true(cfg["MODEL"]["force_PET"])
        self.run_exponential_model = is_true(cfg["MODEL"]["exponential_model"])
        self.run_q_model = is_true(cfg["MODEL"]["q_model"])
        self.run_sigmoid_model = is_true(cfg["MODEL"]["sigmoid_model"])

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
                if updated_event is not None:
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
        # Currently, all the three models need to be fitted to return results

        # _____________________________________________
        # Fit exponential model
        if self.run_exponential_model:
            try:
                popt, r_squared, y_opt = self.fit_exponential_model(event)
                event.add_attributes("exponential", popt, r_squared, y_opt)
            except Exception as e:
                log.debug(f"Exception raised in the thread {self.thread_name}: {e}")
                return None
        # _____________________________________________
        # Fit q model
        if self.run_q_model:
            try:
                popt, r_squared, y_opt = self.fit_q_model(event)
                event.add_attributes("q", popt, r_squared, y_opt, self.force_PET)
            except Exception as e:
                log.debug(f"Exception raised in the thread {self.thread_name}: {e}")
                return None
        # _____________________________________________
        # Fit sigmoid model
        if self.run_sigmoid_model:
            try:
                popt, r_squared, y_opt = self.fit_sigmoid_model(event)
                event.add_attributes("sigmoid", popt, r_squared, y_opt)
            except Exception as e:
                log.debug(f"Exception raised in the thread {self.thread_name}: {e}")
                return None
        # _____________________________________________
        # Finalize results for one event
        # if self.plot_results:
        #     self.plot_drydown_models(event)

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
                # Use normalized y
                # It is easier to fit q model with normalized soil moisture timeseries
                # The parameters will be get de-normalized in the post-analysis
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
        max_delta_theta = self.data.range_sm
        ini_delta_theta = event.subset_sm_range

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

        ### k (should be close to PET/z ###
        min_k = 0
        max_k = np.inf
        ini_k = event.pet / 50

        ### q ###
        min_q = 0.0
        max_q = np.inf
        ini_q = 1.0 + 1.0e-03

        ### delta_theta ###
        min_delta_theta = 0.0
        max_delta_theta = 1.0  # Equivalent of self.data.range_sm as the input theta values are normalized in this code
        ini_delta_theta = event.subset_sm_range / self.data.range_sm

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

    def fit_sigmoid_model(self, event):
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
            # Observed time series data
            t_obs = event.x
            y_obs = event.y
            y_init = y_obs[
                0
            ]  # Initial condition (assuming that the first observed data point is the initial condition)

            # Initial guess for parameters theta50, k, a
            PET = event.pet

            ini_theta50 = 0.5
            ini_k = 1
            ini_a = PET / 50

            min_theta50 = 0.0
            min_k = 0.0
            min_a = 0.0

            max_theta50 = event.max_sm
            max_k = np.inf
            max_a = PET / 50 * 100

            initial_guess = [ini_theta50, ini_k, ini_a]
            bounds = [
                (min_theta50, max_theta50),
                (min_k, max_k),
                (min_a, max_a),
            ]

            # Perform the optimization
            result = minimize(
                objective_function,
                initial_guess,
                args=(y_obs, y_init, t_obs),
                method="L-BFGS-B",
                bounds=bounds,
            )  # You can choose a different method if needed

            # The result contains the optimized parameters
            theta50_best, k_best, a_best = result.x
            best_solution = solve_ivp(
                lambda t, theta: loss_sigmoid(t, theta, theta50_best, k_best, a_best),
                [t_obs[0], t_obs[-1]],
                [y_init],
                t_eval=t_obs,
            )

            # Get the optimal fit
            y_opt = best_solution.y[0]

            # Calculate the residuals
            popt = result.x
            residuals = event.y - y_opt
            ss_res = np.sum(residuals**2)
            r_squared = 1 - ss_res / np.sum((event.y - np.nanmean(event.y)) ** 2)

            return popt, r_squared, y_opt

        except Exception as e:
            log.debug(f"Exception raised in the thread {self.thread_name}: {e}")

    def return_result_df(self):
        """Return results in the pandas dataframe format for easier concatination"""

        results = []
        for event in self.events:
            try:
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
                }

                if self.run_exponential_model:
                    _results.update(
                        {
                            "exp_delta_theta": event.exponential["delta_theta"],
                            "exp_theta_w": event.exponential["theta_w"],
                            "exp_tau": event.exponential["tau"],
                            "exp_r_squared": event.exponential["r_squared"],
                            "exp_y_opt": event.exponential["y_opt"],
                        }
                    )

                if self.run_q_model:
                    if not self.force_PET:
                        q_k = event.q["k"]
                    else:
                        q_k = event.pet
                    _results.update(
                        {
                            "q_k": q_k,
                            "q_q": event.q["q"],
                            "q_delta_theta": event.q["delta_theta"],
                            "q_r_squared": event.q["r_squared"],
                            "q_y_opt": event.q["y_opt"],
                        }
                    )

                if self.run_sigmoid_model:
                    _results.update(
                        {
                            "sigmoid_theta50": event.sigmoid["theta50"],
                            "sigmoid_k": event.sigmoid["k"],
                            "sigmoid_a": event.sigmoid["a"],
                            "sigmoid_r_squared": event.sigmoid["r_squared"],
                            "sigmoid_y_opt": event.sigmoid["y_opt"],
                        }
                    )

                # Now, _results contains only the relevant fields based on the boolean flags.
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
            fig, ax = plt.subplots(figsize=(5, 5))

        # ______________________________________
        # Plot observed data
        if plot_mode == "single":
            ax.scatter(x, event.y)

        # ______________________________________
        # Plot exponential model
        if self.run_exponential_model:
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
        if self.run_q_model:
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
        # Plot sigmoid model
        if self.run_sigmoid_model:
            try:
                ax.plot(
                    x,
                    event.sigmoid["y_opt"],
                    alpha=0.7,
                    linestyle="--",
                    color="blue",
                    label=f"sigmoid: R^2={event.sigmoid['r_squared']:.2f}; k={event.sigmoid['k']:.2f}",
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
            # ______________________________________
            # Plot exponential model
            if self.run_exponential_model:
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

            # ______________________________________
            # Plot q model
            if self.run_q_model:
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

            # ______________________________________
            # Plot sigmoid model
            if self.run_sigmoid_model:
                try:
                    sigmoid_param = f"sigmoid model: R^2={event.sigmoid['r_squared']:.2f}; k={event.sigmoid['k']:.2f}"
                    ax.text(
                        x[0],
                        event.sigmoid["y_opt"][0] - 0.03,
                        f"{sigmoid_param}",
                        fontsize=12,
                        ha="left",
                        va="bottom",
                        color="blue",
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

            plt.tight_layout()
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


"""Old codes


def neg_log_likelihood(params, t, y):
    delta_theta, theta_w, tau, sigma = params
    y_hat = exponential_model(t, delta_theta, theta_w, tau)
    residuals = y - y_hat
    ssr = np.sum(residuals**2)
    n = len(y)
    sigma2 = ssr / n
    ll = -(n / 2) * np.log(2 * np.pi * sigma2) - (1 / (2 * sigma2)) * ssr
    return -ll

"""
