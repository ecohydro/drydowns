import numpy as np
import pandas as pd

import threading

from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# from .event import Event
# from .towerevent import SensorEvent


from .functions import objective_function, exponential_model, q_model, loss_sigmoid
from .mylogger import getLogger

# Create a logger
log = getLogger(__name__)

"""

Name:           model.py
Compatibility:  Python 3.7.0
Description:    Description of what program does

URL:            https://

Requires:       list of libraries required

Dev ToDo:       None

AUTHOR:         Ryoko Araki (initial dev); Bryn Morgan (refactor)
ORGANIZATION:   University of California, Santa Barbara
Contact:        raraki@ucsb.edu
Copyright:      (c) Ryoko Araki & Bryn Morgan 2024


"""
# class DrydownModelHandler:

#     models = {
#         'exponential' : ExponentialModel,
#         'q' : NonlinearModel,
#         'sigmoid' : SigmoidModel
#     }

#     def __init__(self, cfg, data, events):

#         self.cfg = cfg
#         self.data = data
#         self.events = events

#         self.specs = self.get_specs()


#         if self.cfg["run_mode"] == "parallel":
#             current_thread = threading.current_thread()
#             current_thread.name = (f"{self.data.id[0]}, {self.data.id[1]}")
#             self.thread_name = current_thread.name
#         else:
#             self.thread_name = "main thread"

#     def get_specs(self):
#         specs = {
#             'force_PET' : self.cfg.getboolean('force_PET'),
#             'fit_theta_star' : self.cfg.getboolean('fit_theta_star'),
#             # 'run_mode' : self.cfg.get('run_mode'),
#         }
#         return specs

#     # def get_models(self):
#     #     mod_dict = {
#     #         k : self.cfg.getboolean(k + '_model') for k in ['exponential', 'q', 'sigmoid']
#     #     }
#     #     self.cfg.getboolean('exponential_model')

#     def fit_events(self):
#         for event in self.events:
#             self.fit_event(event)

#     def fit_event(self, event):
#         for k in self.models.keys():
#             if self.cfg.getboolean(k + '_model'):
#                 # self._fit_event(event, k)
#                 obj = self.models[k]
#                 model = obj(self.cfg, self.data, event)
#                 model.fit_event(event)

#     def get_results(self):
#         # results = [
#         #     self.get_event_results(event) for event in self.events if self.get_event_results(event)
#         # ]
#         results = []
#         for event in self.events:
#             try:
#                 results.append(self._get_event_results(event))
#             except Exception as e:
#                 log.debug(f"Exception raised in the thread {self.thread_name}: {e}")
#                 continue
#         df = pd.DataFrame(results)
#         # if not results:
#         #     df = pd.DataFrame()
#         return df

#     # def get_event_results(self, event):
#     #     try:
#     #         results = self._get_event_results(event)
#     #     except Exception as e:
#     #         log.debug(f"Exception raised in the thread {self.thread_name}: {e}")
#     #         results = None
#     #     return results

#     def _get_event_results(self, event):
#         if isinstance(self.data, SensorData):
#             col_ids = ('SITE_ID', 'Sensor_ID')
#         elif isinstance(self.data, SMAPData):
#             col_ids = ("EASE_row_index", "EASE_column_index")

#         results = {
#             col_ids[0] : self.data.id[0],
#             col_ids[1] : self.data.id[1],
#             **event.describe(),
#             'min_sm' : self.data.min_sm,
#             'max_sm' : self.data.max_sm,
#             'theta_fc' : self.data.theta_fc,
#             'porosity' : self.data.n,
#             'pet' : event.pet,
#         }
#         try:
#             results.update({
#                 'et' : event.get_et(),
#                 'total_et' : np.sum(event.get_et()),
#                 'precip' : event.calc_precip(),
#             })
#         except:
#             pass

#         for mod,abbrev in zip(self.models.keys(), ['exp', 'q', 'sig']):
#             if self.cfg.getboolean(mod + '_model') & hasattr(event, mod):
#                 mod_results = getattr(event, mod)
#                 results.update({f"{abbrev}_{k}" : v for k, v in mod_results.items()})

#         return results


class DrydownModel:

    popt_dict = {}

    def __init__(self, cfg, data, event):

        self.cfg = cfg
        self.data = data
        self.event = event

        self.specs = self.get_specs()

        self._norm = False

        self.type = None

        if self.cfg["run_mode"] == "parallel":
            current_thread = threading.current_thread()
            current_thread.name = f"{self.data.id[0]}, {self.data.id[1]}"
            self.thread_name = current_thread.name
        else:
            self.thread_name = "main thread"

    def get_specs(self):
        specs = {
            "force_PET": self.cfg.getboolean("force_PET"),
            "fit_theta_star": self.cfg.getboolean("fit_theta_star"),
            # 'run_mode' : self.cfg.get('run_mode'),
        }
        return specs

    def fit_event(self, event):
        """
        Run the model for the given event.

        Parameters
        ----------
        event : Event
            The event to run the model for.

        Returns
        -------
        tuple
            The optimal parameters, the r-squared value, and the optimal fit.

        """
        try:
            # Fit the model
            popt, r_squared, y_opt = self.fit(event)
            # Get results
            results = self.get_results(event, popt, r_squared, y_opt)
            # Add the results to the event
            event.add_results(self.type, results)
        except Exception as e:
            log.debug(f"Exception raised in the thread {self.thread_name}: {e}")
            results = None
        return results

        # return self.results

    # @property
    # def results(self):
    #     if not hasattr(self, '_results'):
    #         self._results = self.get_results(
    #             self, self.event, self.popt, self.r_squared, self.theta_opt
    #         )
    #     return self._results

    def get_results(self, event, popt, r_squared, y_opt):
        return self._get_results(popt, r_squared, y_opt)

    def _get_results(self, popt, r_squared, y_opt):
        results = {v: popt[k] for k, v in self.popt_dict.items() if k < len(popt)}
        results["r_squared"] = r_squared
        results["theta_opt"] = y_opt.tolist()
        return results

    def fit(self, event):
        """
        Fit the nonlinear model to the data.

        Parameters
        ----------
        event : Event
            The event to fit the model to.
        norm : bool, optional
            Whether to normalize the data before fitting. The default is False.

        Returns
        -------
        tuple
            The optimal parameters, the r-squared value, and the optimal fit.

        """
        # if force_PET:
        #     mod = lambda t, delta_theta, q: self.model(t, delta_theta, event.pet, q, 1.0, 0.0)
        # if not force_PET:
        #     mod = lambda t, delta_theta, k, q: self.model(t, delta_theta, k, q, 1.0, 0.0)
        # if fit_theta_star:
        #     mod = lambda t, delta_theta, k, q, theta_star: self.model(t, delta_theta, k, q, theta_star, 0.0)

        # mod = lambda t, *params : self.model(t, *params, *args)

        # self.popt, self.r_squared, self.theta_opt = self._fit_model(
        #     event, self.model, self.bounds, self.p0, self._norm
        # )
        # return self.popt, self.r_squared, self.theta_opt
        return self._fit_model(event, self.model, self.bounds, self.p0, self._norm)

    def _fit_model(self, event, model, bounds, p0, norm=False):
        """
        Base function for fitting models

        Parameters
        ----------
        event : Event
            The event to fit the model to.
        model : function
            The model to fit to the data.
        bounds : list
            The bounds for the parameters.
        p0 : list
            The initial guess for the parameters.
        norm : bool, optional
            Whether to normalize the data before fitting. The default is False.

        Returns
        -------
        tuple
            The optimal parameters, the r-squared value, and the optimal fit.

        """
        try:
            if norm:
                # It's easier to fit q model with normalized soil moisture timeseries
                y_fit = event.theta_norm
            else:
                y_fit = event.theta

            # Fit the model
            popt, _ = curve_fit(
                f=model, xdata=event.t, ydata=y_fit, p0=p0, bounds=bounds
            )

            # Get the optimal fit
            y_opt = model(event.t, *popt)

            # Denormalize
            if norm:
                # theta
                y_opt = y_opt * (event.theta_star - event.theta_w) + event.theta_w
                # delta_theta
                popt[0] = popt[0] * (event.theta_star - event.theta_w)

            # Calculate the residuals
            residuals = event.theta - y_opt  # event.y - y_opt
            ss_res = np.sum(residuals**2)
            # r_squared = 1 - ss_res / np.sum((event.y - np.nanmean(event.y)) ** 2)
            r_squared = 1 - ss_res / np.sum(
                (event.theta - np.nanmean(event.theta)) ** 2
            )

            return popt, r_squared, y_opt

        except Exception as e:
            log.debug(f"Exception raised in the thread {self.thread_name}: {e}")

    @property
    def bounds(self):
        if not hasattr(self, "_bounds"):
            self._bounds = self._get_bounds(self.params)
        return self._bounds

    @property
    def params(self):
        if not hasattr(self, "_params"):
            # self._params = self.get_params(self.event)
            self.set_params(self.event)
        return self._params

    @property
    def p0(self):
        if not hasattr(self, "_p0"):
            self._p0 = self._get_p0(self.params)
        return self._p0

    def _get_bounds(self, params: list):
        log.info(f"Getting bounds for event {self.event} model")
        bounds = ([p[0] for p in params], [p[1] for p in params])
        return bounds

    def _get_p0(self, params: list):
        return [p[2] for p in params]

    def theta_0(self, event):
        return np.array(
            [
                event.theta_w,  # min_theta_0
                # event.theta_star, # max_theta_0
                np.minimum(  # max_theta_0
                    event.theta_star, self.data.theta_fc
                ),  # max_theta_0
                event.event_range + event.theta_w,  # ini_theta_0
            ]
        )

    def theta_w(self, event):
        return np.array(
            [
                self.data.min_sm,  # min_theta_w
                event.event_min,  # max_theta_w
                (self.data.min_sm + event.event_min) / 2,  # ini_theta_w
            ]
        )

    def et_max(self, event):
        return np.array(
            [
                0,  # min_et_max
                100.0,  # max_et_max
                event.pet,  # ini_et_max
                # event.theta_star - event.theta_w
            ]
        )

    def k(self, event=None, et_max=None):
        if not et_max:
            et_max = self.et_max(event)
        return et_max / (self.data.z * 1000)

    def delta_theta(self, event, theta_0=None):
        if theta_0 is None:
            theta_0 = self.theta_0(event)
        return theta_0 - event.theta_w


class ExponentialModel(DrydownModel):

    popt_dict = {
        0: "delta_theta",
        1: "theta_w",
        2: "tau",
    }

    def __init__(self, cfg, data, event):

        super().__init__(cfg, data, event)

        self.type = "exponential"
        # Set the model
        self.model = exponential_model

        self._norm = False

    def set_params(self, event, norm=False):
        """
        Get the parameters of the exponential model.

        Parameters
        ----------
        event : Event
            The event to get the parameters for.
        norm : bool, optional
            Whether to normalize the data before fitting. The default is False.

        Returns
        -------
        tuple
            The optimal parameters, the r-squared value, and the optimal fit.

        """
        theta_0 = self.theta_0(event)

        delta_theta = self.delta_theta(event, theta_0)

        theta_w = self.theta_w(event)

        tau = self.tau()

        params = [delta_theta, theta_w, tau]

        self._params = params
        # return params

    def tau(self):
        return np.array([0.0, np.inf, 1.0])  # min_tau  # max_tau  # ini_tau

    # Exponential
    def get_results(self, event, popt, r_squared, y_opt):
        results = self._get_results(popt, r_squared, y_opt)
        results.update(
            {
                "theta_0": results["delta_theta"] + results["theta_w"],
                "k": (event.theta_star - results["theta_w"]) / results["tau"],
                # 'ET_max' : (self.data.z * 1000) * (self.event.theta_star - results['theta_w']) / results['tau']
            }
        )
        results["ET_max"] = (self.data.z * 1000) * results["k"]

        return results


class NonlinearModel(DrydownModel):

    popt_dict = {
        0: "delta_theta",
        1: "k",  #'q',
        2: "q",  #'k',
        3: "theta_star",
    }

    def __init__(self, cfg, data, event):

        super().__init__(cfg, data, event)

        self.type = "q"
        # Set the model
        self.model = q_model

        if not self.specs["fit_theta_star"]:
            self._norm = True

    def __repr__(self):
        return f"NonlinearModel"

    # @property
    # def params(self):
    #     if not hasattr(self, '_params'):
    #         # self._params = self.get_params(self.event)
    #         self.set_params(self.event)
    #     return self._params

    # @property
    # def args(self):
    #     if not hasattr(self, '_args'):
    #         self.set_params(self.event)
    #     return self._args

    @property
    def args(self):
        if not hasattr(self, "_args"):
            self.set_params(self.event)
        return self._args

    def set_params(self, event, norm=True):
        """
        Get the parameters of the nonlinear model.

        Parameters
        ----------
        event : Event
            The event to get the parameters for.
        norm : bool, optional
            Whether to normalize the data before fitting. The default is False.

        Returns
        -------
        tuple
            The optimal parameters, the r-squared value, and the optimal fit.

        """
        # theta_0 = self.theta_0(event)
        # et_max = self.et_max(event)
        log.info(f"Setting params for event {event.start_date}")
        delta_theta = self.delta_theta(event)
        k = self.k(event)

        q = self.q()

        theta_w = event.theta_w  # self.theta_w(event)
        theta_star = self.theta_star(event)

        # Get list of parameters
        params = [delta_theta, k, q]  # parameters to be fitted
        args = [theta_star, theta_w]  # arguments to be passed (constant params)

        if self.specs["fit_theta_star"]:
            theta_star = self.theta_star(event)
            params.append(theta_star)
            args = [theta_w]
            mod = lambda t, delta_theta, k, q, theta_star: q_model(  # self.model(
                t, delta_theta, k, q, theta_star, self.data.min_sm  # event.theta_w
            )
        elif self._norm:
            params = self.normalize_params([delta_theta, k], event) + [q]
            # params = [params[i] for i in [0, 2, 1]]     # put back in order: delta_theta, q, k
            args = [
                self.normalize_theta(theta, event) for theta in args
            ]  # or just [1.0, 0.0]

            mod = lambda t, delta_theta, k, q: q_model(  # self.model(
                t, delta_theta, k, q, 1.0, 0.0
            )

            if self.specs["force_PET"]:
                params = params[:-1]
                args = [event.pet] + args

                mod = lambda t, delta_theta, q: q_model(  # self.model(
                    t, delta_theta, q, event.pet, 1.0, 0.0
                )

        self.model = mod  # lambda t, *params: self.model(t, *params, *args)

        self._params = params
        self._args = args
        # return params, args

    def normalize_params(self, params, event):
        # p_norm = p / (event.theta_star - event.theta_w)
        return [self._normalize_param(p, event) for p in params]

    def _normalize_param(self, a, event):
        return a / (event.theta_star - event.theta_w)

    def normalize_theta(self, theta, event):
        return (theta - event.theta_w) / (event.theta_star - event.theta_w)

    def q(self):
        return np.array([0.0, np.inf, 1.0 + 1.0e-03])  # min_q  # max_q  # ini_q

    def theta_star(self, event):
        return np.array(
            [
                0.0,  # min_theta_star
                self.data.theta_fc,  # max_theta_star
                event.theta_star,  # ini_theta_star
            ]
        )

    def get_results(self, event, popt, r_squared, y_opt):
        results = self._get_results(popt, r_squared, y_opt)
        results.update(
            {
                "theta_0": results["delta_theta"] + event.theta_w,
            }
        )
        if "k" in results:
            # Denormalize
            # TODO: Figure out how to do this not here.
            if self._norm:
                results["k"] * (event.theta_star - event.theta_w)
            results.update({"ET_max": (self.data.z * 1000) * results["k"]})
        return results


# Types of runs:
# 1. Fit theta_star w/o normalization
# 2. Fit ET w/ normalization
# 3. Force ETmax w/ normalization


class SigmoidModel(DrydownModel):

    popt_dict = {
        0: "theta50",
        1: "k",
        2: "a",
    }

    def __init__(self, cfg, data, event):

        super().__init__(cfg, data, event)

        self.type = "sigmoid"
        # Set the model
        self.model = loss_sigmoid

        if not self.specs["fit_theta_star"]:
            self._norm = True
        # # Set the bounds
        # self.bounds = (
        #     [0, 0, 0], # [a, b, c]
        #     [np.inf, np.inf, np.inf]
        # )

        # # Set the initial guess
        # self.p0 = [1, 1, 1]

    def __repr__(self):
        return f"SigmoidModel"

    # @property
    # def params(self):
    #     if not hasattr(self, '_params'):
    #         # self._params = self.get_params(self.event)
    #         self.set_params(self.event)
    #     return self._params

    # @property
    # def args(self):
    #     if not hasattr(self, '_args'):
    #         self.set_params(self.event)
    #     return self._args

    @property
    def args(self):
        if not hasattr(self, "_args"):
            self.set_params(self.event)
        return self._args

    def fit(self, event):
        """
        Fit the nonlinear model to the data.

        Parameters
        ----------
        event : Event
            The event to fit the model to.

        Returns
        -------
        tuple
            The optimal parameters, the r-squared value, and the optimal fit.

        """
        # self.popt, self.r_squared, self.theta_opt = self._fit_model(event)
        # return self.popt, self.r_squared, self.theta_opt
        return self._fit_model(event)

    # TODO: Update this; for now it's just copied.
    def _fit_model(self, event):
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
            # y_obs = event.y
            y_obs = event.theta
            y_init = y_obs[
                0
            ]  # Initial condition (assuming that the first observed data point is the initial condition)

            # Initial guess for parameters theta50, k, a
            PET = event.pet

            ini_theta50 = 0.5
            ini_k = 1
            ini_a = PET / (self.data.z * 1e3)  # 50

            min_theta50 = 0.0
            min_k = 0.0
            min_a = 0.0

            max_theta50 = event.theta_star  # event.max_sm
            max_k = np.inf
            max_a = PET / (self.data.z * 1e3) * 100  # 50 * 100

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
                lambda t, theta: self.model(t, theta, theta50_best, k_best, a_best),
                [t_obs[0], t_obs[-1]],
                [y_init],
                t_eval=t_obs,
            )

            # Get the optimal fit
            y_opt = best_solution.y[0]

            # Calculate the residuals
            popt = result.x
            # residuals = event.y - y_opt
            residuals = event.theta - y_opt
            ss_res = np.sum(residuals**2)
            # r_squared = 1 - ss_res / np.sum((event.y - np.nanmean(event.y)) ** 2)
            r_squared = 1 - ss_res / np.sum(
                (event.theta - np.nanmean(event.theta)) ** 2
            )

            return popt, r_squared, y_opt

        except Exception as e:
            log.debug(f"Exception raised in the thread {self.thread_name}: {e}")
