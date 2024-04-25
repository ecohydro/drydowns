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


class DrydownModel:

    args = []

    def __init__(self, cfg, data, event):

        self.cfg = cfg
        self.data = data
        self.event = event

        self.specs = self.get_specs()

        self._norm = False

        self.type = None

        if self.cfg["run_mode"] == "parallel":
            current_thread = threading.current_thread()
            current_thread.name = (f"{self.data.id[0]}, {self.data.id[1]}")
            self.thread_name = current_thread.name
        else:
            self.thread_name = "main thread"

    def get_specs(self):
        _specs = self._get_specs()
        return {k : _specs.get(k, True) for k in self.args}

    def _get_specs(self):
        specs = {
            'delta_theta' : self.cfg.getboolean('fit_theta_0'),
            'k' : self.cfg.getboolean('fit_et'),
            'theta_star' : self.cfg.getboolean('fit_theta_star'),
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
        return self._fit_model(event, self.model, self.bounds, self.p0, self._norm)

    # COME BACK TO THIS
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
            # y
            y_fit = event.theta_norm if norm else event.theta
            # Fit the model
            popt, _ = curve_fit(
                f=model, xdata=event.t, ydata=y_fit, p0=p0, bounds=bounds
            )
            # Get optimal fit
            y_opt = model(event.t, *popt)
            # Denormalize
            if norm:
                y_opt = self._denormalize_theta(y_opt, event)
            # Calculate residuals
            residuals = event.theta - y_opt
            ss_res = np.sum(residuals**2)
            r_squared = 1 - ss_res / np.sum((event.theta - np.nanmean(event.theta)) ** 2)
            
            return popt, r_squared, y_opt
        
        except Exception as e:
            log.debug(f"Exception raised in the thread {self.thread_name}: {e}")


    def get_results(self, event, popt, r_squared, y_opt):
        return self._get_results(event, popt, r_squared, y_opt)

    def _get_results(self, event, popt, r_squared, y_opt):
        # Get results
        keys = [p for p in self.args if self.specs[p]]
        results = {p : popt[i] for i, p in enumerate(keys)}
        # results = {p : popt[i] for i,p in enumerate(self.args) if self.specs[p]}
        if not 'delta_theta' in results:
            results.update({
                'delta_theta' : self.params['delta_theta'],
            })
        # Denormalize if necessary
        if self._norm:
            self.denormalize_params(results, event)
        results['r_squared'] = r_squared
        results['theta_opt'] = y_opt.tolist()
        return results

    def normalize_params(self, params, event):
        for name, p in params.items():
            if name in ['delta_theta', 'k']:
                params[name] = self._normalize_param(p, event)
            elif name in ['theta_0', 'theta_star', 'theta_w']:
                params[name] = self._normalize_theta(p, event)
        # return params

    def _normalize_param(self, a, event):
        return a / (event.theta_star - event.theta_w)

    def _normalize_theta(self, theta, event):
        return (theta - event.theta_w) / (event.theta_star - event.theta_w)

    def denormalize_params(self, params, event):
        for name, p in params.items():
            if name in ['delta_theta', 'k']:
                params[name] = self._denormalize_param(p, event)
            elif name in ['theta_0', 'theta_star', 'theta_w']:
                params[name] = self._denormalize_theta(p, event)
        # return params
    def _denormalize_param(self, a, event):
        return a * (event.theta_star - event.theta_w)
    
    def _denormalize_theta(self, theta, event):
        return theta * (event.theta_star - event.theta_w) + event.theta_w

    @property
    def model(self):
        if not hasattr(self, '_model'):
            self._model = self._get_model()
        return self._model
    
    def _get_model(self):
        raise NotImplementedError

    @property
    def params(self):
        if not hasattr(self, '_params'):
            # self._params = self.get_params(self.event)
            self.set_params(self.event)
        return self._params
    
    @property
    def bounds(self):
        if not hasattr(self, '_bounds'):
            self._bounds = self._get_bounds(self.params)
        return self._bounds

    @property
    def p0(self):
        if not hasattr(self, '_p0'):
            self._p0 = self._get_p0(self.params)
        return self._p0

    def set_params(self):
        raise NotImplementedError

    def _get_bounds(self, params : dict):
        log.info(f"Getting bounds for event {self.event} model")
        bounds = (
            [params[k][0] for k in self.args if self.specs[k]],
            [params[k][1] for k in self.args if self.specs[k]]
        )
        return bounds

    def _get_p0(self, params : dict):
        return [params[k][2] for k in self.args if self.specs[k]]
    

    def theta_0(self, event):
        if self.specs['delta_theta']:
            return np.array([
                event.theta_w,  # min_theta_0
                # event.theta_star, # max_theta_0
                np.minimum(     # max_theta_0
                    event.theta_star, self.data.theta_fc
                ),              # max_theta_0
                # event.event_range + event.theta_w # ini_theta_0
                np.minimum(
                    event.event_max, self.data.theta_fc
                )
            ])
        else:
            return np.minimum(event.event_max, self.data.theta_fc)


    def theta_w(self, event):
        return np.array([
            self.data.min_sm, # min_theta_w
            event.event_min, # max_theta_w
            (self.data.min_sm + event.event_min) / 2 # ini_theta_w
        ])

    def et_max(self, event):
        if not self.specs['k']:
            return event.PET
        else:
            return np.array([
                0,          # min_et_max
                # 100.,       # max_et_max
                event.pet,  # max_et_max
                # event.pet   # ini_et_max
                (event.pet) / 2    # ini_et_max (???)
                #event.theta_star - event.theta_w
            ])
        
    def k(self, event=None, et_max=None):
        if not et_max:
            et_max = self.et_max(event)
        return et_max / (self.data.z*1000)

    def delta_theta(self, event=None, theta_0=None):
        if theta_0 is None:
            theta_0 = self.theta_0(event)
        return theta_0 - event.theta_w




class ExponentialModel(DrydownModel):

    args = ['delta_theta', 'theta_w', 'tau']

    def __init__(self, cfg, data, event):
        super().__init__(cfg, data, event)

        self.type = 'exponential'
        # Set the model
        # self.model = exponential_model

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

        params = {
            'delta_theta' : delta_theta,
            'theta_0' : theta_0,
            'theta_w' : theta_w,
            'tau' : tau
        }

        self._params = params

    def _get_model(self):
        if not self.specs['delta_theta']:
            return lambda t, theta_w, tau: exponential_model(
                t, self.params['delta_theta'], theta_w, tau
            )
        else:
            return exponential_model

    def tau(self):
        return np.array([
            0., # min_tau
            np.inf, # max_tau
            1.0 # ini_tau
        ])
    
    def get_results(self, event, popt, r_squared, y_opt):
        results = self._get_results(event, popt, r_squared, y_opt)
        results.update({
            'theta_0' : results['delta_theta'] + results['theta_w'],
            'k' : (event.theta_star - results['theta_w']) / results['tau'],
            # 'ET_max' : (self.data.z * 1000) * (self.event.theta_star - results['theta_w']) / results['tau']
        })
        results['ET_max'] = (self.data.z * 1000) * results['k']

        return results



class NonlinearModel(DrydownModel):

    args = ['delta_theta', 'k', 'q', 'theta_star']

    def __init__(self, cfg, data, event):
        super().__init__(cfg, data, event)
        
        self.type = 'q'

        # self.model = q_model

        self._norm = not self.specs['theta_star']   # if fitting theta_star, don't normalize


    def set_params(self, event):
        """
        Get the parameters of the nonlinear model.

        Parameters
        ----------
        event : Event
            The event to get the parameters for.

        Returns
        -------
        tuple
            The optimal parameters, the r-squared value, and the optimal fit.

        """
        delta_theta = self.delta_theta(event)

        k = self.k(event)

        q = self.q()

        theta_star = self.theta_star(event)

        theta_w = event.theta_w

        params = {
            'delta_theta' : delta_theta,
            'k' : k,
            'q' : q,
            'theta_star' : theta_star,
            'theta_w' : theta_w,
        }

        if self._norm:
            self.normalize_params(params, event)

        self._params = params

    def _get_model(self):
        if self.specs['theta_star']: 
            # THIS IS THE DEFAULT OLD VERSION (STAR)
            if self.specs['k'] and self.specs['delta_theta']:
                return lambda t, delta_theta, k, q, theta_star: q_model(
                    t, delta_theta, k, q, 
                    theta_star, self.params['theta_w']
                )
            elif self.specs['k'] and not self.specs['delta_theta']:
                return lambda t, k, q, theta_star: q_model(
                    t, self.params['delta_theta'], k, q, 
                    theta_star, self.params['theta_w']
                )
            elif not self.specs['k'] and self.specs['delta_theta']:
                return lambda t, delta_theta, q, theta_star: q_model(
                    t, delta_theta, self.params['k'], q, 
                    theta_star, self.params['theta_w']
                )
            elif not self.specs['k'] and not self.specs['delta_theta']:
                return lambda t, q, theta_star: q_model(
                    t, self.params['delta_theta'], self.params['k'], q, 
                    theta_star, self.params['theta_w']
                )
        else:
            # THIS IS THE DEFAULT OLD VERSION (ET)
            if self.specs['k'] and self.specs['delta_theta']:
                return lambda t, delta_theta, k, q: q_model(
                    t, delta_theta, k, q, 
                    self.params['theta_star'], self.params['theta_w']
                )
            elif self.specs['k'] and not self.specs['delta_theta']:
                return lambda t, k, q: q_model(
                    t, self.params['delta_theta'], k, q, 
                    self.params['theta_star'], self.params['theta_w']
                )
            elif not self.specs['k'] and self.specs['delta_theta']:
                return lambda t, delta_theta, q: q_model(
                    t, delta_theta, self.params['k'], q, 
                    self.params['theta_star'], self.params['theta_w']
                )
            elif not self.specs['k'] and not self.specs['delta_theta']:
                return lambda t, q: q_model(
                    t, self.params['delta_theta'], self.params['k'], q, 
                    self.params['theta_star'], self.params['theta_w']
                )
            
    def get_results(self, event, popt, r_squared, y_opt):
        results = self._get_results(event, popt, r_squared, y_opt)

        results.update({
            'theta_0' : results['delta_theta'] + event.theta_w,
        })
        if 'k' in results:
            # Denormalize 
            # This shouldn't be necessary anymore, but need to verify...
            # if self._norm:
            #     results['k'] = results['k'] * (event.theta_star - event.theta_w)
            results.update({
                'ET_max' : (self.data.z * 1000) * results['k']
            })
        return results

    def q(self):
        return np.array([
            0.,             # min_q
            np.inf,         # max_q
            1.0 + 1.0e-03   # ini_q
        ])

    def theta_star(self, event):
        if not self.specs['theta_star']:
            return event.theta_star
        else:
            return np.array([
                0.0, # min_theta_star
                self.data.theta_fc, # max_theta_star
                event.theta_star # ini_theta_star
            ])



class SigmoidModel(DrydownModel):
    
    popt_dict = {
        0 : 'theta50',
        1 : 'k',
        2 : 'a',
    }

    def __init__(self, cfg, data, event):

        super().__init__(cfg, data, event)

        self.type = 'sigmoid'
        # Set the model
        self.model = loss_sigmoid

        if not self.specs['fit_theta_star']:
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
        if not hasattr(self, '_args'):
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
            ini_a = PET / (self.data.z * 1e3) #50

            min_theta50 = 0.0
            min_k = 0.0
            min_a = 0.0

            max_theta50 = event.theta_star #event.max_sm
            max_k = np.inf
            max_a = PET / (self.data.z*1e3) * 100 #50 * 100

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
            r_squared = 1 - ss_res / np.sum((event.theta - np.nanmean(event.theta)) ** 2)

            return popt, r_squared, y_opt

        except Exception as e:
            log.debug(f"Exception raised in the thread {self.thread_name}: {e}")
