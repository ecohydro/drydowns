

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize


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


# def q_model(t, k, q, delta_theta, theta_star=1.0, theta_w=0.0):
def q_model(t, delta_theta, q, k, theta_star=1.0, theta_w=0.0):
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


def loss_sigmoid(t, theta, theta_50, k, a):
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
        -k * (theta - theta_50), -np.inf, 10000
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
    theta_50, k, a = parameters
    sol = solve_ivp(
        lambda t, theta: loss_sigmoid(t, theta, theta_50, k, a),
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