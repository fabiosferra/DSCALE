from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
import scipy.optimize as sc
from scipy.stats import linregress


class FitFunction(ABC):
    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def fit(self, x: Union[np.array, pd.Series], y: Union[np.array, pd.Series]) -> None:
        return

    @abstractmethod
    def predict_y(self, x) -> pd.Series:
        return ...


class LogLogFunc(FitFunction):
    name = "Log-Log"

    def __init__(self, alpha=None, beta=None):
        self.name = "Log-Log"
        self.alpha = alpha or 0
        self.beta = beta or None
        self.r_squared = None
        self.alpha_harm = 0

    def fit(self, x, y):
        fit_res = linregress(np.log(x), np.log(y))
        self.alpha = fit_res.intercept
        self.beta = fit_res.slope
        self.r_squared = fit_res.rvalue**2

    def predict_y(self, x):
        # NOTE: different harmonization here (not the same as logistic and linear), as Final Energy (we use a log-log func for this) needs to be always above zero
        return np.exp(self.alpha + np.log(x) * self.beta + self.alpha_harm)

    def __repr__(self):
        return f"Y(x)=exp({self.alpha+self.alpha_harm}+ln(x)*{self.beta}"


class LogisticFunc(FitFunction):
    name = "logistic"
    # NOTE: See in this link example of alterantive implementation of LogisticFunc: https://stackoverflow.com/questions/75810526/curve-fit-and-extrapolate-for-sigmoid-function-in-python
    def __init__(self, alpha=None, gamma=None, beta=None, x_0=None):
        self.name = "logistic"
        self.alpha_harm = 0  # needed for harmonization step to match base year
        self.K = alpha # or None
        self.A = gamma #or None
        self.beta = beta # or None
        self.x_0 = x_0 #or None
        self.r_squared = None

    def logistic_fit(self, x, K, A, beta, x0):
        return A + ((K - A) / (1 + np.exp(-beta * (x - x0))))

    def fit(self, x: pd.Series, y: pd.Series):
        # Initial guess for the logistic function:
        K_estimate = 1  # austria["Y"].max()
        A_estimate = 0  # austria["Y"].min()
        x0_estimate = 50 # x.min() # 50 is a good initial guess for the inflection point of GDP per capita
        slope_estimate = (
            ((y.iloc[-1] - y.iloc[0]) / (x.iloc[-1] - x.iloc[0]))
            * 4
            / (K_estimate - A_estimate)
        )


        # Reasonable bounds for the logistic function:
        K_bound_upper = 1
        K_bound_lower = 0.9 # Maximum S-shape value ranging from 0.9 to 1
        A_bound = 0
        x0_bound = 0

        # Reasonable bounds for the slope are between -100 and 100
        # You can try it out here: https://www.geogebra.org/m/gfc5avbp
        # Example below: a logistic function with a slope of -100 (and inflection point = 10). Maximum value is 1 (and minimum is zero)
        # e.g using: p(x)=((1)/(1+ℯ^(-100 (x-10))))  (copy paste into geogebra)
        slope_upper = 100
        slope_lower = -100
        if slope_estimate < 0:
            slope_upper = 0
        else:
            slope_lower = 0


        # Do the actual fit based on initial guess and bounds
        popt, pcov = sc.curve_fit(
            self.logistic_fit,
            x,
            y,
            [K_estimate, A_estimate, slope_estimate, x0_estimate],
            bounds=(
                 [K_bound_lower, A_bound, slope_lower, -np.inf],
                 [K_bound_upper, np.inf, slope_upper, np.inf],
                # [0, A_bound, -1, x.min()/2],
                # [1, 1e3, 1, x.max()*100],
            ),
        )
        self.K = popt[0]
        self.A = popt[1]
        self.beta = popt[2]
        self.x_0 = popt[3]

        # move R2 calc to separate helper function
        # reference:
        # https://stackoverflow.com/questions/19189362/getting-the-r-squared-value-using-curve-fit
        residuals = y - self.logistic_fit(x, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        self.r_squared = 1 - (ss_res / ss_tot)

    def predict_y(self, x):
        return np.maximum(
            0,
            self.alpha_harm + self.logistic_fit(x, self.K, self.A, self.beta, self.x_0),
        )

    def __repr__(self):
        return f"Y(x)={self.alpha_harm}+{self.A}+({self.K}-{self.A})/(1+exp(-({self.beta})*(x-{self.x_0}))"


class LinearFunc(FitFunction):
    name = "Linear"

    def __init__(self, alpha=None, beta=None):
        self.name = "Linear"
        self.alpha = alpha or None
        self.beta = beta or None
        self.r_squared = None
        self.alpha_harm = 0

    def fit(self, x, y):
        fit_res = linregress(x, y)
        self.alpha = fit_res.intercept
        self.beta = fit_res.slope
        self.r_squared = fit_res.rvalue**2

    def predict_y(self, x):
        return np.maximum(0, self.alpha_harm + self.alpha + x * self.beta)

    def __repr__(self):
        return f"Y(x)={self.alpha}+x*{self.beta}"


func_dict = {
    "log-log": LogLogFunc,
    "linear": LinearFunc,
    "logistic": LogisticFunc,
}
