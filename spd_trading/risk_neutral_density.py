import numpy as np
from scipy.stats import norm
from statsmodels.nonparametric.bandwidths import bw_silverman
from localpoly.base import LocalPolynomialRegression, LocalPolynomialRegressionCV

from .utils.smoothing import bspline
from .utils.density import pointwise_density_trafo_K2M


# ----------------------------------------------------------------------------------------------------------- CALCULATOR


def create_bandwidth_range(X, num=10):
    """Creates a range of bandwidths around the Silverman Bandwidth. Is used as a parameter grid for bandwidth
    optimization for smoothing algorithms.

    Args:
        X (np.array): X position of values
        num (int, optional): Length of grid. Defaults to 10.

    Returns:
        np.array: Equal grid of bandwidths around Silverman bandwidth.
    """
    bw_silver = bw_silverman(X)
    if bw_silver > 10:
        lower_bound = max(0.5 * bw_silver, 100)
    else:
        lower_bound = max(0.5 * bw_silver, 0.03)
    x_bandwidth = np.linspace(lower_bound, 7 * bw_silver, num)
    return x_bandwidth, bw_silver, lower_bound


def rookley_method(M, S, K, o, o1, o2, r, tau):
    """Uses Rookleys Method to calculate the Risk Neutral Density from an option table. The method can be found in the
    original paper. It calculates the second derivative of the option price by strike price (analytical solution) where
    some dimentionality reduction is used, that has to be resolved in the last step.

    Args:
        M (float)   : Moneyness = S / K
        S (float)   : Price of underlying
        K (float)   : Strike Price of option
        o (float)   : implied volatility (iv, might be value of smoothed iv surface at M)
        o1 (float)  : value of first derivative of iv surface at M
        o2 (float)  : value of second derivative of iv surface at M
        r (float)   : risk free interest rate
        tau (float) : time until maturity (in years)

    Returns:
        float: value of Risk Neutral Density at Strike Price K.
        (usually the result is projected to M, use Density Transformation)
    """
    st = np.sqrt(tau)
    rt = r * tau
    ert = np.exp(rt)

    d1 = (np.log(M) + (r + 1 / 2 * o ** 2) * tau) / (o * st)
    d2 = d1 - o * st

    del_d1_M = 1 / (M * o * st)
    del_d2_M = del_d1_M
    del_d1_o = -(np.log(M) + rt) / (o ** 2 * st) + st / 2
    del_d2_o = -(np.log(M) + rt) / (o ** 2 * st) - st / 2

    d_d1_M = del_d1_M + del_d1_o * o1
    d_d2_M = del_d2_M + del_d2_o * o1

    dd_d1_M = (
        -(1 / (M * o * st)) * (1 / M + o1 / o)
        + o2 * (st / 2 - (np.log(M) + rt) / (o ** 2 * st))
        + o1 * (2 * o1 * (np.log(M) + rt) / (o ** 3 * st) - 1 / (M * o ** 2 * st))
    )
    dd_d2_M = (
        -(1 / (M * o * st)) * (1 / M + o1 / o)
        - o2 * (st / 2 + (np.log(M) + rt) / (o ** 2 * st))
        + o1 * (2 * o1 * (np.log(M) + rt) / (o ** 3 * st) - 1 / (M * o ** 2 * st))
    )

    d_c_M = norm.pdf(d1) * d_d1_M - 1 / ert * norm.pdf(d2) / M * d_d2_M + 1 / ert * norm.cdf(d2) / (M ** 2)
    dd_c_M = (
        norm.pdf(d1) * (dd_d1_M - d1 * (d_d1_M) ** 2)
        - norm.pdf(d2) / (ert * M) * (dd_d2_M - 2 / M * d_d2_M - d2 * (d_d2_M) ** 2)
        - 2 * norm.cdf(d2) / (ert * M ** 3)
    )

    dd_c_K = dd_c_M * (M / K) ** 2 + 2 * d_c_M * (M / K ** 2)
    q = ert * S * dd_c_K

    return q


class Calculator:
    """The Calculator Class for the Risk Neutral Density.

    Stores all relevant parameters for traceability and reproducability.
    - Rookleys method
    - Local polynomial estimation
    - Density transformation

    Attributes:
        data: The option table as a pd.DataFrame. Must include columns ["M", "iv", "S", "P", "K", "option", "tau"]
        tau_day: Time to maturity in days (tau * 365)
        date: Date of the option table.
        sampling: Whether the dataset should be partitioned “random” or as “slicing”. Defaults to “random”
        h_m: bandwidth for local polynomial estimation for iv-smile-fit
        h_m2: bandwidth for local polynomial estimation for q_M fit
        h_k: bandwidth for local polynomial estimation for q_K fit
        r: risk free interest rate
    """

    def __init__(
        self,
        data,
        tau_day,
        date,
        sampling="random",
        n_sections=15,
        loss="MSE",
        kernel="gaussian",
        h_m=None,
        h_m2=None,
        h_k=None,
        r=0,
    ):
        self.data = data
        self.tau_day = tau_day
        self.date = date

        # parameters for LocalPolynomialRegression
        self.sampling = sampling
        self.n_sections = n_sections
        self.loss = loss
        self.kernel = kernel
        self.h_m = h_m
        self.h_m2 = h_m2
        self.h_k = h_k
        self.r = r

        # parameters that are created during run
        self.tau = self.data.tau.iloc[0]
        self.K = None
        self.M = None
        self.q_M = None
        self.q_K = None
        self.M_smile = None
        self.smile = None
        self.first = None
        self.second = None

    def curve_fit(self, X, y, h=None):
        """Uses local polynomial estimation to create a fit and estimate its first and second derivative.
        If no bandwidth h is given, first a Cross Validation for a range of bandwidths is performed. The final fit is
        performed with the optimal bandwidth (by MSE).

        Args:
            X (np.array): X-values of data that is to be fitted (explanatory variable)
            y (np.array): y-values of data that is to be fitted (observations)
            h ([type], optional): Bandwidth for local polynomial estimation. If not specified, h will be determined by
                Cross Validation. Defaults to None.

        Returns:
            dict: Results of fit. "parameters" ("h","bandwidths","MSE"), "fit" ("X", "y", "first", "second")
        """
        if h is None:
            list_of_bandwidths, bw_silver, lower_bound = create_bandwidth_range(X)
            model_cv = LocalPolynomialRegressionCV(
                X=X,
                y=y,
                kernel=self.kernel,
                n_sections=self.n_sections,
                loss=self.loss,
                sampling=self.sampling,
            )

            cv_results = model_cv.bandwidth_cv(list_of_bandwidths)
            parameters = {
                "h": cv_results["fine results"]["h"],
                "bandwidths": cv_results["fine results"]["bandwidths"],
                "MSE": cv_results["fine results"]["MSE"],
            }
            print(f"Optimal Bandwidth: {parameters['h']}")

        else:
            parameters = {
                "h": h,
                "bandwidths": None,
                "MSE": None,
            }

        model = LocalPolynomialRegression(X=X, y=y, h=parameters["h"], kernel=self.kernel, gridsize=100)
        prediction_interval = (X.min(), X.max())
        fit_results = model.fit(prediction_interval)
        results = {
            "parameters": parameters,
            "fit": fit_results,
        }
        return results

    def calc_rnd(self):
        """Pipeline that calculates the Risk Neutral Density using Rookley's Method.

        Tools used: Local Polynomial Smoothing and Density Transformation.
        The final result is saved in self.M, self.q_M

        | step 0 : fit iv-smile to iv-over-M option values
        | step 1 : alculate spd for every option-point "Rookley's method"
        | step 2 : Rookley results (points in K-domain) - fit density curve
        | step 3 : transform density POINTS from K- to M-domain
        | step 4 : density points in M-domain - fit density curve
        | (All fits are obtained by local polynomial estimation)
        """
        # step 0: fit iv-smile to iv-over-M option values
        X = np.array(self.data.M)
        y = np.array(self.data.iv)
        results = self.curve_fit(
            X,
            y,
            self.h_m,
        )  # h_m : Union[None, float]
        self.h_m = results["parameters"]["h"]
        self.M_smile = results["fit"]["X"]
        self.smile = results["fit"]["fit"]
        self.first = results["fit"]["first"]
        self.second = results["fit"]["second"]

        # ------------------------------------ B-SPLINE on SMILE, FIRST, SECOND
        print("fit bspline to derivatives for rookley method")
        pars, spline, points = bspline(self.M_smile, self.smile, sections=8, degree=3)
        # derivatives
        first_fct = spline.derivative(1)
        second_fct = spline.derivative(2)

        # step 1: calculate spd for every option-point "Rookley's method"
        print("calculate q_K (Rookley Method)")
        self.data["q"] = self.data.apply(
            lambda row: rookley_method(
                row.M,
                row.S,
                row.K,
                spline(row.M),
                first_fct(row.M),
                second_fct(row.M),
                self.r,
                self.tau,
            ),
            axis=1,
        )

        # step 2: Rookley results (points in K-domain) - fit density curve
        print("locpoly fit to rookley result q_K")
        X = np.array(self.data.K)
        y = np.array(self.data.q)

        results = self.curve_fit(
            X,
            y,
            self.h_k,
        )  # h_k : Union[None, float]
        self.h_k = results["parameters"]["h"]
        self.q_K = results["fit"]["fit"]
        self.K = results["fit"]["X"]

        # step 3: transform density POINTS from K- to M-domain
        print("density transform rookley points q_K to q_M")
        self.data["q_M"] = pointwise_density_trafo_K2M(self.K, self.q_K, self.data.S, self.data.M)

        # step 4: density points in M-domain - fit density curve
        print("locpoly fit to q_M")
        X = np.array(self.data.M)
        y = np.array(self.data.q_M)

        results = self.curve_fit(
            X,
            y,
            self.h_m2,
        )  # h_m : Union[None, float]
        self.h_m2 = results["parameters"]["h"]
        self.q_M = results["fit"]["fit"]
        self.M = results["fit"]["X"]
        return


# ----------------------------------------------------------------------------------------------------------------- PLOT
from matplotlib import pyplot as plt


class Plot:
    def __init__(self, x=0.5):
        self.x = x

    def rookleyMethod(self, RND):
        fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(12, 4))

        # smile
        ax0.scatter(RND.data.M, RND.data.iv, c="r", s=4)
        ax0.plot(RND.M_smile, RND.smile)
        ax0.set_xlabel("Moneyness")
        ax0.set_ylabel("implied volatility")
        ax0.set_xlim(1 - self.x, 1 + self.x)

        # derivatives
        ax1.plot(RND.M_smile, RND.smile)
        ax1.plot(RND.M_smile, RND.first)
        ax1.plot(RND.M_smile, RND.second)
        ax1.set_xlabel("Moneyness")
        ax1.set_xlim(1 - self.x, 1 + self.x)
        ax1.set_ylim(-4)

        # density q_k
        ax2.scatter(RND.data.K, RND.data.q, c="r", s=4)
        ax2.plot(RND.K, RND.q_K)
        ax2.set_xlabel("Strike Price")
        ax2.set_ylabel("risk neutral density")

        # density q_m
        ax3.scatter(RND.data.M, RND.data.q_M, c="r", s=4)
        ax3.plot(RND.M, RND.q_M)
        ax3.set_xlabel("Moneyness")
        ax3.set_ylabel("risk neutral density")
        ax3.set_xlim(1 - self.x, 1 + self.x)

        plt.tight_layout()
        return fig
