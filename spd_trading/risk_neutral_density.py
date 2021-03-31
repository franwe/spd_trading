import numpy as np
from scipy.stats import norm
from statsmodels.nonparametric.bandwidths import bw_silverman

from .utils.smoothing import (
    create_fit,
    bandwidth_cv,
    bspline,
    local_polynomial_estimation,
)
from .utils.density import pointwise_density_trafo_K2M


# ----------------------------------------------------------------------------------------------------------- CALCULATOR


def create_bandwidth_range(X, bins_max=30, num=10):
    bw_silver = bw_silverman(X)
    if bw_silver > 10:
        lower_bound = max(0.5 * bw_silver, 100)
    else:
        lower_bound = max(0.5 * bw_silver, 0.03)
    x_bandwidth = np.linspace(lower_bound, 7 * bw_silver, num)
    print("------ Silverman: ", bw_silver)
    return x_bandwidth, bw_silver, lower_bound


def rookley_method(M, S, K, o, o1, o2, r, tau):
    """from Applied Quant. Finance - Chapter 8"""
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
    def __init__(self, data, tau_day, date, h_m=None, h_m2=None, h_k=None, r=0):
        self.data = data
        self.tau_day = tau_day
        self.date = date
        self.h_m = h_m
        self.h_m2 = h_m2
        self.h_k = h_k
        self.r = r

        self.tau = self.data.tau.iloc[0]
        self.K = None
        self.M = None
        self.q_M = None
        self.q_K = None
        self.M_smile = None
        self.smile = None
        self.first = None
        self.second = None

    def bandwidth_and_fit(self, X, y):
        x_bandwidth, bw_silver, lower_bound = create_bandwidth_range(X)
        cv_results = bandwidth_cv(X, y, x_bandwidth, smoothing=local_polynomial_estimation)
        h = cv_results["fine results"]["h"]

        X_domain, fit, first, second, h = create_fit(X, y, h)
        results = {
            "parameters": {
                "h": cv_results["fine results"]["h"],
                "bandwidths": cv_results["fine results"]["bandwidths"],
                "MSE": cv_results["fine results"]["MSE"],
            },
            "fit": {"y": fit, "first": first, "second": second, "X": X_domain},
        }
        return results

    def curve_fit(self, X, y, h=None):
        if h is None:
            x_bandwidth, bw_silver, lower_bound = create_bandwidth_range(X)
            cv_results = bandwidth_cv(X, y, x_bandwidth, smoothing=local_polynomial_estimation)

            parameters = {
                "h": cv_results["fine results"]["h"],
                "bandwidths": cv_results["fine results"]["bandwidths"],
                "MSE": cv_results["fine results"]["MSE"],
            }

        else:
            parameters = {
                "h": h,
                "bandwidths": None,
                "MSE": None,
            }

        X_domain, fit, first, second, h = create_fit(X, y, parameters["h"])
        results = {
            "parameters": parameters,
            "fit": {
                "y": fit,
                "first": first,
                "second": second,
                "X": X_domain,
            },
        }
        return results

    def calc_rnd(self):
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
        self.smile = results["fit"]["y"]
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
        self.q_K = results["fit"]["y"]
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
        self.q_M = results["fit"]["y"]
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
