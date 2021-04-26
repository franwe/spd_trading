from matplotlib import pyplot as plt
import pandas as pd
from itertools import groupby
from operator import itemgetter

from .utils.smoothing import interpolate_to_same_interval, remove_tails


class Calculator:
    """The Calculator Class for the Pricing Kernel.

    Stores all relevant parameters for traceability and reproducability.
    :math:`K=\\frac{rnd}{hd}`

    Args:
        tau_day (int): Time to maturity in days (tau * 365)
        date (str): Date of the option table.
        RND (spd_trading.risk_neutral_density.Calculator): Trained instance of class ``spd_trading.risk_neutral_density.Calculator``.
        HD (spd_trading.historical_density.Calculator): Trained instance of class ``spd_trading.historical_density.Calculator``.
        similarity_threshold (float, optional): Creates similarity area around 1. If densities are too similar, the Kernel is K=1.
            Defaults to 0.15.
        cut_tail_percent (float, optional): Percent threshold of when to cut off tails. Defaults to 0.02.

    Attributes:
        M (np.array): Moneyness
        rnd_curve (np.array): Risk Neutral Density interpolated to M-Values
        hd_curve (np.array): Historical Density interpolated to M-Values
        kernel (np.array): Kernel
        sell_intervals (list of tuples): M-intervals for which to sell options according to kernel
        buy_intervals (list of tuples): M-intervals for which to buy options according to kernel
    """

    def __init__(self, tau_day, date, RND, HD, similarity_threshold=0.15, cut_tail_percent=0.02):
        self.tau_day = tau_day
        self.date = date
        self.RND = RND
        self.HD = HD
        self.similarity_threshold = similarity_threshold
        self.cut_tail_percent = cut_tail_percent

        # parameters that are created during run
        self.M = None
        self.rnd_curve = None
        self.hd_curve = None
        self.kernel = None
        self.sell_intervals = None
        self.buy_intervals = None

    def calc_kernel(self):
        """Calculates the Kernel.

        | First, interpolates rnd and hd to the same M-values. Then removes tiny tails.
        | Finally calculates the Kernel: :math:`K = \\frac{rnd}{hd}`.

        """
        self.hd_curve, self.rnd_curve, self.M = interpolate_to_same_interval(
            self.HD.M,
            self.HD.q_M,
            self.RND.M,
            self.RND.q_M,
            interval=[self.RND.data.M.min() * 0.99, self.RND.data.M.max() * 1.01],
        )
        self.rnd_curve, self.hd_curve, self.M = remove_tails(
            self.rnd_curve, self.hd_curve, self.M, self.cut_tail_percent
        )
        self.hd_curve, self.rnd_curve, self.M = remove_tails(
            self.hd_curve, self.rnd_curve, self.M, self.cut_tail_percent
        )

        self.kernel = self.rnd_curve / self.hd_curve

    def _M_bounds_from_list(self, lst, df):
        groups = [[i for i, _ in group] for key, group in groupby(enumerate(lst), key=itemgetter(1)) if key]
        M_bounds = []
        for group in groups:
            M_bounds.append((df.M[group[0]], df.M[group[-1]]))
        return M_bounds

    def calc_trading_intervals(self):
        """Determines the Moneyness intervalls for which to buy and sell options according to the kernel."""
        df = pd.DataFrame({"M": self.M, "K": self.kernel})
        df["buy"] = df.K < (1 - self.similarity_threshold)
        df["sell"] = df.K > (1 + self.similarity_threshold)

        self.sell_intervals = self._M_bounds_from_list(df.sell.tolist(), df)
        self.buy_intervals = self._M_bounds_from_list(df.buy.tolist(), df)


class Plot:
    """The Plotting class for Kernel.

    Args:
        x (float, optional): The Moneyness interval for the plots. :math:`M = [1-x, 1+x]`. Defaults to 0.5.
    """

    def __init__(self, x=0.5):
        self.x = x

    def kernelplot(self, Kernel):
        """Visualization of computation of the Kernel and its derived trading intervals.

        | Left: Risk Neutral Density (red line) and Historical Density (blue line) on evaluation day, option data is
            represented as dots (red: calls, blue: puts).
        | Middle: Construction of Trading Strategy based on the Pricing Kernel. The Pricing Kernel (black line)
            indicates whether options should be bought (red interval) or sold (blue interval).
            In grey the similarity area around :math:`K=1`.

        Args:
            Kernel (spd_trading.kernel.Calculator): Instance of class ``spd_trading.kernel.Calculator``.

        Returns:
            Figure: Matplotlib figure.
        """
        day = Kernel.date
        tau_day = Kernel.tau_day
        call_mask = Kernel.RND.data.option == "C"
        Kernel.RND.data["color"] = "blue"  # blue - put
        Kernel.RND.data.loc[call_mask, "color"] = "red"  # red - call

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        # ---------------------------------------------------------------------------------------- Moneyness - Moneyness
        ax = axes[0]
        ax.scatter(Kernel.RND.data.M, Kernel.RND.data.q_M, 5, c=Kernel.RND.data.color)
        ax.plot(Kernel.M, Kernel.rnd_curve, "-", c="r")
        ax.plot(Kernel.M, Kernel.hd_curve, "-", c="b")

        ax.text(
            0.99,
            0.99,
            str(day) + "\n" + r"$\tau$ = " + str(tau_day),
            horizontalalignment="right",
            verticalalignment="top",
            transform=ax.transAxes,
        )
        ax.set_xlim((1 - self.x), (1 + self.x))
        # if y_lim:
        #     ax.set_ylim(0, y_lim["M"])
        ax.set_ylim(0)
        ax.vlines(1, 0, Kernel.RND.data.q_M.max())
        ax.set_xlabel("Moneyness M")

        # -------------------------------------------------------------------------------------- Kernel K = q/p = rnd/hd
        ax = axes[1]
        ax.plot(Kernel.M, Kernel.kernel, "-", c="k")
        ax.axhspan(1 - Kernel.similarity_threshold, 1 + Kernel.similarity_threshold, color="grey", alpha=0.1)
        for interval in Kernel.buy_intervals:
            ax.axvspan(interval[0], interval[1], color="r", alpha=0.1)
        for interval in Kernel.sell_intervals:
            ax.axvspan(interval[0], interval[1], color="b", alpha=0.1)

        ax.text(
            0.99,
            0.99,
            str(day) + "\n" + r"$\tau$ = " + str(tau_day),
            horizontalalignment="right",
            verticalalignment="top",
            transform=ax.transAxes,
        )
        ax.set_xlim((1 - self.x), (1 + self.x))
        ax.set_ylim(0, 2)
        ax.set_ylabel("K = rnd / hd")
        ax.set_xlabel("Moneyness")
        plt.tight_layout()
        return fig
