from matplotlib import pyplot as plt
import pandas as pd
from itertools import groupby
from operator import itemgetter

from .utils.smoothing import interpolate_to_same_interval, remove_tails


class Calculator:
    def __init__(self, tau_day, date, RND, HD, similarity_threshold=0.15, cut_tail_percent=0.02):
        self.tau_day = tau_day
        self.date = date
        self.RND = RND
        self.HD = HD
        self.similarity_threshold = similarity_threshold
        self.cut_tail_percent = cut_tail_percent

        # parameters that are created during run
        self.M = None
        self.rnd = None
        self.hd = None
        self.kernel = None
        self.sell_intervals = None
        self.buy_intervals = None

    def calc_kernel(self):
        """Calculates the Kernel.

        | First, interpolates rnd and hd to the same M-values. Then removes tiny tails.
        | Finally calculates the Kernel: :math:`K = \\frac{rnd}{hd}`.

        """
        self.hd, self.rnd, self.M = interpolate_to_same_interval(
            self.HD.M,
            self.HD.q_M,
            self.RND.M,
            self.RND.q_M,
            interval=[self.RND.data.M.min() * 0.99, self.RND.data.M.max() * 1.01],
        )
        self.rnd, self.hd, self.M = remove_tails(self.rnd, self.hd, self.M, self.cut_tail_percent)
        self.hd, self.rnd, self.M = remove_tails(self.hd, self.rnd, self.M, self.cut_tail_percent)

        self.kernel = self.rnd / self.hd

    def _M_bounds_from_list(self, lst, df):
        groups = [[i for i, _ in group] for key, group in groupby(enumerate(lst), key=itemgetter(1)) if key]
        M_bounds = []
        for group in groups:
            M_bounds.append((df.M[group[0]], df.M[group[-1]]))
        return M_bounds

    def calc_trading_intervals(self):
        df = pd.DataFrame({"M": self.M, "K": self.kernel})
        df["buy"] = df.K < (1 - self.similarity_threshold)
        df["sell"] = df.K > (1 + self.similarity_threshold)

        self.sell_intervals = self._M_bounds_from_list(df.sell.tolist(), df)
        self.buy_intervals = self._M_bounds_from_list(df.buy.tolist(), df)


class Plot:
    def __init__(self, x=0.5):
        self.x = x

    def kernelplot(self, Kernel):
        day = Kernel.date
        tau_day = Kernel.tau_day
        call_mask = Kernel.RND.data.option == "C"
        Kernel.RND.data["color"] = "blue"  # blue - put
        Kernel.RND.data.loc[call_mask, "color"] = "red"  # red - call

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        # ---------------------------------------------------------------------------------------- Moneyness - Moneyness
        ax = axes[0]
        ax.scatter(Kernel.RND.data.M, Kernel.RND.data.q_M, 5, c=Kernel.RND.data.color)
        ax.plot(Kernel.M, Kernel.rnd, "-", c="r")
        ax.plot(Kernel.M, Kernel.hd, "-", c="b")

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
