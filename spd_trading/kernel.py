from matplotlib import pyplot as plt

from .utils.density import hd_rnd_domain


class Plot:
    def __init__(self, x=0.5):
        self.x = x

    def kernelplot(self, RND, HD):
        day = RND.date
        tau_day = RND.tau_day
        call_mask = RND.data.option == "C"
        RND.data["color"] = "blue"  # blue - put
        RND.data.loc[call_mask, "color"] = "red"  # red - call

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        # ----------------------------------------------- Moneyness - Moneyness
        ax = axes[0]
        ax.scatter(RND.data.M, RND.data.q_M, 5, c=RND.data.color)
        ax.plot(RND.M, RND.q_M, "-", c="r")
        ax.plot(HD.M, HD.q_M, "-", c="b")

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
        ax.vlines(1, 0, RND.data.q_M.max())
        ax.set_xlabel("Moneyness M")

        # --------------------------------------------- Kernel K = q/p = rnd/hd
        hd_curve, rnd_curve, M = hd_rnd_domain(
            HD,
            RND,
            interval=[RND.data.M.min() * 0.99, RND.data.M.max() * 1.01],
        )
        K = rnd_curve / hd_curve
        ax = axes[1]
        ax.plot(M, K, "-", c="k")
        ax.axhspan(0.7, 1.3, color="grey", alpha=0.5)
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
