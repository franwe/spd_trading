import os
import pandas as pd
from matplotlib import pyplot as plt

from spd_trading import risk_neutral_density as rnd

# ---------------------------------------------------------------------------------------------------------------- SETUP
RND_TESTDATA_FILENAME = os.path.join(".", "examples", "data", "rnd_input_data.csv")

rnd_input_data = pd.read_csv(RND_TESTDATA_FILENAME)
evaluation_day = "2020-03-05"  # known from data processing
evaluation_tau = 8  # known from data processing

# ----------------------------------------------------------------------------------------------------------------- MAIN
RND = rnd.Calculator(
    data=rnd_input_data,
    tau_day=evaluation_tau,
    date=evaluation_day,
    sampling="slicing",
    n_sections=15,
    loss="MSE",
    kernel="gaussian",
    h_m=0.088,  # set None if unknown, then `bandwidth_cv`
    h_k=215.068,  # set None if unknown, then `bandwidth_cv`
    h_m2=0.036,  # set None if unknown, then `bandwidth_cv`
)
RND.calc_rnd()

RndPlot = rnd.Plot()  # Rookley Method algorithm plot
fig_method = RndPlot.rookleyMethod(RND)

plt.show()
