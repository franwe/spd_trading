import os
import pandas as pd
from matplotlib import pyplot as plt
from spd_trading import historical_density as hd

import logging

logging.basicConfig(level=logging.INFO)
HD_TESTDATA_FILENAME = os.path.join(".", "examples", "data", "hd_input_data.csv")

hd_input_data = pd.read_csv(HD_TESTDATA_FILENAME)
evaluation_day = "2020-03-05"  # known from data processing
evaluation_tau = 8  # known from data processing
evaluation_S0 = hd_input_data.loc[
    hd_input_data.date_str == evaluation_day, "price"
].item()  # either take from index or replace by other value

HD = hd.Calculator(
    data=hd_input_data,
    S0=evaluation_S0,
    garch_data_folder=os.path.join(".", "examples", "data"),
    tau_day=evaluation_tau,
    date=evaluation_day,
    n_fits=40,
    simulations=50,
    overwrite_model=False,
    overwrite_simulations=False,
)
HD.get_hd(variate_GARCH_parameters=True)

HdPlot = hd.Plot(x=0.35)  # Rookley Method algorithm plot
fig_denstiy = HdPlot.density(HD)

plt.show()
