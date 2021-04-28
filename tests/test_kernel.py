import os

from spd_trading import risk_neutral_density as rnd
from spd_trading import historical_density as hd
from spd_trading import kernel as ker


def test_rnd_hd_kernel(rnd_input_data, hd_input_data, evaluation_day, evaluation_tau):

    # ----------------------------------------------------- RISK NEUTRAL DENSITY
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
    RND.get_rnd()

    assert RND.q_M["x"] is not None
    assert RND.q_M["y"] is not None

    evaluation_S0 = hd_input_data.loc[
        hd_input_data.date_str == evaluation_day, "price"
    ].item()  # either take from index or replace by other value

    # ------------------------------------------------------- HISTORICAL DENSITY
    HD = hd.Calculator(
        data=hd_input_data,
        S0=evaluation_S0,
        garch_data_folder=os.path.join(".", "examples", "data"),
        tau_day=evaluation_tau,
        date=evaluation_day,
        n_fits=30,
        simulations=10,
        overwrite_model=True,
        overwrite_simulations=True,
    )
    HD.get_hd(variate_GARCH_parameters=True)

    assert HD.q_M["x"] is not None
    assert HD.q_M["y"] is not None

    # ------------------------------------------------------------------- KERNEL
    Kernel = ker.Calculator(
        tau_day=evaluation_tau, date=evaluation_day, RND=RND, HD=HD, similarity_threshold=0.15, cut_tail_percent=0.02
    )
    Kernel.calc_kernel()
    Kernel.calc_trading_intervals()

    assert Kernel.kernel["x"] is not None
    assert Kernel.kernel["y"] is not None
    assert Kernel.trading_intervals["buy"] is not None
    assert Kernel.trading_intervals["sell"] is not None
