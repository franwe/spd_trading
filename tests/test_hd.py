import os
from spd_trading import historical_density as hd


def test_get_hd(hd_input_data, evaluation_day, evaluation_tau):
    evaluation_S0 = hd_input_data.loc[
        hd_input_data.date_str == evaluation_day, "price"
    ].item()  # either take from index or replace by other value

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

    assert HD.q_M is not None
    assert HD.M is not None
