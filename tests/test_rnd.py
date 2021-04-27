from spd_trading import risk_neutral_density as rnd


def test_get_rnd(rnd_input_data, evaluation_day, evaluation_tau):
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

    assert RND.q_M is not None
    assert RND.M is not None
