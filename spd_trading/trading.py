from matplotlib import pyplot as plt

from .utils.density import hd_rnd_domain


class Plot:
    def __init__(self, x=0.5):
        self.x = x

    def strategy(self, RND, HD):
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


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


import os
from os.path import join
import numpy as np
from datetime import datetime, timedelta
import pickle
from itertools import groupby
from operator import itemgetter
import pandas as pd

from .utils.connect_db import connect_db, get_as_df

cwd = os.getcwd() + os.sep
source_data = join(cwd, "data", "00-raw") + os.sep
save_data = join(cwd, "data", "03-1_trades") + os.sep
save_plots = join(cwd, "plots") + os.sep
garch_data = join(cwd, "data", "02-2_hd_GARCH") + os.sep


def _M_bounds_from_list(lst, df):
    groups = [
        [i for i, _ in group]
        for key, group in groupby(enumerate(lst), key=itemgetter(1))
        if key
    ]
    M_bounds = []
    for group in groups:
        M_bounds.append((df.M[group[0]], df.M[group[-1]]))
    return M_bounds


def get_buy_sell_bounds(rnd, hd, M, K_bound=0):
    K = rnd / hd
    df = pd.DataFrame({"M": M, "K": K})
    df["buy"] = df.K < (1 - K_bound)
    df["sell"] = df.K > (1 + K_bound)

    M_bounds_sell = _M_bounds_from_list(df.sell.tolist(), df)
    M_bounds_buy = _M_bounds_from_list(df.buy.tolist(), df)
    return M_bounds_sell, M_bounds_buy


def _get_payoff(K, ST, option):
    if option == "C":
        return max(ST - K, 0)
    elif option == "P":
        return max(K - ST, 0)


def execute_options(data, day, tau_day):
    eval_day = datetime.strptime(day, "%Y-%m-%d") + timedelta(days=tau_day)
    db = connect_db()

    coll = db["BTCUSD_deribit"]
    query = {"date_str": str(eval_day.date())}
    ST = get_as_df(coll, query)["price"].iloc[0]

    query = {"date_str": day}
    S0 = get_as_df(coll, query)["price"].iloc[0]

    data["ST"] = ST
    data["opt_payoff"] = data.apply(
        lambda row: _get_payoff(row.K, ST, row.option), axis=1
    )
    print("--- S0: {} --- ST: {} --- M: {}".format(S0, ST, S0 / ST))
    return data


def _calculate_fee(P, S, max_fee_BTC=0.0004, max_fee_pct=0.2):
    option_bound = max_fee_pct * P
    underlying_bound = max_fee_BTC * S
    fee = min(underlying_bound, option_bound)
    return fee


def _payoff_call(action, K, price, ST):
    if action == "buy":
        return np.maximum(np.zeros(len(ST)), (ST - K)) - price
    elif action == "sell":
        return -1 * (np.maximum(np.zeros(len(ST)), (ST - K)) - price)


def _payoff_put(action, K, price, ST):
    if action == "buy":
        return np.maximum(np.zeros(len(ST)), (K - ST)) - price
    elif action == "sell":
        return -1 * (np.maximum(np.zeros(len(ST)), (K - ST)) - price)


def _get_option_payoff(option, action, K, price, ST):
    if option == "C":
        return _payoff_call(action, K, price, ST)
    elif option == "P":
        return _payoff_put(action, K, price, ST)


def _trading_payoffs(data):
    buy_mask = data.action == "buy"

    data["trading_fee"] = data.apply(
        lambda row: _calculate_fee(row.P, row.S, max_fee_BTC=0.0004), axis=1
    )
    data["t0_payoff"] = data["P"]
    data.loc[buy_mask, "t0_payoff"] = -1 * data.loc[buy_mask, "P"]
    data["t0_payoff"] = data["t0_payoff"] - data["trading_fee"]

    data["T_payoff"] = -1 * data["opt_payoff"]
    data.loc[buy_mask, "T_payoff"] = +1 * data.loc[buy_mask, "opt_payoff"]
    data["delivery_fee"] = data.apply(
        lambda row: _calculate_fee(row.T_payoff, row.S, max_fee_BTC=0.0002),
        axis=1,
    )
    data.loc[~buy_mask, "delivery_fee"] = 0  # only applies to TAKER ORDERS
    data["T_payoff"] = data["T_payoff"] - data["delivery_fee"]

    data["total"] = data.t0_payoff + data.T_payoff

    S0 = data.S.mean()
    ST = data.ST.iloc[0]
    lower_S = min(S0 * 0.5, ST * 0.9)
    upper_S = max(S0 * 1.5, ST * 1.1)

    S = np.linspace(lower_S, upper_S, 200)

    data["payoffs"] = data.apply(
        lambda row: _get_option_payoff(
            row.option, row.action, row.K, row.P, S
        ),
        axis=1,
    )
    return data


def add_results_to_table(
    df_results, results, trading_day, trading_tau, deviates_from_one_ratio
):
    if len(results) == 0:
        entry = {
            "date": trading_day,
            "tau_day": trading_tau,
            "t0_payoff": 0,
            "T_payoff": 0,
            "total": 0,
            "trade": "-",
            "kernel": deviates_from_one_ratio,
        }
        df_results = df_results.append(entry, ignore_index=True)

    else:
        for key in results:
            df_trades = results[key]

            entry = {
                "date": trading_day,
                "tau_day": trading_tau,
                "t0_payoff": df_trades.t0_payoff.sum(),
                "T_payoff": df_trades.T_payoff.sum(),
                "total": df_trades.total.sum(),
                "trade": key,
                "kernel": deviates_from_one_ratio,
            }
            df_results = df_results.append(entry, ignore_index=True)
    return df_results


# ---------------------------------------------------------- TRADING STRATEGIES


def options_in_interval(
    option, moneyness, action, df, left, right, near_bound
):
    if (moneyness == "ATM") & (option == "C"):
        mon_left = 1 - near_bound
        mon_right = 1 + near_bound
        which_element = 0
    elif (moneyness == "ATM") & (option == "P"):
        mon_left = 1 - near_bound
        mon_right = 1 + near_bound
        which_element = -1

    elif (moneyness == "OTM") & (option == "C"):
        mon_left = 0
        mon_right = 1 - near_bound
        which_element = -1
    elif (moneyness == "OTM") & (option == "P"):
        mon_left = 1 + near_bound
        mon_right = 10
        which_element = 0

    candidates = df[
        (df.M > left)
        & (df.M < right)  # option interval
        & (df.M > mon_left)
        & (df.M < mon_right)
        & (df.option == option)
    ]
    candidate = candidates.iloc[which_element]
    candidate["action"] = action
    return candidate


def K1(df_tau, M_bounds_buy, M_bounds_sell, near_bound):
    df = df_tau[
        [
            "M",
            "option",
            "P",
            "K",
            "S",
            "iv",
            "P_BTC",
            "color",
            "opt_payoff",
            "ST",
        ]
    ].sort_values(by="M")
    otm_call, otm_call_action = pd.Series(), "buy"
    otm_put, otm_put_action = pd.Series(), "buy"
    atm_call, atm_call_action = pd.Series(), "sell"
    atm_put, atm_put_action = pd.Series(), "sell"

    for interval in M_bounds_buy:
        left, right = interval
        try:
            otm_call = options_in_interval(
                "C", "OTM", otm_call_action, df, left, right, near_bound
            )
        except IndexError:
            pass

    for interval in M_bounds_sell:
        left, right = interval
        try:
            atm_call = options_in_interval(
                "C", "ATM", atm_call_action, df, left, right, near_bound
            )
        except IndexError:
            pass

    for interval in M_bounds_sell:
        left, right = interval
        try:
            atm_put = options_in_interval(
                "P", "ATM", atm_put_action, df, left, right, near_bound
            )
        except IndexError:
            pass

    for interval in M_bounds_buy:
        left, right = interval
        try:
            otm_put = options_in_interval(
                "P", "OTM", otm_put_action, df, left, right, near_bound
            )
        except IndexError:
            pass

    if any([otm_call.empty, atm_call.empty, atm_put.empty, otm_put.empty]):
        pass
    else:
        df_trades = pd.DataFrame([otm_call, atm_call, atm_put, otm_put])
        return _trading_payoffs(df_trades)


def K2(df_tau, M_bounds_buy, M_bounds_sell, near_bound):
    df = df_tau[
        [
            "M",
            "option",
            "P",
            "K",
            "S",
            "iv",
            "P_BTC",
            "color",
            "opt_payoff",
            "ST",
        ]
    ].sort_values(by="M")
    otm_call, otm_call_action = pd.Series(), "sell"
    otm_put, otm_put_action = pd.Series(), "sell"
    atm_call, atm_call_action = pd.Series(), "buy"
    atm_put, atm_put_action = pd.Series(), "buy"

    for interval in M_bounds_sell:
        left, right = interval
        try:
            otm_call = options_in_interval(
                "C", "OTM", otm_call_action, df, left, right, near_bound
            )
        except IndexError:
            pass

    for interval in M_bounds_buy:
        left, right = interval
        try:
            atm_call = options_in_interval(
                "C", "ATM", atm_call_action, df, left, right, near_bound
            )
        except IndexError:
            pass

    for interval in M_bounds_buy:
        left, right = interval
        try:
            atm_put = options_in_interval(
                "P", "ATM", atm_put_action, df, left, right, near_bound
            )
        except IndexError:
            pass

    for interval in M_bounds_sell:
        left, right = interval
        try:
            otm_put = options_in_interval(
                "P", "OTM", otm_put_action, df, left, right, near_bound
            )
        except IndexError:
            pass

    if any([otm_call.empty, atm_call.empty, atm_put.empty, otm_put.empty]):
        pass
    else:
        df_trades = pd.DataFrame([otm_call, atm_call, atm_put, otm_put])
        return _trading_payoffs(df_trades)


def S1(df_tau, M_bounds_buy, M_bounds_sell, near_bound):
    df = df_tau[
        [
            "M",
            "option",
            "P",
            "K",
            "S",
            "iv",
            "P_BTC",
            "color",
            "opt_payoff",
            "ST",
        ]
    ].sort_values(by="M")
    otm_call, otm_call_action = pd.Series(), "buy"
    otm_put, otm_put_action = pd.Series(), "sell"

    for interval in M_bounds_buy:
        left, right = interval
        try:
            otm_call = options_in_interval(
                "C", "OTM", otm_call_action, df, left, right, near_bound
            )
        except IndexError:
            pass

    for interval in M_bounds_sell:
        left, right = interval
        try:
            otm_put = options_in_interval(
                "P", "OTM", otm_put_action, df, left, right, near_bound
            )
        except IndexError:
            pass

    if any([otm_call.empty, otm_put.empty]):
        pass
    elif (len(M_bounds_buy) > 1) or (len(M_bounds_sell) > 1):
        print(" ---- too many intervals")
        pass
    else:
        df_trades = pd.DataFrame([otm_call, otm_put])
        return _trading_payoffs(df_trades)


def S2(df_tau, M_bounds_buy, M_bounds_sell, near_bound):
    df = df_tau[
        [
            "M",
            "option",
            "P",
            "K",
            "S",
            "iv",
            "P_BTC",
            "color",
            "opt_payoff",
            "ST",
        ]
    ].sort_values(by="M")
    otm_call, otm_call_action = pd.Series(), "sell"
    otm_put, otm_put_action = pd.Series(), "buy"

    for interval in M_bounds_sell:
        left, right = interval
        try:
            otm_call = options_in_interval(
                "C", "OTM", otm_call_action, df, left, right, near_bound
            )
        except IndexError:
            pass

    for interval in M_bounds_buy:
        left, right = interval
        try:
            otm_put = options_in_interval(
                "P", "OTM", otm_put_action, df, left, right, near_bound
            )
        except IndexError:
            pass

    if any([otm_call.empty, otm_put.empty]):
        pass
    elif (len(M_bounds_buy) > 1) or (len(M_bounds_sell) > 1):
        print(" ---- too many intervals")
        pass
    else:
        df_trades = pd.DataFrame([otm_call, otm_put])
        return _trading_payoffs(df_trades)


# ------------------------------------------------------- DATASTUFF - LOAD SAVE
def load_rnd_hd_from_pickle(data_path, day, tau_day):
    with open(data_path + "T-{}_{}.pkl".format(tau_day, day), "rb") as handle:
        data = pickle.load(handle)

    RND = data["RND"]
    hd, rnd, M = data["hd"], data["rnd"], data["M"]
    kernel = data["kernel"]
    return RND, hd, rnd, kernel, M


def save_trades_to_pickle(
    data_path,
    trading_day,
    trading_tau,
    rnd,
    rnd_points,
    hd,
    kernel,
    M,
    K_bound,
    M_bounds_buy,
    M_bounds_sell,
    df_all,
    results,
):
    if len(results) == 0:
        content = {
            "day": trading_day,
            "tau_day": trading_tau,
            "trade": "-",
            "rnd": rnd,
            "rnd_points": rnd_points,  # M, q, color
            "hd": hd,
            "kernel": kernel,
            "M": M,
            "K_bound": K_bound,
            "M_bounds_buy": M_bounds_buy,
            "M_bounds_sell": M_bounds_sell,
            "df_all": df_all,
            "df_trades": None,
        }

        filename = "T-{}_{}_{}.pkl".format(trading_tau, trading_day, "-")
        with open(data_path + filename, "wb") as handle:
            pickle.dump(content, handle)
        return
    else:
        for trade in results:
            df_trades = results[trade]
            content = {
                "day": trading_day,
                "tau_day": trading_tau,
                "trade": trade,
                "rnd": rnd,
                "rnd_points": rnd_points,  # M, q, color
                "hd": hd,
                "kernel": kernel,
                "M": M,
                "K_bound": K_bound,
                "M_bounds_buy": M_bounds_buy,
                "M_bounds_sell": M_bounds_sell,
                "df_all": df_all,
                "df_trades": df_trades,
            }

            filename = "T-{}_{}_{}.pkl".format(trading_tau, trading_day, trade)
            with open(data_path + filename, "wb") as handle:
                pickle.dump(content, handle)
    return


def load_trades_from_pickle(data_path, day, tau_day, trade):
    with open(
        data_path + "T-{}_{}_{}.pkl".format(tau_day, day, trade), "rb"
    ) as handle:
        data = pickle.load(handle)
    return data
