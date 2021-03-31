# ---------------------------------------------------------------------------------------------------------------- GARCH
import numpy as np
from arch import arch_model
import time
import copy
import pickle
import os

from .utils.density import density_estimation


class GARCH:
    def __init__(
        self,
        data,
        data_name,
        n,
        garch_data_folder,
        overwrite_garchmodel=False,
        window_length=365,
        z_h=0.1,
    ):
        self.data = data  # timeseries (here: log_returns)
        self.z_h = z_h
        self.data_name = data_name
        self.window_length = window_length
        self.n = n
        self.no_of_paths_to_save = 50
        self.overwrite_garchmodel = overwrite_garchmodel
        self.filename_garchmodel = os.path.join(
            garch_data_folder,
            "GARCH_Model_{}_window_length-{}_n-{}".format(self.data_name, self.window_length, self.n),
        )

    def load(self):
        with open(self.filename_garchmodel, "rb") as f:
            tmp_dict = pickle.load(f)
            f.close()
        self.__dict__.clear()
        self.__dict__.update(tmp_dict)
        return

    def save(self):
        with open(self.filename_garchmodel, "wb") as f:
            pickle.dump(self.__dict__, f, 2)
            f.close()
        return

    def _GARCH_fit(self, data, q=1, p=1):
        model = arch_model(data, q=q, p=p)
        res = model.fit(disp="off")

        pars = (
            res.params["mu"],
            res.params["omega"],
            res.params["alpha[1]"],
            res.params["beta[1]"],
        )
        std_err = (
            res.std_err["mu"],
            res.std_err["omega"],
            res.std_err["alpha[1]"],
            res.std_err["beta[1]"],
        )
        return res, pars, std_err

    def fit_GARCH(self):
        if os.path.exists(self.filename_garchmodel) and (self.overwrite_garchmodel == False):
            print(
                " -------------- use existing GARCH model: ",
                self.filename_garchmodel,
            )
            return
        start = self.window_length + self.n
        end = self.n

        parameters = np.zeros((self.n, 4))
        parameter_bounds = np.zeros((self.n, 4))
        z_process = []
        e_process = []
        sigma2_process = []
        for i in range(0, self.n):
            window = self.data[end - i : start - i]
            data = window - np.mean(window)

            res, parameters[i, :], parameter_bounds[i, :] = self._GARCH_fit(data)

            _, omega, alpha, beta = [
                res.params["mu"],
                res.params["omega"],
                res.params["alpha[1]"],
                res.params["beta[1]"],
            ]
            if i == 0:
                sigma2_tm1 = omega / (1 - alpha - beta)
            else:
                sigma2_tm1 = sigma2_process[-1]

            e_t = data.tolist()[-1]  # last observed log-return, mean adjust.
            e_tm1 = data.tolist()[-2]  # previous observed log-return
            sigma2_t = omega + alpha * e_tm1 ** 2 + beta * sigma2_tm1
            z_t = e_t / np.sqrt(sigma2_t)

            e_process.append(e_t)
            z_process.append(z_t)
            sigma2_process.append(sigma2_t)

        self.parameters = parameters
        self.parameter_bounds = parameter_bounds
        self.e_process = e_process
        self.z_process = z_process
        self.sigma2_process = sigma2_process

        # ------------------------------------------- kernel density estimation
        self.z_values = np.linspace(min(self.z_process), max(self.z_process), 500)
        h_dyn = self.z_h * (np.max(z_process) - np.min(z_process))
        self.z_dens = density_estimation(np.array(z_process), np.array(self.z_values), h=h_dyn).tolist()

        print("------------- save GARCH model: ", self.filename_garchmodel)
        self.save()
        return

    def _GARCH_simulate(self, pars, horizon):
        """stepwise GARCH simulation until burnin + horizon

        Args:
            pars (tuple): (mu, omega, alpha, beta)

        Returns:
            [type]: [description]
        """
        mu, omega, alpha, beta = pars
        burnin = horizon * 2
        sigma2 = [omega / (1 - alpha - beta)]
        e = [self.data.tolist()[-1] - mu]  # last observed log-return mean adj.
        weights = self.z_dens / (np.sum(self.z_dens))

        for _ in range(horizon + burnin):
            sigma2_tp1 = omega + alpha * e[-1] ** 2 + beta * sigma2[-1]
            z_tp1 = np.random.choice(self.z_values, 1, p=weights)[0]
            e_tp1 = z_tp1 * np.sqrt(sigma2_tp1)
            sigma2.append(sigma2_tp1)
            e.append(e_tp1)
        return sigma2[-horizon:], e[-horizon:]

    def _variate_pars(self, pars, bounds):
        new_pars = []
        i = 0
        for par, bound in zip(pars, bounds):
            var = bound ** 2 / self.n
            new_par = np.random.normal(par, var, 1)[0]
            if (new_par <= 0) and (i >= 1):
                print("new_par too small ", new_par)
                new_par = 0.01
            new_pars.append(new_par)
            i += 1
        return new_pars

    def simulate_paths(self, horizon, M, variate=True):
        print(" -------------- simulate paths for: ", self.data_name, horizon, M)
        if os.path.exists(self.filename_garchmodel) and (self.overwrite_garchmodel == False):
            print(
                "    ----------- use existing GARCH model: ",
                self.filename_garchmodel,
            )
        else:
            print("    ----------- fit new GARCH model")
            self.fit_GARCH()

        self.load()
        pars = np.mean(self.parameters, axis=0).tolist()  # mean
        bounds = np.std(self.parameters, axis=0).tolist()  # std of mean par
        print("garch parameters :  ", pars)
        np.random.seed(1)  # for reproducability in _variate_pars()

        new_pars = copy.deepcopy(pars)  # set pars for first round of simulation
        save_sigma = np.zeros((self.no_of_paths_to_save, horizon))
        save_e = np.zeros((self.no_of_paths_to_save, horizon))
        all_summed_returns = np.zeros(M)
        all_tau_mu = np.zeros(M)
        tick = time.time()
        for i in range(M):
            if (i + 1) % (M * 0.1) == 0:
                print("{}/{} - runtime: {} min".format(i + 1, M, round((time.time() - tick) / 60)))
            if ((i + 1) % (M * 0.05) == 0) & variate:
                new_pars = self._variate_pars(pars, bounds)
            sigma2, e = self._GARCH_simulate(new_pars, horizon)
            all_summed_returns[i] = np.sum(e)
            all_tau_mu[i] = horizon * pars[0]
            if i < self.no_of_paths_to_save:
                save_sigma[i, :] = sigma2
                save_e[i, :] = e

        self.save_sigma = save_sigma
        self.save_e = save_e
        self.all_summed_returns = all_summed_returns
        self.all_tau_mu = all_tau_mu
        return all_summed_returns, all_tau_mu


# ----------------------------------------------------------------------------------------------------------- CALCULATOR
import pandas as pd

from .utils.density import density_trafo_K2M


class Calculator(GARCH):
    def __init__(
        self,
        data,
        S0,
        garch_data_folder,
        tau_day,
        date,
        n,
        cutoff=0.5,
        overwrite=True,
        target="price",
        window_length=365,
        h=0.15,
        M=5000,
    ):
        self.data = data
        self.target = target
        self.S0 = S0
        self.tau_day = tau_day
        self.date = date
        self.garch_data_folder = garch_data_folder
        self.cutoff = cutoff
        self.overwrite = overwrite
        self.log_returns = self._get_log_returns()
        self.M = M
        self.h = h
        self.GARCH = GARCH(
            data=self.log_returns,
            window_length=window_length,
            data_name=self.date,
            n=n,
            garch_data_folder=self.garch_data_folder,
            z_h=0.1,
        )

    def _get_log_returns(self):
        n = self.data.shape[0]
        data = self.data.reset_index()
        first = data.loc[: n - 2, self.target].reset_index()
        second = data.loc[1:, self.target].reset_index()
        historical_returns = (second / first)[self.target]
        return np.log(historical_returns) * 100

    def _calculate_path(self, all_summed_returns, all_tau_mu):
        S_T = self.S0 * np.exp(all_summed_returns / 100 + all_tau_mu / 100)
        return S_T

    def get_hd(self, variate=True):
        self.filename = "T-{}_{}_Ksim.csv".format(self.tau_day, self.date)
        # simulate M paths
        if os.path.exists(self.garch_data_folder + self.filename) and (self.overwrite == False):
            print("-------------- use existing Simulations ", self.filename)
            pass
        else:
            print("-------------- create new Simulations")
            all_summed_returns, all_tau_mu = self.GARCH.simulate_paths(self.tau_day, self.M, variate)
            self.ST = self._calculate_path(all_summed_returns, all_tau_mu)
            pd.Series(self.ST).to_csv(os.path.join(self.garch_data_folder, self.filename), index=False)

        self.ST = pd.read_csv(os.path.join(self.garch_data_folder, self.filename))
        S_arr = np.array(self.ST)
        self.K = np.linspace(self.S0 * (1 - self.cutoff), self.S0 * (1 + self.cutoff), 100)
        self.q_K = density_estimation(S_arr, self.K, h=self.S0 * self.h)
        self.M = np.linspace((1 - self.cutoff), (1 + self.cutoff), 100)

        M_arr = np.array(self.S0 / self.ST)
        self.q_M = density_estimation(M_arr, self.M, h=self.h)
        self.M2, self.q_M2 = density_trafo_K2M(self.K, self.q_K, self.S0)