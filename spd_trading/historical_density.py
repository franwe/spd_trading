# ---------------------------------------------------------------------------------------------------------------- GARCH
import numpy as np
from arch import arch_model
import time
import copy
import pickle
import os
import logging

from .utils.density import density_estimation


class GARCH:
    """The GARCH model class, which fits a model to the historical data and simulates sample paths based on that model

    Args:
        data (np.array): The timeseries that should be fitted (usually log returns)
        data_name (str): Name of the dataset, will be used in filename.
        n_fits (int): How many sliding windows the data is devided into.
        garch_data_folder (str): Path to folder where to save/load GARCH model.
        overwrite_model (bool, optional): Whether to overwrite the GARCH model. Defaults to True.
        window_length (int, optional): Length of each sliding window. Defaults to 365.
        z_h (float, optional): Bandwidth *factor* for Kernel Density Estimation. Defaults to 0.1.

    Attributes:
        parameters (np.array): The parameters :math:`\\Theta = (\\omega, \\alpha, \\beta)` of GARCH model.
        z_dens (np.array): The distribution of innovations :math:`\\mathcal{Z} = \\left\\{z_0, z_1, ...., z_T \\right\\}`
        simulated_log_returns (np.array): simulated log returns at horizon (also: maturity) :math:`\\tau`
        simulated_tau_mu (np.array): product of :math:`\\tau \\cdot \\mu` for each simulation
    """

    def __init__(
        self,
        data,
        data_name,
        n_fits,
        garch_data_folder,
        overwrite_model=True,
        window_length=365,
        z_h=0.1,
    ):
        self.data = data  # timeseries (here: log_returns)
        self.z_h = z_h
        self.data_name = data_name
        self.window_length = window_length
        self.n_fits = n_fits
        self.overwrite_model = overwrite_model
        self.filename_model = os.path.join(
            garch_data_folder,
            "GARCH_Model_{}_window_length-{}_n-{}".format(self.data_name, self.window_length, self.n_fits),
        )

        # parameters that are created during run
        self.parameters = None
        self.e_process = None
        self.z_process = None
        self.sigma2_process = None
        self.z_dens = None
        self.simulated_log_returns = None
        self.simulated_tau_mu = None

    def _load(self):
        """Loads a GARCH model."""
        with open(self.filename_model, "rb") as f:
            tmp_dict = pickle.load(f)
            f.close()
        self.__dict__.clear()
        self.__dict__.update(tmp_dict)
        return

    def _save(self):
        """Saves a GARCH model."""
        with open(self.filename_model, "wb") as f:
            pickle.dump(self.__dict__, f, 2)
            f.close()
        return

    def fit(self):
        """Fits a GARCH(1,1) model to the data. This

        For *n_fits* sliding windows, the parameters :math:`\\Theta = (\\omega, \\alpha, \\beta)` and the
        distribution of innovations :math:`\\mathcal{Z} = \\left\\{z_0, z_1, ...., z_T \\right\\}` are estimated.
        The results are saved in *self*.
        """
        if os.path.exists(self.filename_model) and (self.overwrite_model == False):
            logging.info(f" -------------- use existing GARCH model: {self.filename_model}")
            return
        start = self.window_length + self.n_fits
        end = self.n_fits

        parameters = np.zeros((self.n_fits, 4))
        z_process = []
        e_process = []
        sigma2_process = []
        for i in range(0, self.n_fits):
            window = self.data[end - i : start - i]
            data = window - np.mean(window)

            model = arch_model(data, q=1, p=1)
            GARCH_fit = model.fit(disp="off")

            mu, omega, alpha, beta = [
                GARCH_fit.params["mu"],
                GARCH_fit.params["omega"],
                GARCH_fit.params["alpha[1]"],
                GARCH_fit.params["beta[1]"],
            ]
            parameters[i, :] = [mu, omega, alpha, beta]

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
        self.e_process = e_process
        self.z_process = z_process
        self.sigma2_process = sigma2_process

        # ------------------------------------------- kernel density estimation

        z_dens_x = np.linspace(min(self.z_process), max(self.z_process), 500)
        h_dyn = self.z_h * (np.max(z_process) - np.min(z_process))
        z_dens_y = density_estimation(np.array(z_process), np.array(z_dens_x), h=h_dyn).tolist()
        self.z_dens = {"x": z_dens_x, "y": z_dens_y}

        logging.info(f"------------- save GARCH model: {self.filename_model}")
        self._save()
        return

    def _GARCH_simulate(self, pars, horizon):
        """stepwise GARCH simulation until burnin + horizon

        Args:
            pars (tuple): (mu, omega, alpha, beta)

        Returns:
            tuple of lists: simulated sigma2- and e-process
        """
        mu, omega, alpha, beta = pars
        burnin = horizon * 2
        sigma2 = [omega / (1 - alpha - beta)]
        e = [self.data.tolist()[-1] - mu]  # last observed log-return mean adj.
        weights = self.z_dens["y"] / (np.sum(self.z_dens["y"]))

        for _ in range(horizon + burnin):
            sigma2_tp1 = omega + alpha * e[-1] ** 2 + beta * sigma2[-1]
            z_tp1 = np.random.choice(self.z_dens["x"], 1, p=weights)[0]
            e_tp1 = z_tp1 * np.sqrt(sigma2_tp1)
            sigma2.append(sigma2_tp1)
            e.append(e_tp1)
        return sigma2[-horizon:], e[-horizon:]

    def _variate_pars(self, pars, bounds):
        """Variation of parameters for fit of GARCH model.

        The GARCH fit (pars) was obtained from *n_fits* moving windows and therefore varies itself.

        Args:
            pars (np.array): *n_fits* parameters of GARCH model
            bounds (np.array): bounds to the parameters

        Returns:
            np.array: Slightly varied parameters.
        """
        new_pars = []
        i = 0
        for par, bound in zip(pars, bounds):
            var = bound ** 2 / self.n_fits
            new_par = np.random.normal(par, var, 1)[0]
            if (new_par <= 0) and (i >= 1):
                logging.warning(f"new_par too small {new_par}")
                new_par = 0.01
            new_pars.append(new_par)
            i += 1
        return new_pars

    def simulate_paths(self, horizon, simulations, variate_GARCH_parameters=True):
        """Monte Carlo Simulation - Simulate paths using the fitted GARCH(1,1) model.

        Args:
            horizon (int): How many steps of paths to simulate
            variate (bool, optional): Whether GARCH parameters should be variated. Defaults to True.

        Returns:
            tuple: simulated_log_returns, simulated_tau_mu
        """
        logging.info(f" -------------- simulate paths for: {self.data_name}, {horizon}, {simulations}")
        if os.path.exists(self.filename_model) and (self.overwrite_model == False):
            logging.info(f"    ----------- use existing GARCH model: {self.filename_model}")
        else:
            logging.info("    ----------- fit new GARCH model")
            self.fit()

        self._load()
        pars = np.mean(self.parameters, axis=0).tolist()  # mean
        bounds = np.std(self.parameters, axis=0).tolist()  # std of parameters
        logging.info(f"garch parameters : {pars}")
        np.random.seed(1)  # for reproducability in _variate_pars()

        new_pars = copy.deepcopy(pars)  # set pars for first round of simulation
        simulated_log_returns = np.zeros(simulations)
        simulated_tau_mu = np.zeros(simulations)
        tick = time.time()
        for i in range(simulations):
            if (i + 1) % (simulations * 0.1) == 0:
                logging.info(f"{i + 1}/{simulations} - runtime: {round((time.time() - tick) / 60)} min")
            if ((i + 1) % (simulations * 0.05) == 0) & variate_GARCH_parameters:
                new_pars = self._variate_pars(pars, bounds)
            sigma2, e = self._GARCH_simulate(new_pars, horizon)
            simulated_log_returns[i] = np.sum(e)
            simulated_tau_mu[i] = horizon * pars[0]

        self.simulated_log_returns = simulated_log_returns  # summed because we have log-returns
        self.simulated_tau_mu = simulated_tau_mu
        return simulated_log_returns, simulated_tau_mu


# ----------------------------------------------------------------------------------------------------------- CALCULATOR
import pandas as pd


class Calculator(GARCH):
    """The Calculator Class for the Historical Density.

    The Historical Density is estimated via a Monte Carlo simulation, where the sample paths are simulated by a
    GARCH(1,1) model.

    Args:
        data (np.array): Timeseries of the underlying.
        tau_day (int): Time to maturity in days, also: *horizon*.
        date (str): Evaluation date, the last day of timeseries.
        S0 (float): Price of underlying at :math:`t=0`, which is on evaluation date
        garch_data_folder (str): Path to folder where to save/load GARCH model.
        n_fits (int): How many sliding windows the data is devided into.
        overwrite_model (bool, optional): Whether to overwrite the GARCH model. Defaults to True.
        overwrite_simulations (bool, optional): Whether to overwrite the simulations. Defaults to True.
        target (str, optional): Column name of the timeseries. Defaults to "price".
        window_length (int, optional): Length of each sliding window in GARCH fit. Defaults to 365.
        h (float, optional): Bandwidth for Kernel Density Estimation. Defaults to 0.15.
        simulations (int, optional): How many paths to simulate. Defaults to 5000.

    Attributes:
        log_returns (np.array): The daily log returns of the price index timeseries. Timeseries for GARCH model.
        GARCH (spd_trading.historical_density.GARCH): Instance of class.
        ST (np.array): Simulated prices of underlying, according to GARCH model.
        q_M (np.array): Density in Moneyness domain M=S0/ST.
    """

    def __init__(
        self,
        data,
        tau_day,
        date,
        S0,
        garch_data_folder,
        n_fits,
        cutoff=0.5,
        overwrite_model=True,
        overwrite_simulations=True,
        target="price",
        window_length=365,
        h=0.15,
        simulations=5000,
    ):
        self.data = data
        self.tau_day = tau_day
        self.date = date
        self.S0 = S0
        self.garch_data_folder = garch_data_folder
        self.cutoff = cutoff
        self.overwrite_simulations = overwrite_simulations
        self.target = target
        self.simulations = simulations
        self.h = h

        # parameters that are created during run
        self.log_returns = self._get_log_returns()
        self.GARCH = GARCH(
            data=self.log_returns,
            window_length=window_length,
            data_name=self.date,
            n_fits=n_fits,
            overwrite_model=overwrite_model,
            garch_data_folder=self.garch_data_folder,
            z_h=0.1,
        )
        self.ST = None
        self.q_M = None

    def _get_log_returns(self):
        """Calculate logarithmic 1-day returns from data timeseries."""
        n = self.data.shape[0]
        data = self.data.reset_index()
        first = data.loc[: n - 2, self.target].reset_index()
        second = data.loc[1:, self.target].reset_index()
        historical_returns = (second / first)[self.target]
        return np.log(historical_returns) * 100

    def _calculate_path(self, simulated_log_returns, simulated_tau_mu):
        """Calculates the underlyings' prices of maturity based on the simulations.

        Args:
            simulated_log_returns (np.array): simulated log returns at horizon (also: maturity) :math:`\\tau`
            simulated_tau_mu (np.array): product of :math:`\\tau \\cdot \\mu` for each simulation

        Returns:
            np.array: Simulated prices of underlying at maturity :math:`\\tau`
        """
        S_T = self.S0 * np.exp(simulated_log_returns / 100 + simulated_tau_mu / 100)
        return S_T

    def get_hd(self, variate_GARCH_parameters=True):
        """Estimates the Historical Density via a GARCH(1,1) model and Monte Carlo simulation.

        Args:
            variate_GARCH_parameters (bool, optional): Whether the GARCH parameters should be variated slightly during
                the Monte Carlo Simulation. Defaults to True.
        """
        self.filename = "T-{}_{}_Ksim.csv".format(self.tau_day, self.date)
        if os.path.exists(os.path.join(self.garch_data_folder, self.filename)) and (
            self.overwrite_simulations == False
        ):
            logging.info(f"-------------- use existing Simulations {self.filename}")
            pass
        else:
            logging.info("-------------- create new Simulations")
            simulated_log_returns, simulated_tau_mu = self.GARCH.simulate_paths(
                self.tau_day, self.simulations, variate_GARCH_parameters
            )
            self.ST = self._calculate_path(simulated_log_returns, simulated_tau_mu)
            pd.Series(self.ST).to_csv(os.path.join(self.garch_data_folder, self.filename), index=False)

        self.ST = pd.read_csv(os.path.join(self.garch_data_folder, self.filename))
        M = np.linspace((1 - self.cutoff), (1 + self.cutoff), 100)
        simulated_paths_in_moneyness = np.array(self.S0 / self.ST)
        self.q_M = {"x": M, "y": density_estimation(simulated_paths_in_moneyness, M, h=self.h)}


from matplotlib import pyplot as plt


class Plot:
    """The Plotting class for Historical Density.

    Args:
        x (float, optional): The Moneyness interval for the plots. :math:`M = [1-x, 1+x]`. Defaults to 0.5.
    """

    def __init__(self, x=0.5):
        self.x = x

    def density(self, HD):
        """Visualization of the Historical Density.

        Args:
            HD (spd_trading.historical_density.Calculator): Instance of class ``spd_trading.historical_density.Calculator``.

        Returns:
            Figure: Matplotlib figure.
        """
        fig, (ax0) = plt.subplots(1, 1, figsize=(6, 4))

        # density q_m
        ax0.plot(HD.q_M["x"], HD.q_M["y"])

        ax0.set_xlabel("Moneyness")
        ax0.set_ylabel("Historical Density")
        ax0.set_xlim(1 - self.x, 1 + self.x)
        ax0.set_ylim(0)

        plt.tight_layout()
        return fig
