import pytest
from spd_trading.config.core import config
import pandas as pd


@pytest.fixture
def hd_input_data():
    return pd.read_csv(config.app_config.hd_input_data_file)


@pytest.fixture
def rnd_input_data():
    return pd.read_csv(config.app_config.rnd_input_data_file)


@pytest.fixture
def evaluation_day():
    return "2020-03-05"


@pytest.fixture
def evaluation_tau():
    return 8
