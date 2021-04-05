import numpy as np

from spd_trading.config.core import config

rnd_options_schema = {
    "M": {
        "range": {"min": 0, "max": np.inf},
        "dtype": float,
    },
    "iv": {
        "range": {"min": 0, "max": np.inf},
        "dtype": float,
    },
    "K": {
        "range": {"min": 0, "max": np.inf},
        "dtype": float,
    },
    "S": {
        "range": {"min": 0, "max": np.inf},
        "dtype": float,
    },
    "P": {
        "range": {"min": 0, "max": np.inf},
        "dtype": float,
    },
    "tau": {
        "range": {"min": 0, "max": np.inf},
        "dtype": float,
    },
    "option": {
        "range": {"min": "C", "max": "P"},
        "dtype": object,  # np.dtype str -> object
    },
}


def test_input_data_columns(rnd_input_data):
    for feature in config.model_config.rnd_input_features:
        assert feature in rnd_input_data.columns


def test_input_data_ranges(rnd_input_data):
    max_values = rnd_input_data.max()
    min_values = rnd_input_data.min()

    for feature in rnd_options_schema.keys():
        print(feature)
        assert max_values[feature] <= rnd_options_schema[feature]["range"]["max"]
        assert min_values[feature] >= rnd_options_schema[feature]["range"]["min"]


def test_input_data_types(rnd_input_data):
    data_types = rnd_input_data.dtypes  # pandas dtypes method

    for feature in rnd_options_schema.keys():
        assert data_types[feature] == rnd_options_schema[feature]["dtype"]
