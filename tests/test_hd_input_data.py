import numpy as np

from spd_trading.config.core import config


hd_index_schema = {
    "date_str": {"range": {"min": "1000-01-01", "max": "9999-12-31"}, "dtype": object},  # np.dtype str -> object
    "price": {
        "range": {"min": 0, "max": np.inf},
        "dtype": float,
    },
}


def test_input_data_columns(hd_input_data):
    for feature in config.model_config.hd_input_features:
        assert feature in hd_input_data.columns


def test_input_data_ranges(hd_input_data):
    max_values = hd_input_data.max()
    min_values = hd_input_data.min()

    for feature in hd_index_schema.keys():
        print(feature)
        assert max_values[feature] <= hd_index_schema[feature]["range"]["max"]
        assert min_values[feature] >= hd_index_schema[feature]["range"]["min"]


def test_input_data_types(hd_input_data):
    data_types = hd_input_data.dtypes  # pandas dtypes method

    for feature in hd_index_schema.keys():
        assert data_types[feature] == hd_index_schema[feature]["dtype"]
