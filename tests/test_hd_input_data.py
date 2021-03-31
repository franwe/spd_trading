import pandas as pd
import numpy as np

import unittest
import os

TESTDATA_FILENAME = os.path.join(".", "data", "hd_input_data.csv")

rnd_options_schema = {
    "date_str": {"range": {"min": "1000-01-01", "max": "9999-12-31"}, "dtype": object},  # np.dtype str -> object
    "price": {
        "range": {"min": 0, "max": np.inf},
        "dtype": float,
    },
}


class TestRndInputData(unittest.TestCase):
    def setUp(self):
        self.data = pd.read_csv(TESTDATA_FILENAME)

    def test_input_data_columns(self):
        df_columns = self.data.columns.tolist()
        for col in rnd_options_schema.keys():
            self.assertTrue(col in df_columns)

    def test_input_data_ranges(self):
        max_values = self.data.max()
        min_values = self.data.min()

        for feature in rnd_options_schema.keys():
            self.assertTrue(max_values[feature] <= rnd_options_schema[feature]["range"]["max"])
            self.assertTrue(min_values[feature] >= rnd_options_schema[feature]["range"]["min"])

    def test_input_data_types(self):
        data_types = self.data.dtypes  # pandas dtypes method

        for feature in rnd_options_schema.keys():
            self.assertEqual(data_types[feature], rnd_options_schema[feature]["dtype"])


if __name__ == "__main__":
    unittest.main()
