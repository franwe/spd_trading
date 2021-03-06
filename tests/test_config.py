from spd_trading.config.core import (
    create_and_validate_config,
    fetch_config_from_yaml,
)

import pytest
from pydantic import ValidationError


TEST_CONFIG_TEXT = """
package_name: spd_trading
rnd_input_data_file: rnd_input_data.csv
hd_input_data_file: hd_input_data.csv
target: q_M
rnd_input_features:
  - M
hd_input_features:
  - price
numerical_vars:
  - M
categorical_vars:
  - option
numerical_na_not_allowed:
  - M
random_state: 1
loss: ls
allowed_loss_functions:
  - ls
  - MSE
"""

INVALID_TEST_CONFIG_TEXT = """
package_name: spd_trading
rnd_input_data_file: rnd_input_data.csv
hd_input_data_file: hd_input_data.csv
target: q_M
rnd_input_features:
  - M
hd_input_features:
  - price
numerical_vars:
  - M
categorical_vars:
  - option
numerical_na_not_allowed:
  - M
random_state: 1
loss: ls
allowed_loss_functions:
  - MSE
"""


def test_fetch_config_structure(tmpdir):
    # Given
    config_1 = tmpdir.mkdir("sub").join("sample_config.yaml")
    config_1.write(TEST_CONFIG_TEXT)
    parsed_config = fetch_config_from_yaml(cfg_path=config_1)

    # When
    config = create_and_validate_config(parsed_config=parsed_config)

    # Then
    assert config.model_config
    assert config.app_config


def test_config_validation_raises_error_for_invalid_config(tmpdir):
    # Given
    config_1 = tmpdir.mkdir("sub").join("sample_config.yaml")

    # invalid config attempts to set a prohibited loss
    # function which we validate against an allowed set of
    # loss function parameters.
    config_1.write(INVALID_TEST_CONFIG_TEXT)
    parsed_config = fetch_config_from_yaml(cfg_path=config_1)

    # When
    with pytest.raises(ValidationError) as excinfo:
        create_and_validate_config(parsed_config=parsed_config)

    # Then
    assert "not in the allowed set" in str(excinfo.value)


def test_missing_config_field_raises_validation_error(tmpdir):
    # Given
    # We make use of the pytest built-in tmpdir fixture
    config_1 = tmpdir.mkdir("sub").join("sample_config.yaml")
    TEST_CONFIG_TEXT = """package_name: spd_trading"""
    config_1.write(TEST_CONFIG_TEXT)
    parsed_config = fetch_config_from_yaml(cfg_path=config_1)

    # When
    with pytest.raises(ValidationError) as excinfo:
        create_and_validate_config(parsed_config=parsed_config)

    # Then
    assert "field required" in str(excinfo.value)
