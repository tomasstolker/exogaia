import numpy as np
import pytest

from exogaia.utils import (
    binary_bias,
    calc_mass_from_sma,
    calc_sma_from_ti,
    param_dict_to_list,
    param_list_to_dict,
)


@pytest.mark.parametrize("n_param", [5, 7, 9, 12])
def test_parameter_dictionary_round_trip(n_param: int) -> None:
    names = ["ra_offset", "dec_offset", "parallax", "pm_ra", "pm_dec"]
    if n_param in (7, 9):
        names += ["pm_dot_ra", "pm_dot_dec"]
    if n_param == 9:
        names += ["pm_dotdot_ra", "pm_dotdot_dec"]
    if n_param == 12:
        names += ["per", "ecc", "tau", "sma", "inc", "aop", "pan"]

    values = np.arange(1.0, n_param + 1.0)
    parameters = dict(zip(names, values))

    assert param_dict_to_list(parameters) == list(values)
    assert param_list_to_dict(values) == parameters


def test_parameter_conversion_rejects_unsupported_lengths() -> None:
    with pytest.raises(ValueError):
        param_dict_to_list({"value": 1.0})
    with pytest.raises(ValueError):
        param_list_to_dict([1.0])


def test_calc_sma_from_thiele_innes_constants() -> None:
    model_param = np.zeros(9)
    model_param[5:9] = [3.0, 0.0, 0.0, 4.0]
    covariance = np.eye(9)

    sma, sma_sigma = calc_sma_from_ti(model_param, covariance)

    assert sma == pytest.approx(4.0)
    assert sma_sigma == pytest.approx(1.0)


def test_calc_mass_from_sma_returns_root_of_mass_function() -> None:
    mass_function, companion_mass = calc_mass_from_sma(
        sma_0=1.0, period=365.25, parallax=100.0, primary_mass=1.0
    )

    assert mass_function == pytest.approx(1.0e-6)
    assert companion_mass > 0.0
    assert companion_mass**3 / (1.0 + companion_mass) ** 2 == pytest.approx(
        mass_function
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sma_0": -1.0, "period": 1.0, "parallax": 1.0, "primary_mass": 1.0},
        {"sma_0": 1.0, "period": 0.0, "parallax": 1.0, "primary_mass": 1.0},
        {"sma_0": 1.0, "period": 1.0, "parallax": 0.0, "primary_mass": 1.0},
        {"sma_0": 1.0, "period": 1.0, "parallax": 1.0, "primary_mass": 0.0},
    ],
)
def test_calc_mass_from_sma_validates_inputs(kwargs) -> None:
    with pytest.raises(ValueError):
        calc_mass_from_sma(**kwargs)


def test_binary_bias_covers_resolution_regimes() -> None:
    separation = np.array([0.0, 18.0, 360.0])

    bias = binary_bias(
        separation,
        mass_ratio=0.5,
        flux_ratio=0.5,
        verbose=False,
    )

    assert bias[0] == pytest.approx(0.0)
    assert np.isfinite(bias[1])
    assert bias[2] == pytest.approx(-120.0)


@pytest.mark.parametrize("mass_ratio, flux_ratio", [(-1.0, 0.5), (0.5, 1.1)])
def test_binary_bias_validates_ratios(mass_ratio: float, flux_ratio: float) -> None:
    with pytest.raises(ValueError):
        binary_bias(np.array([1.0]), mass_ratio, flux_ratio, verbose=False)
