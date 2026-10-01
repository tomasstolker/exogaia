import numpy as np
import pytest

from exogaia.planets import OccurrenceRate


def constant_occurrence_rate(sma, mass_planet, mass_star):
    return np.ones_like(np.broadcast_arrays(sma, mass_planet, mass_star)[0])


def test_occurrence_rate_integrates_callable_density() -> None:
    model = OccurrenceRate(
        primary_mass=1.0,
        occ_rate=constant_occurrence_rate,
        sma_range=(1.0, 10.0),
        mass_range=(2.0, 8.0),
        verbose=False,
    )

    result = model.integrate_occ_rate(mass_star=1.0)

    assert result == pytest.approx(np.log(10.0) * np.log(4.0), rel=1e-3)


def test_occurrence_rate_samples_required_planet_in_range() -> None:
    model = OccurrenceRate(
        primary_mass=np.array([1.0, 1.2]),
        occ_rate=constant_occurrence_rate,
        sma_range=(1.0, 10.0),
        mass_range=(2.0, 8.0),
        verbose=False,
    )

    sma, mass, hosts = model.sample_planets(require_planet=True, seed=4)

    assert sma.size == mass.size == hosts.size
    assert np.all((sma >= 1.0) & (sma <= 10.0))
    assert np.all((mass > 0.0))
    assert np.all(np.isin(hosts, [0, 1]))


def test_occurrence_rate_rejects_invalid_primary_mass() -> None:
    with pytest.raises(ValueError):
        OccurrenceRate(primary_mass=0.0, verbose=False)
