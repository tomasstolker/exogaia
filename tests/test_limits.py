from exogaia import limits
import numpy as np
import pytest


class FakeGaiaAstrometry:
    def __init__(self, primary_mass, gaia_release, verbose):
        self.primary_mass = primary_mass
        self.gaia_release = gaia_release
        self.verbose = verbose

    def query_source(self, source_id, gaia_release):
        self.source_id = source_id
        self.query_release = gaia_release

    def simulate_data(self, **kwargs):
        return None


class FakeLeastSquares:
    def __init__(self, epoch_astrometry):
        self.best_param = np.zeros(12)
        self.best_param[5] = 10.0
        self.param_cov = np.eye(12)
        self.fit_success = True

    def accel_7param(self, **kwargs):
        self.best_param[5] = 10.0

    def accel_9param(self, **kwargs):
        self.best_param[5:9] = 10.0

    def orbit_grid(self, **kwargs):
        return None

    def orbit_fit(self, **kwargs):
        return None


def test_completeness_map_initializes_from_source(monkeypatch) -> None:
    monkeypatch.setattr(limits, "GaiaAstrometry", FakeGaiaAstrometry)

    completeness = limits.CompletenessMap(
        source_id=123,
        primary_mass=(1.0, 0.1),
        gaia_release="DR4",
    )

    assert completeness.source_id == 123
    assert completeness.epoch_astrom.source_id == 123
    assert completeness.epoch_astrom.query_release == "DR3"


@pytest.mark.parametrize("det_type", ["accel_7param", "accel_9param", "orbit"])
def test_calc_completeness_detection_modes(monkeypatch, tmp_path, det_type) -> None:
    monkeypatch.setattr(limits, "GaiaAstrometry", FakeGaiaAstrometry)
    monkeypatch.setattr(limits, "LeastSquares", FakeLeastSquares)

    completeness = limits.CompletenessMap(123, (1.0, 0.1))
    figure = completeness.calc_completeness(
        det_type=det_type,
        n_sigma=1.0,
        n_samples=1,
        mass_points=np.array([0.01]),
        sma_points=np.array([1.0]),
        plot_file=str(tmp_path / f"{det_type}.png"),
    )

    assert figure.axes


@pytest.mark.parametrize("kwargs", [{"n_sigma": 0.0}, {"n_samples": 0}])
def test_calc_completeness_validates_counts(monkeypatch, kwargs) -> None:
    monkeypatch.setattr(limits, "GaiaAstrometry", FakeGaiaAstrometry)
    completeness = limits.CompletenessMap(123, (1.0, 0.1))

    with pytest.raises(ValueError):
        completeness.calc_completeness(**kwargs)
