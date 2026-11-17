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

    completeness = limits.GaiaCompleteness(
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

    completeness = limits.GaiaCompleteness(123, (1.0, 0.1))
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
    completeness = limits.GaiaCompleteness(123, (1.0, 0.1))

    with pytest.raises(ValueError):
        completeness.calc_completeness(**kwargs)


class FakePMAnomaly:
    def __init__(self, gaia_astrometry, hipparcos_astrometry, verbose):
        self.dpm_hip_hg = np.array([1.0, 0.0])
        self.dpm_gaia_hg = np.array([0.0, 1.0])
        self.dpm_model = np.array([1.0, 0.0, 0.0, 1.0])

    def calc_pm_anomaly(self, model_param, add_noise=False):
        assert add_noise
        assert model_param["sma"] > 0.0


@pytest.mark.parametrize("det_type", ["joint", "hipparcos", "gaia"])
def test_hipparcos_gaia_completeness(monkeypatch, tmp_path, det_type) -> None:
    from types import SimpleNamespace

    monkeypatch.setattr(limits, "HipparcosGaiaPMAnomaly", FakePMAnomaly)

    gaia = SimpleNamespace(primary_mass=(1.0, 0.1), parallax=10.0)
    hip = SimpleNamespace(
        hgca_cov_hip=0.01 * np.eye(2),
        hgca_cov_gaia=0.01 * np.eye(2),
        hgca_cov_hg=0.01 * np.eye(2),
    )

    completeness = limits.HipparcosGaiaCompleteness(gaia, hip)
    completeness.calc_completeness(
        det_type=det_type,
        n_sigma=3.0,
        n_samples=2,
        mass_points=np.array([0.01]),
        sma_points=np.array([1.0]),
        plot_file=str(tmp_path / "hg.png"),
    )

    assert completeness.compl_map.shape == (1, 1)
    assert completeness.compl_map[0, 0] == 1.0


def _fake_astrometry(rng, n_obs, year_start, year_end, ref_epoch):
    import pandas as pd

    angle = rng.uniform(0.0, np.pi, n_obs)
    obs_time = np.sort(rng.uniform(year_start, year_end, n_obs))

    return pd.DataFrame(
        {
            "obs_time_tcb": obs_time,
            "relative_time_year": obs_time - ref_epoch,
            "centroid_pos_error_al": rng.uniform(0.2, 0.5, n_obs),
            "sin_scan_ang": np.sin(angle),
            "cos_scan_ang": np.cos(angle),
            "parallax_factor_al": rng.uniform(-1.0, 1.0, n_obs),
        }
    )


def test_hipparcos_gaia_orbit_map_matches_calc_chi2(tmp_path) -> None:
    from types import SimpleNamespace

    from astropy.time import Time
    from exogaia.inference import HipparcosGaiaOrbitMap
    from exogaia.models import KeplerModel

    rng = np.random.default_rng(1)

    gaia = SimpleNamespace(
        primary_mass=(1.0, 0.1),
        parallax=20.0,
        gaia_release="DR3",
        source_id=1,
        ref_epoch=Time(2016.0, format="jyear", scale="tcb"),
        data_table=_fake_astrometry(rng, 60, 2014.6, 2017.4, 2016.0),
    )

    hip = SimpleNamespace(
        hip_id=1,
        ref_epoch=Time(1991.25, format="jyear", scale="tcb"),
        hgca_epoch_hip=np.array([1991.3, 1991.2]),
        hgca_epoch_gaia=np.array([2016.1, 2016.0]),
        hgca_pm_hip=np.array([10.0, -5.0]),
        hgca_pm_gaia=np.array([10.2, -5.1]),
        hgca_pm_hg=np.array([10.1, -5.0]),
        hgca_cov_hip=0.04 * np.eye(2),
        hgca_cov_gaia=0.01 * np.eye(2),
        hgca_cov_hg=0.005 * np.eye(2),
        hgca_chisq=1.0,
        data_table=_fake_astrometry(rng, 100, 1989.9, 1993.2, 1991.25),
    )

    orbit_map = HipparcosGaiaOrbitMap(gaia, hip)

    with pytest.raises(RuntimeError):
        orbit_map.plot_position(2020.0)

    orbit_map.sample_orbits(n_samples=6, seed=3)

    assert np.isclose(orbit_map.weights.sum(), 1.0)

    # Compare with the scalar implementation
    prim_mass = gaia.primary_mass[0]
    pars = orbit_map.orbit_param

    for i in range(6):
        ratio = pars["mass"][i] / prim_mass

        model_param = {
            "per": pars["per"][i],
            "ecc": pars["ecc"][i],
            "tau": pars["tau"][i],
            "sma": pars["sma_au"][i] * gaia.parallax * ratio / (1.0 + ratio),
            "inc": pars["inc"][i],
            "aop": (pars["aop"][i] + np.pi) % (2.0 * np.pi),
            "pan": pars["pan"][i],
        }

        expected = orbit_map.pm_anomaly.calc_chi2(model_param)

        assert np.isclose(orbit_map.chi2[i], expected, rtol=1e-6)

    for astrometry, epochs, fit_param, fit_cov in (
        (
            hip,
            hip.hgca_epoch_hip,
            orbit_map.pm_anomaly.hip_fit_param,
            orbit_map.pm_anomaly.hip_fit_cov,
        ),
        (
            gaia,
            hip.hgca_epoch_gaia,
            orbit_map.pm_anomaly.gaia_fit_param,
            orbit_map.pm_anomaly.gaia_fit_cov,
        ),
    ):
        table = astrometry.data_table
        times = table["obs_time_tcb"].to_numpy()
        sin_angle = table["sin_scan_ang"].to_numpy()
        cos_angle = table["cos_scan_ang"].to_numpy()
        error = table["centroid_pos_error_al"].to_numpy()
        design = np.column_stack(
            (
                sin_angle,
                cos_angle,
                table["parallax_factor_al"],
                (times - epochs[0]) * sin_angle,
                (times - epochs[1]) * cos_angle,
            )
        )
        orbit = KeplerModel(astrometry, ref_epoch=2016.0, verbose=False)
        ra, dec = orbit.calc_orbit(model_param, obs_time=times)
        positions = ra * sin_angle + dec * cos_angle
        weighted_design = design / error[:, None]
        expected_param = np.linalg.lstsq(
            weighted_design, positions / error, rcond=None
        )[0]
        expected_cov = np.linalg.inv(weighted_design.T @ weighted_design)
        np.testing.assert_allclose(fit_param, expected_param, atol=1e-10)
        np.testing.assert_allclose(fit_cov, expected_cov, atol=1e-10)

    figure = orbit_map.plot_position(
        2020.0,
        true_position=(0.2, 30.0),
        filter_sigma=1.0,
        plot_file=str(tmp_path / "m.png"),
    )

    assert figure.axes
