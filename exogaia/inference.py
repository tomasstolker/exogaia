"""
Module with the ``HipparcosGaiaOrbitMap`` class.
"""

from numbers import Real

import matplotlib.pyplot as plt
import numpy as np

from astropy.time import Time
from beartype import beartype, typing
from matplotlib.figure import Figure
from scipy.linalg import cho_factor, cho_solve
from scipy.ndimage import gaussian_filter
from tqdm.auto import tqdm

from exogaia.models import HipparcosGaiaPMAnomaly, KeplerModel
from exogaia.utils import print_section


class HipparcosGaiaOrbitMap:
    """
    Class for mapping the predicted companion position that is
    consistent with the Hipparcos-Gaia proper-motion anomaly.
    """

    @beartype
    def __init__(
        self,
        gaia_astrometry,
        hipparcos_astrometry,
    ) -> None:
        """
        Parameters
        ----------
        gaia_astrometry : GaiaAstrometry
            Gaia DR3 astrometry with epoch data, parallax, and
            the primary mass set.
        hipparcos_astrometry : HipparcosAstrometry
            Hipparcos epoch astrometry for which ``add_hgca()``
            has been called.

        Returns
        -------
        NoneType
            None
        """

        if gaia_astrometry.primary_mass is None:
            raise ValueError("The 'primary_mass' of 'gaia_astrometry' is not set.")

        if gaia_astrometry.parallax is None:
            raise ValueError(
                "The parallax of 'gaia_astrometry' is not set. Run "
                "'query_source' first."
            )

        self.gaia_astrometry = gaia_astrometry
        self.hipparcos_astrometry = hipparcos_astrometry

        self.pm_anomaly = HipparcosGaiaPMAnomaly(
            gaia_astrometry=gaia_astrometry,
            hipparcos_astrometry=hipparcos_astrometry,
            verbose=False,
        )

        self.orbit_param = None
        self.chi2 = None
        self.weights = None

    @beartype
    def sample_orbits(
        self,
        n_samples: int = 10000,
        mass_range: typing.Tuple[Real, Real] = (0.001, 0.5),
        sma_range: typing.Tuple[Real, Real] = (0.1, 100.0),
        seed: typing.Optional[int] = None,
    ) -> None:
        """
        Draw orbits from the priors and calculate the HGCA chi-squared
        of each orbit. The results are stored as attributes.

        The priors are log-uniform for the companion mass and the
        relative semi-major axis, a Beta distribution for the
        eccentricity (Kipping 2013), isotropic for the orientation,
        and uniform for ``tau``, ``aop``, and ``pan``.

        Parameters
        ----------
        n_samples : int
            Number of orbits (default: 10000).
        mass_range : tuple(float, float)
            Range of the companion mass (Msun).
        sma_range : tuple(float, float)
            Range of the relative semi-major axis (au).
        seed : int, None
            Seed for the random number generator.

        Returns
        -------
        NoneType
            None
        """

        print_section("Sample orbits")

        if n_samples <= 0:
            raise ValueError("'n_samples' should be positive")

        if min(mass_range) <= 0.0 or min(sma_range) <= 0.0:
            raise ValueError("The mass and semi-major axis ranges should be positive")

        rng = np.random.default_rng(seed)

        prim_mass = self.gaia_astrometry.primary_mass[0]
        parallax = self.gaia_astrometry.parallax

        mass = 10.0 ** rng.uniform(*np.log10(mass_range), n_samples)
        sma = 10.0 ** rng.uniform(*np.log10(sma_range), n_samples)

        self.orbit_param = {
            "mass": mass,
            "sma_au": sma,
            "per": np.sqrt(sma**3 / (prim_mass + mass)) * 365.25,
            "ecc": rng.beta(a=0.867, b=3.03, size=n_samples),  # (Kipping 2013)
            "tau": rng.uniform(0.0, 1.0, n_samples),
            "inc": np.arccos(rng.uniform(-1.0, 1.0, n_samples)),
            "aop": rng.uniform(0.0, 2.0 * np.pi, n_samples),
            "pan": rng.uniform(0.0, 2.0 * np.pi, n_samples),
        }

        mass_ratio = mass / prim_mass

        # Reflex motion of the primary, opposite to the companion

        sma_reflex = sma * parallax * mass_ratio / (1.0 + mass_ratio)
        aop_reflex = (self.orbit_param["aop"] + np.pi) % (2.0 * np.pi)

        hip = self.hipparcos_astrometry
        pm_anomaly = self.pm_anomaly

        # The HGCA chi-squared is the same as in
        # HipparcosGaiaPMAnomaly.calc_chi2 but vectorized. The
        # five-parameter fit is linear in the data, so the fitted
        # parameters are a fixed projection of the orbital positions.

        fit_proj = []

        for epoch_astrom, hgca_epoch in (
            (hip, pm_anomaly.hgca_epoch_hip),
            (self.gaia_astrometry, pm_anomaly.hgca_epoch_gaia),
        ):
            table = epoch_astrom.data_table
            err = table["centroid_pos_error_al"].to_numpy(dtype=float)
            year = table["obs_time_tcb"].to_numpy(dtype=float)
            sin_ang = table["sin_scan_ang"].to_numpy(dtype=float)
            cos_ang = table["cos_scan_ang"].to_numpy(dtype=float)
            plx_fac = table["parallax_factor_al"].to_numpy(dtype=float)

            design = np.column_stack(
                [
                    sin_ang,
                    cos_ang,
                    plx_fac,
                    (year - hgca_epoch[0]) * sin_ang,
                    (year - hgca_epoch[1]) * cos_ang,
                ]
            )

            weighted_design = design / err[:, None] ** 2
            proj = np.linalg.solve(design.T @ weighted_design, weighted_design.T)

            fit_proj.append((year, sin_ang, cos_ang, proj))

        cov_hg = hip.hgca_cov_hg
        cov_hip = hip.hgca_cov_hip + cov_hg
        cov_gaia = hip.hgca_cov_gaia + cov_hg
        cov_joint = np.block([[cov_hip, cov_hg], [cov_hg, cov_gaia]])

        obs_joint = np.concatenate(
            (
                hip.hgca_pm_hip - hip.hgca_pm_hg,
                hip.hgca_pm_gaia - hip.hgca_pm_hg,
            )
        )

        cho_fac = cho_factor(cov_joint, lower=True)

        orbit_ref_epoch = pm_anomaly.orbit_ref_epoch
        baseline = pm_anomaly.hgca_baseline

        self.chi2 = np.zeros(n_samples)

        chunk_size = 2000

        for start in tqdm(range(0, n_samples, chunk_size)):
            chunk = slice(start, start + chunk_size)

            chunk_param = {key: val[chunk] for key, val in self.orbit_param.items()}

            fit_list = []

            for year, sin_ang, cos_ang, proj in fit_proj:
                ra, dec = KeplerModel.calc_orbit_batch(
                    chunk_param,
                    sma_reflex[chunk],
                    aop_reflex[chunk],
                    year,
                    orbit_ref_epoch,
                )

                fit_list.append((ra * sin_ang + dec * cos_ang) @ proj.T)

            hip_fit, gaia_fit = fit_list

            pm_hg = (gaia_fit[:, :2] - hip_fit[:, :2]) / baseline

            resid = np.hstack((hip_fit[:, 3:5] - pm_hg, gaia_fit[:, 3:5] - pm_hg))
            resid = obs_joint[None, :] - resid

            self.chi2[chunk] = np.sum(resid * cho_solve(cho_fac, resid.T).T, axis=1)

        # Importance weights from the likelihood of the anomaly

        weights = np.exp(-0.5 * (self.chi2 - np.min(self.chi2)))
        self.weights = weights / np.sum(weights)

        n_eff = 1.0 / np.sum(self.weights**2)

        print(f"\nMinimum chi-squared: {np.min(self.chi2):.2f}")
        print(f"Effective sample size: {n_eff:.0f} of {n_samples}")

    @beartype
    def plot_position(
        self,
        epoch: typing.Union[Real, Time],
        n_bins: typing.Tuple[int, int] = (60, 36),
        sep_range: typing.Tuple[Real, Real] = (0.01, 10.0),
        true_position: typing.Optional[typing.Tuple[Real, Real]] = None,
        filter_sigma: typing.Optional[Real] = None,
        plot_file: typing.Optional[str] = None,
    ) -> Figure:
        """
        Plot the weighted two-dimensional histogram of the predicted
        companion separation and position angle at an epoch.

        Parameters
        ----------
        epoch : float, Time
            Epoch as float (Julian year, TCB) or as Time object.
        n_bins : tuple(int, int)
            Number of bins for the separation and position angle.
        sep_range : tuple(float, float)
            Separation range (arcsec).
        true_position : tuple(float, float), None
            Measured separation (arcsec) and position angle (deg) to
            mark in the plot.
        filter_sigma : float, None
            Width (bins) of an optional Gaussian filter to smooth
            the histogram. It wraps around in position angle.
        plot_file : str, None
            File path for saving the plot. The plot is shown
            interactively if ``None``.

        Returns
        -------
        Figure
            Matplotlib ``Figure`` object.
        """

        print_section("Plot position")

        if self.weights is None:
            raise RuntimeError("Run 'sample_orbits' first.")

        parallax = self.gaia_astrometry.parallax

        # Relative position of the companion (mas)

        if isinstance(epoch, Time):
            epoch = epoch.tcb.jyear

        print(f"Epoch = {epoch:.2f}")

        if filter_sigma is not None:
            print(f"Filter width= {filter_sigma:.2f}")

        d_ra, d_dec = KeplerModel.calc_orbit_batch(
            self.orbit_param,
            self.orbit_param["sma_au"] * parallax,
            self.orbit_param["aop"],
            np.array([epoch]),
            self.pm_anomaly.orbit_ref_epoch,
        )

        sep = np.hypot(d_ra[:, 0], d_dec[:, 0]) / 1e3
        pos_ang = np.degrees(np.arctan2(d_ra[:, 0], d_dec[:, 0])) % 360.0

        sep_edges = np.logspace(
            np.log10(sep_range[0]), np.log10(sep_range[1]), n_bins[0] + 1
        )
        pa_edges = np.linspace(0.0, 360.0, n_bins[1] + 1)

        hist, _, _ = np.histogram2d(
            sep, pos_ang, bins=[sep_edges, pa_edges], weights=self.weights
        )

        if filter_sigma is not None:
            # Wrap in position angle, which is periodic
            hist = gaussian_filter(hist, sigma=filter_sigma, mode=("nearest", "wrap"))

        fig, ax = plt.subplots(figsize=(5, 3))

        mesh = ax.pcolormesh(sep_edges, pa_edges, hist.T / hist.max(), cmap="inferno")

        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label("Relative probability", fontsize=12)

        if true_position is not None:
            ax.axvline(true_position[0], color="lightyellow", ls="--", lw=1.0)
            ax.axhline(true_position[1], color="lightyellow", ls="--", lw=1.0)

        ax.set_xscale("log")
        ax.set_xlabel("Separation (arcsec)", fontsize=12)
        ax.set_ylabel("Position angle (deg)", fontsize=12)
        ax.set_ylim(0.0, 360.0)
        ax.set_title(f"Predicted position at {epoch:.2f}", fontsize=10.0)

        if plot_file is None:
            plt.show()
        else:
            print(f"\nOutput file: {plot_file}")
            fig.savefig(plot_file)

        return fig
