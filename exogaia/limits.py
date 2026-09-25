"""
Module with the ``CompletenessMap`` class.
"""

from numbers import Real

import matplotlib.pyplot as plt
import numpy as np

from beartype import beartype, typing
from matplotlib.figure import Figure
from scipy.ndimage import gaussian_filter

from tqdm.auto import tqdm

from exogaia.data import GaiaAstrometry
from exogaia.leastsq import LeastSquares
from exogaia.utils import print_section


class CompletenessMap:
    """
    Class for computing a completeness map of detecting a
    proper motion acceleration as function of planetary
    mass and semi-major axis.
    """

    @beartype
    def __init__(
        self,
        source_id: typing.Union[int, str],
        primary_mass: typing.Tuple[Real, Real],
        gaia_release: typing.Literal["DR1", "DR2", "DR3", "DR4", "DR5"] = "DR4",
    ) -> None:
        """
        Parameters
        ----------
        source_id : int, str
            Gaia source ID in DR3.
        primary_mass : tuple(float, float)
            Primary mass and uncertainty (Msun).
        gaia_release : str
            Gaia release (DR3, DR4, DR5) for which the completeness
            map will be computed.

        Returns
        -------
        NoneType
            None
        """

        self.source_id = source_id
        self.primary_mass = primary_mass
        self.gaia_release = gaia_release

        self.epoch_astrom = GaiaAstrometry(
            primary_mass=self.primary_mass,
            gaia_release=self.gaia_release,
            verbose=False,
        )

        print_section("Completeness map", bound_char="=")

        print(f"Gaia ID = {self.source_id}")
        print(f"Gaia release = {self.gaia_release}")
        print(
            f"\nPrimary mass = {self.primary_mass[0]:.2f} +/- {self.primary_mass[1]:.2f}"
        )

        self.epoch_astrom.query_source(source_id=self.source_id, gaia_release="DR3")

    @beartype
    def calc_completeness(
        self,
        det_type: typing.Literal[
            "accel_7param",
            "accel_9param",
            "orbit",
        ] = "accel_7param",
        n_sigma: Real = 3.0,
        n_samples: int = 30,
        mass_points: typing.Optional[np.ndarray] = None,
        sma_points: typing.Optional[np.ndarray] = None,
        filter_sigma: typing.Optional[Real] = None,
        plot_file: typing.Optional[str] = None,
    ) -> Figure:
        """
        Compute and plot a detection completeness map.

        The detection completeness is estimated by sampling random
        orbits on a grid of companion masses and semi-major axes.
        For each grid point, ``n_samples`` realizations are generated
        and fit with either a 7-parameter acceleration model, a
        9-parameter acceleration model, or a full orbital model.
        Completeness is defined as the fraction of realizations that
        satisfy the selected detection criterion.

        Parameters
        ----------
        det_type : str
            Detection type. Supported values are ``"accel_7param"``,
            ``"accel_9param"``, and ``"orbit"``. Acceleration and jerk
            detections are based on the significance of their two-dimensional
            vectors using the full covariance matrix. Orbital detections are
            based on the fitted orbital period and its uncertainty.
        n_sigma : float, optional
            Detection significance threshold (default: 3.0).
        n_samples : int
            Number of Monte Carlo realizations per grid point
            (default: 30).
        mass_points : np.ndarray, None
            Grid of companion masses (Msun). If ``None``, a logarithmic
            grid between 0.001 and 0.1 Msun is used.
        sma_points : np.ndarray, None
            Grid of semi-major axes (au). If ``None``, a logarithmic
            grid between 0.1 and 100 au is used.
        filter_sigma : float, None
            Width of an optional Gaussian filter used to reduce Monte
            Carlo sampling noise. The width is specified in grid points.
            No filter is applied if ``None``.
        plot_file : str, None
            File path for saving the completeness map. If ``None``,
            the plot is shown interactively.

        Returns
        -------
        Figure
            Matplotlib ``Figure`` object containing the completeness map.

        Notes
        -----
        For ``det_type="accel_7param"``, the acceleration significance is

            sqrt(a.T @ C_a^-1 @ a),

        where ``a`` is the two-dimensional acceleration vector and ``C_a``
        is its covariance matrix. The same criterion is applied to the jerk
        vector for ``det_type="accel_9param"``.

        For ``det_type="orbit"``, a realization is currently considered
        detected when

            period / sigma_period > n_sigma.

        The resulting map gives the detection completeness as a function
        of companion mass and semi-major axis.
        """

        print_section("Calculate completeness")

        if mass_points is None:
            # Grid points for companion mass (Msun)
            mass_points = np.logspace(-3, -1, 50)

        if sma_points is None:
            # Grid points for semi-major axis (au)
            sma_points = 10.0 ** np.linspace(np.log10(0.1), np.log10(100.0), 50)

        compl_map = np.zeros((mass_points.size, sma_points.size))

        pbar = tqdm(total=mass_points.size * sma_points.size)

        n_detect = 0

        for mass2_idx, mass2_item in enumerate(mass_points):
            for sma_idx, sma_item in enumerate(sma_points):
                for _ in range(n_samples):
                    model_param = {"sma": sma_item}

                    self.epoch_astrom.simulate_data(
                        model_param=model_param,
                        mass_2=mass2_item,
                    )

                    least_sq = LeastSquares(epoch_astrometry=self.epoch_astrom)

                    if det_type == "accel_7param":
                        least_sq.accel_7param(plot_file=None, verbose=False)

                        # Acceleration dmu/dt (mas/yr^2)
                        accel_components = least_sq.best_param[5:7]
                        accel = np.linalg.norm(accel_components)

                        # Gradient of |a| with respect to the RA and Dec components
                        grad_accel = accel_components / accel

                        # Covariance matrix of the RA and Dec acceleration components
                        cov_accel = least_sq.param_cov[5:7, 5:7]

                        # Uncertainty on the total acceleration
                        # sigma_accel = np.sqrt(grad_accel @ cov_accel @ grad_accel)

                        # Significance of the 2D acceleration vector
                        # using its full covariance matrix
                        snr_accel = np.sqrt(
                            accel_components
                            @ np.linalg.solve(cov_accel, accel_components)
                        )

                        if snr_accel > n_sigma:
                            n_detect += 1

                    elif det_type == "accel_9param":
                        least_sq.accel_9param(plot_file=None, verbose=False)

                        # Jerk d²mu/dt² (mas/yr^3)
                        # Parameters 7 and 8 are the RA and Dec jerk components.
                        jerk_components = least_sq.best_param[7:9]
                        jerk = np.linalg.norm(jerk_components)

                        # Gradient of |j| with respect to the RA and Dec components
                        grad_jerk = jerk_components / jerk

                        # Covariance matrix of the RA and Dec jerk components
                        cov_jerk = least_sq.param_cov[7:9, 7:9]

                        # Propagate the component uncertainties and covariance into the
                        # uncertainty on the total jerk.
                        # sigma_jerk = np.sqrt(grad_jerk @ cov_jerk @ grad_jerk)

                        # Significance of the 2D jerk vector
                        # using its full covariance matrix
                        snr_jerk = np.sqrt(
                            jerk_components @ np.linalg.solve(cov_jerk, jerk_components)
                        )

                        if snr_jerk > n_sigma:
                            n_detect += 1

                    else:
                        least_sq.orbit_grid(
                            n_points=20,
                            map_type="chi2_det",
                            plot_file=None,
                            verbose=False,
                        )

                        least_sq.orbit_fit(
                            inc_jitter=False, plot_file=None, verbose=False
                        )

                        if least_sq.fit_success:
                            period = least_sq.best_param[9]
                            sigma_period = np.sqrt(np.diag(least_sq.param_cov))[9]

                            if period / sigma_period > n_sigma:
                                n_detect += 1

                compl_map[mass2_idx, sma_idx] = float(n_detect) / float(n_samples)

                pbar.update(1)

        if filter_sigma is not None:
            # Apply Gaussian filter to smooth out Monte Carlo noise
            compl_map = gaussian_filter(compl_map, sigma=filter_sigma)

        fig, ax = plt.subplots(figsize=(5, 3))

        mesh = ax.pcolormesh(
            sma_points, mass_points, 100.0 * compl_map, vmin=0.0, vmax=100.0
        )

        cbar = plt.colorbar(mesh, ax=ax)
        cbar.set_label("Completeness (%)", fontsize=12)

        ax.set_xlabel("Semi-major axis (au)", fontsize=12)
        ax.set_ylabel(r"Companion mass ($M_\odot$)", fontsize=12)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(
            rf"{self.gaia_release} ${n_sigma}\sigma$ completeness ({det_type})",
            fontsize=10.0,
        )

        if plot_file is None:
            plt.show()
        else:
            plt.savefig(plot_file)
            plt.close(fig)

        return fig
