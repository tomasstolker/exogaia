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

from exogaia.core import ExoGaia
from exogaia.data import EpochAstrometry
from exogaia.leastsq import LeastSquares


class CompletenessMap(ExoGaia):
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
        gaia_release: str = "DR4",
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

        self.epoch_astrom = EpochAstrometry(
            primary_mass=self.primary_mass,
            gaia_release=self.gaia_release,
            verbose=False,
        )

        self.print_section("Completeness map")

        print(f"Gaia ID = {self.source_id}")
        print(f"Gaia release = {self.gaia_release}")
        print(
            f"\nPrimary mass = {self.primary_mass[0]:.2f} +/- {self.primary_mass[1]:.2f}"
        )

        self.epoch_astrom.query_source(source_id=self.source_id, gaia_release="DR3")

    @beartype
    def calc_completeness(
        self,
        det_type: str = "accel_7param",
        n_sigma: Real = 3.0,
        n_samples: int = 30,
        mass_points: typing.Optional[np.ndarray] = None,
        sma_points: typing.Optional[np.ndarray] = None,
        filter_sigma: typing.Optional[Real] = None,
        plot_file: typing.Optional[str] = None,
    ) -> Figure:
        """
        Compute and plot a completeness map. This method
        estimates the detection completeness by sampling
        random orbits for a grid of companion masses and
        semi-major axes. Either a 7-parameter
        acceleration, 9-parameter acceleration, or full
        orbit model is fit, and the fraction of
        realizations with a detection significance larger
        than ``n_sigma`` is adopted as the completeness.

        Parameters
        ----------
        det_type : str
            Detection type: 'accel_7param', 'accel_9param',
            or 'orbit', for respectively using the precision
            on the acceleration, jerk, and orbital period, to
            determine if a simulated source is detected
            with a significance of ``n_sigma``.
        n_sigma : float, optional
            Detection threshold in units of acceleration
            signal-to-noise (default: 5.0).
        n_samples : int
            Number of Monte Carlo realizations per grid
            point (default: 30).
        mass_points : np.ndarray, None
            Grid of companion masses (Msun). If ``None``, a
            linear grid between 0.001 and 0.1 Msun is used.
        sma_points : np.ndarray, None
            Grid of semi-major axes (au). If ``None``, a
            logarithmic grid between 0.1 and 100 au is used.
        filter_sigma : float, None
            Width of the optional Gaussian filter that is applied
            to smooth away the Monte Carlo sampling noise. The
            width is in number of grid points, so a value of 1.0
            works usually well if for example the number of grid
            points is 50 in both the mass and semi-major axis
            dimension. No filter is applied if the argument is
            set to ``None``.
        plot_file : str, None
            If provided, the completeness map is saved to this
            file. If ``None``, the plot is shown interactively.

        Returns
        -------
        Figure
            The Matplotlib ``Figure`` object that can be used
            for further adjustments of the plot.

        Notes
        -----
        - Completeness is defined as the fraction of simulations
          for which the total acceleration amplitude satisfies:

              accel / sigma_accel > n_sigma

        - The resulting map shows completeness (%) as a function
          of companion mass (Msun) and semi-major axis (au).
        """

        self.print_section("Calculate completeness")

        if mass_points is None:
            # Grid points for companion mass (Msun)
            mass_points = np.linspace(0.001, 0.1, 50)

        if sma_points is None:
            # Grid points for semi-major axis (au)
            sma_points = 10.0 ** np.linspace(np.log10(0.1), np.log10(100.0), 50)

        compl_map = np.zeros((mass_points.size, sma_points.size))

        pbar = tqdm(total=mass_points.size * sma_points.size)

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
                        sigma_accel = np.sqrt(grad_accel @ cov_accel @ grad_accel)

                        if accel / sigma_accel > n_sigma:
                            compl_map[mass2_idx, sma_idx] += 1.0 / float(n_samples)

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
                        sigma_jerk = np.sqrt(grad_jerk @ cov_jerk @ grad_jerk)

                        if sigma_jerk > 0.0 and jerk / sigma_jerk > n_sigma:
                            compl_map[mass2_idx, sma_idx] += 1.0 / float(n_samples)

                    elif det_type == "orbit":
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
                                compl_map[mass2_idx, sma_idx] += 1.0 / float(n_samples)

                    else:
                        raise ValueError(
                            f"Setting 'det_type'='{det_type}' is not valid. "
                            "Please set the argument of 'det_type' to "
                            "'accel_7param', 'accel_9param', or 'orbit'."
                        )

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
        ax.set_title(rf"${n_sigma}\sigma$ completeness ({det_type})", fontsize=10.0)

        if plot_file is None:
            plt.show()
        else:
            plt.savefig(plot_file)
            plt.close(fig)

        return fig
