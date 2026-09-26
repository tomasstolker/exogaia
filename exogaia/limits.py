"""
Module with the ``CompletenessMap`` class.
"""

from numbers import Real

import matplotlib.pyplot as plt
import numpy as np

from beartype import beartype, typing
from matplotlib.figure import Figure
from scipy.ndimage import gaussian_filter
from scipy.stats import chi2, norm
from tqdm.auto import tqdm

from exogaia.data import GaiaAstrometry
from exogaia.leastsq import LeastSquares
from exogaia.utils import print_section


class CompletenessMap:
    """
    Class for computing companion detection completeness maps
    as a function of companion mass and semi-major axis.
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
            Gaia release (DR1, DR2, DR3, DR4, DR5) for which the
            completeness map will be computed.

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
        9-parameter acceleration + jerk model, or a full orbital model.
        Completeness is defined as the fraction of realizations that
        satisfy the selected detection criterion. Companions are
        assumed to contribute zero flux to the photocenter,
        regardless of the companion mass.

        Parameters
        ----------
        det_type : str
            Detection type. Supported values are ``"accel_7param"``,
            ``"accel_9param"``, and ``"orbit"``. For the 7-parameter
            model, detection is based on the joint significance of the
            two acceleration components. For the 9-parameter model,
            detection is based on the joint significance of the two
            acceleration and two jerk components. The full covariance
            matrix is used in both cases. Orbital detections are
            currently based on the fitted orbital period and its
            uncertainty.
        n_sigma : float, optional
            Gaussian-equivalent detection significance threshold
            (default: 3.0). For the acceleration models, this is
            converted to the corresponding chi-square threshold for
            the number of components being tested.
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
        For ``det_type="accel_7param"``, the detection statistic is

            Q = a.T @ C_a^-1 @ a,

        where ``a`` is the two-dimensional acceleration vector and
        ``C_a`` is its covariance matrix. Under the null hypothesis,
        ``Q`` follows a chi-square distribution with two degrees of
        freedom.

        For ``det_type="accel_9param"``, the detection statistic is
        computed from the four-dimensional vector containing the two
        acceleration and two jerk components. The full 4x4 covariance
        matrix is used, and the statistic follows a chi-square
        distribution with four degrees of freedom under the null
        hypothesis.

        In both cases, ``n_sigma`` is interpreted as a two-sided
        Gaussian-equivalent significance and converted to the
        corresponding chi-square threshold.

        For ``det_type="orbit"``, a realization is currently
        considered detected when

            period / sigma_period > n_sigma.

        The resulting map gives the detection completeness as a
        function of companion mass and semi-major axis.
        """

        print_section("Calculate completeness")

        if n_sigma <= 0.0:
            raise ValueError("'n_sigma' should be positive")

        if n_samples <= 0:
            raise ValueError("'n_samples' should be positive")

        print(f"Detection type: {det_type}")
        print(f"Detection significance: {n_sigma:.1f} sigma")

        # Gaussian-equivalent false-alarm probability
        false_alarm_prob = 2.0 * norm.sf(n_sigma)

        if det_type == "accel_7param":
            n_dof_det = 2

        elif det_type == "accel_9param":
            n_dof_det = 4

        else:
            n_dof_det = None

        if n_dof_det is not None:
            chi2_threshold = chi2.isf(
                false_alarm_prob,
                df=n_dof_det,
            )

            print(f"Detection degrees of freedom: {n_dof_det}")
            print(f"Chi-square threshold: {chi2_threshold:.2f}")

        print()

        if mass_points is None:
            # Grid points for companion mass (Msun)
            mass_points = np.logspace(-3, -1, 50)

        if sma_points is None:
            # Grid points for semi-major axis (au)
            sma_points = np.logspace(-1, 2, 50)

        if np.any(mass_points <= 0.0):
            raise ValueError("All companion masses should be positive")

        if np.any(sma_points <= 0.0):
            raise ValueError("All semi-major axes should be positive")

        compl_map = np.zeros((mass_points.size, sma_points.size))

        pbar = tqdm(total=mass_points.size * sma_points.size)

        for mass2_idx, mass2_item in enumerate(mass_points):
            for sma_idx, sma_item in enumerate(sma_points):
                n_detect = 0

                for _ in range(n_samples):
                    model_param = {"sma": sma_item}

                    self.epoch_astrom.simulate_data(
                        model_param=model_param,
                        mass_2=mass2_item,
                        flux_ratio=0.0,
                    )

                    least_sq = LeastSquares(epoch_astrometry=self.epoch_astrom)

                    if det_type == "accel_7param":
                        least_sq.accel_7param(
                            plot_file=None,
                            verbose=False,
                        )

                        # RA and Dec acceleration components (mas/yr^2)

                        signal_param = least_sq.best_param[5:7]

                        # Full covariance matrix of the acceleration

                        signal_cov = least_sq.param_cov[5:7, 5:7]

                        # Chi-square significance of the 2D acceleration vector

                        chi2_signal = signal_param @ np.linalg.solve(
                            signal_cov,
                            signal_param,
                        )

                        if chi2_signal > chi2_threshold:
                            n_detect += 1

                    elif det_type == "accel_9param":
                        least_sq.accel_9param(
                            plot_file=None,
                            verbose=False,
                        )

                        # RA and Dec acceleration and jerk components
                        # Parameters 5:7 are acceleration (mas/yr^2)
                        # and parameters 7:9 are jerk (mas/yr^3).

                        signal_param = least_sq.best_param[5:9]

                        # Full covariance matrix of acceleration and jerk

                        signal_cov = least_sq.param_cov[5:9, 5:9]

                        # Chi-square significance of the joint
                        # 4D acceleration + jerk vector

                        chi2_signal = signal_param @ np.linalg.solve(
                            signal_cov,
                            signal_param,
                        )

                        if chi2_signal > chi2_threshold:
                            n_detect += 1

                    else:
                        least_sq.orbit_grid(
                            n_points=20,
                            map_type="chi2_det",
                            plot_file=None,
                            verbose=False,
                        )

                        least_sq.orbit_fit(
                            inc_jitter=False,
                            plot_file=None,
                            verbose=False,
                        )

                        if least_sq.fit_success:
                            # Orbital period and uncertainty
                            period = least_sq.best_param[5]
                            sigma_period = np.sqrt(least_sq.param_cov[5, 5])

                            # This only checks the significance of the period
                            if sigma_period > 0.0 and period / sigma_period > n_sigma:
                                n_detect += 1

                compl_map[mass2_idx, sma_idx] = float(n_detect) / float(n_samples)

                pbar.update(1)

        pbar.close()

        if filter_sigma is not None:
            # Apply Gaussian filter to smooth out Monte Carlo noise
            compl_map = gaussian_filter(
                compl_map,
                sigma=filter_sigma,
            )

        fig, ax = plt.subplots(figsize=(5, 3))

        mesh = ax.pcolormesh(
            sma_points,
            mass_points,
            100.0 * compl_map,
            vmin=0.0,
            vmax=100.0,
        )

        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label("Completeness (%)", fontsize=12)

        ax.set_xlabel("Semi-major axis (au)", fontsize=12)
        ax.set_ylabel(r"Companion mass ($M_\odot$)", fontsize=12)
        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.set_title(
            rf"Gaia {self.gaia_release} "
            rf"${n_sigma}\sigma$ completeness ({det_type})",
            fontsize=10.0,
        )

        if plot_file is None:
            plt.show()
        else:
            print(f"\nOutput file: {plot_file}")
            fig.savefig(plot_file)

        return fig
