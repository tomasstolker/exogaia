"""
Module with the ``CompletenessMap`` class.
"""

import matplotlib.pyplot as plt
import numpy as np

from astropy import units as u

# from astropy.io import fits
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
        primary_mass: typing.Tuple[float, float],
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
            primary_mass=self.primary_mass, gaia_release=self.gaia_release
        )

        self.epoch_astrom.query_source(source_id=self.source_id, gaia_release="DR3")

    @beartype
    def calc_completeness(
        self,
        n_sigma: typing.Union[float, int] = 5,
        n_samples: int = 30,
        mass_points: typing.Optional[np.ndarray] = None,
        sma_points: typing.Optional[np.ndarray] = None,
        filter_sigma: typing.Optional[float] = None,
        plot_file: typing.Optional[str] = None,
    ) -> Figure:
        """
        Compute and plot a completeness map for astrometric
        accelerations. This method estimates the detection
        completeness by sampling random orbits for a grid of
        companion masses and semi-major axes. A 9-parameter
        acceleration model is fit, and the fraction of
        realizations with a detection significance larger
        than ``n_sigma`` adopted as completeness.

        Parameters
        ----------
        n_sigma : float, optional
            Detection threshold in units of acceleration signal-to-noise
            (default is 5.0).
        n_samples : int, optional
            Number of Monte Carlo realizations per grid point. Note that this
            value is internally overridden to 50 in the current implementation.
        mass_points : numpy.ndarray or None, optional
            Grid of secondary (companion) masses in solar masses. If None,
            a linear grid between 0.001 and 0.1 Msun is used.
        sma_points : numpy.ndarray or None, optional
            Grid of semi-major axes in astronomical units. If None, a logarithmic
            grid between 0.1 and 100 au is used.
        filter_sigma : float, None
            Width of the optional Gaussian filter that is applied to
            smooth away the Monte Carlo sampling noise. The width is
            in number of grid points, so a value of 1.0 works usually
            well if for example the number of grid points is 50 in
            both the mass and semi-major axis dimension. No filter is
            applied if the argument is set to ``None``.
        plot_file : str or None, optional
            If provided, the completeness map is saved to this file. If None,
            the plot is shown interactively.

        Returns
        -------
        Figure
            The Matplotlib ``Figure`` object that can be used
            for further adjustments of the plot.

        Notes
        -----
        - Completeness is defined as the fraction of simulations for which
          the total acceleration amplitude satisfies::

              accel / sigma_accel > n_sigma

        - The resulting map shows completeness (%) as a function of companion
          mass (in Jupiter masses) and semi-major axis.
        - This method assumes IMPLICITLY that the primary mass and astrometric
          setup are already configured in the object state.
        """

        self.print_section("Completeness map")

        if mass_points is None:
            # Grid points for companion mass (Msun)
            mass_points = np.linspace(0.001, 0.1, 50)

        if sma_points is None:
            # Grid points for semi-major axis (au)
            sma_points = 10.0 ** np.linspace(np.log10(0.1), np.log10(100.0), 50)

        compl_map = np.zeros((mass_points.size, sma_points.size))

        pbar = tqdm(total=mass_points.size * sma_points.size)

        for m2_idx, m2_item in enumerate(mass_points):
            for sma_idx, sma_item in enumerate(sma_points):
                for _ in range(n_samples):
                    self.epoch_astrom.simulate_data(
                        mass_1=self.primary_mass[0],
                        mass_2=m2_item,
                        sma=sma_item,
                        verbose=False,
                    )

                    least_sq = LeastSquares(epoch_astrometry=self.epoch_astrom)

                    least_sq.accel_9param(plot_file=None, verbose=False)

                    # Acceleration dmu/dt (mas/yr^2)
                    # Quadratic sum of the RA and Dec components
                    accel = np.sqrt(
                        least_sq.best_param[5] ** 2 + least_sq.best_param[6] ** 2
                    )

                    # Gradient with respect to the RA and Dec components
                    # So delta(a)/delta(a_RA) and delta(a)/delta(a_Dec)
                    # with a = sqrt(a_RA^2 + a_Dec^2)
                    grad_accel = np.array(
                        [least_sq.best_param[5] / accel, least_sq.best_param[6] / accel]
                    )

                    # Covariance matrix for acceleration in RA and Dec
                    cov_accel = least_sq.param_cov[5:7, 5:7]

                    # Propagate RA and Dec acceleration uncertainty into
                    # uncertainty on total acceleration, while folding in
                    # the covariances between the RA and Dec acceleration
                    sigma_accel = np.sqrt(grad_accel @ cov_accel @ grad_accel)

                    if accel / sigma_accel > n_sigma:
                        compl_map[m2_idx, sma_idx] += 1.0 / float(n_samples)

                pbar.update(1)

        # fits.writeto("test.fits", compl_map, overwrite=True)
        # compl_map = fits.getdata("test.fits")

        if filter_sigma is not None:
            # Apply Gaussian filter to smooth out Monte Carlo noise
            compl_map = gaussian_filter(compl_map, sigma=filter_sigma)

        fig, ax = plt.subplots(figsize=(5, 3))

        mass_jup = (mass_points * u.M_sun).to(u.M_jup)

        mesh = ax.pcolormesh(
            sma_points, mass_jup, 100.0 * compl_map, vmin=0.0, vmax=100.0
        )

        cbar = plt.colorbar(mesh, ax=ax)
        cbar.set_label("Completeness (%)", fontsize=12)

        ax.set_xlabel("Semi-major axis (au)", fontsize=12)
        ax.set_ylabel(r"Companion mass ($M_\mathrm{J}$)", fontsize=12)
        ax.set_xscale("log")
        ax.set_title(rf"${n_sigma}\sigma$ acceleration completeness", fontsize=10.0)

        if plot_file is None:
            plt.show()
        else:
            plt.savefig(plot_file)

        return fig
