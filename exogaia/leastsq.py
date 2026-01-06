"""
Module with the ``LeastSquares`` class.
"""

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# from matplotlib import cm
from matplotlib.gridspec import GridSpec
from matplotlib.colorbar import Colorbar

# from matplotlib.colors import Normalize
from matplotlib.figure import Figure

# from scipy.linalg import cho_factor, cho_solve
from tqdm.auto import tqdm
from typeguard import typechecked

from exogaia.core import ExoGaia
from exogaia.data import EpochAstrometry
from exogaia.models import BinaryModel, StarModel


class LeastSquares(ExoGaia):
    """
    Class for least-squares model fit of epoch astrometry.
    """

    @typechecked
    def __init__(self, epoch_astrometry: EpochAstrometry = None) -> None:
        """
        Parameters
        ----------
        epoch_astrometry : EpochAstrometry
            ``EpochAstrometry`` object that contains the data.

        Returns
        -------
        NoneType
            None
        """

        self.epoch_astrometry = epoch_astrometry
        self.primary_mass = epoch_astrometry.primary_mass
        self.data_table = epoch_astrometry.data_table
        self.ref_epoch = epoch_astrometry.ref_epoch
        self.time_start = epoch_astrometry.time_start
        self.time_end = epoch_astrometry.time_end
        self.inv_cov = np.diag(1.0 / self.data_table["centroid_pos_error_al"] ** 2)

    @typechecked
    def least_squares(
        self,
        design: np.ndarray,
        obs_pos: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Method for calculating a least-squares fit for a given
        design matrix and 1D position measurements.

        Parameters
        ----------
        design : np.ndarray
            Array with the design matrix for the linear fit.
        obs_pos : np.ndarray, None
            Array with the 1D astrometric measurements in mas. The
            position measurements are selected from the
            ``EpochAstrometry`` if  the argument is set to ``None``.
        verbose : bool
            Print some information (default: False).

        Returns
        -------
        np.ndarray
            Array with the best-fit model astrometry.
        np.ndarray
            Array with the best-fit parameters
        np.ndarray
            Array with the parameter uncertainties.
        float
            RUWE of the fit.
        """

        # Epoch astrometry data
        if obs_pos is None:
            obs_pos = self.data_table["centroid_pos_al"].to_numpy()

        obs_err = self.data_table["centroid_pos_error_al"].to_numpy()

        # Number of model parameters
        n_param = design.shape[1]

        # Normal equations components
        # a_matrix = design.T @ self.inv_cov @ design
        # b_matrix = design.T @ self.inv_cov @ obs_pos

        # Cholesky factorization
        # a_matrix = cho_fac cho_fac^T
        # cho_fac = cho_factor(a_matrix, lower=True)

        # Solves (a_matrix cho_fac) = b_matrix
        # best_param = cho_solve(cho_fac, b_matrix)
        # best_model = design @ best_param

        # Linear model: obs_pos = design x best_param
        # Minimize weighted chi^2: chi^2 = (y - A theta)^T C^-1 (y - A theta)
        # y = obs_pos, A = design, theta = params
        # Setting delta_chi^2/delta_theta = 0
        # Solution: theta = (A^T C^-1 A)^-1 A^T C^-1 y
        best_param = np.linalg.solve(
            design.T @ self.inv_cov @ design, design.T @ self.inv_cov @ obs_pos
        )
        best_model = design @ best_param

        # Fit residuals
        fit_res = obs_pos - best_model

        # Number of data points
        n_obs = len(obs_pos)

        # Number of degrees of freedom
        n_dof = n_obs - n_param

        # Reduced chi^2
        chi2_red = np.sum(fit_res**2 / obs_err**2) / n_dof

        # RUWE
        ruwe = np.sqrt(chi2_red)

        # F2 estimator, which obeys a normal distribution N(0,1)
        # See Equation 1 in Halbwachs et al. (2023)
        f2_stat = np.sqrt(9.0 * n_dof / 2.0) * (
            chi2_red ** (1.0 / 3.0) + 2.0 / (9.0 * n_dof) - 1
        )

        # Uncertainty inflation such that F2 will be zero
        # See Equation 2 in Halbwachs et al. (2023)
        infl_fact = np.sqrt(chi2_red / ((1.0 - 2.0 / (9.0 * n_dof)) ** 3))

        # Parameter covariances
        # param_cov = cho_solve(cho_fac, np.eye(a_matrix.shape[0]))
        # param_sig = np.sqrt(np.diag(param_cov))

        # Parameter covariances
        cov_matrix = np.linalg.inv(design.T @ self.inv_cov @ design)

        # Covariance inflation
        if infl_fact > 1.0:
            cov_matrix *= infl_fact

        # Uncorrelated uncertainties
        param_sig = np.sqrt(np.diag(cov_matrix))

        @typechecked
        def significance(
            param_1: float, param_2: float, sigma_1: float, sigma_2: float, rho: float
        ) -> float:
            """
            Compute the combined significance of two correlated
            parameters, taking into account their uncertainties
            and correlation.

            Parameters
            ----------
            param_1 : float
                The value of the first parameter.
            param_2 : float
                The value of the second parameter.
            sigma_1 : float
                The uncertainty of the first parameter.
            sigma_2 : float
                The uncertainty of the second parameter.
            rho : float
                The correlation coefficient between param_1 and param_2,
                must be between -1 and 1.

            Returns
            -------
            float
                The combined significance of the two parameters.
            """

            return (
                1
                / (sigma_1 * sigma_2)
                * np.sqrt(
                    (
                        param_1**2 * sigma_2**2
                        + param_2**2 * sigma_1**2
                        - 2 * param_1 * param_2 * rho * sigma_1 * sigma_2
                    )
                    / (1 - rho**2)
                )
            )

        # Significance of adding acceleration parameters
        # Calculated as: vector_magnitude / vector_uncertainty
        # See Equation 3 in Halbwachs et al. (2023)

        if n_param in [7, 9]:
            rho = cov_matrix[5][6] / (param_sig[5] * param_sig[6])

            sig_7param = significance(
                best_param[5], best_param[6], param_sig[5], param_sig[6], rho
            )

        else:
            sig_7param = None

        if n_param == 9:
            rho = cov_matrix[7][8] / (param_sig[7] * param_sig[8])

            sig_9param = significance(
                best_param[7], best_param[8], param_sig[7], param_sig[8], rho
            )

        else:
            sig_9param = None

        if verbose:
            print(f"Reduced chi^2 = {chi2_red:.3f}")
            print(f"RUWE = {ruwe:.3f}")
            print(f"F2 estimator = {f2_stat:.3f}")
            print(f"Inflation factor = {infl_fact:.3f}")

            if sig_7param is not None:
                print(f"Significance dmu/dt = {sig_7param:.3f}")

            if sig_9param is not None:
                print(f"Significance d^2mu/d^2t = {sig_9param:.3f}")

        return best_model, best_param, param_sig, ruwe

    @typechecked
    def singl_5param(
        self, plot_file: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Method for a least-squares fit of the the epoch astrometry
        with a 5-parameter model for a single star.

        Parameters
        ----------
        plot_file : str, None
            File name of the plot with the results. No plot
            is created when the argument is set to ``None``.

        Returns
        -------
        np.ndarray
            Array with the best-fit model astrometry.
        np.ndarray
            Array with the best-fit parameters
        np.ndarray
            Array with the parameter uncertainties.
        float
            RUWE of the fit.
        """

        self.print_section("Single star (5-parameters)")

        # Epoch astrometry data
        obs_time = self.data_table["obs_time_tcb"].to_numpy()
        obs_pos = self.data_table["centroid_pos_al"].to_numpy()
        obs_err = self.data_table["centroid_pos_error_al"].to_numpy()
        rel_yr = self.data_table["relative_time_year"].to_numpy()
        scan_ang = self.data_table["scan_pos_angle"].to_numpy()
        par_fac = self.data_table["parallax_factor_al"].to_numpy()

        # Design matrix for least-squares fit
        design = np.column_stack(
            [
                np.sin(scan_ang),
                np.cos(scan_ang),
                par_fac,
                rel_yr * np.sin(scan_ang),
                rel_yr * np.cos(scan_ang),
            ]
        )

        best_model, best_param, param_sig, ruwe = self.least_squares(design)

        residuals = obs_pos - best_model

        res_ra, res_dec = np.sin(scan_ang) * residuals, np.cos(scan_ang) * residuals

        print("\nBest-fit parameters:")
        print(f"   - RA = {best_param[0]:.3f} deg +/- {param_sig[0]:.3f} mas")
        print(f"   - Dec = {best_param[1]:.3f} deg +/- {param_sig[1]:.3f} mas")
        print(f"   - Parallax = {best_param[2]:.3f} +/- {param_sig[2]:.3f} mas")
        print(f"   - mu in RA = {best_param[3]:.3f} +/- {param_sig[3]:.3f} mas/yr")
        print(f"   - mu in Dec = {best_param[4]:.3f} +/- {param_sig[4]:.3f} mas/yr")

        star_model = StarModel(epoch_astrometry=self.epoch_astrometry)

        delta_ra_obs, delta_dec_obs, _ = star_model.calc_model(
            model_param=best_param, obs_time=None
        )

        time_full = np.linspace(self.time_start.jyear, self.time_end.jyear, 1000)
        delta_ra_full, delta_dec_full, _ = star_model.calc_model(
            model_param=best_param, obs_time=time_full
        )

        # Create plot with residuals

        if plot_file is not None:
            _, axs = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw={"wspace": -0.05})

            axs[0].set_aspect("equal", adjustable="box")

            axs[0].plot(
                delta_ra_full,
                delta_dec_full,
                ls="-",
                lw=1.0,
                marker="none",
                color="black",
                zorder=1,
            )

            axs[0].plot(
                delta_ra_obs[0] + res_ra[0],
                delta_dec_obs[0] + res_dec[0],
                ls="none",
                marker="s",
                ms=5.0,
                mew=1.2,
                color="tab:green",
                mec="black",
                zorder=3,
            )

            axs[0].plot(
                delta_ra_obs[1:-1] + res_ra[1:-1],
                delta_dec_obs[1:-1] + res_dec[1:-1],
                ls="none",
                marker="s",
                ms=5.0,
                mew=1.2,
                color="tab:purple",
                mec="black",
                zorder=2,
            )

            axs[0].plot(
                delta_ra_obs[-1] + res_ra[-1],
                delta_dec_obs[-1] + res_dec[-1],
                ls="none",
                marker="s",
                ms=5.0,
                mew=1.2,
                color="tab:red",
                mec="black",
                zorder=3,
            )

            axs[0].set_title(
                rf"RA = {best_param[0]:.3f} deg $\pm$ {param_sig[0]:.3f} mas" + "\n"
                rf"Dec = {best_param[1]:.3f} deg $\pm$ {param_sig[1]:.3f} mas" + "\n"
                rf"$\varpi$ = {best_param[2]:.3f} $\pm$ {param_sig[2]:.3f} mas" + "\n"
                rf"$\mu_\mathrm{{RA}}$ = {best_param[3]:.3f} $\pm$ {param_sig[3]:.3f} mas/yr"
                + "\n"
                rf"$\mu_\mathrm{{Dec}}$ = {best_param[4]:.3f} $\pm$ {param_sig[4]:.3f} mas/yr"
            )

            axs[1].errorbar(
                obs_time[0],
                residuals[0],
                yerr=obs_err[0],
                ls="none",
                marker="s",
                ms=5.0,
                mew=1.2,
                elinewidth=1.2,
                color="tab:green",
                ecolor="black",
                mec="black",
                zorder=2,
            )

            axs[1].errorbar(
                obs_time[1:-1],
                residuals[1:-1],
                yerr=obs_err[1:-1],
                ls="none",
                marker="s",
                ms=5.0,
                mew=1.2,
                elinewidth=1.2,
                color="tab:purple",
                ecolor="black",
                mec="black",
                zorder=1,
            )

            axs[1].errorbar(
                obs_time[-1],
                residuals[-1],
                yerr=obs_err[-1],
                ls="none",
                marker="s",
                ms=5.0,
                mew=1.2,
                elinewidth=1.2,
                color="tab:red",
                ecolor="black",
                mec="black",
                zorder=2,
            )

            axs[0].set_xlabel(r"$\Delta\alpha$ (mas)")
            axs[0].set_ylabel(r"$\Delta\delta$ (mas)")
            axs[0].invert_xaxis()

            axs[1].set_xlabel("Time (yr)")
            axs[1].set_ylabel("Residuals (mas)")
            axs[1].invert_xaxis()

            axs[1].text(
                0.03,
                0.92,
                f"RUWE = {ruwe:.3f}",
                ha="left",
                va="center",
                transform=axs[1].transAxes,
                fontsize=12,
            )

            plt.savefig(plot_file)

        return best_model, best_param, param_sig, ruwe

    @typechecked
    def accel_7param(
        self, plot_file: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Method for a least-squares fit of the the epoch astrometry
        with a 7-parameter model for a star with a proper motion
        acceleration.

        Parameters
        ----------
        plot_file : str, None
            File name of the plot with the results. No plot
            is created when the argument is set to ``None``.

        Returns
        -------
        np.ndarray
            Array with the best-fit model astrometry.
        np.ndarray
            Array with the best-fit parameters
        np.ndarray
            Array with the parameter uncertainties.
        float
            RUWE of the fit.
        """

        self.print_section("Binary star (7-parameters)")

        # Epoch astrometry data
        obs_time = self.data_table["obs_time_tcb"].to_numpy()
        obs_pos = self.data_table["centroid_pos_al"].to_numpy()
        obs_err = self.data_table["centroid_pos_error_al"].to_numpy()
        rel_yr = self.data_table["relative_time_year"].to_numpy()
        scan_ang = self.data_table["scan_pos_angle"].to_numpy()
        par_fac = self.data_table["parallax_factor_al"].to_numpy()

        # Design matrix for least-squares fit
        design = np.column_stack(
            [
                np.sin(scan_ang),
                np.cos(scan_ang),
                par_fac,
                rel_yr * np.sin(scan_ang),
                rel_yr * np.cos(scan_ang),
                0.5 * rel_yr**2 * np.sin(scan_ang),
                0.5 * rel_yr**2 * np.cos(scan_ang),
            ]
        )

        best_model, best_param, param_sig, ruwe = self.least_squares(design)

        residuals = obs_pos - best_model

        res_ra, res_dec = (
            np.sin(scan_ang) * residuals,
            np.cos(scan_ang) * residuals,
        )

        print("\nBest-fit parameters:")
        print(f"   - RA = {best_param[0]:.3f} deg +/- {param_sig[0]:.3f} mas")
        print(f"   - Dec = {best_param[1]:.3f} deg +/- {param_sig[1]:.3f} mas")
        print(f"   - Parallax = {best_param[2]:.3f} +/- {param_sig[2]:.3f} mas")
        print(f"   - mu in RA = {best_param[3]:.3f} +/- {param_sig[3]:.3f} mas/yr")
        print(f"   - mu in Dec = {best_param[4]:.3f} +/- {param_sig[4]:.3f} mas/yr")
        print(
            f"   - dmu/dt in RA = {best_param[5]:.3f} +/- {param_sig[5]:.3f} mas/yr^2"
        )
        print(
            f"   - dmu/dt in Dec = {best_param[6]:.3f} +/- {param_sig[6]:.3f} mas/yr^2"
        )

        # Stellar track

        star_model = StarModel(epoch_astrometry=self.epoch_astrometry)

        delta_ra_obs, delta_dec_obs, _ = star_model.calc_model(
            model_param=best_param, obs_time=None
        )

        time_full = np.linspace(self.time_start.jyear, self.time_end.jyear, 1000)
        delta_ra_full, delta_dec_full, _ = star_model.calc_model(
            model_param=best_param, obs_time=time_full
        )

        # Stellar track, without acceleration

        star_no_accel = StarModel(epoch_astrometry=self.epoch_astrometry)

        delta_ra_no_accel, delta_dec_no_accel, _ = star_no_accel.calc_model(
            model_param=best_param[:5], obs_time=None
        )

        delta_ra_no_accel_full, delta_dec_no_accel_full, _ = star_no_accel.calc_model(
            model_param=best_param[:5], obs_time=time_full
        )

        # Acceleration

        delta_ra_accel = delta_ra_full - delta_ra_no_accel_full
        delta_dec_accel = delta_dec_full - delta_dec_no_accel_full

        # Create plot with residuals

        if plot_file is not None:
            _, axs = plt.subplots(1, 3, figsize=(18, 4), gridspec_kw={"wspace": 0.25})

            # axs[0].set_aspect("equal", adjustable="box")
            # axs[1].set_aspect("equal", adjustable="box")

            axs[0].plot(
                delta_ra_full,
                delta_dec_full,
                ls="-",
                lw=1.5,
                marker="none",
                color="black",
                zorder=1,
            )

            axs[0].plot(
                delta_ra_obs[0] + res_ra[0],
                delta_dec_obs[0] + res_dec[0],
                ls="none",
                marker="s",
                ms=5.0,
                mew=1.2,
                color="tab:green",
                mec="black",
                zorder=3,
            )

            axs[0].plot(
                delta_ra_obs[1:-1] + res_ra[1:-1],
                delta_dec_obs[1:-1] + res_dec[1:-1],
                ls="none",
                marker="s",
                ms=5.0,
                mew=1.2,
                color="tab:purple",
                mec="black",
                zorder=2,
            )

            axs[0].plot(
                delta_ra_obs[-1] + res_ra[-1],
                delta_dec_obs[-1] + res_dec[-1],
                ls="none",
                marker="s",
                ms=5.0,
                mew=1.2,
                color="tab:red",
                mec="black",
                zorder=3,
            )

            axs[0].set_title(
                rf"RA = {best_param[0]:.3f} deg $\pm$ {param_sig[0]:.3f} mas" + "\n"
                rf"Dec = {best_param[1]:.3f} deg $\pm$ {param_sig[1]:.3f} mas" + "\n"
                rf"$\varpi$ = {best_param[2]:.3f} $\pm$ {param_sig[2]:.3f} mas" + "\n"
                rf"$\mu_\mathrm{{RA}}$ = {best_param[3]:.3f} $\pm$ {param_sig[3]:.3f} mas/yr"
                + "\n"
                rf"$\mu_\mathrm{{Dec}}$ = {best_param[4]:.3f} $\pm$ {param_sig[4]:.3f} mas/yr"
            )

            axs[0].invert_xaxis()
            axs[0].set_xlabel(r"$\Delta\alpha$ (mas)")
            axs[0].set_ylabel(r"$\Delta\delta$ (mas)")

            axs[1].plot(
                1e3 * delta_ra_accel,
                1e3 * delta_dec_accel,
                ls="-",
                lw=1.5,
                marker="none",
                color="black",
            )

            for i, res_item in enumerate(residuals):
                if i == 0:
                    color = "tab:green"
                    zorder = 3

                elif i == len(residuals) - 1:
                    color = "tab:red"
                    zorder = 3

                else:
                    color = "tab:purple"
                    zorder = 2

                x1 = (
                    delta_ra_obs[i]
                    - delta_ra_no_accel[i]
                    + np.sin(scan_ang[i]) * (res_item + obs_err[i])
                )
                x2 = (
                    delta_ra_obs[i]
                    - delta_ra_no_accel[i]
                    + np.sin(scan_ang[i]) * (res_item - obs_err[i])
                )
                y1 = (
                    delta_dec_obs[i]
                    - delta_dec_no_accel[i]
                    + np.cos(scan_ang[i]) * (res_item + obs_err[i])
                )
                y2 = (
                    delta_dec_obs[i]
                    - delta_dec_no_accel[i]
                    + np.cos(scan_ang[i]) * (res_item - obs_err[i])
                )

                axs[1].plot(
                    [1e3 * x1, 1e3 * x2],
                    [1e3 * y1, 1e3 * y2],
                    "-",
                    lw=1,
                    color=color,
                    zorder=zorder,
                )

            axs[1].plot(
                1e3 * (delta_ra_obs + res_ra - delta_ra_no_accel)[0],
                1e3 * (delta_dec_obs + res_dec - delta_dec_no_accel)[0],
                ls="none",
                marker="s",
                ms=5.0,
                mew=1.2,
                color="tab:green",
                mec="black",
                zorder=3,
            )

            axs[1].plot(
                1e3 * (delta_ra_obs + res_ra - delta_ra_no_accel)[1:-1],
                1e3 * (delta_dec_obs + res_dec - delta_dec_no_accel)[1:-1],
                ls="none",
                marker="s",
                ms=5.0,
                mew=1.2,
                color="tab:purple",
                mec="black",
                zorder=2,
            )

            axs[1].plot(
                1e3 * (delta_ra_obs + res_ra - delta_ra_no_accel)[-1],
                1e3 * (delta_dec_obs + res_dec - delta_dec_no_accel)[-1],
                ls="none",
                marker="s",
                ms=5.0,
                mew=1.2,
                color="tab:red",
                mec="black",
                zorder=3,
            )

            axs[1].set_title(
                rf"$\dot{{\mu}}_\mathrm{{RA}}$ = {1e3*best_param[5]:.3f} "
                rf"$\pm$ {1e3*param_sig[5]:.3f} $\mu$as/yr$^2$" + "\n"
                rf"$\dot{{\mu}}_\mathrm{{Dec}}$ = {1e3*best_param[6]:.3f} "
                rf"$\pm$ {1e3*param_sig[6]:.3f} $\mu$as/yr$^2$"
            )

            axs[1].invert_xaxis()
            axs[1].set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
            axs[1].set_ylabel(r"$\Delta\delta$ ($\mu$as)")

            axs[2].errorbar(
                obs_time[0],
                residuals[0],
                yerr=obs_err[0],
                ls="none",
                marker="s",
                ms=5.0,
                mew=1.2,
                elinewidth=1.2,
                color="tab:green",
                ecolor="black",
                mec="black",
                zorder=3,
            )

            axs[2].errorbar(
                obs_time[1:-1],
                residuals[1:-1],
                yerr=obs_err[1:-1],
                ls="none",
                marker="s",
                ms=5.0,
                mew=1.2,
                elinewidth=1.2,
                color="tab:purple",
                ecolor="black",
                mec="black",
                zorder=2,
            )

            axs[2].errorbar(
                obs_time[-1],
                residuals[-1],
                yerr=obs_err[-1],
                ls="none",
                marker="s",
                ms=5.0,
                mew=1.2,
                elinewidth=1.2,
                color="tab:red",
                ecolor="black",
                mec="black",
                zorder=3,
            )

            axs[2].set_xlabel("Time (yr)")
            axs[2].set_ylabel("Residuals (mas)")
            axs[2].text(
                0.04,
                0.92,
                f"RUWE = {ruwe:.3f}",
                ha="left",
                va="center",
                transform=axs[2].transAxes,
                fontsize=12,
            )

            plt.savefig(plot_file)

        return best_model, best_param, param_sig, ruwe

    @typechecked
    def accel_9param(
        self, plot_file: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Method for a least-squares fit of the the epoch astrometry
        with a 9-parameter model for a star with a proper motion
        acceleration and second-order derivative.

        Parameters
        ----------
        plot_file : str, None
            File name of the plot with the results. No plot
            is created when the argument is set to ``None``.

        Returns
        -------
        np.ndarray
            Array with the best-fit model astrometry.
        np.ndarray
            Array with the best-fit parameters
        np.ndarray
            Array with the parameter uncertainties.
        float
            RUWE of the fit.
        """

        self.print_section("Binary star (9-parameters)")

        # Epoch astrometry data
        obs_time = self.data_table["obs_time_tcb"].to_numpy()
        obs_pos = self.data_table["centroid_pos_al"].to_numpy()
        obs_err = self.data_table["centroid_pos_error_al"].to_numpy()
        rel_yr = self.data_table["relative_time_year"].to_numpy()
        scan_ang = self.data_table["scan_pos_angle"].to_numpy()
        par_fac = self.data_table["parallax_factor_al"].to_numpy()

        # Design matrix for least-squares fit
        design = np.column_stack(
            [
                np.sin(scan_ang),
                np.cos(scan_ang),
                par_fac,
                rel_yr * np.sin(scan_ang),
                rel_yr * np.cos(scan_ang),
                0.5 * rel_yr**2 * np.sin(scan_ang),
                0.5 * rel_yr**2 * np.cos(scan_ang),
                (1.0 / 6.0) * rel_yr**3 * np.sin(scan_ang),
                (1.0 / 6.0) * rel_yr**3 * np.cos(scan_ang),
            ]
        )

        best_model, best_param, param_sig, ruwe = self.least_squares(design)

        residuals = obs_pos - best_model

        res_ra, res_dec = (
            np.sin(scan_ang) * residuals,
            np.cos(scan_ang) * residuals,
        )

        print("\nBest-fit parameters:")
        print(f"   - RA = {best_param[0]:.3f} deg +/- {param_sig[0]:.3f} mas")
        print(f"   - Dec = {best_param[1]:.3f} deg +/- {param_sig[1]:.3f} mas")
        print(f"   - Parallax = {best_param[2]:.3f} +/- {param_sig[2]:.3f} mas")
        print(f"   - mu in RA = {best_param[3]:.3f} +/- {param_sig[3]:.3f} mas/yr")
        print(f"   - mu in Dec = {best_param[4]:.3f} +/- {param_sig[4]:.3f} mas/yr")
        print(
            f"   - dmu/dt in RA = {best_param[5]:.3f} +/- {param_sig[5]:.3f} mas/yr^2"
        )
        print(
            f"   - dmu/dt in Dec = {best_param[6]:.3f} +/- {param_sig[6]:.3f} mas/yr^2"
        )
        print(
            f"   - d^2mu/d^2t in RA = {best_param[7]:.3f} +/- {param_sig[7]:.3f} mas/yr^3"
        )
        print(
            f"   - d^2mu/d^2t in Dec = {best_param[8]:.3f} +/- {param_sig[8]:.3f} mas/yr^3"
        )

        # Stellar track

        star_model = StarModel(epoch_astrometry=self.epoch_astrometry)

        delta_ra_obs, delta_dec_obs, _ = star_model.calc_model(
            model_param=best_param, obs_time=None
        )

        time_full = np.linspace(self.time_start.jyear, self.time_end.jyear, 1000)
        delta_ra_full, delta_dec_full, _ = star_model.calc_model(
            model_param=best_param, obs_time=time_full
        )

        # Stellar track, without acceleration

        star_no_accel = StarModel(epoch_astrometry=self.epoch_astrometry)

        delta_ra_no_accel, delta_dec_no_accel, _ = star_no_accel.calc_model(
            model_param=best_param[:5], obs_time=None
        )

        delta_ra_no_accel_full, delta_dec_no_accel_full, _ = star_no_accel.calc_model(
            model_param=best_param[:5], obs_time=time_full
        )

        # Acceleration

        delta_ra_accel = delta_ra_full - delta_ra_no_accel_full
        delta_dec_accel = delta_dec_full - delta_dec_no_accel_full

        # Create plot with residuals

        if plot_file is not None:
            _, axs = plt.subplots(1, 3, figsize=(18, 4), gridspec_kw={"wspace": 0.25})

            # axs[0].set_aspect("equal", adjustable="box")
            # axs[1].set_aspect("equal", adjustable="box")

            axs[0].plot(
                delta_ra_full,
                delta_dec_full,
                ls="-",
                lw=1.5,
                marker="none",
                color="black",
            )

            axs[0].plot(
                delta_ra_obs[0] + res_ra[0],
                delta_dec_obs[0] + res_dec[0],
                ls="none",
                marker="s",
                ms=5.0,
                mew=1.2,
                color="tab:green",
                mec="black",
                zorder=3,
            )

            axs[0].plot(
                delta_ra_obs[1:-1] + res_ra[1:-1],
                delta_dec_obs[1:-1] + res_dec[1:-1],
                ls="none",
                marker="s",
                ms=5.0,
                mew=1.2,
                color="tab:purple",
                mec="black",
                zorder=2,
            )

            axs[0].plot(
                delta_ra_obs[-1] + res_ra[-1],
                delta_dec_obs[-1] + res_dec[-1],
                ls="none",
                marker="s",
                ms=5.0,
                mew=1.2,
                color="tab:red",
                mec="black",
                zorder=3,
            )

            axs[0].set_title(
                rf"RA = {best_param[0]:.3f} deg $\pm$ {param_sig[0]:.3f} mas" + "\n"
                rf"Dec = {best_param[1]:.3f} deg $\pm$ {param_sig[1]:.3f} mas" + "\n"
                rf"$\varpi$ = {best_param[2]:.3f} $\pm$ {param_sig[2]:.3f} mas" + "\n"
                rf"$\mu_\mathrm{{RA}}$ = {best_param[3]:.3f} $\pm$ {param_sig[3]:.3f} mas/yr"
                + "\n"
                rf"$\mu_\mathrm{{Dec}}$ = {best_param[4]:.3f} $\pm$ {param_sig[4]:.3f} mas/yr"
            )

            axs[0].invert_xaxis()
            axs[0].set_xlabel(r"$\Delta\alpha$ (mas)")
            axs[0].set_ylabel(r"$\Delta\delta$ (mas)")

            axs[1].plot(
                1e3 * delta_ra_accel,
                1e3 * delta_dec_accel,
                ls="-",
                lw=1.5,
                marker="none",
                color="black",
            )

            for i, res_item in enumerate(residuals):
                if i == 0:
                    color = "tab:green"
                    zorder = 3

                elif i == len(residuals) - 1:
                    color = "tab:red"
                    zorder = 3

                else:
                    color = "tab:purple"
                    zorder = 2

                x1 = (
                    delta_ra_obs[i]
                    - delta_ra_no_accel[i]
                    + np.sin(scan_ang[i]) * (res_item + obs_err[i])
                )
                x2 = (
                    delta_ra_obs[i]
                    - delta_ra_no_accel[i]
                    + np.sin(scan_ang[i]) * (res_item - obs_err[i])
                )
                y1 = (
                    delta_dec_obs[i]
                    - delta_dec_no_accel[i]
                    + np.cos(scan_ang[i]) * (res_item + obs_err[i])
                )
                y2 = (
                    delta_dec_obs[i]
                    - delta_dec_no_accel[i]
                    + np.cos(scan_ang[i]) * (res_item - obs_err[i])
                )

                axs[1].plot(
                    [1e3 * x1, 1e3 * x2],
                    [1e3 * y1, 1e3 * y2],
                    "-",
                    lw=1,
                    color=color,
                    zorder=zorder,
                )

            axs[1].plot(
                1e3 * (delta_ra_obs + res_ra - delta_ra_no_accel)[0],
                1e3 * (delta_dec_obs + res_dec - delta_dec_no_accel)[0],
                ls="none",
                marker="s",
                ms=5.0,
                mew=1.2,
                color="tab:green",
                mec="black",
                zorder=3,
            )

            axs[1].plot(
                1e3 * (delta_ra_obs + res_ra - delta_ra_no_accel)[1:-1],
                1e3 * (delta_dec_obs + res_dec - delta_dec_no_accel)[1:-1],
                ls="none",
                marker="s",
                ms=5.0,
                mew=1.2,
                color="tab:purple",
                mec="black",
                zorder=2,
            )

            axs[1].plot(
                1e3 * (delta_ra_obs + res_ra - delta_ra_no_accel)[-1],
                1e3 * (delta_dec_obs + res_dec - delta_dec_no_accel)[-1],
                ls="none",
                marker="s",
                ms=5.0,
                mew=1.2,
                color="tab:red",
                mec="black",
                zorder=3,
            )

            axs[1].set_title(
                rf"$\dot{{\mu}}_\mathrm{{RA}}$ = {1e3*best_param[5]:.3f} "
                rf"$\pm$ {1e3*param_sig[5]:.3f} $\mu$as/yr$^2$" + "\n"
                rf"$\dot{{\mu}}_\mathrm{{Dec}}$ = {1e3*best_param[6]:.3f} "
                rf"$\pm$ {1e3*param_sig[6]:.3f} $\mu$as/yr$^2$" + "\n"
                rf"$\ddot{{\mu}}_\mathrm{{RA}}$ = {1e3*best_param[7]:.3f} "
                rf"$\pm$ {1e3*param_sig[7]:.3f} $\mu$as/yr$^3$" + "\n"
                rf"$\ddot{{\mu}}_\mathrm{{Dec}}$ = {1e3*best_param[8]:.3f} "
                rf"$\pm$ {1e3*param_sig[8]:.3f} $\mu$as/yr$^3$"
            )

            axs[1].invert_xaxis()
            axs[1].set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
            axs[1].set_ylabel(r"$\Delta\delta$ ($\mu$as)")

            axs[2].errorbar(
                obs_time[0],
                residuals[0],
                yerr=obs_err[0],
                ls="none",
                marker="s",
                ms=5.0,
                mew=1.2,
                elinewidth=1.2,
                color="tab:green",
                ecolor="black",
                mec="black",
                zorder=3,
            )

            axs[2].errorbar(
                obs_time[1:-1],
                residuals[1:-1],
                yerr=obs_err[1:-1],
                ls="none",
                marker="s",
                ms=5.0,
                mew=1.2,
                elinewidth=1.2,
                color="tab:purple",
                ecolor="black",
                mec="black",
                zorder=2,
            )

            axs[2].errorbar(
                obs_time[-1],
                residuals[-1],
                yerr=obs_err[-1],
                ls="none",
                marker="s",
                ms=5.0,
                mew=1.2,
                elinewidth=1.2,
                color="tab:red",
                ecolor="black",
                mec="black",
                zorder=3,
            )

            axs[2].set_xlabel("Time (yr)")
            axs[2].set_ylabel("Residuals (mas)")
            axs[2].text(
                0.04,
                0.92,
                f"RUWE = {ruwe:.3f}",
                ha="left",
                va="center",
                transform=axs[2].transAxes,
                fontsize=12,
            )

            plt.savefig(plot_file)

        return best_model, best_param, param_sig, ruwe

    def orbit_grid(self, plot_file: Optional[str] = None) -> Figure:
        """
        Method for exploring a grid of orbits of varying semi-major
        axis, eccentricity, and time of periastron. The semi-major
        axis is converted in a period, using the ``primary_mass``
        from the ``EpochAstrometry`` and ignoring the secondary mass.
        For each combination, the Kepler equation is solved. The other
        parameters (RA, Dec, parallax, proper motion, Thiele-Innes
        constants) are all linear and optimized with a least-squares.
        The output plot shows the RUWE for each semi-major axis and
        eccentricity pair, with the time of periastron selected that
        minimizes the RUWE.

        Parameters
        ----------
        plot_file : str, None
            File name of the plot with the results. No plot
            is created when the argument is set to ``None``.

        Returns
        -------
        Figure
            Matplotlib ``Figure`` object.
        """

        self.print_section("Orbit (12-parameters)")

        # Epoch astrometry data
        # obs_time = self.data_table["obs_time_tcb"].to_numpy()
        obs_pos = self.data_table["centroid_pos_al"].to_numpy()
        # obs_err = self.data_table["centroid_pos_error_al"].to_numpy()
        rel_yr = self.data_table["relative_time_year"].to_numpy()
        scan_ang = self.data_table["scan_pos_angle"].to_numpy()
        par_fac = self.data_table["parallax_factor_al"].to_numpy()

        # Grid for the log10(a/au)
        loga_list = np.linspace(np.log10(0.1), np.log10(30.0), 50)

        # Grid for the eccentricity
        ecc_list = np.linspace(0.0, 1.0, 50, endpoint=False)

        # Grid for the relative time of periastron
        tau_list = np.linspace(0.0, 0.1, 50, endpoint=False)

        binary_model = BinaryModel(
            epoch_astrometry=self.epoch_astrometry, verbose=False
        )

        ruwe_grid = np.zeros((loga_list.size, ecc_list.size, tau_list.size))
        tau_grid = np.zeros((loga_list.size, ecc_list.size, tau_list.size))

        global_ruwe = np.inf
        # global_model = None
        global_param = None
        global_sigma = None
        global_orbit = None

        pbar = tqdm(total=len(loga_list) * len(ecc_list) * len(tau_list))

        for loga_idx, loga_item in enumerate(loga_list):
            for ecc_idx, ecc_item in enumerate(ecc_list):
                for tau_idx, tau_item in enumerate(tau_list):
                    # Semi-major axis (au)
                    sma = 10.0**loga_item

                    # Primary mass, ignore secondary mass (Msun)
                    m1 = self.primary_mass[0]
                    m2 = 0.0

                    # Orbital period (days)
                    period = np.sqrt(sma**3 / (m1 + m2)) * 365.25

                    # Solve the Kepler equation
                    x_orb, y_orb = binary_model.solve_kepler(period, ecc_item, tau_item)

                    # Linear least-squares fit

                    design = np.column_stack(
                        [
                            np.sin(scan_ang),
                            np.cos(scan_ang),
                            par_fac,
                            rel_yr * np.sin(scan_ang),
                            rel_yr * np.cos(scan_ang),
                            x_orb * np.sin(scan_ang),  # Thiele-Innes B
                            y_orb * np.sin(scan_ang),  # Thiele-Innes G
                            x_orb * np.cos(scan_ang),  # Thiele-Innes A
                            y_orb * np.cos(scan_ang),  # Thiele-Innes F
                        ]
                    )

                    best_model, best_param, param_sig, ruwe = self.least_squares(
                        design, obs_pos=obs_pos, verbose=False
                    )

                    if ruwe < global_ruwe:
                        global_ruwe = ruwe
                        # global_model = best_model
                        global_param = best_param
                        global_sigma = param_sig
                        global_orbit = [sma, ecc_item, tau_item]

                    # x_sky = best_param[0] * x_orb + best_param[1] * y_orb
                    # y_sky = best_param[2] * x_orb + best_param[3] * y_orb
                    # plt.plot(x_sky, y_sky, 'o')
                    # plt.plot(x_sky + (best_model-obs_pos)*np.sin(scan_ang),
                    #          y_sky + (best_model-obs_pos)*np.cos(scan_ang), 'o')
                    # plt.show()

                    # Store RUWE as goodness-of-fit statistics
                    ruwe_grid[loga_idx, ecc_idx, tau_idx] = ruwe

                    # Store tau for contour plot
                    tau_grid[loga_idx, ecc_idx, tau_idx] = tau_item

            pbar.update(len(ecc_list) * len(tau_list))

        # fig, ax = plt.subplots(figsize=(7, 3))
        #
        # cmap = cm.viridis
        # norm = Normalize(vmin=0.0, vmax=1.0)
        #
        # ax.errorbar(
        #     obs_time,
        #     obs_pos-global_model,
        #     yerr=obs_err,
        #     ls="none",
        #     marker="s",
        #     color="black",
        #     ms=1.0,
        # )
        #
        # sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        # plt.colorbar(sm, ax=ax)
        # plt.savefig("test.png")
        # plt.close()

        print(f"Best-fit parameters (RUWE = {global_ruwe:.3f}):")
        print(f"   - RA = {global_param[0]:.3f} deg +/- {global_sigma[0]:.3f} mas")
        print(f"   - Dec = {global_param[1]:.3f} deg +/- {global_sigma[1]:.3f} mas")
        print(f"   - Parallax = {global_param[2]:.3f} +/- {global_sigma[2]:.3f} mas")
        print(f"   - mu in RA = {global_param[3]:.3f} +/- {global_sigma[3]:.3f} mas/yr")
        print(
            f"   - mu in Dec = {global_param[4]:.3f} +/- {global_sigma[4]:.3f} mas/yr"
        )
        print(f"   - Thiele-Innes B = {global_param[5]:.3f} +/- {global_sigma[5]:.3f}")
        print(f"   - Thiele-Innes G = {global_param[6]:.3f} +/- {global_sigma[6]:.3f}")
        print(f"   - Thiele-Innes A = {global_param[7]:.3f} +/- {global_sigma[7]:.3f}")
        print(f"   - Thiele-Innes F = {global_param[8]:.3f} +/- {global_sigma[8]:.3f}")
        print(f"   - Semi-major axis = {global_orbit[0]:.3f} au")
        print(f"   - Eccentricity = {global_orbit[1]:.3f}")
        print(f"   - Relative time of periastron = {global_orbit[2]:.2f}")

        # Select minimum RUWE along the 3rd axis to create a 2D array

        ruwe_grid_2d = np.nanmin(ruwe_grid, axis=2)

        # Select indices with the minimum RUWE along the 3rd axis

        min_idx = np.argmin(ruwe_grid, axis=2)

        # Create a grid with the best-fit tau for each sma-ecc pair

        tau_best = np.zeros((loga_list.size, ecc_list.size))
        for loga_idx, loga_item in enumerate(loga_list):
            for ecc_idx, ecc_item in enumerate(ecc_list):
                tau_idx = min_idx[loga_idx, ecc_idx]
                tau_best[loga_idx, ecc_idx] = tau_grid[loga_idx, ecc_idx, tau_idx]

        # Create goodness-of-fit plot

        fig = plt.figure(figsize=(4, 3))

        grid_spec = GridSpec(1, 2, width_ratios=[4.0, 0.25])
        grid_spec.update(wspace=0.07, hspace=0, left=0, right=1, bottom=0, top=1)

        ax = plt.subplot(grid_spec[0, 0])
        ax_cb = plt.subplot(grid_spec[0, 1])

        x_grid, y_grid = np.meshgrid(10.0**loga_list, ecc_list)

        # Transpose to make eccentricity rows and semi-major axis columns

        c = ax.contourf(x_grid, y_grid, ruwe_grid_2d.T, levels=30)

        cb = Colorbar(
            ax=ax_cb,
            mappable=c,
            orientation="vertical",
            ticklocation="right",
            format="%.2f",
        )

        cb.ax.minorticks_on()

        cb.ax.tick_params(
            which="major",
            width=0.8,
            length=5,
            labelsize=12,
            direction="in",
            color="black",
        )

        cb.ax.set_ylabel(
            "RUWE",
            rotation=270,
            labelpad=22,
            fontsize=13.0,
        )

        # Transpose to make eccentricity rows and semi-major axis columns

        # cs = ax.contour(
        #     x_grid,
        #     y_grid,
        #     tau_best.T,
        #     levels=10,
        #     colors="white",
        #     linewidths=0.7,
        # )
        #
        # ax.clabel(cs, cs.levels, inline=True, fontsize=8, fmt="%1.1f")

        ax.set_xlabel(r"$\log{a/\mathrm{au}}$")
        ax.set_ylabel(r"Eccentricity")
        ax.set_xscale("log")

        if plot_file is None:
            plt.show()
        else:
            plt.savefig(plot_file)

        return fig
