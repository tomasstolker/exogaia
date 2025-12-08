"""
Module with the ``exogaia`` tool.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from scipy.linalg import cho_factor, cho_solve
from typeguard import typechecked

from exogaia.core import ExoGaia
from exogaia.data import EpochAstrometry
from exogaia.models import StarModel


class LeastSquares(ExoGaia):
    """
    Class for least-squares fit of epoch astrometry.
    """

    @typechecked
    def __init__(self, epoch_astrometry: EpochAstrometry = None) -> None:
        """
        Returns
        -------
        NoneType
            None
        """

        self.epoch_astrometry = epoch_astrometry
        self.data_table = epoch_astrometry.data_table
        self.ref_epoch = epoch_astrometry.ref_epoch
        self.time_start = epoch_astrometry.time_start
        self.time_end = epoch_astrometry.time_end
        self.inv_cov = np.diag(1.0 / self.data_table["centroid_pos_error_al"] ** 2)

    def least_squares(self, design: np.ndarray):
        """
        Method for calculating the least-squares for an input design
        matrix and the Gaia epoch astrometry.
        """

        # Epoch astrometry data
        obs_pos = self.data_table["centroid_pos_al"]
        obs_err = self.data_table["centroid_pos_error_al"]

        # Number of model parameters
        n_param = design.shape[1]

        # Normal equations components
        a_matrix = design.T @ self.inv_cov @ design
        b_matrix = design.T @ self.inv_cov @ obs_pos

        # Normal equations components
        a_matrix = design.T @ self.inv_cov @ design
        b_matrix = design.T @ self.inv_cov @ obs_pos

        # Cholesky factorization
        # a_matrix = cho_fac cho_fac^T
        cho_fac = cho_factor(a_matrix, lower=True)

        # Solves (a_matrix cho_fac) = b_matrix
        best_param = cho_solve(cho_fac, b_matrix)
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

        # F2 goodness-of fit statistic from Wilson–Hilferty transformation
        # See Eq. 11 in El-Badry et al. (2024)
        infl_fact = np.sqrt(chi2_red / ((1.0 - 2.0 / (9.0 * n_dof)) ** 3))

        # Parameter covariances
        param_cov = cho_solve(cho_fac, np.eye(a_matrix.shape[0]))
        param_sig = infl_fact * np.sqrt(np.diag(param_cov))

        print(f"Reduced chi^2 = {chi2_red:.2f}")
        print(f"RUWE = {ruwe:.2f}")
        print(f"Inflation factor = {infl_fact:.2f}")

        return best_model, best_param, param_sig, ruwe

    @typechecked
    def singl_5param(self, plot_residuals: Optional[str] = None):
        """
        Fit epoch astrometry with a 5-parameter model.
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
        print(f"   - ref. RA (deg) = {best_param[0]:.3f} +/- {param_sig[0]:.3f}")
        print(f"   - ref. Dec (deg) = {best_param[1]:.3f} +/- {param_sig[1]:.3f}")
        print(f"   - Parallax (mas) = {best_param[2]:.3f} +/- {param_sig[2]:.3f}")
        print(f"   - mu in RA (mas/yr) = {best_param[3]:.3f} +/- {param_sig[3]:.3f}")
        print(f"   - mu in Dec (mas/yr) = {best_param[4]:.3f} +/- {param_sig[4]:.3f}")

        star_model = StarModel(
            star_param=best_param, epoch_astrometry=self.epoch_astrometry
        )

        delta_ra_obs, delta_dec_obs = star_model.calc_model(obs_time=None)

        time_full = np.linspace(self.time_start.jyear, self.time_end.jyear, 1000)
        delta_ra_full, delta_dec_full = star_model.calc_model(obs_time=time_full)

        # Create plot with residuals

        if plot_residuals is not None:
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
                rf"$\varpi$ = {best_param[2]:.2f} $\pm$ {param_sig[2]:.2f} mas/yr"
                + "\n"
                rf"$\mu_\mathrm{{RA}}$ = {best_param[3]:.2f} $\pm$ {param_sig[3]:.2f} mas/yr"
                + "\n"
                rf"$\mu_\mathrm{{Dec}}$ = {best_param[4]:.2f} $\pm$ {param_sig[4]:.2f} mas/yr"
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

            axs[0].set_xlabel(r"$\alpha$ (deg)")
            axs[0].set_ylabel(r"$\delta$ (deg)")
            axs[1].set_xlabel("Time (yr)")
            axs[1].set_ylabel("Residuals (mas)")
            axs[1].text(
                0.03,
                0.92,
                f"RUWE = {ruwe:.2f}",
                ha="left",
                va="center",
                transform=axs[1].transAxes,
                fontsize=12,
            )

            plt.savefig(plot_residuals)

        return best_model, best_param, param_sig

    @typechecked
    def accel_7param(self, plot_residuals: Optional[str] = None):
        """
        Fit epoch astrometry with a 7-parameter model.
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
        print(f"   - ref. RA (deg) = {best_param[0]:.3f} +/- {param_sig[0]:.3f}")
        print(f"   - ref. Dec (mas) = {best_param[1]:.3f} +/- {param_sig[1]:.3f}")
        print(f"   - Parallax (mas) = {best_param[2]:.3f} +/- {param_sig[2]:.3f}")
        print(f"   - mu in RA (mas/yr) = {best_param[3]:.3f} +/- {param_sig[3]:.3f}")
        print(f"   - mu in Dec (mas/yr) = {best_param[4]:.3f} +/- {param_sig[4]:.3f}")
        print(
            f"   - dmu/dt in RA (mas/yr^2) = {best_param[5]:.3f} +/- {param_sig[5]:.3f}"
        )
        print(
            f"   - dmu/dt in Dec (mas/yr^2) = {best_param[6]:.3f} +/- {param_sig[6]:.3f}"
        )

        # Stellar track

        star_model = StarModel(
            star_param=best_param, epoch_astrometry=self.epoch_astrometry
        )

        delta_ra_obs, delta_dec_obs = star_model.calc_model(obs_time=None)

        time_full = np.linspace(self.time_start.jyear, self.time_end.jyear, 1000)
        delta_ra_full, delta_dec_full = star_model.calc_model(obs_time=time_full)

        # Stellar track, without acceleration

        star_no_accel = StarModel(
            star_param=best_param[:5], epoch_astrometry=self.epoch_astrometry
        )

        delta_ra_no_accel, delta_dec_no_accel = star_no_accel.calc_model(obs_time=None)

        delta_ra_no_accel_full, delta_dec_no_accel_full = star_no_accel.calc_model(
            obs_time=time_full
        )

        # Acceleration

        delta_ra_accel = delta_ra_full - delta_ra_no_accel_full
        delta_dec_accel = delta_dec_full - delta_dec_no_accel_full

        # Create plot with residuals

        if plot_residuals is not None:
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
                rf"$\varpi$ = {best_param[2]:.2f} $\pm$ {param_sig[2]:.2f} mas/yr"
                + "\n"
                rf"$\mu_\mathrm{{RA}}$ = {best_param[3]:.2f} $\pm$ {param_sig[3]:.2f} mas/yr"
                + "\n"
                rf"$\mu_\mathrm{{Dec}}$ = {best_param[4]:.2f} $\pm$ {param_sig[4]:.2f} mas/yr"
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
                rf"$\dot{{\mu}}_\mathrm{{RA}}$ = {1e3*best_param[5]:.2f} "
                rf"$\pm$ {1e3*param_sig[5]:.2f} $\mu$as/yr$^2$" + "\n"
                rf"$\dot{{\mu}}_\mathrm{{Dec}}$ = {1e3*best_param[6]:.2f} "
                rf"$\pm$ {1e3*param_sig[6]:.2f} $\mu$as/yr$^2$"
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
                f"RUWE = {ruwe:.2f}",
                ha="left",
                va="center",
                transform=axs[2].transAxes,
                fontsize=12,
            )

            plt.savefig(plot_residuals)

        return best_model, best_param, param_sig

    @typechecked
    def accel_9param(self, plot_residuals: Optional[str] = None):
        """
        Fit epoch astrometry with a 9-parameter model.
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
        print(f"   - ref. RA (deg) = {best_param[0]:.3f} +/- {param_sig[0]:.3f}")
        print(f"   - ref. Dec (deg) = {best_param[1]:.3f} +/- {param_sig[1]:.3f}")
        print(f"   - Parallax (mas) = {best_param[2]:.3f} +/- {param_sig[2]:.3f}")
        print(f"   - mu in RA (mas/yr) = {best_param[3]:.3f} +/- {param_sig[3]:.3f}")
        print(f"   - mu in Dec (mas/yr) = {best_param[4]:.3f} +/- {param_sig[4]:.3f}")
        print(
            f"   - dmu/dt in RA (mas/yr^2) = {best_param[5]:.3f} +/- {param_sig[5]:.3f}"
        )
        print(
            f"   - dmu/dt in Dec (mas/yr^2) = {best_param[6]:.3f} +/- {param_sig[6]:.3f}"
        )
        print(
            f"   - d^2mu/d^2t in RA (mas/yr^3) = {best_param[7]:.3f} +/- {param_sig[7]:.3f}"
        )
        print(
            f"   - d^2mu/d^2t in Dec (mas/yr^3) = {best_param[8]:.3f} +/- {param_sig[8]:.3f}"
        )

        # Stellar track

        star_model = StarModel(
            star_param=best_param, epoch_astrometry=self.epoch_astrometry
        )

        delta_ra_obs, delta_dec_obs = star_model.calc_model(obs_time=None)

        time_full = np.linspace(self.time_start.jyear, self.time_end.jyear, 1000)
        delta_ra_full, delta_dec_full = star_model.calc_model(obs_time=time_full)

        # Stellar track, without acceleration

        star_no_accel = StarModel(
            star_param=best_param[:5], epoch_astrometry=self.epoch_astrometry
        )

        delta_ra_no_accel, delta_dec_no_accel = star_no_accel.calc_model(obs_time=None)

        delta_ra_no_accel_full, delta_dec_no_accel_full = star_no_accel.calc_model(
            obs_time=time_full
        )

        # Acceleration

        delta_ra_accel = delta_ra_full - delta_ra_no_accel_full
        delta_dec_accel = delta_dec_full - delta_dec_no_accel_full

        # Create plot with residuals

        if plot_residuals is not None:
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
                rf"$\varpi$ = {best_param[2]:.2f} $\pm$ {param_sig[2]:.2f} mas/yr"
                + "\n"
                rf"$\mu_\mathrm{{RA}}$ = {best_param[3]:.2f} $\pm$ {param_sig[3]:.2f} mas/yr"
                + "\n"
                rf"$\mu_\mathrm{{Dec}}$ = {best_param[4]:.2f} $\pm$ {param_sig[4]:.2f} mas/yr"
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
                rf"$\dot{{\mu}}_\mathrm{{RA}}$ = {1e3*best_param[5]:.2f} "
                rf"$\pm$ {1e3*param_sig[5]:.2f} $\mu$as/yr$^2$" + "\n"
                rf"$\dot{{\mu}}_\mathrm{{Dec}}$ = {1e3*best_param[6]:.2f} "
                rf"$\pm$ {1e3*param_sig[6]:.2f} $\mu$as/yr$^2$" + "\n"
                rf"$\ddot{{\mu}}_\mathrm{{RA}}$ = {1e3*best_param[7]:.2f} "
                rf"$\pm$ {1e3*param_sig[7]:.2f} $\mu$as/yr$^3$" + "\n"
                rf"$\ddot{{\mu}}_\mathrm{{Dec}}$ = {1e3*best_param[8]:.2f} "
                rf"$\pm$ {1e3*param_sig[8]:.2f} $\mu$as/yr$^3$"
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
                f"RUWE = {ruwe:.2f}",
                ha="left",
                va="center",
                transform=axs[2].transAxes,
                fontsize=12,
            )

            plt.savefig(plot_residuals)

        return best_model, best_param, param_sig
