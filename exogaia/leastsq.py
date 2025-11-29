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

        self.data_table = epoch_astrometry.data_table
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

        # Number of degrees of freedomg
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

        return best_model, best_param, param_sig

    @typechecked
    def singl_5param(self, plot_residuals: Optional[str] = None):
        """
        Fit epoch astrometry with a 5-parameter model.
        """

        self.print_section("Single star (5-parameters)")

        # Epoch astrometry data
        obs_yr = self.data_table["relative_time_year"]
        obs_pos = self.data_table["centroid_pos_al"]
        obs_err = self.data_table["centroid_pos_error_al"]

        # Design matrix for least-squares fit
        design = np.column_stack(
            [
                np.sin(self.data_table["scan_pos_angle"]),
                np.cos(self.data_table["scan_pos_angle"]),
                self.data_table["parallax_factor_al"],
                self.data_table["relative_time_year"]
                * np.sin(self.data_table["scan_pos_angle"]),
                self.data_table["relative_time_year"]
                * np.cos(self.data_table["scan_pos_angle"]),
            ]
        )

        best_model, best_param, param_sig = self.least_squares(design)

        print("\nBest-fit parameters:")
        print(f"   - ref. RA (mas) = {best_param[0]:.3f} +/- {param_sig[0]:.3f}")
        print(f"   - ref. Dec (mas) = {best_param[1]:.3f} +/- {param_sig[1]:.3f}")
        print(f"   - Parallax (mas) = {best_param[2]:.3f} +/- {param_sig[2]:.3f}")
        print(f"   - mu in RA (mas/yr) = {best_param[3]:.3f} +/- {param_sig[3]:.3f}")
        print(f"   - mu in Dec (mas/yr) = {best_param[4]:.3f} +/- {param_sig[4]:.3f}")

        # Create plot with residuals

        if plot_residuals is not None:
            plt.figure(figsize=(6, 3))
            plt.errorbar(
                obs_yr,
                obs_pos - best_model,
                yerr=obs_err,
                ls="none",
                marker="s",
                ms=5.0,
                mew=1.2,
                elinewidth=1.2,
                color="tab:purple",
                ecolor="black",
                mec="black",
            )
            plt.xlabel("Time (yr)")
            plt.ylabel("Residual")
            plt.savefig(plot_residuals)

        return best_model, best_param, param_sig

    @typechecked
    def accel_7param(self, plot_residuals: Optional[str] = None):
        """
        Fit epoch astrometry with a 7-parameter model.
        """

        self.print_section("Binary star (7-parameters)")

        # Epoch astrometry data
        obs_yr = self.data_table["relative_time_year"]
        obs_pos = self.data_table["centroid_pos_al"]
        obs_err = self.data_table["centroid_pos_error_al"]

        # Design matrix for least-squares fit
        design = np.column_stack(
            [
                np.sin(self.data_table["scan_pos_angle"]),
                np.cos(self.data_table["scan_pos_angle"]),
                self.data_table["parallax_factor_al"],
                self.data_table["relative_time_year"]
                * np.sin(self.data_table["scan_pos_angle"]),
                self.data_table["relative_time_year"]
                * np.cos(self.data_table["scan_pos_angle"]),
                0.5
                * self.data_table["relative_time_year"] ** 2
                * np.sin(self.data_table["scan_pos_angle"]),
                0.5
                * self.data_table["relative_time_year"] ** 2
                * np.cos(self.data_table["scan_pos_angle"]),
            ]
        )

        best_model, best_param, param_sig = self.least_squares(design)

        print("\nBest-fit parameters:")
        print(f"   - ref. RA (mas) = {best_param[0]:.3f} +/- {param_sig[0]:.3f}")
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

        # Create plot with residuals

        if plot_residuals is not None:
            plt.figure(figsize=(6, 3))
            plt.errorbar(
                obs_yr,
                obs_pos - best_model,
                yerr=obs_err,
                ls="none",
                marker="s",
                ms=5.0,
                mew=1.2,
                elinewidth=1.2,
                color="tab:purple",
                ecolor="black",
                mec="black",
            )
            plt.xlabel("Time (yr)")
            plt.ylabel("Residual")
            plt.savefig(plot_residuals)

        return best_model, best_param, param_sig

    @typechecked
    def accel_9param(self, plot_residuals: Optional[str] = None):
        """
        Fit epoch astrometry with a 9-parameter model.
        """

        self.print_section("Binary star (9-parameters)")

        # Epoch astrometry data
        obs_yr = self.data_table["relative_time_year"]
        obs_pos = self.data_table["centroid_pos_al"]
        obs_err = self.data_table["centroid_pos_error_al"]

        # Design matrix for least-squares fit
        design = np.column_stack(
            [
                np.sin(self.data_table["scan_pos_angle"]),
                np.cos(self.data_table["scan_pos_angle"]),
                self.data_table["parallax_factor_al"],
                self.data_table["relative_time_year"]
                * np.sin(self.data_table["scan_pos_angle"]),
                self.data_table["relative_time_year"]
                * np.cos(self.data_table["scan_pos_angle"]),
                0.5
                * self.data_table["relative_time_year"] ** 2
                * np.sin(self.data_table["scan_pos_angle"]),
                0.5
                * self.data_table["relative_time_year"] ** 2
                * np.cos(self.data_table["scan_pos_angle"]),
                (1.0 / 6.0)
                * self.data_table["relative_time_year"] ** 3
                * np.sin(self.data_table["scan_pos_angle"]),
                (1.0 / 6.0)
                * self.data_table["relative_time_year"] ** 3
                * np.cos(self.data_table["scan_pos_angle"]),
            ]
        )

        best_model, best_param, param_sig = self.least_squares(design)

        print("\nBest-fit parameters:")
        print(f"   - ref. RA (mas) = {best_param[0]:.3f} +/- {param_sig[0]:.3f}")
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
        print(
            f"   - d^2mu/d^2t in RA (mas/yr^3) = {best_param[7]:.3f} +/- {param_sig[7]:.3f}"
        )
        print(
            f"   - d^2mu/d^2t in Dec (mas/yr^3) = {best_param[8]:.3f} +/- {param_sig[8]:.3f}"
        )

        # Create plot with residuals

        if plot_residuals is not None:
            plt.figure(figsize=(6, 3))
            plt.errorbar(
                obs_yr,
                obs_pos - best_model,
                yerr=obs_err,
                ls="none",
                marker="s",
                ms=5.0,
                mew=1.2,
                elinewidth=1.2,
                color="tab:purple",
                ecolor="black",
                mec="black",
            )
            plt.xlabel("Time (yr)")
            plt.ylabel("Residual")
            plt.savefig(plot_residuals)

        return best_model, best_param, param_sig

    # def test_orbit(self):
    #     """
    #     Method for plotting the acceleration.
    #     """
    #
    #     best_model, param, _ = self.accel_9param()
    #     dt = self.data_table["relative_time_year"]
    #     dt -= dt[0]
    #     dra = 0.5 * param[4] * dt**2 + (1 / 6) * param[6] * dt**3
    #     ddec = 0.5 * param[5] * dt**2 + (1 / 6) * param[7] * dt**3
    #
    #     fit_res = self.data_table["centroid_pos_al"] - best_model
    #     res_ra, res_dec = (
    #         np.sin(self.data_table["scan_pos_angle"]) * fit_res,
    #         np.cos(self.data_table["scan_pos_angle"]) * fit_res,
    #     )
    #
    #     plt.figure(figsize=(5, 3))
    #     plt.plot(dra + res_ra, ddec + res_dec, "o", ms=1.0, color="tab:gray")
    #     plt.plot(
    #         dra,
    #         ddec,
    #         marker="s",
    #         ls="-",
    #         ms=5.0,
    #         mew=1.2,
    #         markerfacecolor="tab:purple",
    #         mec="black",
    #         color="black",
    #     )
    #
    #     for i in range(len(fit_res)):
    #         x1 = dra[i] + np.sin(self.data_table["scan_pos_angle"])[i] * (
    #             fit_res[i] + self.data_table["centroid_pos_error_al"][i]
    #         )
    #         x2 = dra[i] + np.sin(self.data_table["scan_pos_angle"])[i] * (
    #             fit_res[i] - self.data_table["centroid_pos_error_al"][i]
    #         )
    #         y1 = ddec[i] + np.cos(self.data_table["scan_pos_angle"])[i] * (
    #             fit_res[i] + self.data_table["centroid_pos_error_al"][i]
    #         )
    #         y2 = ddec[i] + np.cos(self.data_table["scan_pos_angle"])[i] * (
    #             fit_res[i] - self.data_table["centroid_pos_error_al"][i]
    #         )
    #         plt.plot([x1, x2], [y1, y2], "-", lw=1, color="tab:gray")
    #
    #     plt.xlabel("RA (mas)")
    #     plt.ylabel("Dec (mas)")
    #     # plt.xlim(-2, 2)
    #     # plt.ylim(-2, 2)
    #     plt.savefig("test.png")
