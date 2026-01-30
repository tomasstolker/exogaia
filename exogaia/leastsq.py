"""
Module with the ``LeastSquares`` class.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from astropy import units as u
from beartype import beartype, typing
from matplotlib.gridspec import GridSpec
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from nsstools import NssSource
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import brentq, minimize, least_squares
from tqdm.auto import tqdm

# from matplotlib import cm
# from matplotlib.colors import Normalize

from exogaia.core import ExoGaia
from exogaia.data import EpochAstrometry
from exogaia.models import KeplerModel, StarModel


class LeastSquares(ExoGaia):
    """
    Class for least-squares model fit of epoch astrometry. The best-fit
    parameters and astrometric model, and the corresponding
    parameter covariance matrix and RUWE are stored as the
    ``best_model``, ``best_param``, ``param_cov``, and ``ruwe``
    attributes after running any of the class methods.
    """

    @beartype
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
        pos_err = self.data_table["centroid_pos_error_al"].to_numpy()
        self.inv_cov = np.diag(1.0 / pos_err**2)

        # Initialize attributes for least-squares results
        self.best_model = None
        self.best_param = None
        self.param_cov = None
        self.ruwe = None
        self.chi2 = None
        self.chi2_red = None
        self.n_dof = None

    def __repr__(self):
        """
        String representation of the best-fit model parameters.

        Returns
        -------
        str
            Model parameters from the latest fit.
        """

        class_name = self.__class__.__name__

        if self.best_param is None:
            return f"{class_name}(No parameters available)"

        param_sets = {
            5: (
                ["Δα", "Δδ", "ϖ", "μ_α", "μ_δ"],
                ["mas", "mas", "mas", "mas/yr", "mas/yr"],
            ),
            7: (
                ["Δα", "Δδ", "ϖ", "μ_α", "μ_δ", "a_α", "a_δ"],
                ["mas", "mas", "mas", "mas/yr", "mas/yr", "mas/yr^2", "mas/yr^2"],
            ),
            9: (
                ["Δα", "Δδ", "ϖ", "μ_α", "μ_δ", "a_α", "a_δ", "ȧ_α", "ȧ_δ"],
                [
                    "mas",
                    "mas",
                    "mas",
                    "mas/yr",
                    "mas/yr",
                    "mas/yr^2",
                    "mas/yr^2",
                    "mas/yr^3",
                    "mas/yr^3",
                ],
            ),
            12: (
                [
                    "Δα",
                    "Δδ",
                    "ϖ",
                    "μ_α",
                    "μ_δ",
                    "P",
                    "e",
                    "τ",
                    "a",
                    "i",
                    "ω",
                    "Ω",
                ],
                [
                    "mas",
                    "mas",
                    "mas",
                    "mas/yr",
                    "mas/yr",
                    "days",
                    "",
                    "",
                    "mas",
                    "rad",
                    "rad",
                    "rad",
                ],
            ),
        }

        n_param = len(self.best_param)

        param_labels, param_units = param_sets[n_param]

        params = ""
        count_param = 0
        for param_key, param_val, param_unit in zip(
            param_labels, self.best_param, param_units
        ):
            if len(params) > 0:
                params += ", "

            # if count_param == 5:
            #     params += "\n"

            if len(param_unit) == 0:
                params += f"{param_key} = {param_val:.6g}"
            else:
                params += f"{param_key} = {param_val:.6g} {param_unit}"

            count_param += 1

        return f"{class_name}({params})"

    @beartype
    def calc_excess_noise(
        self, verbose: bool = True
    ) -> typing.Tuple[typing.Optional[float], typing.Optional[float]]:
        """
        Compute the astrometric excess noise (AEN).

        The astrometric excess noise is estimated by introducing an
        additive jitter term, ε, to the formal astrometric uncertainties
        and solving for the value of ε that makes the astrometric
        solution statistically consistent with the data. Specifically,
        ε is determined such that the weighted sum of squared residuals,

            Σ_i r_i² / (σ_i² + ε²),

        is equal to the number of degrees of freedom of the astrometric fit.

        The AEN quantifies the amount of additional astrometric scatter
        required beyond the formal measurement uncertainties.

        Parameters
        ----------
        verbose : bool
            Print some information (default: True).

        Returns
        -------
        tuple(float, float)
            Astrometric excess noise and uncertainty (mas).
        """

        if verbose:
            self.print_section("Astrometric excess noise")

        obs_pos = self.data_table["centroid_pos_al"].to_numpy()
        obs_err = self.data_table["centroid_pos_error_al"].to_numpy()

        self.singl_5param(plot_file=None, verbose=False)
        residuals = obs_pos - self.best_model

        @beartype
        def objective(params: np.ndarray) -> float:
            """
            Objective function for estimating astrometric excess
            noise (AEN). The function evaluates the squared
            deviation between the weighted sum of astrometric
            residuals and the number of degrees of freedom, which
            is minimized to determine the AEN.

            Parameters
            ----------
            params : np.ndarray
                Array with one model parameter, the astrometric
                excess noise that is tested by the ``minimize``
                function.

            Returns
            -------
            float
                Squared mismatch between the weighted residual sum
                and the number of degrees of freedom. This quantity
                is minimized to obtain ε.
            """
            model = np.sum(residuals**2 / (obs_err**2 + params[0] ** 2))
            return (model - self.n_dof) ** 2

        # Initial AEN for the optimization
        aen_init = np.sqrt(max(0.1, np.mean(residuals**2) - np.mean(obs_err**2)))

        aen = None
        aen_sigma = None

        if self.chi2_red < 1.0:
            warnings.warn(
                "The reduced chi-squared is smaller than 1.0 so the "
                "residuals do not warrant including AEN in the fit."
            )

        else:
            result = minimize(
                objective, x0=[aen_init], bounds=[(0.0, None)], method="L-BFGS-B"
            )

            if result.success:
                # Best-fit AEN
                aen = result.x[0]

                # Inverse Hessian (1x1 matrix)
                inv_hess = result.hess_inv.todense()

                # Uncertainty on AEN (mas)
                aen_sigma = np.sqrt(inv_hess[0, 0])

                if verbose:
                    print(f"Number of iterations = {result.nit}")
                    print(f"Number of function evaluations = {result.nfev}")
                    print(f"Number of Jacobian evaluations: {result.njev}")
                    print(
                        f"\nAstrometric excess noise (mas) = "
                        f"{aen:.4f} +/- {aen_sigma:.4f}"
                    )

            else:
                warnings.warn(f"The minimization was not successful: {result.message}")

        return aen, aen_sigma

    def least_squares(
        self,
        design: np.ndarray,
        obs_pos: typing.Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
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
            Print some information (default: True).

        Returns
        -------
        np.ndarray
            Array with the best-fit model astrometry.
        np.ndarray
            Array with the best-fit parameters
        np.ndarray
            Array with the parameter covariance matrix.
        float
            RUWE of the fit.
        """

        # Epoch astrometry data
        if obs_pos is None:
            obs_pos = self.data_table["centroid_pos_al"].to_numpy()

        # obs_err = self.data_table["centroid_pos_error_al"].to_numpy()

        # Number of model parameters
        n_param = design.shape[1]

        # Normal matrix equations
        # Symmetric positive definite
        dt_cinv_d = design.T @ self.inv_cov @ design
        dt_cinv_y = design.T @ self.inv_cov @ obs_pos

        # Cholesky factorization
        # a_matrix = cho_fac cho_fac^T
        cho_fac = cho_factor(dt_cinv_d, lower=True)

        # Solves (a_matrix cho_fac) = b_matrix
        best_param = cho_solve(cho_fac, dt_cinv_y)
        best_model = design @ best_param

        # Parameter covariances
        cov_matrix = cho_solve(cho_fac, np.eye(n_param))

        # Linear model: obs_pos = design x best_param
        # Minimize weighted chi^2: chi^2 = (y - A theta)^T C^-1 (y - A theta)
        # y = obs_pos, A = design, theta = params
        # Setting delta_chi^2/delta_theta = 0
        # Solution: theta = (A^T C^-1 A)^-1 A^T C^-1 y
        # This is slower than cho_factor + cho_solve
        # best_param = np.linalg.solve(dt_cinv_d, dt_cinv_y)
        # best_model = design @ best_param
        # cov_matrix = np.linalg.inv(dt_cinv_d)

        # Fit residuals
        fit_res = obs_pos - best_model

        # Number of data points
        n_obs = len(obs_pos)

        # Number of degrees of freedom
        self.n_dof = n_obs - n_param

        # Reduced chi^2
        self.chi2 = fit_res @ self.inv_cov @ fit_res
        self.chi2_red = self.chi2 / self.n_dof

        # RUWE
        if self.epoch_astrometry.sim_data:
            ruwe = np.sqrt(self.chi2_red)
        else:
            ruwe = np.sqrt(self.chi2_red) / self.epoch_astrometry.u0_norm

        # F2 estimator, which obeys a normal distribution N(0,1)
        # See Equation 1 in Halbwachs et al. (2023)
        f2_stat = np.sqrt(9.0 * self.n_dof / 2.0) * (
            self.chi2_red ** (1.0 / 3.0) + 2.0 / (9.0 * self.n_dof) - 1
        )

        # Uncertainty inflation such that F2 will be zero
        # See Equation 2 in Halbwachs et al. (2023)
        infl_fact = np.sqrt(self.chi2_red / ((1.0 - 2.0 / (9.0 * self.n_dof)) ** 3))

        # Parameter covariances
        # param_cov = cho_solve(cho_fac, np.eye(a_matrix.shape[0]))
        # param_sig = np.sqrt(np.diag(param_cov))

        # Covariance inflation
        if infl_fact > 1.0:
            cov_matrix *= infl_fact

        # Uncorrelated uncertainties
        param_sig = np.sqrt(np.diag(cov_matrix))

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
            print(f"Reduced chi^2 = {self.chi2_red:.3f}")
            print(f"RUWE = {ruwe:.3f}")
            print(f"F2 estimator = {f2_stat:.3f}")
            print(f"Inflation factor = {infl_fact:.3f}")

            if sig_7param is not None:
                print(f"Significance dmu/dt = {sig_7param:.3f}")

            if sig_9param is not None:
                print(f"Significance d^2mu/d^2t = {sig_9param:.3f}")

        return best_model, best_param, cov_matrix, ruwe

    @beartype
    def singl_5param(
        self,
        plot_file: typing.Optional[str] = None,
        verbose: bool = True,
    ) -> typing.Optional[Figure]:
        """
        Method for a least-squares fit of the the epoch astrometry
        with a 5-parameter model for a single star.

        Parameters
        ----------
        plot_file : str, None
            File name of the plot with the results. No plot
            is created when the argument is set to ``None``.
        verbose : bool
            Print some information (default: True).

        Returns
        -------
        Figure
            Matplotlib ``Figure`` object.
        """

        if verbose:
            self.print_section("Single star (5 parameters)")

        # Epoch astrometry data
        obs_time = self.data_table["obs_time_tcb"].to_numpy()
        obs_pos = self.data_table["centroid_pos_al"].to_numpy()
        obs_err = self.data_table["centroid_pos_error_al"].to_numpy()
        rel_yr = self.data_table["relative_time_year"].to_numpy()
        sin_scan_ang = self.data_table["sin_scan_ang"].to_numpy()
        cos_scan_ang = self.data_table["cos_scan_ang"].to_numpy()
        par_fac = self.data_table["parallax_factor_al"].to_numpy()

        # Design matrix for least-squares fit
        design = np.column_stack(
            [
                sin_scan_ang,
                cos_scan_ang,
                par_fac,
                rel_yr * sin_scan_ang,
                rel_yr * cos_scan_ang,
            ]
        )

        self.best_model, self.best_param, self.param_cov, self.ruwe = (
            self.least_squares(design, verbose=verbose)
        )

        param_sig = np.sqrt(np.diag(self.param_cov))

        residuals = obs_pos - self.best_model

        res_ra, res_dec = sin_scan_ang * residuals, cos_scan_ang * residuals

        if verbose:
            print("\nBest-fit parameters:")
            print(
                f"   - RA offset = {self.best_param[0]:.3f} +/- {param_sig[0]:.3f} mas"
            )
            print(
                f"   - Dec offset = {self.best_param[1]:.3f} +/- {param_sig[1]:.3f} mas"
            )
            print(
                f"   - Parallax = {self.best_param[2]:.3f} +/- {param_sig[2]:.3f} mas"
            )
            print(
                f"   - mu in RA = {self.best_param[3]:.3f} +/- {param_sig[3]:.3f} mas/yr"
            )
            print(
                f"   - mu in Dec = {self.best_param[4]:.3f} +/- {param_sig[4]:.3f} mas/yr"
            )

        # Create plot with residuals

        fig = None

        if plot_file is not None:
            star_model = StarModel(epoch_astrometry=self.epoch_astrometry)

            delta_ra_obs, delta_dec_obs, _ = star_model.calc_2d_model(
                model_param=self.best_param, obs_time=None
            )

            time_full = np.linspace(
                self.time_start.tcb.jyear, self.time_end.tcb.jyear, 1000
            )

            delta_ra_full, delta_dec_full, _ = star_model.calc_2d_model(
                model_param=self.best_param, obs_time=time_full
            )

            fig, axs = plt.subplots(
                1, 2, figsize=(14, 4), gridspec_kw={"wspace": -0.05}
            )

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
                rf"$\Delta$RA = {self.best_param[0]:.3f} mas $\pm$ {param_sig[0]:.3f} mas"
                + "\n"
                rf"$\Delta$Dec = {self.best_param[1]:.3f} mas $\pm$ {param_sig[1]:.3f} mas"
                + "\n"
                rf"$\varpi$ = {self.best_param[2]:.3f} $\pm$ {param_sig[2]:.3f} mas"
                + "\n"
                rf"$\mu_\mathrm{{RA}}$ = {self.best_param[3]:.3f} $\pm$ {param_sig[3]:.3f} mas/yr"
                + "\n"
                rf"$\mu_\mathrm{{Dec}}$ = {self.best_param[4]:.3f} $\pm$ {param_sig[4]:.3f} mas/yr"
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

            axs[1].text(
                0.03,
                0.92,
                f"RUWE = {self.ruwe:.3f}",
                ha="left",
                va="center",
                transform=axs[1].transAxes,
                fontsize=12,
            )

            plt.savefig(plot_file)

        return fig

    @beartype
    def accel_7param(
        self,
        plot_file: typing.Optional[str] = None,
        verbose: bool = True,
    ) -> typing.Optional[Figure]:
        """
        Method for a least-squares fit of the the epoch astrometry
        with a 7-parameter model for a star with a proper motion
        acceleration.

        Parameters
        ----------
        plot_file : str, None
            File name of the plot with the results. No plot
            is created when the argument is set to ``None``.
        verbose : bool
            Print some information (default: True).

        Returns
        -------
        Figure
            Matplotlib ``Figure`` object.
        """

        if verbose:
            self.print_section("Binary star (7 parameters)")

        # Epoch astrometry data
        obs_time = self.data_table["obs_time_tcb"].to_numpy()
        obs_pos = self.data_table["centroid_pos_al"].to_numpy()
        obs_err = self.data_table["centroid_pos_error_al"].to_numpy()
        rel_yr = self.data_table["relative_time_year"].to_numpy()
        sin_scan_ang = self.data_table["sin_scan_ang"].to_numpy()
        cos_scan_ang = self.data_table["cos_scan_ang"].to_numpy()
        par_fac = self.data_table["parallax_factor_al"].to_numpy()

        # Design matrix for least-squares fit
        design = np.column_stack(
            [
                sin_scan_ang,
                cos_scan_ang,
                par_fac,
                rel_yr * sin_scan_ang,
                rel_yr * cos_scan_ang,
                0.5 * rel_yr**2 * sin_scan_ang,
                0.5 * rel_yr**2 * cos_scan_ang,
            ]
        )

        self.best_model, self.best_param, self.param_cov, self.ruwe = (
            self.least_squares(design, verbose=verbose)
        )

        param_sig = np.sqrt(np.diag(self.param_cov))

        residuals = obs_pos - self.best_model

        res_ra, res_dec = (
            sin_scan_ang * residuals,
            cos_scan_ang * residuals,
        )

        if verbose:
            print("\nBest-fit parameters:")
            print(
                f"   - RA offset = {self.best_param[0]:.3f} +/- {param_sig[0]:.3f} mas"
            )
            print(
                f"   - Dec offset = {self.best_param[1]:.3f} +/- {param_sig[1]:.3f} mas"
            )
            print(
                f"   - Parallax = {self.best_param[2]:.3f} +/- {param_sig[2]:.3f} mas"
            )
            print(
                f"   - mu in RA = {self.best_param[3]:.3f} +/- {param_sig[3]:.3f} mas/yr"
            )
            print(
                f"   - mu in Dec = {self.best_param[4]:.3f} +/- {param_sig[4]:.3f} mas/yr"
            )
            print(
                f"   - dmu/dt in RA = {self.best_param[5]:.3f} +/- {param_sig[5]:.3f} mas/yr^2"
            )
            print(
                f"   - dmu/dt in Dec = {self.best_param[6]:.3f} +/- {param_sig[6]:.3f} mas/yr^2"
            )

        fig = None

        if plot_file is not None:
            # Stellar track

            star_model = StarModel(epoch_astrometry=self.epoch_astrometry)

            delta_ra_obs, delta_dec_obs, _ = star_model.calc_2d_model(
                model_param=self.best_param, obs_time=None
            )

            time_full = np.linspace(
                self.time_start.tcb.jyear, self.time_end.tcb.jyear, 1000
            )

            delta_ra_full, delta_dec_full, _ = star_model.calc_2d_model(
                model_param=self.best_param, obs_time=time_full
            )

            # Stellar track, without acceleration

            star_no_accel = StarModel(epoch_astrometry=self.epoch_astrometry)

            delta_ra_no_accel, delta_dec_no_accel, _ = star_no_accel.calc_2d_model(
                model_param=self.best_param[:5], obs_time=None
            )

            delta_ra_no_accel_full, delta_dec_no_accel_full, _ = (
                star_no_accel.calc_2d_model(
                    model_param=self.best_param[:5], obs_time=time_full
                )
            )

            # Acceleration

            delta_ra_accel = delta_ra_full - delta_ra_no_accel_full
            delta_dec_accel = delta_dec_full - delta_dec_no_accel_full

            # Create plot with residuals

            fig, axs = plt.subplots(1, 3, figsize=(18, 4), gridspec_kw={"wspace": 0.25})

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
                rf"$\Delta$RA = {self.best_param[0]:.3f} mas $\pm$ {param_sig[0]:.3f} mas"
                + "\n"
                rf"$\Delta$Dec = {self.best_param[1]:.3f} mas $\pm$ {param_sig[1]:.3f} mas"
                + "\n"
                rf"$\varpi$ = {self.best_param[2]:.3f} $\pm$ {param_sig[2]:.3f} mas"
                + "\n"
                rf"$\mu_\mathrm{{RA}}$ = {self.best_param[3]:.3f} $\pm$ {param_sig[3]:.3f} mas/yr"
                + "\n"
                rf"$\mu_\mathrm{{Dec}}$ = {self.best_param[4]:.3f} $\pm$ {param_sig[4]:.3f} mas/yr"
            )

            axs[0].set_xlabel(r"$\Delta\alpha$ (mas)")
            axs[0].set_ylabel(r"$\Delta\delta$ (mas)")
            axs[0].invert_xaxis()

            axs[1].plot(
                delta_ra_accel,
                delta_dec_accel,
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
                    + sin_scan_ang[i] * (res_item + obs_err[i])
                )
                x2 = (
                    delta_ra_obs[i]
                    - delta_ra_no_accel[i]
                    + sin_scan_ang[i] * (res_item - obs_err[i])
                )
                y1 = (
                    delta_dec_obs[i]
                    - delta_dec_no_accel[i]
                    + cos_scan_ang[i] * (res_item + obs_err[i])
                )
                y2 = (
                    delta_dec_obs[i]
                    - delta_dec_no_accel[i]
                    + cos_scan_ang[i] * (res_item - obs_err[i])
                )

                axs[1].plot(
                    [x1, x2],
                    [y1, y2],
                    "-",
                    lw=1,
                    color=color,
                    zorder=zorder,
                )

            axs[1].plot(
                (delta_ra_obs + res_ra - delta_ra_no_accel)[0],
                (delta_dec_obs + res_dec - delta_dec_no_accel)[0],
                ls="none",
                marker="s",
                ms=5.0,
                mew=1.2,
                color="tab:green",
                mec="black",
                zorder=3,
            )

            axs[1].plot(
                (delta_ra_obs + res_ra - delta_ra_no_accel)[1:-1],
                (delta_dec_obs + res_dec - delta_dec_no_accel)[1:-1],
                ls="none",
                marker="s",
                ms=5.0,
                mew=1.2,
                color="tab:purple",
                mec="black",
                zorder=2,
            )

            axs[1].plot(
                (delta_ra_obs + res_ra - delta_ra_no_accel)[-1],
                (delta_dec_obs + res_dec - delta_dec_no_accel)[-1],
                ls="none",
                marker="s",
                ms=5.0,
                mew=1.2,
                color="tab:red",
                mec="black",
                zorder=3,
            )

            axs[1].set_title(
                rf"$\dot{{\mu}}_\mathrm{{RA}}$ = {self.best_param[5]:.3f} "
                rf"$\pm$ {param_sig[5]:.3f} $\mu$as/yr$^2$" + "\n"
                rf"$\dot{{\mu}}_\mathrm{{Dec}}$ = {self.best_param[6]:.3f} "
                rf"$\pm$ {param_sig[6]:.3f} $\mu$as/yr$^2$"
            )

            axs[1].set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
            axs[1].set_ylabel(r"$\Delta\delta$ ($\mu$as)")
            axs[1].invert_xaxis()

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
                f"RUWE = {self.ruwe:.3f}",
                ha="left",
                va="center",
                transform=axs[2].transAxes,
                fontsize=12,
            )

            plt.savefig(plot_file)

        return fig

    @beartype
    def accel_9param(
        self,
        plot_file: typing.Optional[str] = None,
        verbose: bool = True,
    ) -> typing.Optional[Figure]:
        """
        Method for a least-squares fit of the the epoch astrometry
        with a 9-parameter model for a star with a proper motion
        acceleration and second-order derivative.

        Parameters
        ----------
        plot_file : str, None
            File name of the plot with the results. No plot
            is created when the argument is set to ``None``.
        verbose : bool
            Print some information (default: True).

        Returns
        -------
        Figure
            Matplotlib ``Figure`` object.
        """

        if verbose:
            self.print_section("Binary star (9 parameters)")

        # Epoch astrometry data
        obs_time = self.data_table["obs_time_tcb"].to_numpy()
        obs_pos = self.data_table["centroid_pos_al"].to_numpy()
        obs_err = self.data_table["centroid_pos_error_al"].to_numpy()
        rel_yr = self.data_table["relative_time_year"].to_numpy()
        sin_scan_ang = self.data_table["sin_scan_ang"].to_numpy()
        cos_scan_ang = self.data_table["cos_scan_ang"].to_numpy()
        par_fac = self.data_table["parallax_factor_al"].to_numpy()

        # Design matrix for least-squares fit
        design = np.column_stack(
            [
                sin_scan_ang,
                cos_scan_ang,
                par_fac,
                rel_yr * sin_scan_ang,
                rel_yr * cos_scan_ang,
                0.5 * rel_yr**2 * sin_scan_ang,
                0.5 * rel_yr**2 * cos_scan_ang,
                (1.0 / 6.0) * rel_yr**3 * sin_scan_ang,
                (1.0 / 6.0) * rel_yr**3 * cos_scan_ang,
            ]
        )

        self.best_model, self.best_param, self.param_cov, self.ruwe = (
            self.least_squares(design, verbose=verbose)
        )

        param_sig = np.sqrt(np.diag(self.param_cov))

        residuals = obs_pos - self.best_model

        res_ra, res_dec = (
            sin_scan_ang * residuals,
            cos_scan_ang * residuals,
        )

        if verbose:
            print("\nBest-fit parameters:")
            print(
                f"   - RA offset = {self.best_param[0]:.3f} +/- {param_sig[0]:.3f} mas"
            )
            print(
                f"   - Dec offset = {self.best_param[1]:.3f} +/- {param_sig[1]:.3f} mas"
            )
            print(
                f"   - Parallax = {self.best_param[2]:.3f} +/- {param_sig[2]:.3f} mas"
            )
            print(
                f"   - mu in RA = {self.best_param[3]:.3f} +/- {param_sig[3]:.3f} mas/yr"
            )
            print(
                f"   - mu in Dec = {self.best_param[4]:.3f} +/- {param_sig[4]:.3f} mas/yr"
            )
            print(
                f"   - dmu/dt in RA = {self.best_param[5]:.3f} +/- {param_sig[5]:.3f} mas/yr^2"
            )
            print(
                f"   - dmu/dt in Dec = {self.best_param[6]:.3f} +/- {param_sig[6]:.3f} mas/yr^2"
            )
            print(
                f"   - d^2mu/d^2t in RA = {self.best_param[7]:.3f} +/- {param_sig[7]:.3f} mas/yr^3"
            )
            print(
                f"   - d^2mu/d^2t in Dec = {self.best_param[8]:.3f} +/- {param_sig[8]:.3f} mas/yr^3"
            )

        fig = None

        if plot_file is not None:
            # Stellar track

            star_model = StarModel(epoch_astrometry=self.epoch_astrometry)

            delta_ra_obs, delta_dec_obs, _ = star_model.calc_2d_model(
                model_param=self.best_param, obs_time=None
            )

            time_full = np.linspace(
                self.time_start.tcb.jyear, self.time_end.tcb.jyear, 1000
            )

            delta_ra_full, delta_dec_full, _ = star_model.calc_2d_model(
                model_param=self.best_param, obs_time=time_full
            )

            # Stellar track, without acceleration

            star_no_accel = StarModel(epoch_astrometry=self.epoch_astrometry)

            delta_ra_no_accel, delta_dec_no_accel, _ = star_no_accel.calc_2d_model(
                model_param=self.best_param[:5], obs_time=None
            )

            delta_ra_no_accel_full, delta_dec_no_accel_full, _ = (
                star_no_accel.calc_2d_model(
                    model_param=self.best_param[:5], obs_time=time_full
                )
            )

            # Acceleration

            delta_ra_accel = delta_ra_full - delta_ra_no_accel_full
            delta_dec_accel = delta_dec_full - delta_dec_no_accel_full

            # Create plot with residuals

            fig, axs = plt.subplots(1, 3, figsize=(18, 4), gridspec_kw={"wspace": 0.25})

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
                rf"$\Delta$RA = {self.best_param[0]:.3f} mas $\pm$ {param_sig[0]:.3f} mas"
                + "\n"
                rf"$\Delta$Dec = {self.best_param[1]:.3f} mas $\pm$ {param_sig[1]:.3f} mas"
                + "\n"
                rf"$\varpi$ = {self.best_param[2]:.3f} $\pm$ {param_sig[2]:.3f} mas"
                + "\n"
                rf"$\mu_\mathrm{{RA}}$ = {self.best_param[3]:.3f} $\pm$ {param_sig[3]:.3f} mas/yr"
                + "\n"
                rf"$\mu_\mathrm{{Dec}}$ = {self.best_param[4]:.3f} $\pm$ {param_sig[4]:.3f} mas/yr"
            )

            axs[0].set_xlabel(r"$\Delta\alpha$ (mas)")
            axs[0].set_ylabel(r"$\Delta\delta$ (mas)")
            axs[0].invert_xaxis()

            axs[1].plot(
                delta_ra_accel,
                delta_dec_accel,
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
                    + sin_scan_ang[i] * (res_item + obs_err[i])
                )
                x2 = (
                    delta_ra_obs[i]
                    - delta_ra_no_accel[i]
                    + sin_scan_ang[i] * (res_item - obs_err[i])
                )
                y1 = (
                    delta_dec_obs[i]
                    - delta_dec_no_accel[i]
                    + cos_scan_ang[i] * (res_item + obs_err[i])
                )
                y2 = (
                    delta_dec_obs[i]
                    - delta_dec_no_accel[i]
                    + cos_scan_ang[i] * (res_item - obs_err[i])
                )

                axs[1].plot(
                    [x1, x2],
                    [y1, y2],
                    "-",
                    lw=1,
                    color=color,
                    zorder=zorder,
                )

            axs[1].plot(
                (delta_ra_obs + res_ra - delta_ra_no_accel)[0],
                (delta_dec_obs + res_dec - delta_dec_no_accel)[0],
                ls="none",
                marker="s",
                ms=5.0,
                mew=1.2,
                color="tab:green",
                mec="black",
                zorder=3,
            )

            axs[1].plot(
                (delta_ra_obs + res_ra - delta_ra_no_accel)[1:-1],
                (delta_dec_obs + res_dec - delta_dec_no_accel)[1:-1],
                ls="none",
                marker="s",
                ms=5.0,
                mew=1.2,
                color="tab:purple",
                mec="black",
                zorder=2,
            )

            axs[1].plot(
                (delta_ra_obs + res_ra - delta_ra_no_accel)[-1],
                (delta_dec_obs + res_dec - delta_dec_no_accel)[-1],
                ls="none",
                marker="s",
                ms=5.0,
                mew=1.2,
                color="tab:red",
                mec="black",
                zorder=3,
            )

            axs[1].set_title(
                rf"$\dot{{\mu}}_\mathrm{{RA}}$ = {self.best_param[5]:.3f} "
                rf"$\pm$ {param_sig[5]:.3f} $\mu$as/yr$^2$" + "\n"
                rf"$\dot{{\mu}}_\mathrm{{Dec}}$ = {self.best_param[6]:.3f} "
                rf"$\pm$ {param_sig[6]:.3f} $\mu$as/yr$^2$" + "\n"
                rf"$\ddot{{\mu}}_\mathrm{{RA}}$ = {self.best_param[7]:.3f} "
                rf"$\pm$ {param_sig[7]:.3f} $\mu$as/yr$^3$" + "\n"
                rf"$\ddot{{\mu}}_\mathrm{{Dec}}$ = {self.best_param[8]:.3f} "
                rf"$\pm$ {param_sig[8]:.3f} $\mu$as/yr$^3$"
            )

            axs[1].set_xlabel(r"$\Delta\alpha$ ($\mu$as)")
            axs[1].set_ylabel(r"$\Delta\delta$ ($\mu$as)")
            axs[1].invert_xaxis()

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
                f"RUWE = {self.ruwe:.3f}",
                ha="left",
                va="center",
                transform=axs[2].transAxes,
                fontsize=12,
            )

            plt.savefig(plot_file)

        return fig

    @beartype
    def orbit_grid(
        self, plot_file: typing.Optional[str] = None, n_points: int = 30
    ) -> typing.Optional[Figure]:
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
        n_points : int
            Number of grid points in the period, eccentricity, and
            epoch of periastron dimensions. The default is 50, so
            calculating a grid with shape (50, 50, 50).

        Returns
        -------
        Figure
            Matplotlib ``Figure`` object.
        """

        self.print_section("Orbit grid (12-parameters)")

        # Epoch astrometry data
        # obs_time = self.data_table["obs_time_tcb"].to_numpy()
        obs_pos = self.data_table["centroid_pos_al"].to_numpy()
        # obs_err = self.data_table["centroid_pos_error_al"].to_numpy()
        rel_yr = self.data_table["relative_time_year"].to_numpy()
        sin_scan_ang = self.data_table["sin_scan_ang"].to_numpy()
        cos_scan_ang = self.data_table["cos_scan_ang"].to_numpy()
        par_fac = self.data_table["parallax_factor_al"].to_numpy()

        # Grid for the log10(P/days)
        logp_list = np.linspace(np.log10(1e2), np.log10(1e5), n_points)

        # Grid for the eccentricity
        ecc_list = np.linspace(1e-6, 1.0 - 1e-6, n_points, endpoint=False)

        # Grid for the relative time of periastron
        tau_list = np.linspace(1e-6, 1.0 - 1e-6, n_points, endpoint=False)

        kepler_model = KeplerModel(
            epoch_astrometry=self.epoch_astrometry, verbose=False
        )

        ruwe_grid = np.zeros((logp_list.size, ecc_list.size, tau_list.size))
        tau_grid = np.zeros((logp_list.size, ecc_list.size, tau_list.size))

        global_ruwe = np.inf
        global_model = None
        global_param = None
        global_cov = None

        pbar = tqdm(total=len(logp_list) * len(ecc_list))

        for logp_idx, logp_item in enumerate(logp_list):
            for ecc_idx, ecc_item in enumerate(ecc_list):
                for tau_idx, tau_item in enumerate(tau_list):
                    # Period (days)
                    period = 10.0**logp_item

                    # Solve the Kepler equation
                    x_orb, y_orb = kepler_model.solve_kepler(period, ecc_item, tau_item)

                    # Linear least-squares fit
                    # See equation 9 in Holl et al. (2023)
                    # delta RA = B x_orb + G y_orb
                    # delta Dec = A x_orb + F y_orb
                    # eta_orb = x_orb (A cos(psi) + B sin(psi)) + y_orb (F cos(psi) + G sin(psi))

                    design = np.column_stack(
                        [
                            sin_scan_ang,  # delta eta / delta RA
                            cos_scan_ang,  # delta eta / delta Dec
                            par_fac,  # delta eta / delta parallax
                            rel_yr * sin_scan_ang,  # delta eta / delta mu_ra
                            rel_yr * cos_scan_ang,  # delta eta / delta mu_dec
                            x_orb * cos_scan_ang,  # delta eta / delta A
                            x_orb * sin_scan_ang,  # delta eta / delta B
                            y_orb * cos_scan_ang,  # delta eta / delta F
                            y_orb * sin_scan_ang,  # delta eta / delta G
                        ]
                    )

                    best_model, best_param, param_cov, ruwe = self.least_squares(
                        design, obs_pos=obs_pos, verbose=False
                    )

                    if ruwe < global_ruwe:
                        global_ruwe = ruwe
                        global_model = best_model
                        global_param = np.hstack(
                            [best_param, period, ecc_item, tau_item]
                        )
                        # The P, e, and tau covariances are not included
                        global_cov = param_cov

                    # x_sky = best_param[0] * x_orb + best_param[1] * y_orb
                    # y_sky = best_param[2] * x_orb + best_param[3] * y_orb
                    # plt.plot(x_sky, y_sky, 'o')
                    # plt.plot(x_sky + (best_model-obs_pos)*sin_scan_ang,
                    #          y_sky + (best_model-obs_pos)*cos_scan_ang, 'o')
                    # plt.show()

                    # Store RUWE as goodness-of-fit statistics
                    ruwe_grid[logp_idx, ecc_idx, tau_idx] = ruwe

                    # Store tau for contour plot
                    tau_grid[logp_idx, ecc_idx, tau_idx] = tau_item

                pbar.update(1)

        pbar.close()

        # fig, ax = plt.subplots(figsize=(7, 3))
        # cmap = cm.viridis
        # norm = Normalize(vmin=0.0, vmax=1.0)
        # ax.errorbar(
        #     obs_time,
        #     obs_pos-global_model,
        #     yerr=obs_err,
        #     ls="none",
        #     marker="s",
        #     color="black",
        #     ms=1.0,
        # )
        # sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        # plt.colorbar(sm, ax=ax)
        # plt.savefig("test.png")
        # plt.close()

        # Uncorrelated uncertainties
        global_sigma = np.sqrt(np.diag(global_cov))

        # Convert Thiele-Innes constants into
        # semi-major axis of the photocenter, a0
        # See equation A.2 in Halbwachs et al. (2023)

        # u = (A^2 + B^2 + F^2 + G^2) / 2
        # v = AG - BF
        # a = sqrt(u + sqrt((u+v)(u-v)))

        u_param = (
            global_param[5] ** 2
            + global_param[6] ** 2
            + global_param[7] ** 2
            + global_param[8] ** 2
        ) / 2.0

        v_param = global_param[5] * global_param[8] - global_param[6] * global_param[7]

        # All components in the 1D position of eta are in mas
        # Therefore, also the Thiele-Innes components

        sma_0 = np.sqrt(u_param + np.sqrt((u_param + v_param) * (u_param - v_param)))

        # Propagate uncertainties from Thiele-Innes constants to sma_0
        # Gradient of sma_0 to u and v

        w_param = np.sqrt(u_param**2 - v_param**2)
        da_du = np.sqrt(w_param + u_param) / (2.0 * w_param)
        da_dv = -v_param / (2.0 * w_param * np.sqrt(w_param + u_param))

        du_dA = global_param[5]  # A
        du_dB = global_param[6]  # B
        du_dF = global_param[7]  # F
        du_dG = global_param[8]  # G

        dv_dA = global_param[8]  # G
        dv_dB = -global_param[7]  # -F
        dv_dF = -global_param[6]  # -B
        dv_dG = global_param[5]  # A

        grad_a = np.array(
            [
                da_du * du_dA + da_dv * dv_dA,
                da_du * du_dB + da_dv * dv_dB,
                da_du * du_dF + da_dv * dv_dF,
                da_du * du_dG + da_dv * dv_dG,
            ]
        )

        # global_cov[5:, 5:] are covariances of the
        # Thiele-Innes constants in the order A, B, F, G
        # TODO Double check if the calculation is correct
        # The uncertainty on sma_0 seems a bit small?

        sma_0_sigma = np.sqrt(grad_a @ global_cov[5:, 5:] @ grad_a)

        # Calculate mass function
        # See equation 1 in Halbwachs et al. (2023)

        f_mass = (sma_0 / global_param[2]) ** 3 / (global_param[9] / 365.25) ** 2

        # Companion mass M2
        # See equation 2 in Gaia colab (2025) on Gaia BH3
        # f_mass = M2 (M2/(M1+M2))**2

        def find_root(mass_2):
            return mass_2**3 / (self.primary_mass[0] + mass_2) ** 2 - f_mass

        mass_2 = brentq(find_root, 0.0, 100.0 * self.primary_mass[0])

        # Relative semi-major axis (i.e. a = a1 + a2) in mas

        sma = ((period / 365.25) ** 2 * (self.primary_mass[0] + mass_2)) ** (1.0 / 3.0)
        sma /= global_param[2]  # (au)

        # Print best-fit parameters

        print(f"\nBest-fit RUWE = {global_ruwe:.3f}")

        print("\nStellar track:")
        print(f"   - RA offset (mas) = {global_param[0]:.3f} +/- {global_sigma[0]:.3f}")
        print(
            f"   - Dec offset (mas) = {global_param[1]:.3f} +/- {global_sigma[1]:.3f}"
        )
        print(f"   - Parallax (mas) = {global_param[2]:.3f} +/- {global_sigma[2]:.3f}")
        print(
            f"   - mu in RA (mas/yr) = {global_param[3]:.3f} +/- {global_sigma[3]:.3f}"
        )
        print(
            f"   - mu in Dec (mas/yr) = {global_param[4]:.3f} +/- {global_sigma[4]:.3f}"
        )

        print("\nThiele-Innes elements:")
        print(f"   - Period (days) = {global_param[9]:.3f}")
        print(f"   - Eccentricity = {global_param[10]:.3f}")
        print(f"   - Relative time of periastron = {global_param[11]:.2f}")
        print(f"   - Thiele-Innes A = {global_param[5]:.3f} +/- {global_sigma[5]:.3f}")
        print(f"   - Thiele-Innes B = {global_param[6]:.3f} +/- {global_sigma[6]:.3f}")
        print(f"   - Thiele-Innes F = {global_param[7]:.3f} +/- {global_sigma[7]:.3f}")
        print(f"   - Thiele-Innes G = {global_param[8]:.3f} +/- {global_sigma[8]:.3f}")

        print("\nDerived parameters:")
        print(
            f"   - Semi-major axis of photocenter (mas) = {sma_0:.3f} +/- {sma_0_sigma:.3f}"
        )
        print(f"   - Relative semi-major axis (au) = {sma:.3f}")
        print(f"   - Mass function (Msun) = {f_mass:.3f}")
        print(f"   - Secondary mass (Msun) = {mass_2:.3f}")

        # Select minimum RUWE along the 3rd axis to create a 2D array

        ruwe_grid_2d = np.nanmin(ruwe_grid, axis=2)

        # Select indices with the minimum RUWE along the 3rd axis

        min_idx = np.argmin(ruwe_grid, axis=2)

        # Create a grid with the best-fit tau for each period-ecc pair

        tau_best = np.zeros((logp_list.size, ecc_list.size))
        for logp_idx, logp_item in enumerate(logp_list):
            for ecc_idx, ecc_item in enumerate(ecc_list):
                tau_idx = min_idx[logp_idx, ecc_idx]
                tau_best[logp_idx, ecc_idx] = tau_grid[logp_idx, ecc_idx, tau_idx]

        # Create goodness-of-fit plot

        fig = plt.figure(figsize=(4, 3))

        grid_spec = GridSpec(1, 2, width_ratios=[4.0, 0.25])
        grid_spec.update(wspace=0.07, hspace=0, left=0, right=1, bottom=0, top=1)

        ax = plt.subplot(grid_spec[0, 0])
        ax_cb = plt.subplot(grid_spec[0, 1])

        x_grid, y_grid = np.meshgrid(10.0**logp_list, ecc_list)

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

        ax.set_xlabel("Period (days)")
        ax.set_ylabel("Eccentricity")
        ax.set_xscale("log")

        if plot_file is not None:
            plt.savefig(plot_file)

        orb_data = {
            "source_id": [self.epoch_astrometry.source_id],
            "nss_solution_type": ["Orbital"],
            "corr_vec": [[]],
            "ra": [global_param[0]],
            "ra_error": [np.nan],
            "dec": [global_param[1]],
            "dec_error": [np.nan],
            "parallax": [global_param[2]],
            "parallax_error": [np.nan],
            "pmra": [global_param[3]],
            "pmra_error": [np.nan],
            "pmdec": [global_param[4]],
            "pmdec_error": [np.nan],
            "a_thiele_innes": [global_param[5]],
            "a_thiele_innes_error": [np.nan],
            "b_thiele_innes": [global_param[6]],
            "b_thiele_innes_error": [np.nan],
            "f_thiele_innes": [global_param[7]],
            "f_thiele_innes_error": [np.nan],
            "g_thiele_innes": [global_param[8]],
            "g_thiele_innes_error": [np.nan],
            "c_thiele_innes": [np.nan],
            "c_thiele_innes_error": [np.nan],
            "h_thiele_innes": [np.nan],
            "h_thiele_innes_error": [np.nan],
            "period": [global_param[9]],
            "period_error": [np.nan],
            "eccentricity": [global_param[10]],
            "eccentricity_error": [np.nan],
            "t_periastron": [global_param[11]],
            "t_periastron_error": [np.nan],
        }

        df_in = pd.DataFrame(orb_data)

        nss_source = NssSource(star=df_in, indice=0)
        df_campbell = nss_source.campbell()

        print("\nCampbell elements:")
        print(f"   - Period = {global_param[9]:.3f} days")
        print(f"   - Eccentricity = {global_param[10]:.3f}")
        print(f"   - Relative time of periastron = {global_param[11]:.3f}")
        print(f"   - Semi-major axis of photocenter (mas) = {sma_0:.3f}")
        print(f"   - Inclination (deg) = {df_campbell['inclination'][0]:.3f}")
        print(
            f"   - Argument of periastron (deg) = {df_campbell['arg_periastron'][0]:.3f}"
        )
        print(f"   - PA of ascending node (deg) = {df_campbell['nodeangle'][0]:.3f}")

        self.best_param = np.hstack(
            [
                global_param[:5],  # [delta_RA, delta_dec, par, mu_RA, mu_Dec]
                global_param[9],  # per (days)
                global_param[10],  # ecc
                global_param[11],  # tau
                sma_0,  # a_0 (mas)
                np.radians(df_campbell["inclination"][0]),  # inc (rad)
                np.radians(df_campbell["arg_periastron"][0]),  # aop (rad)
                np.radians(df_campbell["nodeangle"][0]),  # pan (rad)
            ]
        )

        self.best_model = global_model
        self.ruwe = global_ruwe

        # Do not store because the P, e, and tau
        # covariances are not included
        # self.param_cov = global_cov

        return fig

    @beartype
    def orbit_fit(
        self,
        inc_jitter: bool = False,
        plot_file: typing.Optional[str] = None,
    ) -> typing.Optional[Figure]:
        """
        Fit a Keplerian astrometric orbit using maximum likelihood and
        (optionally) an additive jitter term, and produce a diagnostic plot
        of the fit residuals.

        This method refines an initial 12-parameter Keplerian orbit solution
        by minimizing the full Gaussian negative log-likelihood using
        ``scipy.optimize.least_squares``. When enabled, an additional jitter
        term is fitted and added in quadrature to the formal astrometric
        uncertainties, such that

            var_i = sigma_i^2 + epsilon^2 ,

        where ``epsilon`` is the astrometric jitter (in mas).

        If no prior best-fit solution is available, an initial orbit grid
        search is performed automatically.

        Parameters
        ----------
        inc_jitter : bool
            If ``True``, include an additional jitter term in the likelihood
            and fit for its amplitude. The jitter is added in quadrature to
            the formal astrometric uncertainties.
        plot_file : str or None
            File name for saving the diagnostic plot of the best-fit orbit
            and residuals. If ``None``, no plot is generated or saved.

        Returns
        -------
        Figure
            Matplotlib ``Figure`` object containing the orbit fit and
            astrometric residuals. A ``None`` is returned if the
            argument of ``plot_file`` if ``False``.
        """

        if self.best_param is None or len(self.best_param) != 12:
            self.orbit_grid(plot_file=None, n_points=30)

        self.print_section("Orbit fit (12 parameters)")

        # Epoch astrometry data
        obs_time = self.data_table["obs_time_tcb"].to_numpy()
        obs_err = self.data_table["centroid_pos_error_al"].to_numpy()
        sin_scan_ang = self.data_table["sin_scan_ang"].to_numpy()
        cos_scan_ang = self.data_table["cos_scan_ang"].to_numpy()

        # Initial parameter values

        if inc_jitter:
            # Add the initial jitter (mas)
            fit_param = np.append(self.best_param, 0.1)

        else:
            fit_param = self.best_param.copy()

        # Lower and upper boundary

        lower = np.array(
            [
                -100.0,  # (mas)
                -100.0,  # (mas)
                0.0,  # (mas)
                -1000.0,  # (mas/yr)
                -1000.0,  # (mas/yr)
                10.0,  # (days)
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )

        upper = np.array(
            [
                100.0,  # (mas)
                100.0,  # (mas)
                100.0,  # (mas)
                1000.0,  # (mas/yr)
                1000.0,  # (mas/yr)
                1e5,  # (days)
                0.99999,
                1.0,
                1e4,
                np.pi,
                2.0 * np.pi,
                2.0 * np.pi,
            ]
        )

        if inc_jitter:
            # Add bounds for jitter (mas)
            lower = np.append(lower, 0.0)
            upper = np.append(upper, 10.0)

        bounds = (lower, upper)

        # Normalize the variances to make sure that the
        # np.log of the normalization term is positive
        var_norm = np.median(obs_err**2)

        @beartype
        def like_residuals(params: np.ndarray, obs_err: np.ndarray) -> np.ndarray:
            """
            Compute normalized residuals for a Keplerian orbit model.
            The function evaluates the residuals between the observed
            data and a Keplerian model, normalized by the measurement
            uncertainties.

            Parameters
            ----------
            params : np.ndarray
                Model parameters passed to ``KeplerModel`` and used to
                compute the model residuals.
            obs_err : ndarray
                Epoch astrometry uncertainties (mas).

            Returns
            -------
            np.ndarray
                Residuals normalized by the data uncertainties.
            """

            if inc_jitter:
                var = obs_err**2 + params[-1] ** 2  # (mas^2)
                model_param = params[:-1]

            else:
                var = obs_err**2  # (mas^2)
                model_param = params

            kepler_model = KeplerModel(
                epoch_astrometry=self.epoch_astrometry, verbose=False
            )

            # We construct residuals such that least_squares minimizes
            # the full Gaussian negative log-likelihood:
            #
            #   -ln L = 0.5 * sum[ r_i^2 / var_i + ln(var_i) ]
            #
            # The factor of 0.5 is applied internally by
            # scipy.optimize.least_squares, so it must not appear
            # explicitly in the residuals. Likewise, there is no
            # leading minus sign because least_squares performs
            # a minimization of the summed squared residuals.
            #
            # The signed square root is used so that negative
            # log-variance terms are represented with real residuals
            # while preserving the correct summed-square value.
            like_res = kepler_model.calc_residuals(model_param) / np.sqrt(var)

            log_var = np.log(var / var_norm)
            like_norm = np.sign(log_var) * np.sqrt(np.abs(log_var))

            return np.hstack([like_res, like_norm])

        result = least_squares(
            like_residuals, fit_param, args=(obs_err,), bounds=bounds
        )

        print(f"Number of function evaluations: {result.nfev}")
        print(f"Number of Jacobian evaluations: {result.njev}")

        if result.success:
            self.best_param = result.x

            self.param_cov = np.linalg.inv(result.jac.T @ result.jac)
            param_sig = np.sqrt(np.diag(self.param_cov))

            # Degrees of freedom = n_data - n_param
            dof = result.jac.shape[0] - result.jac.shape[1]

            # See SciPy docs for definition of the cost function
            # https://docs.scipy.org/doc/scipy/reference/
            # generated/scipy.optimize.least_squares.html
            self.chi2 = 2.0 * result.cost
            self.chi2_red = self.chi2 / dof
            print(f"\nReduced chi^2: {self.chi2_red:.3f}")

            if self.epoch_astrometry.sim_data:
                self.ruwe = np.sqrt(self.chi2_red)
            else:
                self.ruwe = np.sqrt(self.chi2_red) / self.epoch_astrometry.u0_norm

            print(f"RUWE: {self.ruwe:.3f}")

            print("\nBest-fit parameters:")

            print(
                "   - RA offset (mas) = "
                f"{self.best_param[0]:.3f} "
                f"+/- {param_sig[0]:.3f}"
            )

            print(
                "   - Dec offset (mas) = "
                f"{self.best_param[1]:.3f} "
                f"+/- {param_sig[1]:.3f}"
            )

            print(
                "   - Parallax (mas) = "
                f"{self.best_param[2]:.3f} "
                f"+/- {param_sig[2]:.3f}"
            )

            print(
                "   - mu in RA (mas/yr) = "
                f"{self.best_param[3]:.3f} "
                f"+/- {param_sig[3]:.3f}"
            )

            print(
                "   - mu in Dec (mas/yr) = "
                f"{self.best_param[4]:.3f} "
                f"+/- {param_sig[4]:.3f}"
            )

            print(
                "   - Period (days) = "
                f"{self.best_param[5]:.3f} "
                f"+/- {param_sig[5]:.3f}"
            )

            print(
                "   - Eccentricity = "
                f"{self.best_param[6]:.3f} "
                f"+/- {param_sig[6]:.3f}"
            )

            print(
                "   - Relative time of periastron = "
                f"{self.best_param[7]:.3f} "
                f"+/- {param_sig[7]:.3f}"
            )

            print(
                "   - Semi-major axis (mas) = "
                f"{self.best_param[8]:.3f} "
                f"+/- {param_sig[8]:.3f}"
            )

            print(
                "   - Inclination (deg) = "
                f"{np.degrees(self.best_param[9]):.3f} "
                f"+/- {np.degrees(param_sig[9]):.3f}"
            )

            print(
                "   - Argument of periastron (deg) = "
                f"{np.degrees(self.best_param[10]):.3f} "
                f"+/- {np.degrees(param_sig[10]):.3f}"
            )

            print(
                "   - PA of ascending node (deg) = "
                f"{np.degrees(self.best_param[11]):.3f} "
                f"+/- {np.degrees(param_sig[11]):.3f}"
            )

            if inc_jitter:
                print(
                    "   - Jitter (mas) = "
                    f"{self.best_param[12]:.3f} "
                    f"+/- {param_sig[12]:.3f}"
                )

        else:
            warnings.warn(f"The fit was not successful: {result.message}")

        if inc_jitter:
            # Remove the jitter parameter
            self.best_param = self.best_param[:-1]

        fig = None

        if plot_file is not None:
            # Stellar track + orbit model

            kepler_model = KeplerModel(
                epoch_astrometry=self.epoch_astrometry, verbose=False
            )

            delta_ra_obs, delta_dec_obs = kepler_model.calc_2d_model(
                model_param=self.best_param, obs_time=None
            )

            obs_time_full = np.linspace(
                self.time_start.tcb.jyear, self.time_end.tcb.jyear, 1000
            )

            delta_ra_full, delta_dec_full = kepler_model.calc_2d_model(
                model_param=self.best_param, obs_time=obs_time_full
            )

            # Orbit-only model

            delta_ra_orbit, delta_dec_orbit = kepler_model.calc_orbit(
                model_param=self.best_param, obs_time=None
            )

            yr_start = self.ref_epoch
            yr_end = self.ref_epoch + (self.best_param[5] / 365.25) * u.yr

            obs_time_full = np.linspace(yr_start, yr_end, 1000)
            obs_time_full = obs_time_full.tcb.jyear

            delta_ra_orbit_full, delta_dec_orbit_full = kepler_model.calc_orbit(
                model_param=self.best_param, obs_time=obs_time_full
            )

            # Calculate residuals of best-fit model

            residuals = kepler_model.calc_residuals(model_param=self.best_param)

            res_ra, res_dec = (
                sin_scan_ang * residuals,
                cos_scan_ang * residuals,
            )

            # Create plot with residuals

            fig, axs = plt.subplots(1, 3, figsize=(18, 4), gridspec_kw={"wspace": 0.25})

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
                rf"$\Delta$RA = {self.best_param[0]:.3f} mas $\pm$ {param_sig[0]:.3f} mas"
                + "\n"
                rf"$\Delta$Dec = {self.best_param[1]:.3f} mas $\pm$ {param_sig[1]:.3f} mas"
                + "\n"
                rf"$\varpi$ = {self.best_param[2]:.3f} $\pm$ {param_sig[2]:.3f} mas"
                + "\n"
                rf"$\mu_\mathrm{{RA}}$ = {self.best_param[3]:.3f} $\pm$ {param_sig[3]:.3f} mas/yr"
                + "\n"
                rf"$\mu_\mathrm{{Dec}}$ = {self.best_param[4]:.3f} $\pm$ {param_sig[4]:.3f} mas/yr"
            )

            axs[0].set_xlabel(r"$\Delta\alpha$ (mas)")
            axs[0].set_ylabel(r"$\Delta\delta$ (mas)")
            axs[0].invert_xaxis()

            axs[1].plot(
                delta_ra_orbit_full,
                delta_dec_orbit_full,
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

                x1 = delta_ra_orbit[i] + sin_scan_ang[i] * (res_item + obs_err[i])
                x2 = delta_ra_orbit[i] + sin_scan_ang[i] * (res_item - obs_err[i])
                y1 = delta_dec_orbit[i] + cos_scan_ang[i] * (res_item + obs_err[i])
                y2 = delta_dec_orbit[i] + cos_scan_ang[i] * (res_item - obs_err[i])

                axs[1].plot(
                    [x1, x2],
                    [y1, y2],
                    "-",
                    lw=1,
                    color=color,
                    zorder=zorder,
                )

            axs[1].plot(
                (delta_ra_orbit + res_ra)[0],
                (delta_dec_orbit + res_dec)[0],
                ls="none",
                marker="s",
                ms=5.0,
                mew=1.2,
                color="tab:green",
                mec="black",
                zorder=3,
            )

            axs[1].plot(
                (delta_ra_orbit + res_ra)[1:-1],
                (delta_dec_orbit + res_dec)[1:-1],
                ls="none",
                marker="s",
                ms=5.0,
                mew=1.2,
                color="tab:purple",
                mec="black",
                zorder=2,
            )

            axs[1].plot(
                (delta_ra_orbit + res_ra)[-1],
                (delta_dec_orbit + res_dec)[-1],
                ls="none",
                marker="s",
                ms=5.0,
                mew=1.2,
                color="tab:red",
                mec="black",
                zorder=3,
            )

            axs[1].set_title(
                rf"$P = {self.best_param[5]:.3f} \pm "
                rf"{param_sig[5]:.3f}\ \mathrm{{days}}$" + "\n"
                rf"$e = {self.best_param[6]:.3f} \pm "
                rf"{param_sig[6]:.3f}$" + "\n"
                rf"$\tau = {self.best_param[7]:.3f} \pm "
                rf"{param_sig[7]:.3f}$" + "\n"
                rf"$a_0 = {self.best_param[8]:.3f} \pm "
                rf"{param_sig[8]:.3f}\ \mathrm{{mas}}$" + "\n"
                rf"$i = {np.degrees(self.best_param[9]):.3f} \pm "
                rf"{np.degrees(param_sig[9]):.3f}\ \mathrm{{deg}}$" + "\n"
                rf"$\omega = {np.degrees(self.best_param[10]):.3f} \pm "
                rf"{np.degrees(param_sig[10]):.3f}\ \mathrm{{deg}}$" + "\n"
                rf"$\Omega = {np.degrees(self.best_param[11]):.3f} \pm "
                rf"{np.degrees(param_sig[11]):.3f}\ \mathrm{{deg}}$"
            )

            axs[1].set_xlabel(r"$\Delta\alpha$ (mas)")
            axs[1].set_ylabel(r"$\Delta\delta$ (mas)")
            axs[1].invert_xaxis()

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
                f"RUWE = {self.ruwe:.3f}",
                ha="left",
                va="center",
                transform=axs[2].transAxes,
                fontsize=12,
            )

            plt.savefig(plot_file)

        return fig
