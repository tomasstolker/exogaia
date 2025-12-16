"""
Module with the ``FitResults`` class.
"""

import pickle

from typing import List

import matplotlib.pyplot as plt
import numpy as np

from corner import corner
from matplotlib.figure import Figure
from scipy.stats import norm
from typeguard import typechecked

from exogaia.core import ExoGaia
from exogaia.models import BinaryModel


class FitResults(ExoGaia):
    """
    Class for plotting fit results.
    """

    @typechecked
    def __init__(self, pickle_file: str = None) -> None:
        """
        Returns
        -------
        NoneType
            None
        """

        self.pickle_file = pickle_file

        with open(self.pickle_file, "rb") as open_file:
            pickle_data = pickle.load(open_file)

        self.samples = pickle_data["samples"]
        self.ln_like = pickle_data["ln_like"]
        self.ln_z = pickle_data["ln_z"]
        self.data_table = pickle_data["data_table"]
        self.epoch_astrometry = pickle_data["epoch_astrometry"]

    @typechecked
    def plot_posterior(
        self, truths: List[float] = None, output_file: str = None
    ) -> Figure:
        """
        Plot posterior
        """

        self.print_section("Plot posterior")

        post_samples = np.copy(self.samples)
        post_samples[:, 5] = post_samples[:, 5]

        # Convert inc, aop, pan from rad to deg
        post_samples[:, 7:10] = np.degrees(post_samples[:, 7:10])

        n_params = post_samples.shape[1]

        # Quantiles for the 1D distributions (-1, 1 sigma)
        quantiles = [norm.cdf(n_sigma) for n_sigma in [-1, 1]]

        # Quantiles for the titles (-1, 0, 1 sigma)
        title_quantiles = [norm.cdf(n_sigma) for n_sigma in [-1, 0, 1]]

        # Credible regions for the 2D distributions (1, 2 sigma)
        # Radial integral of the Gaussian probability
        levels = [1.0 - np.exp(-0.5 * n_sigma**2) for n_sigma in [1, 2]]

        # Exclude 1% of the outliers from the posterior plot for clarity
        range_select = np.full(n_params, 0.99)

        labels = [
            r"$\alpha \cos\delta$ (mas)",
            r"$\delta$ (mas)",
            r"$\mu_\alpha \cos\delta$ (mas/yr)",
            r"$\mu_\delta$ (mas/yr)",
            r"$\varpi$ (mas)",
            r"$a$",
            r"$e$",
            r"$i$",
            r"$\omega$",
            r"$\Omega$",
            r"$t_\mathrm{p}$",
            r"$M_1$",
            r"$M_2$",
        ]

        params = [
            r"$\alpha \cos\delta$",
            r"$\delta$",
            r"$\varpi$",
            r"$\mu_\alpha \cos\delta$",
            r"$\mu_\delta$",
            r"$a$",
            r"$e$",
            r"$i$",
            r"$\omega$",
            r"$\Omega$",
            r"$t_\mathrm{p}$",
            r"$M_1$",
            r"$M_2$",
        ]

        units = [
            "(deg)",
            "(deg)",
            "(mas)",
            "(mas/yr)",
            "(mas/yr)",
            "(au)",
            None,
            "(deg)",
            "(deg)",
            "(deg)",
            "(days)",
            r"($M_\odot$)",
            r"($M_\odot$)",
        ]

        titles = []
        for i in range(n_params):
            q_16, q_50, q_84 = np.percentile(post_samples[:, i], [16.0, 50.0, 84])
            q_minus, q_plus = q_50 - q_16, q_84 - q_50

            if i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
                fmt = "{0:.2f}".format
            else:
                fmt = "{0:.1f}".format

            best_fit = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
            best_fit = best_fit.format(fmt(q_50), fmt(q_minus), fmt(q_plus))

            if units[i] is None:
                titles.append(f"{params[i]} = {best_fit}")
            else:
                titles.append(f"{params[i]} = {best_fit} {units[i]}")

        fig = corner(
            post_samples,
            truths=truths,
            truth_color="cornflowerblue",
            labels=labels,
            titles=titles,
            title_quantiles=title_quantiles,
            title_fmt=None,
            show_titles=True,
            smooth=None,
            smooth_1d=None,
            quantiles=quantiles,
            levels=levels,
            range=range_select,
            labelpad=-0.12,
        )

        for ax in fig.axes:
            ax.title.set_fontsize(14.0)
            ax.xaxis.label.set_fontsize(15.0)
            ax.yaxis.label.set_fontsize(15.0)
            ax.tick_params(axis="both", labelsize=13.0)

        if output_file is None:
            plt.show()
        else:
            fig.savefig(output_file)

        return fig

    @typechecked
    def plot_residuals(self, output_file: str = None) -> Figure:
        """
        Plot residuals
        """

        self.print_section("Plot residuals")

        # Epoch astrometry data
        obs_yr = self.data_table["relative_time_year"]
        obs_pos = self.data_table["centroid_pos_al"]
        obs_err = self.data_table["centroid_pos_error_al"]

        # Best sample
        # best_params = np.median(self.samples, axis=0)
        max_idx = np.argmax(self.ln_like)
        best_params = self.samples[max_idx, :]

        binary_model = BinaryModel(epoch_astrometry=self.epoch_astrometry)
        best_model = binary_model.calc_model(best_params)

        fig = plt.figure(figsize=(6, 3))

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
        plt.ylabel("Residuals (mas)")

        if output_file is None:
            plt.show()
        else:
            plt.savefig(output_file)

        return fig

    @typechecked
    def plot_orbit(self, output_file: str = None) -> Figure:
        """
        Orbit plot
        """

        # Epoch astrometry data
        obs_err = self.data_table["centroid_pos_error_al"]
        scan_ang = self.data_table["scan_pos_angle"]

        # Best sample
        # best_params = np.median(self.samples, axis=0)
        max_idx = np.argmax(self.ln_like)
        best_params = self.samples[max_idx, :]

        period = np.sqrt(best_params[5] ** 3 / best_params[11]) * 365.25
        obs_time = np.linspace(0.0, period, 1000)

        bin_model = BinaryModel(epoch_astrometry=self.epoch_astrometry, verbose=False)

        delta_ra_full, delta_dec_full = bin_model.calc_orbit(
            best_params, obs_time=obs_time
        )

        delta_ra, delta_dec = bin_model.calc_orbit(best_params, obs_time=None)
        residuals = bin_model.calc_residuals(best_params)

        self.print_section("Plot orbit")

        res_ra, res_dec = (
            np.sin(scan_ang) * residuals,
            np.cos(scan_ang) * residuals,
        )

        fig = plt.figure(figsize=(4, 4))
        ax = plt.gca()

        plt.plot(
            1e3 * delta_ra_full,
            1e3 * delta_dec_full,
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

            x1 = delta_ra[i] + np.sin(scan_ang[i]) * (res_item + obs_err[i])
            x2 = delta_ra[i] + np.sin(scan_ang[i]) * (res_item - obs_err[i])
            y1 = delta_dec[i] + np.cos(scan_ang[i]) * (res_item + obs_err[i])
            y2 = delta_dec[i] + np.cos(scan_ang[i]) * (res_item - obs_err[i])

            plt.plot(
                [1e3 * x1, 1e3 * x2],
                [1e3 * y1, 1e3 * y2],
                "-",
                lw=1,
                color=color,
                zorder=zorder,
            )

            plt.plot(
                1e3 * delta_ra[i] + 1e3 * np.sin(scan_ang[i]) * res_item,
                1e3 * delta_dec[i] + 1e3 * np.cos(scan_ang[i]) * res_item,
                ls="none",
                marker="s",
                ms=5.0,
                mew=1.2,
                color=color,
                mec="black",
                zorder=zorder,
            )

        plt.xlabel(r"$\Delta$RA ($\mu$as)")
        plt.ylabel(r"$\Delta$Dec ($\mu$as)")

        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()

        lim_list = np.array([x_lim[0], x_lim[1], y_lim[0], y_lim[1]])
        lim_max = np.amax(np.abs(lim_list))

        plt.xlim(lim_max, -lim_max)
        plt.ylim(-lim_max, lim_max)

        if output_file is None:
            plt.show()
        else:
            plt.savefig(output_file)

        return fig
