"""
Module with the ``FitResults`` class.
"""

import pickle

from typing import List

import matplotlib.pyplot as plt
import numpy as np

from corner import corner
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
        # self.primary_mass = pickle_data["primary_mass"]
        self.epoch_astrometry = pickle_data["epoch_astrometry"]
        self.primary_mass = (1.0, 0.1)

        # Convert inc, aop, pan from rad to deg
        self.samples[:, 7:10] = np.degrees(self.samples[:, 7:10])

    def plot_posterior(self, truths: List[float] = None, output_file: str = None):
        """
        Plot posterior
        """

        self.print_section("Plot posterior")

        n_params = self.samples.shape[1]

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
            "(mas)",
            "(mas",
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
            q_16, q_50, q_84 = np.percentile(self.samples[:, i], [16.0, 50.0, 84])
            q_minus, q_plus = q_50 - q_16, q_84 - q_50

            if i in [0, 1, 2, 3, 4]:
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
            self.samples,
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

        if output_file is not None:
            fig.savefig(output_file)

        return fig

    def plot_residuals(self, output_file: str = None):
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

        if output_file is not None:
            fig.savefig(output_file)

        return fig

    def plot_orbit(self, output_file: str = None):
        """
        Orbit plot
        """

        # Epoch astrometry data
        obs_pos = self.data_table["centroid_pos_error_al"]
        scan_ang = self.data_table["scan_pos_angle"]

        # Best sample
        # best_params = np.median(self.samples, axis=0)
        max_idx = np.argmax(self.ln_like)
        best_params = self.samples[max_idx, :]
        best_params = [
            0.0,
            0.0,
            10.0,
            20.0,
            -20.0,
            1.0,
            0.0,
            np.radians(0.0),
            np.radians(0.0),
            np.radians(0.0),
            0.0,
            1.1,
        ]

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

        plt.plot(delta_ra + res_ra, delta_dec + res_dec, "o", ms=2.0, color="tab:gray")

        plt.plot(
            delta_ra_full,
            delta_dec_full,
            ls="-",
            lw=1.0,
            marker="none",
            color="black",
        )

        for i, res_item in enumerate(residuals):
            x1 = delta_ra[i] + np.sin(scan_ang[i]) * (res_item + obs_pos[i])
            x2 = delta_ra[i] + np.sin(scan_ang[i]) * (res_item - obs_pos[i])
            y1 = delta_dec[i] + np.cos(scan_ang[i]) * (res_item + obs_pos[i])
            y2 = delta_dec[i] + np.cos(scan_ang[i]) * (res_item - obs_pos[i])
            plt.plot([x1, x2], [y1, y2], "-", lw=1, color="tab:gray")

        plt.xlabel(r"$\Delta$RA (mas)")
        plt.ylabel(r"$\Delta$Dec (mas)")

        if output_file is not None:
            plt.savefig(output_file)

        return fig
