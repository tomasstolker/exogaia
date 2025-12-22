"""
Module with the ``FitResults`` class.
"""

import pickle

from typing import List, Optional

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
    def __init__(self, pickle_file: str, burnin: Optional[int] = None) -> None:
        """
        Parameters
        ----------
        pickle_file : str
            Pickle file that contains the results from the fit.
        burnin : int, None
            Number of samples to discard for each walker, i.e. the
            burnin of the MCMC sampling with ``emcee``. No burnin is
            removed if the argument is set to ``None``. This parameter
            has only an effect on the fit results from
            :class:`~exogaia.sampler.MCMCSampler`.

        Returns
        -------
        NoneType
            None
        """

        self.print_section("Fit results")

        self.pickle_file = pickle_file

        with open(self.pickle_file, "rb") as open_file:
            pickle_data = pickle.load(open_file)

        self.orig_samples = pickle_data["samples"]
        self.samples = pickle_data["samples"]
        self.n_params = self.samples.shape[-1]

        self.ln_like = pickle_data["ln_like"]
        self.ln_z = pickle_data["ln_z"]

        self.data_table = pickle_data["data_table"]
        self.epoch_astrometry = pickle_data["epoch_astrometry"]

        print(f"Number of parameters: {self.n_params}")
        print(f"Samples shape: {self.samples.shape}")

        if self.samples.ndim == 3:
            if burnin is not None:
                self.samples = self.samples[burnin:, :, :]
                self.ln_like = self.ln_like[burnin:, :]

                print(f"\nBurnin: {burnin}")
                print(f"Samples shape: {self.samples.shape}")

            self.samples = self.samples.reshape(-1, self.n_params)
            self.ln_like = self.ln_like.reshape(-1)
            print(f"\nReshaped samples: {self.samples.shape}")

        elif self.samples.ndim == 4:
            if burnin is not None:
                self.samples = self.samples[:, burnin:, :, :]
                self.ln_like = self.ln_like[:, burnin:, :]

                print(f"\nBurnin: {burnin}")
                print(f"Samples shape: {self.samples.shape}")

            self.samples = self.samples.reshape(-1, self.n_params)
            self.ln_like = self.ln_like.reshape(-1)
            print(f"\nReshaped samples: {self.samples.shape}")

        self.labels = [
            r"$\alpha \cos\delta$ (mas)",
            r"$\delta$ (mas)",
            r"$\mu_\alpha \cos\delta$ (mas/yr)",
            r"$\mu_\delta$ (mas/yr)",
            r"$\varpi$ (mas)",
            r"$\log{a/\mathrm{au}}$",
            r"$e$",
            r"$i$ (deg)",
            r"$\omega$ (deg)",
            r"$\Omega$ (deg)",
            r"$t_\mathrm{p}$",
            r"$M_1$ ($M_\odot$)",
            r"$\log{M_2/M_\odot}$",
        ]

    @typechecked
    def plot_walkers(
        self, n_walkers: int = 30, thin: Optional[int] = None, plot_file: str = None
    ) -> Figure:
        """
        Function for plotting the tracks by the MCMC walkers.

        Parameters
        ----------
        n_walkers : int
            Number of walkers to plot (default: 30).
        thin : int, None
            Thin each walker chain by selecting every ``thin`` step. The full
            chain is plotted if the argument is set to ``None``.
        plot_file : str, None
            File name for the output plot. The plot is shown
            instead of stored if the arguments is set to ``None``.

        Returns
        -------
        Figure
            The Matplotlib ``Figure`` object that can be used
            for further adjustments of the plot.
        """

        self.print_section("Plot walkers")

        if thin is None:
            thin = 1

        print(f"Number of walkers: {n_walkers}")
        print(f"Thin value: {thin}")

        fig, axs = plt.subplots(13, 1, figsize=(5, 20))

        if self.orig_samples.ndim == 4:
            samples_arr = np.swapaxes(self.orig_samples, 0, 1)
            samples_arr = samples_arr.reshape(samples_arr.shape[0], -1, self.n_params)

        else:
            samples_arr = np.copy(self.orig_samples)

        rand_walk = np.random.randint(0, high=samples_arr.shape[1], size=n_walkers)

        for param_idx in range(samples_arr.shape[2]):
            for walk_idx in rand_walk:
                walk_track = samples_arr[::thin, walk_idx, param_idx]

                if param_idx in [5, 12]:
                    walk_track = np.log10(walk_track)

                elif param_idx in [7, 8, 9]:
                    walk_track = np.degrees(walk_track)

                axs[param_idx].plot(
                    walk_track,
                    ls="-",
                    lw=0.3,
                    marker="none",
                    color="tab:purple",
                    alpha=0.1,
                )

                axs[param_idx].set_xlim(0.0, walk_track.size)

                if param_idx == 12:
                    axs[param_idx].set_xlabel("Step number")
                else:
                    axs[param_idx].tick_params(labelbottom=False)

                axs[param_idx].set_ylabel(self.labels[param_idx])

        if plot_file is None:
            plt.show()
        else:
            print(f"Output file: {plot_file}")
            plt.savefig(plot_file)

        return fig

    @typechecked
    def plot_posterior(
        self, truths: List[float] = None, plot_file: str = None
    ) -> Figure:
        """
        Function for plotting the posterior distributions.

        Parameters
        ----------
        truths : list(float), None
            Optional list with the true parameter values that
            will be included in the corner plot. No truths
            are included in the plot if the argument is set
            to ``None``.
        plot_file : str, None
            File name for the output plot. The plot is shown
            instead of stored if the arguments is set to ``None``.

        Returns
        -------
        Figure
            The Matplotlib ``Figure`` object that can be used
            for further adjustments of the plot.
        """

        self.print_section("Plot posterior")

        post_samples = np.copy(self.samples)

        # Convert sma to log10(sma)
        post_samples[:, 5] = np.log10(post_samples[:, 5])
        truths[5] = np.log10(truths[5])

        # Convert mass_2 to log10(mass_2)
        post_samples[:, 12] = np.log10(post_samples[:, 12])
        truths[12] = np.log10(truths[12])

        # Convert inc, aop, pan from rad to deg
        post_samples[:, 7:10] = np.degrees(post_samples[:, 7:10])

        # Quantiles for the 1D distributions (-1, 1 sigma)
        quantiles = [norm.cdf(n_sigma) for n_sigma in [-1, 1]]

        # Quantiles for the titles (-1, 0, 1 sigma)
        title_quantiles = [norm.cdf(n_sigma) for n_sigma in [-1, 0, 1]]

        # Credible regions for the 2D distributions (1, 2 sigma)
        # Radial integral of the Gaussian probability
        levels = [1.0 - np.exp(-0.5 * n_sigma**2) for n_sigma in [1, 2]]

        # Exclude 1% of the outliers from the posterior plot for clarity
        range_select = np.full(self.n_params, 0.99)

        params = [
            r"$\alpha \cos\delta$",
            r"$\delta$",
            r"$\varpi$",
            r"$\mu_\alpha \cos\delta$",
            r"$\mu_\delta$",
            # r"$a$",
            r"$\log{a/\mathrm{au}}$",
            r"$e$",
            r"$i$",
            r"$\omega$",
            r"$\Omega$",
            r"$t_\mathrm{p}$",
            r"$M_1$",
            # r"$M_2$",
            r"$\log{M_2/M_\odot}$",
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
        for i in range(self.n_params):
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
            labels=self.labels,
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

        if plot_file is None:
            plt.show()
        else:
            print(f"Output file: {plot_file}")
            fig.savefig(plot_file)

        return fig

    @typechecked
    def plot_residuals(self, plot_file: str = None) -> Figure:
        """
        Function for plotting the residuals of the sample
        that has the maximum likelihood.

        Parameters
        ----------
        plot_file : str, None
            File name for the output plot. The plot is shown
            instead of stored if the arguments is set to ``None``.

        Returns
        -------
        Figure
            The Matplotlib ``Figure`` object that can be used
            for further adjustments of the plot.
        """

        self.print_section("Plot residuals")

        # Epoch astrometry data
        obs_yr = self.data_table["relative_time_year"]
        obs_pos = self.data_table["centroid_pos_al"]
        obs_err = self.data_table["centroid_pos_error_al"]

        # Best sample
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

        if plot_file is None:
            plt.show()
        else:
            plt.savefig(plot_file)

        return fig

    @typechecked
    def plot_orbit(self, plot_file: str = None) -> Figure:
        """
        Function for plotting the stellar orbit based on the
        parameters with the maximum likelihood.

        Parameters
        ----------
        plot_file : str, None
            File name for the output plot. The plot is shown
            instead of stored if the arguments is set to ``None``.

        Returns
        -------
        Figure
            The Matplotlib ``Figure`` object that can be used
            for further adjustments of the plot.
        """

        # Epoch astrometry data
        obs_err = self.data_table["centroid_pos_error_al"]
        scan_ang = self.data_table["scan_pos_angle"]

        # Best sample
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

        if plot_file is None:
            plt.show()
        else:
            plt.savefig(plot_file)

        return fig
