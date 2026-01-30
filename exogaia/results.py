"""
Module with the ``SamplingResults`` class.
"""

import pickle

import matplotlib.pyplot as plt
import numpy as np

from astropy import units as u
from beartype import beartype, typing
from corner import corner
from matplotlib.figure import Figure
from scipy.stats import norm

from exogaia.core import ExoGaia
from exogaia.models import KeplerModel


class SamplingResults(ExoGaia):
    """
    Class for plotting fit results.
    """

    @beartype
    def __init__(self, pickle_file: str, burnin: typing.Optional[int] = None) -> None:
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
            :class:`~exogaia.samplers.MCMCSampler`.

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
        self.ref_epoch = self.epoch_astrometry.ref_epoch

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

        if self.samples.shape[0] == 0:
            raise ValueError(
                "The posterior array contains zero samples. "
                f"Perhaps the burnin of {burnin} is too large?"
            )

        self.labels = [
            r"$\Delta \alpha$ (mas)",
            r"$\Delta \delta$ (mas)",
            r"$\varpi$ (mas)",
            r"$\mu_\alpha$ (mas/yr)",
            r"$\mu_\delta$ (mas/yr)",
            # r"$\log{a/\mathrm{au}}$",
            r"$\log{P/\mathrm{days}}$",
            r"$e$",
            r"$\tau$",
            r"$\log{a_0/\mathrm{mas}}$",
            r"$i$ (deg)",
            r"$\omega$ (deg)",
            r"$\Omega$ (deg)",
            # r"$M_1$ ($M_\odot$)",
            # r"$\log{M_2/M_\odot}$",
        ]

    @beartype
    def plot_walkers(
        self,
        n_walkers: int = 30,
        thin: typing.Optional[int] = None,
        plot_file: typing.Optional[str] = None,
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

        fig, axs = plt.subplots(self.n_params, 1, figsize=(5, 20))

        if self.orig_samples.ndim == 4:
            samples_arr = np.swapaxes(self.orig_samples, 0, 1)
            samples_arr = samples_arr.reshape(samples_arr.shape[0], -1, self.n_params)

        else:
            samples_arr = np.copy(self.orig_samples)

        rand_walk = np.random.randint(0, high=samples_arr.shape[1], size=n_walkers)

        for param_idx in range(samples_arr.shape[2]):
            for walk_idx in rand_walk:
                walk_track = samples_arr[::thin, walk_idx, param_idx]

                if param_idx in [5, 8]:
                    walk_track = np.log10(walk_track)

                elif param_idx in [9, 10, 11]:
                    walk_track = np.degrees(walk_track)

                axs[param_idx].plot(
                    walk_track,
                    ls="-",
                    lw=0.3,
                    marker="none",
                    color="tab:purple",
                    alpha=0.3,
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

    @beartype
    def plot_posterior(
        self,
        truths: typing.Optional[typing.List[float]] = None,
        plot_file: typing.Optional[str] = None,
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

        if truths is None:
            truths_new = None
        else:
            truths_new = truths.copy()

        # Convert per to log10(per)
        post_samples[:, 5] = np.log10(post_samples[:, 5])
        if truths is not None:
            truths_new[5] = np.log10(truths_new[5])

        # Convert sma_0 to log10(sma_0)
        post_samples[:, 8] = np.log10(post_samples[:, 8])
        if truths is not None:
            truths_new[8] = np.log10(truths_new[8])

        # Convert mass_2 to log10(mass_2)
        # post_samples[:, 12] = np.log10(post_samples[:, 12])
        # if truths is not None:
        #     truths[12] = np.log10(truths[12])

        # Convert inc, aop, pan from rad to deg
        post_samples[:, 9:] = np.degrees(post_samples[:, 9:])
        if truths is not None:
            truths_new[9] = np.degrees(truths_new[9])
            truths_new[10] = np.degrees(truths_new[10])
            truths_new[11] = np.degrees(truths_new[11])

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
            r"$\Delta \alpha$",
            r"$\Delta \delta$",
            r"$\varpi$",
            r"$\mu_\alpha$",
            r"$\mu_\delta$",
            # r"$\log{a/\mathrm{au}}$",
            r"$\log{P/\mathrm{days}}$",
            r"$e$",
            r"$\tau$",
            r"$\log{a_0/\mathrm{mas}}$",
            r"$i$",
            r"$\omega$",
            r"$\Omega$",
            # r"$t_\mathrm{p}$",
            # r"$M_1$",
            # r"$M_2$",
            r"$\log{M_2/M_\odot}$",
        ]

        units = [
            "(mas)",
            "(mas)",
            "(mas)",
            "(mas/yr)",
            "(mas/yr)",
            None,
            None,
            None,
            None,
            "(deg)",
            "(deg)",
            "(deg)",
            # r"($M_\odot$)",
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
            truths=truths_new,
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

    @beartype
    def plot_residuals(self, plot_file: typing.Optional[str] = None) -> Figure:
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
        obs_time = self.data_table["obs_time_tcb"].to_numpy()
        obs_pos = self.data_table["centroid_pos_al"].to_numpy()
        obs_err = self.data_table["centroid_pos_error_al"].to_numpy()

        # Best sample
        max_idx = np.argmax(self.ln_like)
        best_params = self.samples[max_idx, :]

        kepler_model = KeplerModel(epoch_astrometry=self.epoch_astrometry)
        best_model = kepler_model.calc_1d_model(best_params)

        residuals = obs_pos - best_model

        # Number of degrees of freedom
        n_dof = len(obs_pos) - len(best_params)

        # Reduced chi^2
        chi2_red = np.sum(residuals**2 / obs_err**2) / n_dof

        # RUWE
        if self.epoch_astrometry.sim_data:
            ruwe = np.sqrt(chi2_red)
        else:
            ruwe = np.sqrt(chi2_red) / self.epoch_astrometry.u0_norm

        fig = plt.figure(figsize=(6, 3))
        ax = plt.gca()

        plt.errorbar(
            obs_time,
            residuals,
            yerr=obs_err,
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

        plt.errorbar(
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

        plt.errorbar(
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

        plt.text(
            0.04,
            0.92,
            f"RUWE = {ruwe:.3f}",
            ha="left",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )

        plt.xlabel("Time (yr)")
        plt.ylabel("Residuals (mas)")

        if plot_file is None:
            plt.show()
        else:
            plt.savefig(plot_file)

        return fig

    @beartype
    def plot_orbit(self, plot_file: typing.Optional[str] = None) -> Figure:
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
        obs_err = self.data_table["centroid_pos_error_al"].to_numpy()
        sin_scan_ang = self.data_table["sin_scan_ang"].to_numpy()
        cos_scan_ang = self.data_table["cos_scan_ang"].to_numpy()

        # Best sample
        max_idx = np.argmax(self.ln_like)
        best_params = self.samples[max_idx, :]

        # period = np.sqrt(best_params[5] ** 3 / best_params[11]) * 365.25
        # obs_time = np.linspace(0.0, period, 1000)

        yr_start = self.ref_epoch
        yr_end = self.ref_epoch + (best_params[5] / 365.25) * u.yr

        obs_time_full = np.linspace(yr_start, yr_end, 1000)
        obs_time_full = obs_time_full.tcb.jyear

        kepler_model = KeplerModel(
            epoch_astrometry=self.epoch_astrometry, verbose=False
        )

        delta_ra_full, delta_dec_full = kepler_model.calc_orbit(
            best_params, obs_time=obs_time_full
        )

        delta_ra, delta_dec = kepler_model.calc_orbit(best_params, obs_time=None)
        residuals = kepler_model.calc_residuals(best_params)

        self.print_section("Plot orbit")

        fig = plt.figure(figsize=(4, 4))
        ax = plt.gca()

        plt.plot(
            delta_ra_full,
            delta_dec_full,
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

            x1 = delta_ra[i] + sin_scan_ang[i] * (res_item + obs_err[i])
            x2 = delta_ra[i] + sin_scan_ang[i] * (res_item - obs_err[i])
            y1 = delta_dec[i] + cos_scan_ang[i] * (res_item + obs_err[i])
            y2 = delta_dec[i] + cos_scan_ang[i] * (res_item - obs_err[i])

            plt.plot(
                [x1, x2],
                [y1, y2],
                "-",
                lw=1,
                color=color,
                zorder=zorder,
            )

            plt.plot(
                delta_ra[i] + sin_scan_ang[i] * res_item,
                delta_dec[i] + cos_scan_ang[i] * res_item,
                ls="none",
                marker="s",
                ms=5.0,
                mew=1.2,
                color=color,
                mec="black",
                zorder=zorder,
            )

        plt.xlabel(r"$\Delta$RA (mas)")
        plt.ylabel(r"$\Delta$Dec (mas)")

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
