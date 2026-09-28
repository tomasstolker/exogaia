"""
Module with the ``SamplingResults`` class.
"""

import pickle
import warnings

from numbers import Real

import matplotlib.pyplot as plt
import numpy as np

from astropy.time import Time
from beartype import beartype, typing
from corner import corner
from matplotlib.figure import Figure
from scipy.stats import gaussian_kde, norm

from exogaia.models import KeplerModel
from exogaia.utils import param_dict_to_list, param_list_to_dict, print_section


class SamplingResults:
    """
    Class for plotting fit results.
    """

    @beartype
    def __init__(self, pickle_file: str, burnin: typing.Optional[int] = None) -> None:
        """
        Parameters
        ----------
        pickle_file : str
            Pickle file that contains the sampling results.
        burnin : int, None
            Number of initial MCMC steps to discard from each walker.
            No burnin is removed if the argument is set to ``None``.
            This parameter only affects MCMC samples. It is ignored for
            nested-sampling posterior samples.

        Returns
        -------
        NoneType
            None
        """

        print_section("Fit results", bound_char="=")

        self.pickle_file = pickle_file
        self.burnin = burnin

        # Load sampling results
        with open(self.pickle_file, "rb") as open_file:
            pickle_data = pickle.load(open_file)

        required_keys = {
            "samples",
            "ln_like",
            "ln_z",
            "data_table",
            "epoch_astrometry",
            "param_indices",
        }

        missing_keys = required_keys - pickle_data.keys()

        if missing_keys:
            raise KeyError(
                "The sampling results are missing the "
                f"following entries: {sorted(missing_keys)}"
            )

        # Samples and log-likelihood

        self.orig_samples = np.asarray(pickle_data["samples"])
        self.samples = np.copy(self.orig_samples)

        self.orig_ln_like = np.asarray(pickle_data["ln_like"])
        self.ln_like = np.copy(self.orig_ln_like)

        # Other fit information

        self.ln_z = pickle_data["ln_z"]
        self.data_table = pickle_data["data_table"]
        self.epoch_astrometry = pickle_data["epoch_astrometry"]
        self.ref_epoch = self.epoch_astrometry.ref_epoch
        self.param_indices = pickle_data["param_indices"]

        # Check sample dimensions

        if self.samples.ndim not in (2, 3):
            raise ValueError(
                "Expected samples with 2 or 3 dimensions, "
                f"but received {self.samples.ndim}."
            )

        self.n_params = self.samples.shape[-1]

        # SamplingResults currently assumes the 12-parameter orbital model

        if self.n_params != 12:
            raise ValueError(
                "SamplingResults currently expects 12 model "
                f"parameters, but received {self.n_params}."
            )

        # The log-likelihood array should contain one value for each sample

        expected_ln_like_shape = self.samples.shape[:-1]

        if self.ln_like.shape != expected_ln_like_shape:
            raise ValueError(
                "The shape of 'ln_like' is inconsistent with the "
                f"samples. Expected {expected_ln_like_shape}, "
                f"but received {self.ln_like.shape}."
            )

        print(f"Number of parameters: {self.n_params}")
        print(f"Samples shape: {self.samples.shape}")

        # Validate burnin

        if burnin is not None and burnin < 0:
            raise ValueError("'burnin' should be zero or a positive integer.")

        # MCMC output:
        # (n_steps, n_walkers, n_params)

        if self.samples.ndim == 3:
            if burnin is not None:
                if burnin >= self.samples.shape[0]:
                    raise ValueError(
                        f"'burnin'={burnin} removes all "
                        f"{self.samples.shape[0]} MCMC steps."
                    )

                self.samples = self.samples[burnin:]
                self.ln_like = self.ln_like[burnin:]

                print(f"\nBurnin: {burnin}")
                print(f"Samples shape: {self.samples.shape}")

            # Combine steps and walkers into a single posterior array

            self.samples = self.samples.reshape(-1, self.n_params)
            self.ln_like = self.ln_like.reshape(-1)

            print(f"\nReshaped samples: {self.samples.shape}")

        # Nested-sampling output:
        # (n_samples, n_params)

        else:
            if burnin is not None:
                warnings.warn(
                    "'burnin' is ignored for nested-sampling posterior samples.",
                )

        if self.samples.shape[0] == 0:
            raise ValueError(
                "The posterior array contains zero samples. "
                f"Perhaps the burnin of {burnin} is too large?"
            )

        if not np.all(np.isfinite(self.ln_like)):
            warnings.warn(
                "The log-likelihood array contains non-finite values.",
            )

        self.labels = [
            r"$\Delta \alpha$ (mas)",
            r"$\Delta \delta$ (mas)",
            r"$\varpi$ (mas)",
            r"$\mu_\alpha$ (mas/yr)",
            r"$\mu_\delta$ (mas/yr)",
            # r"$\log_{10}(P/\mathrm{days})$",
            r"$P$",
            r"$e$",
            r"$\tau$",
            # r"$\log_{10}(a_0/\mathrm{mas})$",
            r"$a_0$",
            r"$i$ (deg)",
            r"$\omega$ (deg)",
            r"$\Omega$ (deg)",
        ]

        # Maximum likelihood and posterior

        print()

        max_idx = np.argmax(self.ln_like)
        best_param = self.samples[max_idx]

        for param_name, param_idx in self.param_indices.items():
            q16, q50, q84 = np.percentile(self.samples[:, param_idx], [16, 50, 84])
            best_val = best_param[param_idx]

            if param_name in ("inc", "aop", "pan"):
                q16, q50, q84, best_val = np.degrees([q16, q50, q84, best_val])

            print(
                f"{param_name:10s}: ML = {best_val:.4g}, "
                f"posterior = {q50:.4g} "
                f"[-{q50-q16:.3g}, +{q84-q50:.3g}]"
            )

    @beartype
    def plot_walkers(
        self,
        n_walkers: int = 30,
        thin: typing.Optional[int] = None,
        plot_file: typing.Optional[str] = None,
    ) -> typing.Optional[Figure]:
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
            instead of stored if the argument is set to ``None``.

        Returns
        -------
        Figure
            The Matplotlib ``Figure`` object that can be used
            for further adjustments of the plot.
        """

        print_section("Plot walkers")

        if n_walkers <= 0:
            raise ValueError("'n_walkers' should be positive.")

        if thin is None:
            thin = 1

        elif thin <= 0:
            raise ValueError("'thin' should be positive.")

        print(f"Number of walkers: {n_walkers}")
        print(f"Thin value: {thin}")

        if self.orig_samples.ndim == 2:
            warnings.warn(
                "Walker traces are not available for "
                "nested-sampling posterior samples.",
            )

            return None

        if self.orig_samples.ndim == 3:
            samples_arr = np.copy(self.orig_samples)

        else:
            raise ValueError(
                "Expected samples with 2 or dimensions, "
                f"but received {self.orig_samples.ndim}."
            )

        fig, axs = plt.subplots(self.n_params, 1, figsize=(5, 20))

        rng = np.random.default_rng()

        rand_walk = rng.choice(
            samples_arr.shape[1],
            size=min(n_walkers, samples_arr.shape[1]),
            replace=False,
        )

        step_idx = np.arange(0, samples_arr.shape[0], thin)

        for param_idx in range(samples_arr.shape[2]):
            for walk_idx in rand_walk:
                walk_track = samples_arr[::thin, walk_idx, param_idx]

                if param_idx in [5, 8]:
                    walk_track = np.log10(walk_track)

                elif param_idx in [9, 10, 11]:
                    walk_track = np.degrees(walk_track)

                axs[param_idx].plot(
                    step_idx,
                    walk_track,
                    ls="-",
                    lw=0.3,
                    marker="none",
                    color="tab:purple",
                    alpha=0.3,
                )

                axs[param_idx].set_xlim(step_idx[0], step_idx[-1])

                if param_idx == self.n_params - 1:
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
        truths: typing.Optional[typing.Dict[str, Real]] = None,
        outlier_fraction: typing.Optional[Real] = None,
        plot_file: typing.Optional[str] = None,
    ) -> Figure:
        """
        Function for plotting the posterior distributions.

        Parameters
        ----------
        truths : dict, None
            Dictionary with the true parameter values that
            will be included in the corner plot. No truths
            are included in the plot if the argument is
            set to ``None``.
        plot_file : str, None
            File name for the output plot. The plot is shown
            instead of stored if the argument is set to ``None``.
        outlier_fraction : float, None
            Exclude the outlier fraction from the posterior
            samples, for example setting ``outlier_fraction=0.01``
            will exclude 1 percent of the most outliers. This
            can be useful if a few outlier samples determine
            the ranges of the plot axes and posterior binning.

        Returns
        -------
        Figure
            The Matplotlib ``Figure`` object that can be used
            for further adjustments of the plot.
        """

        print_section("Plot posterior")

        post_samples = np.copy(self.samples)

        if truths is None:
            truths_new = None
        else:
            truths_new = param_dict_to_list(truths)

        # Convert per to log10(per)

        # post_samples[:, 5] = np.log10(post_samples[:, 5])
        #
        # if truths is not None:
        #     truths_new[5] = np.log10(truths_new[5])

        # Convert sma_0 to log10(sma_0)

        # post_samples[:, 8] = np.log10(post_samples[:, 8])
        #
        # if truths is not None:
        #     truths_new[8] = np.log10(truths_new[8])

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

        # Exclude outlier fraction from the posterior plot for clarity

        if outlier_fraction is not None:
            range_select = np.full(self.n_params, 1.0 - outlier_fraction)
        else:
            range_select = None

        params = [
            r"$\Delta \alpha$",
            r"$\Delta \delta$",
            r"$\varpi$",
            r"$\mu_\alpha$",
            r"$\mu_\delta$",
            # r"$\log{P/\mathrm{days}}$",
            r"$P$",
            r"$e$",
            r"$\tau$",
            # r"$\log{a_0/\mathrm{mas}}$",
            r"$a_0$",
            r"$i$",
            r"$\omega$",
            r"$\Omega$",
        ]

        units = [
            "(mas)",
            "(mas)",
            "(mas)",
            "(mas/yr)",
            "(mas/yr)",
            # None,
            "(days)",
            None,
            None,
            # None,
            "(mas)",
            "(deg)",
            "(deg)",
            "(deg)",
        ]

        titles = []
        for i in range(self.n_params):
            q_16, q_50, q_84 = np.quantile(post_samples[:, i], title_quantiles)
            q_minus, q_plus = q_50 - q_16, q_84 - q_50

            fmt = "{0:.2f}".format

            best_fit = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
            best_fit = best_fit.format(fmt(q_50), fmt(q_minus), fmt(q_plus))

            if units[i] is None:
                titles.append(f"{params[i]} = {best_fit}")
            else:
                titles.append(f"{params[i]} = {best_fit} {units[i]}")

        fig = corner(
            post_samples,
            truths=truths_new,
            truth_color="mediumseagreen",
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
            # labelpad=0.1,
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
    def plot_residuals(
        self,
        n_samples: int = 100,
        seed: typing.Optional[int] = None,
        plot_file: typing.Optional[str] = None,
    ) -> Figure:
        """
        Plot the residuals of the maximum-likelihood sample
        together with residuals from random posterior samples.

        Parameters
        ----------
        n_samples : int
            Number of random posterior samples for which the residuals
            are plotted (default: 100).
        seed : int, None
            Seed for the random number generator used to select posterior
            samples. A random seed is used if set to ``None``.
        plot_file : str, None
            File name for the output plot. The plot is shown instead of
            saved if the argument is set to ``None``.

        Returns
        -------
        Figure
            The Matplotlib ``Figure`` object that can be used for further
            adjustments of the plot.
        """

        print_section("Plot residuals")

        print(f"Number of samples: {n_samples}")

        # Epoch astrometry data

        obs_time = self.data_table["obs_time_tcb"].to_numpy()
        obs_pos = self.data_table["centroid_pos_al"].to_numpy()
        obs_err = self.data_table["centroid_pos_error_al"].to_numpy()

        # Best sample

        max_idx = np.argmax(self.ln_like)
        best_param = self.samples[max_idx, :]
        model_param = param_list_to_dict(best_param)

        kepler_model = KeplerModel(
            epoch_astrometry=self.epoch_astrometry, verbose=False
        )

        residuals = kepler_model.calc_residuals(model_param)

        # Number of degrees of freedom

        n_dof = len(obs_pos) - len(best_param)

        # Reduced chi^2

        chi2_red = np.sum(residuals**2 / obs_err**2) / n_dof

        # RUWE

        if self.epoch_astrometry.sim_data:
            # For simulated data, the uncertainties
            # are already calibrated, so u0 = 1
            ruwe = np.sqrt(chi2_red)

        else:
            ruwe = np.sqrt(chi2_red) / self.epoch_astrometry.u0_norm

        # Random posterior samples

        rng = np.random.default_rng(seed)

        n_samples = min(n_samples, self.samples.shape[0])

        sample_idx = rng.choice(
            self.samples.shape[0],
            size=n_samples,
            replace=False,
        )

        fig = plt.figure(figsize=(6, 3))

        for idx in sample_idx:
            sample_param = param_list_to_dict(self.samples[idx])

            res_sample = kepler_model.calc_residuals(sample_param)

            plt.plot(
                obs_time,
                res_sample,
                ls="none",
                marker="o",
                ms=4.0,
                mew=0.0,
                alpha=0.1,
                color="darkviolet",
                zorder=0,
            )

        plt.errorbar(
            obs_time,
            residuals,
            yerr=obs_err,
            ls="none",
            marker="s",
            ms=4.0,
            mew=1.0,
            elinewidth=1.0,
            color="goldenrod",
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
            ms=4.0,
            mew=1.0,
            elinewidth=1.0,
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
            ms=4.0,
            mew=1.0,
            elinewidth=1.0,
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
            transform=plt.gca().transAxes,
            fontsize=12,
        )

        plt.xlabel("Time (yr)")
        plt.ylabel("Residuals (mas)")

        if plot_file is None:
            plt.show()
        else:
            print(f"\nOutput file: {plot_file}")
            plt.savefig(plot_file)

        return fig

    @beartype
    def plot_orbit(
        self,
        n_samples: int = 50,
        truths: typing.Optional[typing.Dict[str, Real]] = None,
        marker_time: typing.Optional[typing.Union[str, Time]] = None,
        plot_file: typing.Optional[str] = None,
    ) -> Figure:
        """
        Plot the maximum-likelihood photocenter orbit together with
        random orbits drawn from the posterior distribution.

        Parameters
        ----------
        n_samples : int
            Number of random posterior orbit samples to plot (default: 50).
        truths : dict, None
            Dictionary with the true model parameters. If provided, the
            corresponding orbit is plotted together with the true position
            at ``marker_time`` when specified. No true orbit or position is
            plotted if set to ``None``.
        marker_time : str, Time, None
            Date and time at which to evaluate the orbital position. If
            provided, the posterior distribution of the position on that
            date are plotted. The true position is also plotted if
            ``truths`` is provided. A string is interpreted as UTC and
            converted to an ``astropy.time.Time`` object. No positions
            at a specific epoch are plotted if set to ``None``.
        plot_file : str, None
            File name for the output plot. The plot is shown instead of
            stored if the argument is set to ``None``.

        Returns
        -------
        Figure
            The Matplotlib ``Figure`` object that can be used for further
            adjustments of the plot.
        """

        # Epoch astrometry data
        obs_err = self.data_table["centroid_pos_error_al"].to_numpy()
        sin_scan_ang = self.data_table["sin_scan_ang"].to_numpy()
        cos_scan_ang = self.data_table["cos_scan_ang"].to_numpy()

        # Maximum-likelihood sample

        max_idx = np.argmax(self.ln_like)
        best_param = self.samples[max_idx, :]

        print_section("Plot orbit")

        fig = plt.figure(figsize=(4, 4))
        ax = plt.gca()

        rng = np.random.default_rng()

        if n_samples <= 0:
            raise ValueError("'n_samples' should be positive.")

        n_samples = min(n_samples, self.samples.shape[0])

        sample_indices = rng.choice(
            self.samples.shape[0],
            size=n_samples,
            replace=False,
        )

        kepler_model = KeplerModel(
            epoch_astrometry=self.epoch_astrometry, verbose=False
        )

        for sample_idx in sample_indices:
            sample_param = param_list_to_dict(self.samples[sample_idx])

            obs_time_sample = np.linspace(
                self.ref_epoch.tcb.jyear,
                self.ref_epoch.tcb.jyear + sample_param["per"] / 365.25,
                500,
            )

            delta_ra_sample, delta_dec_sample = kepler_model.calc_orbit(
                sample_param,
                obs_time=obs_time_sample,
            )

            plt.plot(
                delta_ra_sample,
                delta_dec_sample,
                lw=0.5,
                alpha=0.2,
                color="dimgray",
                zorder=1,
            )

        # Orbit with true parameters

        if truths is not None:
            obs_time_truth = np.linspace(
                self.ref_epoch.tcb.jyear,
                self.ref_epoch.tcb.jyear + truths["per"] / 365.25,
                10000,
            )

            delta_ra_truth, delta_dec_truth = kepler_model.calc_orbit(
                truths,
                obs_time=obs_time_truth,
            )

            plt.plot(
                delta_ra_truth,
                delta_dec_truth,
                ls="-",
                lw=0.7,
                marker="none",
                color="mediumaquamarine",
                label="True orbit",
                zorder=2,
            )

        # Orbit with maximum likelihood

        model_param = param_list_to_dict(best_param)

        # Calculate full orbit for maximum-likelihood parameters

        obs_time_full = np.linspace(
            self.ref_epoch.tcb.jyear,
            self.ref_epoch.tcb.jyear + best_param[5] / 365.25,
            10000,
        )

        delta_ra_full, delta_dec_full = kepler_model.calc_orbit(
            model_param, obs_time=obs_time_full
        )

        # Plot orbit with maximum likelihood

        plt.plot(
            delta_ra_full,
            delta_dec_full,
            ls="-",
            lw=1.0,
            marker="none",
            color="black",
            zorder=1,
            label="Maximum likelihood",
        )

        # Position at time of periastron (in Julian years)

        t_per = (
            self.ref_epoch.tcb.jyear
            + (model_param["per"] * model_param["tau"]) / 365.25
        )

        delta_ra_per, delta_dec_per = kepler_model.calc_orbit(
            model_param,
            obs_time=np.array([t_per]),
        )

        # Position at time of apastron (in Julian years)

        period_years = model_param["per"] / 365.25

        delta_ra_ap, delta_dec_ap = kepler_model.calc_orbit(
            model_param,
            obs_time=np.array([t_per + 0.5 * period_years]),
        )

        # Plot line of apsides

        plt.plot(
            [delta_ra_per, delta_ra_ap],
            [delta_dec_per, delta_dec_ap],
            ls=":",
            lw=0.8,
            marker="none",
            color="tab:gray",
            label="Line of apsides",
            zorder=2,
        )

        # Signed distance perpendicular to the line of nodes

        pan = model_param["pan"]
        node_perp = delta_ra_full * np.cos(pan) - delta_dec_full * np.sin(pan)

        # Find the two crossings of the orbit with the line of nodes

        cross_idx = np.where(node_perp[:-1] * node_perp[1:] < 0.0)[0]

        # Linear interpolation to the exact crossing

        node_points = []

        for idx in cross_idx:
            frac = -node_perp[idx] / (node_perp[idx + 1] - node_perp[idx])

            node_ra = delta_ra_full[idx] + frac * (
                delta_ra_full[idx + 1] - delta_ra_full[idx]
            )

            node_dec = delta_dec_full[idx] + frac * (
                delta_dec_full[idx + 1] - delta_dec_full[idx]
            )

            node_points.append((node_ra, node_dec))

        # Plot line of nodes

        if len(node_points) == 2:
            plt.plot(
                [node_points[0][0], node_points[1][0]],
                [node_points[0][1], node_points[1][1]],
                ls="--",
                lw=0.8,
                color="tab:gray",
                label="Line of nodes",
                zorder=2,
            )

        else:
            warnings.warn(
                "Expected two intersections of the orbit "
                "with the line of nodes, but found "
                f"{len(node_points)}. The line of nodes "
                "is not plotted.",
            )

        delta_ra, delta_dec = kepler_model.calc_orbit(model_param, obs_time=None)
        residuals = kepler_model.calc_residuals(model_param)

        for i, res_item in enumerate(residuals):
            if i == 0:
                color = "tab:green"
                zorder = 3

            elif i == len(residuals) - 1:
                color = "tab:red"
                zorder = 3

            else:
                color = "goldenrod"
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
                ms=4.0,
                mew=1.0,
                color=color,
                mec="black",
                zorder=zorder,
            )

        # Plot barycenter

        plt.plot(
            0.0,
            0.0,
            marker="x",
            ms=5.0,
            mew=1.5,
            ls="none",
            color="black",
            mec="black",
            label="Barycenter",
            zorder=3,
        )

        # Plot periastron

        plt.plot(
            delta_ra_per,
            delta_dec_per,
            marker="x",
            ms=5.0,
            mew=1.5,
            ls="none",
            color="indianred",
            zorder=3,
            label=rf"Periastron ($t_\mathrm{{per}} = {t_per:.2f}$)",
        )

        # Create and plot posterior positions at marker_time

        if marker_time is not None:
            if isinstance(marker_time, str):
                marker_time = Time(marker_time, format="isot", scale="utc")

            marker_jyear = marker_time.tcb.jyear

            delta_ra_post = np.empty(self.samples.shape[0])
            delta_dec_post = np.empty(self.samples.shape[0])

            for sample_idx, sample_item in enumerate(self.samples):
                sample_param = param_list_to_dict(sample_item)

                delta_ra_tmp, delta_dec_tmp = kepler_model.calc_orbit(
                    sample_param,
                    obs_time=np.array([marker_jyear]),
                )

                delta_ra_post[sample_idx] = delta_ra_tmp[0]
                delta_dec_post[sample_idx] = delta_dec_tmp[0]

            # plt.scatter(
            #     delta_ra_post,
            #     delta_dec_post,
            #     s=5,
            #     alpha=0.05,
            #     color="dimgray",
            #     edgecolors="none",
            #     label=(f"Posterior ({marker_time.utc.strftime('%d %b %Y')})"),
            # )

            # KDE of posterior position

            positions = np.vstack((delta_ra_post, delta_dec_post))
            kde = gaussian_kde(positions)

            # Evaluation grid with padding

            x_pad = 0.15 * np.ptp(delta_ra_post)
            y_pad = 0.15 * np.ptp(delta_dec_post)

            x_grid = np.linspace(
                np.min(delta_ra_post) - x_pad,
                np.max(delta_ra_post) + x_pad,
                250,
            )

            y_grid = np.linspace(
                np.min(delta_dec_post) - y_pad,
                np.max(delta_dec_post) + y_pad,
                250,
            )

            xx, yy = np.meshgrid(x_grid, y_grid)

            density = kde(np.vstack((xx.ravel(), yy.ravel()))).reshape(xx.shape)

            # Credible density thresholds

            density_sorted = np.sort(density.ravel())[::-1]
            cumsum = np.cumsum(density_sorted)
            cumsum /= cumsum[-1]

            level_997 = density_sorted[np.searchsorted(cumsum, 0.997)]

            # Mask very low-density regions

            density_masked = np.ma.masked_less(density, level_997)

            # Filled KDE

            plt.contourf(
                xx,
                yy,
                density_masked,
                levels=np.linspace(level_997, density.max(), 15),
                cmap="Purples",
                alpha=1.0,
                zorder=0,
            )

            # Legend entry

            plt.plot(
                [],
                [],
                color="mediumpurple",
                lw=5.0,
                alpha=0.75,
                label=(f"{marker_time.utc.strftime('%d %b %Y')} (posterior)"),
                zorder=3,
            )

            # Mark orbital position at specified time

            if truths is not None:
                delta_ra_marker, delta_dec_marker = kepler_model.calc_orbit(
                    truths,
                    obs_time=np.array([marker_jyear]),
                )

                plt.plot(
                    delta_ra_marker,
                    delta_dec_marker,
                    ls="none",
                    marker="+",
                    ms=7.0,
                    mec="seagreen",
                    mew=1.5,
                    color="seagreen",
                    label=(f"{marker_time.utc.strftime('%d %b %Y')} (truth)"),
                    zorder=3,
                )

            # Plot maximum-likelihood position at marker_time

            delta_ra_marker, delta_dec_marker = kepler_model.calc_orbit(
                model_param,
                obs_time=np.array([marker_jyear]),
            )

            plt.plot(
                delta_ra_marker,
                delta_dec_marker,
                ls="none",
                marker="+",
                ms=7.0,
                mec="indianred",
                mew=1.5,
                color="indianred",
                label=(f"{marker_time.utc.strftime('%d %b %Y')} (ML)"),
                zorder=3,
            )

        plt.xlabel(r"$\Delta$RA (mas)")
        plt.ylabel(r"$\Delta$Dec (mas)")

        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()

        lim_list = np.array([x_lim[0], x_lim[1], y_lim[0], y_lim[1]])
        lim_max = np.amax(np.abs(lim_list))

        plt.xlim(lim_max, -lim_max)
        plt.ylim(-lim_max, lim_max)

        ax.set_aspect("equal", adjustable="box")

        plt.legend(loc="upper left", frameon=False, fontsize=6, ncol=2)

        if plot_file is None:
            plt.show()
        else:
            print(f"\nOutput file: {plot_file}")
            plt.savefig(plot_file)

        return fig
