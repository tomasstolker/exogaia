"""
Module for setting up parameter priors.
"""

from abc import ABC, abstractmethod
from numbers import Real

import numpy as np

from beartype import beartype, typing
from scipy.stats import norm, truncnorm


class Prior(ABC):
    """
    Interface for prior subclasses.
    """

    @abstractmethod
    def draw_samples(self, n_samples):
        """
        Abstract method for drawing prior samples.
        """

    @abstractmethod
    def transform_samples(self, unit_samples):
        """
        Abstract method for transforming unit samples.
        """


class UniformPrior(Prior):
    """
    Class for a uniform prior.
    """

    @beartype
    def __init__(self, min_val: Real, max_val: Real) -> None:
        """
        Parameters
        ----------
        min_val : float
            Minimum value.
        max_val : float
            Maximum value.

        Returns
        -------
        NoneType
            None
        """

        self.min_val = min_val
        self.max_val = max_val
        self.rng = np.random.default_rng()

    @beartype
    def __repr__(self) -> str:
        """
        String representation of the class.

        Returns
        -------
        str
            Details on the prior.
        """

        return f"Uniform: [{self.min_val:.2f}, {self.max_val:.2f}]"

    @beartype
    def draw_samples(self, n_samples: int) -> np.ndarray:
        """
        Method for drawing random samples from the prior distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples to draw.

        Returns
        -------
        np.ndarray
            Array with the drawn samples.
        """

        samples = self.rng.uniform(low=0.0, high=1.0, size=n_samples)

        return self.transform_samples(samples)

    @beartype
    def transform_samples(self, unit_samples: np.ndarray) -> np.ndarray:
        """
        Method for transforming the unit samples into parameter samples.

        Parameters
        ----------
        unit_samples : np.ndarray
            Array with unit samples.

        Returns
        -------
        np.ndarray
            Array with parameters samples.
        """

        return self.min_val + (self.max_val - self.min_val) * unit_samples


class LogUniformPrior(Prior):
    """
    Class for a log-uniform prior.
    """

    @beartype
    def __init__(self, min_val: Real, max_val: Real) -> None:
        """
        Parameters
        ----------
        min_val : float
            Minimum value.
        max_val : float
            Maximum value.

        Returns
        -------
        NoneType
            None
        """

        self.log_min = np.log10(min_val)
        self.log_max = np.log10(max_val)
        self.rng = np.random.default_rng()

    @beartype
    def __repr__(self) -> str:
        """
        String representation of the class.

        Returns
        -------
        str
            Details on the prior.
        """

        return f"LogUniform: [{self.log_min:.2f}, {self.log_max:.2f}]"

    @beartype
    def draw_samples(self, n_samples: int) -> np.ndarray:
        """
        Method for drawing random samples from the prior distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples to draw.

        Returns
        -------
        np.ndarray
            Array with the drawn samples.
        """

        samples = self.rng.uniform(low=0.0, high=1.0, size=n_samples)

        return self.transform_samples(samples)

    @beartype
    def transform_samples(self, unit_samples: np.ndarray) -> np.ndarray:
        """
        Method for transforming the unit samples into parameter samples.

        Parameters
        ----------
        unit_samples : np.ndarray
            Array with unit samples.

        Returns
        -------
        np.ndarray
            Array with parameters samples.
        """

        samples = self.log_min + (self.log_max - self.log_min) * unit_samples

        return 10.0**samples


class NormalPrior(Prior):
    """
    Class for a normal prior.
    """

    @beartype
    def __init__(
        self,
        mu: Real,
        sigma: Real,
        truncate_zero: bool = False,
        truncate_upper: typing.Optional[Real] = None,
    ) -> None:
        """
        Parameters
        ----------
        mu : float
            Mean of the normal distribution.
        sigma : float
            Standard deviation of the normal distribution.
        truncate_zero : bool
            Truncate the normal distribution at zero.
        truncate_upper : float, None
            Truncate the normal distribution at the upper
            end of the distribution at the value of
            ``truncate_upper``. Only applied when the
            argument of ``truncate_zero`` is set to
            ``True``. The upper truncation is not applied
            if the argument is set to ``None``.

        Returns
        -------
        NoneType
            None
        """

        self.mu = mu
        self.sigma = sigma
        self.truncate_zero = truncate_zero
        self.truncate_upper = truncate_upper
        self.rng = np.random.default_rng()

    @beartype
    def __repr__(self) -> str:
        """
        String representation of the class.

        Returns
        -------
        str
            Details on the prior.
        """

        return f"Normal: [{self.mu:.2f}, {self.sigma:.2f}]"

    @beartype
    def draw_samples(self, n_samples: int) -> np.ndarray:
        """
        Method for drawing random samples from the prior distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples to draw.

        Returns
        -------
        np.ndarray
            Array with the drawn samples.
        """

        samples = self.rng.uniform(low=0.0, high=1.0, size=n_samples)

        return self.transform_samples(samples)

    @beartype
    def transform_samples(self, unit_samples: np.ndarray) -> np.ndarray:
        """
        Method for transforming the unit samples into parameter samples.

        Parameters
        ----------
        unit_samples : np.ndarray
            Array with unit samples.

        Returns
        -------
        np.ndarray
            Array with parameters samples.
        """

        if self.truncate_zero:
            if self.truncate_upper is None:
                lower_bound, upper_bound = 0.0, np.inf
            else:
                lower_bound, upper_bound = 0.0, self.truncate_upper

            a = (lower_bound - self.mu) / self.sigma
            b = (upper_bound - self.mu) / self.sigma

            samples = truncnorm.ppf(unit_samples, a, b, loc=self.mu, scale=self.sigma)

            samples = truncnorm.isf(
                unit_samples, a, np.inf, loc=self.mu, scale=self.sigma
            )

        else:
            samples = self.mu + self.sigma * norm.ppf(unit_samples)

        return samples


class SinPrior(Prior):
    """
    Class for a sine prior.
    """

    @beartype
    def __init__(self) -> None:
        """
        Returns
        -------
        NoneType
            None
        """

        self.rng = np.random.default_rng()

    @beartype
    def __repr__(self) -> str:
        """
        String representation of the class.

        Returns
        -------
        str
            Details on the prior.
        """

        return "Sin: arccos(1 - 2u)"

    @beartype
    def draw_samples(self, n_samples: int) -> np.ndarray:
        """
        Method for drawing random samples from the prior distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples to draw.

        Returns
        -------
        np.ndarray
            Array with the drawn samples.
        """

        samples = self.rng.uniform(low=0.0, high=1.0, size=n_samples)

        return self.transform_samples(samples)

    @beartype
    def transform_samples(self, unit_samples: np.ndarray) -> np.ndarray:
        """
        Method for transforming the unit samples into parameter samples.

        Parameters
        ----------
        unit_samples : np.ndarray
            Array with unit samples.

        Returns
        -------
        np.ndarray
            Array with parameters samples.
        """

        return np.arccos(1.0 - 2.0 * unit_samples)


class FixedPrior(Prior):
    """
    Class for a fixed prior.
    """

    @beartype
    def __init__(self, fix_val: Real) -> None:
        """
        Parameters
        ----------
        fix_val : float
            Fixed value.

        Returns
        -------
        NoneType
            None
        """

        self.fix_val = fix_val

    @beartype
    def __repr__(self) -> str:
        """
        String representation of the class.

        Returns
        -------
        str
            Details on the prior.
        """

        return f"Fixed: {self.fix_val:.2f}"

    @beartype
    def draw_samples(self, n_samples: int) -> np.ndarray:
        """
        Method for drawing random samples from the prior distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples to draw.

        Returns
        -------
        np.ndarray
            Array with the drawn samples.
        """

        return np.full(n_samples, self.fix_val)

    @beartype
    def transform_samples(self, unit_samples: np.ndarray) -> None:
        """
        Method for transforming the unit samples into parameter samples.

        Parameters
        ----------
        unit_samples : np.ndarray
            Array with unit samples.

        Returns
        -------
        NoneType
            None
        """
