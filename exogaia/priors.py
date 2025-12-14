"""
Module for priors.
"""

from abc import ABC, abstractmethod

import numpy as np

from scipy.stats import norm, truncnorm
from typeguard import typechecked


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

    @typechecked
    def __init__(self, min_val: float, max_val: float) -> None:
        self.min_val = min_val
        self.max_val = max_val
        self.rng = np.random.default_rng()

    @typechecked
    def __repr__(self) -> str:
        return f"Uniform: [{self.min_val:.2f}, {self.max_val:.2f}]"

    @typechecked
    def draw_samples(self, n_samples: int) -> np.ndarray:
        samples = self.rng.uniform(low=0.0, high=1.0, size=n_samples)

        return self.transform_samples(samples)

    @typechecked
    def transform_samples(self, unit_samples: np.ndarray) -> np.ndarray:
        return self.min_val + (self.max_val - self.min_val) * unit_samples


class LogUniformPrior(Prior):
    """
    Class for a log-uniform prior.
    """

    @typechecked
    def __init__(self, min_val: float, max_val: float) -> None:
        self.log_min = np.log10(min_val)
        self.log_max = np.log10(max_val)
        self.rng = np.random.default_rng()

    @typechecked
    def __repr__(self) -> str:
        return f"LogUniform: [{self.log_min:.2f}, {self.log_max:.2f}]"

    @typechecked
    def draw_samples(self, n_samples: int) -> np.ndarray:
        samples = self.rng.uniform(low=0.0, high=1.0, size=n_samples)
        return self.transform_samples(samples)

    @typechecked
    def transform_samples(self, unit_samples: np.ndarray) -> np.ndarray:
        samples = self.log_min + (self.log_max - self.log_min) * unit_samples
        return 10.0**samples


class NormalPrior(Prior):
    """
    Class for a normal prior.
    """

    @typechecked
    def __init__(self, mu: float, sigma: float, truncate_zero: bool = False) -> None:
        self.mu = mu
        self.sigma = sigma
        self.truncate_zero = truncate_zero
        self.rng = np.random.default_rng()

    @typechecked
    def __repr__(self) -> str:
        return f"Normal: [{self.mu:.2f}, {self.sigma:.2f}]"

    @typechecked
    def draw_samples(self, n_samples: int) -> np.ndarray:
        samples = self.rng.uniform(low=0.0, high=1.0, size=n_samples)
        return self.transform_samples(samples)

    @typechecked
    def transform_samples(self, unit_samples: np.ndarray) -> np.ndarray:
        if self.truncate_zero:
            lower_bound, upper_bound = 0.0, np.inf

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

    @typechecked
    def __init__(self) -> None:
        self.rng = np.random.default_rng()

    @typechecked
    def __repr__(self) -> str:
        return "Sin: arccos(1 - 2u)"

    @typechecked
    def draw_samples(self, n_samples: int) -> np.ndarray:
        samples = self.rng.uniform(low=0.0, high=1.0, size=n_samples)
        return self.transform_samples(samples)

    @typechecked
    def transform_samples(self, unit_samples: np.ndarray) -> np.ndarray:
        return np.arccos(1.0 - 2.0 * unit_samples)


class FixedPrior(Prior):
    """
    Class for a fixed prior.
    """

    @typechecked
    def __init__(self, fix_val: float) -> None:
        self.fix_val = fix_val

    @typechecked
    def __repr__(self) -> str:
        return f"Fixed: {self.fix_val:.2f}"

    @typechecked
    def draw_samples(self, n_samples: int) -> None:

        return self.fix_val

    @typechecked
    def transform_samples(self, unit_samples: np.ndarray) -> None:
        """
        Method to transform samples.
        """
