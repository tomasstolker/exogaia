import numpy as np
import pytest

from exogaia.priors import (
    FixedPrior,
    LogUniformPrior,
    NormalPrior,
    SinPrior,
    UniformPrior,
)


def test_uniform_prior_transforms_unit_samples() -> None:
    prior = UniformPrior(2.0, 6.0)

    np.testing.assert_allclose(
        prior.transform_samples(np.array([0.0, 0.5, 1.0])), [2.0, 4.0, 6.0]
    )
    assert prior.draw_samples(100).min() >= 2.0
    assert prior.draw_samples(100).max() <= 6.0


def test_log_uniform_prior_is_uniform_in_log_space() -> None:
    prior = LogUniformPrior(1.0, 100.0)

    np.testing.assert_allclose(
        prior.transform_samples(np.array([0.0, 0.5, 1.0])), [1.0, 10.0, 100.0]
    )


def test_normal_prior_and_truncation() -> None:
    prior = NormalPrior(5.0, 2.0)
    np.testing.assert_allclose(prior.transform_samples(np.array([0.5])), [5.0])

    truncated = NormalPrior(-1.0, 2.0, truncate_zero=True, truncate_upper=3.0)
    samples = truncated.draw_samples(200)
    assert np.all((samples >= 0.0) & (samples <= 3.0))


def test_sin_and_fixed_priors() -> None:
    sin_prior = SinPrior()
    np.testing.assert_allclose(
        sin_prior.transform_samples(np.array([0.0, 0.5, 1.0])),
        [0.0, np.pi / 2.0, np.pi],
    )

    fixed_prior = FixedPrior(3.5)
    np.testing.assert_allclose(fixed_prior.draw_samples(3), [3.5, 3.5, 3.5])


@pytest.mark.parametrize(
    "factory, args",
    [
        (UniformPrior, (1.0, 1.0)),
        (LogUniformPrior, (0.0, 1.0)),
        (LogUniformPrior, (2.0, 1.0)),
        (NormalPrior, (0.0, 0.0)),
        (NormalPrior, (0.0, 1.0, True, 0.0)),
    ],
)
def test_prior_validation(factory, args) -> None:
    with pytest.raises(ValueError):
        factory(*args)
