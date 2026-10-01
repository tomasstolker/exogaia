from exogaia.leastsq import LeastSquares


def test_calculate_excess_noise(least_squares: LeastSquares) -> None:
    least_squares.calc_excess_noise()


def test_fit_single_star(least_squares: LeastSquares) -> None:
    least_squares.singl_5param(plot_file=None)


def test_fit_acceleration(least_squares: LeastSquares) -> None:
    least_squares.accel_7param(plot_file=None)


def test_fit_acceleration_and_jerk(least_squares: LeastSquares) -> None:
    least_squares.accel_9param(plot_file=None)


def test_generate_orbit_grid(least_squares: LeastSquares) -> None:
    least_squares.orbit_grid(plot_file=None, n_points=30)


def test_fit_orbit(least_squares: LeastSquares) -> None:
    least_squares.orbit_fit(inc_jitter=False, plot_file=None)


def test_fit_orbit_with_jitter(least_squares: LeastSquares) -> None:
    least_squares.orbit_fit(inc_jitter=True, plot_file=None)

    assert len(least_squares.best_param) == 12
    assert least_squares.param_cov.shape == (12, 12)
