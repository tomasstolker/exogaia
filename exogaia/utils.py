"""
Module with utility functions.
"""

from numbers import Real

import numpy as np
import pandas as pd

from beartype import beartype, typing
from nsstools import NssSource
from scipy.optimize import brentq

# @beartype
# def orbit_sky(
#     nu: typing.Union[Real, np.ndarray],
#     sma: Real,
#     ecc: Real,
#     inc: Real,
#     aop: Real,
#     pan: Real,
# ) -> typing.Tuple[
#     typing.Union[Real, np.ndarray],
#     typing.Union[Real, np.ndarray],
# ]:
#     """
#     Compute the projected sky-plane coordinates of a Keplerian orbit.
#
#     Parameters
#     ----------
#     nu : float or ndarray
#         True anomaly (radians).
#     sma : float
#         Semi-major axis.
#     ecc : float
#         Eccentricity (0 <= e < 1).
#     inc : float
#         Inclination (radians). i = 0 corresponds to a face-on orbit.
#     aop : float
#         Argument of pericenter (radians).
#     pan : float
#         Longitude of the ascending node (radians).
#
#     Returns
#     -------
#     x : float or ndarray
#         Projected x-coordinate on the sky plane.
#         In the units of ``sma``.
#     y : float or ndarray
#         Projected y-coordinate on the sky plane.
#         In the units of ``sma``.
#
#     Notes
#     -----
#     The orbit is first computed in the orbital plane using the standard
#     conic-section equation:
#
#         r = a (1 - e^2) / (1 + e cos(nu))
#
#     The position vector is then rotated by:
#         1) argument of pericenter (aop),
#         2) inclination (inc),
#         3) longitude of ascending node (pan),
#
#     to obtain the sky-plane projection.
#     """
#
#     r = sma * (1.0 - ecc**2.0) / (1.0 + ecc * np.cos(nu))
#
#     # position in orbital plane
#     x_op = r * np.sin(nu)
#     y_op = r * np.cos(nu)
#
#     # rotate by omega
#     x1 = x_op * np.cos(aop) - y_op * np.sin(aop)
#     y1 = x_op * np.sin(aop) + y_op * np.cos(aop)
#
#     # incline
#     x2 = x1 * np.cos(inc)
#
#     # rotate by Omega to sky plane
#     x = x2 * np.cos(-pan) - y1 * np.sin(-pan)
#     y = x2 * np.sin(-pan) + y1 * np.cos(-pan)
#
#     return x, y


def calc_sma_from_ti(
    model_param: np.ndarray,
    param_cov: np.ndarray,
) -> typing.Tuple[Real, Real, Real, Real, Real]:
    """
    Compute the photocenter semi-major axis and derived orbital quantities
    from Thiele–Innes constants.

    The function converts the Thiele–Innes constants (A, B, F, G) into
    the semi-major axis of the photocenter following Eq. A.2 of
    Halbwachs et al. (2023). Uncertainty propagation from the covariance
    matrix of the Thiele–Innes parameters is also performed.

    Using the photocenter semi-major axis and the system parallax and
    period, the astrometric mass function is computed, following
    Eq. 1 of Halbwachs et al. (2023).. The companion mass is then
    obtained by numerically solving the mass function equation for M₂.

    Parameters
    ----------
    model_param : np.ndarray
        Array of fitted model parameters. The relevant entries are
        - ``model_param[2]`` : parallax (mas)
        - ``model_param[5]`` : Thiele–Innes constant A (mas)
        - ``model_param[6]`` : Thiele–Innes constant B (mas)
        - ``model_param[7]`` : Thiele–Innes constant F (mas)
        - ``model_param[8]`` : Thiele–Innes constant G (mas)
        - ``model_param[9]`` : orbital period (days)
    param_cov : np.ndarray
        Covariance matrix of the model parameters. The submatrix
        ``param_cov[5:, 5:]`` should correspond to the covariance
        of the Thiele–Innes constants (A, B, F, G).

    Returns
    -------
    float
        Semi-major axis of the photocenter (mas).
    float
        Uncertainty on the semi-major axis of the photocenter (mas).
    """

    # Convert Thiele-Innes constants into
    # semi-major axis of the photocenter, a0
    # See Eq. A.2 in Halbwachs et al. (2023)

    # u = (A^2 + B^2 + F^2 + G^2) / 2
    # v = AG - BF
    # a = sqrt(u + sqrt((u+v)(u-v)))

    u_param = (
        model_param[5] ** 2
        + model_param[6] ** 2
        + model_param[7] ** 2
        + model_param[8] ** 2
    ) / 2.0

    v_param = model_param[5] * model_param[8] - model_param[6] * model_param[7]

    # All components in the 1D position of eta are in mas
    # Therefore, also the Thiele-Innes components

    sma_0 = np.sqrt(u_param + np.sqrt((u_param + v_param) * (u_param - v_param)))

    # Propagate uncertainties from Thiele-Innes constants to sma_0
    # Gradient of sma_0 to u and v

    w_param = np.sqrt(u_param**2 - v_param**2)
    da_du = np.sqrt(w_param + u_param) / (2.0 * w_param)
    da_dv = -v_param / (2.0 * w_param * np.sqrt(w_param + u_param))

    du_dA = model_param[5]  # A
    du_dB = model_param[6]  # B
    du_dF = model_param[7]  # F
    du_dG = model_param[8]  # G

    dv_dA = model_param[8]  # G
    dv_dB = -model_param[7]  # -F
    dv_dF = -model_param[6]  # -B
    dv_dG = model_param[5]  # A

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

    sma_0_sigma = np.sqrt(grad_a @ param_cov[5:, 5:] @ grad_a)

    return sma_0, sma_0_sigma


def calc_mass_from_sma(
    sma_0: Real, period: Real, parallax: Real, primary_mass: Real
) -> typing.Tuple[Real, Real]:
    """
    Compute the photocenter semi-major axis and derived orbital quantities
    from Thiele–Innes constants.

    The function converts the Thiele–Innes constants (A, B, F, G) into
    the semi-major axis of the photocenter following Eq. A.2 of
    Halbwachs et al. (2023). Uncertainty propagation from the covariance
    matrix of the Thiele–Innes parameters is also performed.

    Using the photocenter semi-major axis and the system parallax and
    period, the astrometric mass function is computed, following
    Eq. 1 of Halbwachs et al. (2023).. The companion mass is then
    obtained by numerically solving the mass function equation for M₂.

    Parameters
    ----------
    sma_0 : float
        Semi-major axis of the photocenter (mas).
    period : float
        Orbital period (days).
    parallax : float
        Parallax (mas).
    primary_mass : float
        Mass of the primary star (Msun).

    Returns
    -------
    float
        Mass function (Msun).
    float
        Companion mass (Msun).
    """

    # Calculate mass function (Msun)
    # See Eq. 1 in Halbwachs et al. (2023)

    f_mass = (sma_0 / parallax) ** 3 / (period / 365.25) ** 2

    # Companion mass (Msun)
    # See Eq. 2 in Gaia colab (2025) on Gaia BH3
    # Calculated from f_mass = M2 (M2/(M1+M2))**2

    @beartype
    def find_root(mass_2: Real) -> Real:
        """
        Equation whose root yields the companion mass ``M₂``

            f(M) = M₂³ / (M₁ + M₂)²

        where ``M₁`` is the primary mass and ``f(M)`` is the
        astrometric mass function.

        Parameters
        ----------
        mass_2 : Real
            Trial companion mass (Msun).

        Returns
        -------
        float
            Value of the mass-function equation evaluated at ``mass_2``.
            The root (value equal to zero) corresponds to the physical
            companion mass.
        """

        return mass_2**3 / (primary_mass + mass_2) ** 2 - f_mass

    upper = max(primary_mass, 4.0 * f_mass)
    mass_2 = brentq(find_root, 0.0, upper)

    return f_mass, mass_2


@beartype
def thiele_innes_to_campbell(
    source_id: typing.Optional[typing.Union[int, np.int64]],
    sma_0: Real,
    model_param: np.ndarray,
    verbose: bool = True,
) -> np.ndarray:
    """
    Convert Thiele–Innes orbital parameters into Campbell elements.

    This function constructs a minimal orbital-solution table containing
    the fitted Thiele–Innes constants and other astrometric parameters,
    and uses ``NssSource.campbell()`` to derive the equivalent Campbell
    orbital elements (inclination, argument of periastron, and position
    angle of the ascending node).

    The resulting parameters are returned in a vector suitable for use
    in an orbital model using Campbell elements.

    Parameters
    ----------
    source_id : int, None
        Gaia source ID.
    sma_0 : Real
        Semi-major axis of the photocenter (mas).
    model_param : np.ndarray
        Array containing the astrometric and orbital parameters.
        The expected ordering is
        - ``model_param[0]`` : right ascension (deg)
        - ``model_param[1]`` : declination (deg)
        - ``model_param[2]`` : parallax (mas)
        - ``model_param[3]`` : proper motion in RA (mas yr⁻¹)
        - ``model_param[4]`` : proper motion in Dec (mas yr⁻¹)
        - ``model_param[5]`` : Thiele–Innes constant A (mas)
        - ``model_param[6]`` : Thiele–Innes constant B (mas)
        - ``model_param[7]`` : Thiele–Innes constant F (mas)
        - ``model_param[8]`` : Thiele–Innes constant G (mas)
        - ``model_param[9]`` : orbital period (days)
        - ``model_param[10]`` : eccentricity
        - ``model_param[11]`` : time of periastron passage
    verbose : bool
        Print some information (default: True).

    Returns
    -------
    np.ndarray
        Parameter vector containing the astrometric parameters and
        Campbell orbital elements in the order

        [ra_offset, dec_offset, parallax, pmra, pmdec,
         period, ecc, tau, sma0, inc, aop, pan]

        The inclination, argument of periastron, and PA of the
        ascending node are returned in radians.
    """

    orb_data = {
        "source_id": [source_id],
        "nss_solution_type": ["Orbital"],
        "corr_vec": [[]],
        "ra": [model_param[0]],
        "ra_error": [np.nan],
        "dec": [model_param[1]],
        "dec_error": [np.nan],
        "parallax": [model_param[2]],
        "parallax_error": [np.nan],
        "pmra": [model_param[3]],
        "pmra_error": [np.nan],
        "pmdec": [model_param[4]],
        "pmdec_error": [np.nan],
        "a_thiele_innes": [model_param[5]],
        "a_thiele_innes_error": [np.nan],
        "b_thiele_innes": [model_param[6]],
        "b_thiele_innes_error": [np.nan],
        "f_thiele_innes": [model_param[7]],
        "f_thiele_innes_error": [np.nan],
        "g_thiele_innes": [model_param[8]],
        "g_thiele_innes_error": [np.nan],
        "c_thiele_innes": [np.nan],
        "c_thiele_innes_error": [np.nan],
        "h_thiele_innes": [np.nan],
        "h_thiele_innes_error": [np.nan],
        "period": [model_param[9]],
        "period_error": [np.nan],
        "eccentricity": [model_param[10]],
        "eccentricity_error": [np.nan],
        "t_periastron": [model_param[11]],
        "t_periastron_error": [np.nan],
    }

    df_in = pd.DataFrame(orb_data)

    nss_source = NssSource(star=df_in, indice=0)
    df_campbell = nss_source.campbell()

    if verbose:
        print("\nConversion to campbell elements:")
        print(f"   - Period = {model_param[9]:.3f} days")
        print(f"   - Eccentricity = {model_param[10]:.3f}")
        print(f"   - Relative time of periastron = {model_param[11]:.3f}")
        print(f"   - Semi-major axis of photocenter (mas) = {sma_0:.3f}")
        print(f"   - Inclination (deg) = {df_campbell['inclination'][0]:.3f}")
        print(
            f"   - Argument of periastron (deg) = {df_campbell['arg_periastron'][0]:.3f}"
        )
        print(f"   - PA of ascending node (deg) = {df_campbell['nodeangle'][0]:.3f}")

    param_list = np.hstack(
        [
            model_param[:5],  # [ra_offset, dec_offset, parallax, pm_ra, pm_dec]
            model_param[9],  # per (days)
            model_param[10],  # ecc
            model_param[11],  # tau
            sma_0,  # a_0 (mas)
            np.radians(df_campbell["inclination"][0]),  # inc (rad)
            np.radians(df_campbell["arg_periastron"][0]),  # aop (rad)
            np.radians(df_campbell["nodeangle"][0]),  # pan (rad)
        ]
    )

    return param_list


@beartype
def param_dict_to_list(param_dict: typing.Dict[str, Real]) -> typing.List[Real]:
    """
    Convert an astrometric parameter dictionary to an ordered list.

    Supports parameter sets of length 5 (single star),
    7 (with acceleration), 9 (with acceleration and
    derivative), or 12 (binary orbit). Parameters are
    returned in the required internal order.

    Parameters
    ----------
    param_dict : dict
        Dictionary with 5, 7, 9, or 12 model parameters.

    Returns
    -------
    list(float)
        List with the model parameter values in the
        required internal order.

    Raises
    ------
    ValueError
        If the number of parameters is not one of the supported cases.
    KeyError
        If required parameters are missing from the dictionary.
    """

    if len(param_dict) not in [5, 7, 9, 12]:
        raise ValueError(
            "Number of parameters in the dictionary is "
            f"{len(param_dict)}, whereas there should "
            "be either 5 (single star), 7 (single star + "
            "acceleration), 9 (single star + acceleration + "
            "derivative), or 12 (binary orbit)."
        )

    param_order = ["ra_offset", "dec_offset", "parallax", "pm_ra", "pm_dec"]

    if len(param_dict) in [7, 9]:
        param_order += ["pm_dot_ra", "pm_dot_dec"]

    if len(param_dict) == 9:
        param_order += ["pm_dotdot_ra", "pm_dotdot_dec"]

    if len(param_dict) == 12:
        param_order += ["per", "ecc", "tau", "sma", "inc", "aop", "pan"]

    missing_param = [k for k in param_order if k not in param_dict]

    if len(missing_param) > 0:
        raise KeyError(f"Missing parameters: {missing_param}")

    param_list = [param_dict[k] for k in param_order if k in param_dict]

    return param_list


@beartype
def param_list_to_dict(
    param_list: typing.Union[typing.List[Real], np.ndarray],
) -> typing.Dict[str, Real]:
    """
    Convert an ordered astrometric parameter list to a dictionary.

    Supports parameter sets of length 5 (single star), 7
    (with acceleration), 9 (with acceleration and derivative),
    or 12 (binary orbit).

    Parameters
    ----------
    param_list : dict
        List with 5, 7, 9, or 12 model parameters, in the
        required internal order.

    Returns
    -------
    dict
        Dictionary with the model parameters

    Raises
    ------
    ValueError
        If the number of parameters is not one of the supported cases.
    """

    n_param = len(param_list)

    if n_param not in [5, 7, 9, 12]:
        raise ValueError(
            f"Number of parameters in the 'param_list' is {n_param}, "
            "whereas there should be either 5 (single star), "
            "7 (single star + acceleration), 9 (single star + "
            "acceleration + derivative), or 12 (binary orbit)."
        )

    param_order = ["ra_offset", "dec_offset", "parallax", "pm_ra", "pm_dec"]

    if n_param in [7, 9]:
        param_order += ["pm_dot_ra", "pm_dot_dec"]

    if n_param == 9:
        param_order += ["pm_dotdot_ra", "pm_dotdot_dec"]

    if n_param == 12:
        param_order += ["per", "ecc", "tau", "sma", "inc", "aop", "pan"]

    return dict(zip(param_order, param_list))


@beartype
def binary_bias(
    delta_eta_rel: np.ndarray,
    mass_ratio: float,
    flux_ratio: float,
    verbose: bool,
) -> np.ndarray:
    """
    Compute the along-scan (AL) observation bias caused by binarity.

    The AL bias is defined as the difference between the measured
    AL position (as obtained in IPD) and the AL position of the
    mass centre of the binary.

    The details are provided in Section 2.4 of "Expected
    astrometric properties of binaries in Gaia (E)DR3" by
    L. Lindegren at `ESA's Gaia public documents
    <https://www.cosmos.esa.int/web/gaia/public-dpac-documents>`_.

    Let

    - ρ : angular separation of the binary
    - θ : binary position angle
    - ψ : scan position angle
    - q : mass ratio (q = M₂ / M₁)
    - f : flux ratio (f = F₂ / F₁)
    - Δη : projected AL separation = ρ cos(ψ − θ)

    The bias depends primarily on Δη, the flux ratio f, and
    the mass ratio q.

    Following the model of L. Lindegren, three regimes are
    distinguished depending on the projected separation
    relative to the Gaia resolution unit, u = 90 mas.

    The bias δη is computed as:

    1. :math:`|\\Delta \\eta / u| \\leq 0.1` (unresolved regime),
    :math:`\\delta \\eta = \\left( \\frac{f}{1+f} - \\frac{q}{1+q} \\right)\\Delta \\eta`

    2. :math:`0.1 < |\\Delta \\eta / u| \\leq 3 - f` (partially resolved regime),
    :math:`\\delta \\eta = u\\, B(f, \\Delta \\eta / u) - \\frac{q}{1+q}\\Delta \\eta`

    3. :math:`|\\Delta \\eta / u| > 3 - f` (resolved regime),
    :math:`\\delta \\eta = - \\frac{q}{1+q}\\Delta \\eta`

    where :math:`B(f, p)` is the dimensionless centroid bias function
    defined in Appendix E of the document by L. Lindegren.

    Parameters
    ----------
    delta_eta : np.ndarray
        Projected AL separation Δη (in mas).

    Returns
    -------
    np.ndarray
        Along-scan observation bias δη (in mas), which is
        the displacement of the measured AL position
        relative to the binary mass centre.
    """

    # Gaia angular resolution (mas)

    res_unit = 90.0

    def bias_func(p_param: Real, f_param: Real) -> Real:
        """
        Compute the astrometric centroid bias for a marginally
        resolved binary assuming a Gaussian line-spread function.

        See for details Appendix E in the `ESA's Gaia public documents
        <https://www.cosmos.esa.int/web/gaia/public-dpac-documents>`_
        by L. Lindegren.

        The model assumes:
        (i) both components have identical Gaussian LSFs,
        (ii) the measured centroid is given by the mode of
        the superposed profiles.

        In units where the along-scan coordinate x is expressed
        in Gaussian standard deviations (u), and with origin at
        the primary, the combined profile is (see Eq. 64):

            g(x) = exp(-x^2 / 2) + f * exp(-(x - p)^2 / 2),

        where:
            f = F_companion / F_primary  (0 ≤ f ≤ 1)
            p = Δη / u                   (projected separation in units of u)

        Setting dg/dx = 0 yields (see Eq. 66):

            x = f p / (f + exp(p^2/2 - p x)),

        which is solved here by successive substitution.

        Parameters
        ----------
        p_param : float
            Projected angular separation in units of the
            Gaussian width u (p = Δη / u).

        f_param : float
            Flux ratio of the companion relative to the
            primary (f = F_companion / F_primary, 0 ≤ f ≤ 1).

        Returns
        -------
        float
            The centroid shift x (in units of u), i.e. the
            bias function B(f, p).

        Notes
        -----
        - The iteration starts from x = 0 and proceeds until successive
          estimates differ by less than 1e-6.
        - For f = 1, x = p/2 is always a solution. For sufficiently large |p|
          (|p| ≳ 1.89), this corresponds to a minimum of the double-peaked
          profile, while the iteration converges to the maximum closer to
          the origin.
        - For small f and small p, the solution approaches the
          photocenter at:

                x = f p / (1 + f).
        """

        max_iter = 1000
        abs_tol = 1e-6

        b = 0.0

        for _ in range(max_iter):
            b_next = (
                f_param * p_param / (f_param + np.exp(0.5 * p_param**2 - p_param * b))
            )

            if abs(b_next - b) < abs_tol:
                break

            b = b_next

        return b_next

    if verbose:
        print("\nApplying AL bias from binarity:")

    obs_al_bias = np.zeros(delta_eta_rel.size)

    abs_p = np.abs(delta_eta_rel / res_unit)

    mask1 = abs_p <= 0.1

    if verbose:
        print(f"   - Unresolved epochs = {np.sum(mask1)} / {len(delta_eta_rel)}")

    obs_al_bias[mask1] = (
        flux_ratio / (1.0 + flux_ratio) - mass_ratio / (1.0 + mass_ratio)
    ) * delta_eta_rel[mask1]

    mask2 = (abs_p > 0.1) & (abs_p <= 3.0 - flux_ratio)

    if verbose:
        print(
            f"   - Marginally resolved epochs = {np.sum(mask2)} / {len(delta_eta_rel)}"
        )

    for i in np.where(mask2)[0]:
        bias = bias_func(p_param=delta_eta_rel[i] / res_unit, f_param=flux_ratio)

        obs_al_bias[i] = (
            res_unit * bias - mass_ratio / (1.0 + mass_ratio) * delta_eta_rel[i]
        )

    mask3 = abs_p > 3.0 - flux_ratio

    if verbose:
        print(f"   - Resolved epochs = {np.sum(mask3)} / {len(delta_eta_rel)}")

    obs_al_bias[mask3] = -mass_ratio / (1.0 + mass_ratio) * delta_eta_rel[mask3]

    return obs_al_bias
