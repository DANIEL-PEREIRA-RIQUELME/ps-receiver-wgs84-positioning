"""
GPS PVT (Position, Velocity, Time) and WGS-84 Multilateration Solver
=====================================================================
Calculates satellite ECEF orbits, applies relativistic and satellite clock corrections,
compensates for Sagnac Earth rotation, solves non-linear pseudorange equations via Gauss-Newton,
and converts coordinates to WGS-84 (Latitude, Longitude, Altitude).
"""

from typing import Tuple, List, Optional, Union, Dict
import math
import numpy as np
import scipy.io as sio

from .config import (
    C, MU_E, OMEGA_DOT_E, F_REL, WGS84_A, WGS84_F, WGS84_E2,
    PI_GPS, DATA_DIR
)
from .ephemeris import Ephemeris


def limit_valid_range(t_in: float) -> float:
    """Wraps ephemeris time difference within [-302400, +302400] seconds."""
    half_range = 302400.0
    if t_in < -half_range:
        return t_in + 2.0 * half_range
    elif t_in > half_range:
        return t_in - 2.0 * half_range
    return t_in


def calc_eccentric_anomaly(eph: Ephemeris, t: float, tol: float = 1e-14) -> float:
    """
    Computes eccentric anomaly E_k by iteratively solving Kepler's equation:
    M_k = E_k - e * sin(E_k)
    """
    a_s = eph.sqrt_a ** 2
    n_0 = math.sqrt(MU_E / (a_s ** 3))
    n = n_0 + eph.delta_n

    t_k = limit_valid_range(t - eph.t_oe)
    m_k = eph.M_0 + n * t_k

    # Iterative Newton-Raphson / fixed-point solver
    e_k = m_k + eph.e * math.sin(m_k)
    e_old = m_k
    err = 1.0

    while err >= tol:
        e_k = m_k + eph.e * math.sin(e_old)
        err = abs(e_k - e_old)
        e_old = e_k

    return e_k


def calc_satellite_clock_bias(eph: Ephemeris, e_k: float, t: float) -> float:
    """
    Calculates total satellite clock offset including 2nd-order polynomial,
    relativistic eccentricity correction, and group delay differential (T_GD).
    """
    a_s = eph.sqrt_a ** 2
    delta_t_rel = F_REL * eph.e * math.sqrt(a_s) * math.sin(e_k)

    dt = limit_valid_range(t - eph.t_oc)
    delta_t_sv = eph.a_f0 + eph.a_f1 * dt + eph.a_f2 * (dt ** 2) + delta_t_rel
    return delta_t_sv - eph.T_GD


def calc_satellite_position(eph: Ephemeris, t: float) -> np.ndarray:
    """
    Computes satellite 3D position in Earth-Centered, Earth-Fixed (ECEF) frame at GPS time t.

    :param eph: Ephemeris parameters.
    :param t: GPS time in seconds.
    :return: 1D array of [x, y, z] in meters.
    """
    e_k = calc_eccentric_anomaly(eph, t)
    t_k = limit_valid_range(t - eph.t_oe)
    a_s = eph.sqrt_a ** 2

    # True anomaly
    sin_nu = math.sqrt(max(0.0, 1.0 - eph.e ** 2)) * math.sin(e_k)
    cos_nu = math.cos(e_k) - eph.e
    nu_k = math.atan2(sin_nu, cos_nu)

    # Argument of latitude
    phi_k = nu_k + eph.w

    # Second harmonic perturbations
    sin_2phi = math.sin(2.0 * phi_k)
    cos_2phi = math.cos(2.0 * phi_k)

    delta_u_k = eph.C_us * sin_2phi + eph.C_uc * cos_2phi
    delta_r_k = eph.C_rs * sin_2phi + eph.C_rc * cos_2phi
    delta_i_k = eph.C_is * sin_2phi + eph.C_ic * cos_2phi

    # Corrected orbital parameters
    u_k = phi_k + delta_u_k
    r_k = a_s * (1.0 - eph.e * math.cos(e_k)) + delta_r_k
    i_k = eph.i_0 + eph.idot * t_k + delta_i_k

    # Position in orbital plane
    x_prim = r_k * math.cos(u_k)
    y_prim = r_k * math.sin(u_k)

    # Longitude of ascending node
    omega_k = eph.Omega_0 + (eph.Omegadot - OMEGA_DOT_E) * t_k - OMEGA_DOT_E * eph.t_oe

    # ECEF Cartesian Coordinates
    x_k = x_prim * math.cos(omega_k) - y_prim * math.cos(i_k) * math.sin(omega_k)
    y_k = x_prim * math.sin(omega_k) + y_prim * math.cos(i_k) * math.cos(omega_k)
    z_k = y_prim * math.sin(i_k)

    return np.array([x_k, y_k, z_k], dtype=float)


def rotate_z(pos: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotates a 3D coordinate vector around the Z-axis by angle_rad."""
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    rot_mat = np.array([
        [cos_a, sin_a, 0.0],
        [-sin_a, cos_a, 0.0],
        [0.0, 0.0, 1.0]
    ])
    return rot_mat.dot(pos.T).T


def solve_range_equations_newton(sat_positions: np.ndarray, corrected_pseudoranges: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Solves non-linear multilateration pseudorange equations using Gauss-Newton iteration.

    :param sat_positions: (N x 3) matrix of satellite ECEF coordinates.
    :param corrected_pseudoranges: 1D array of N corrected pseudoranges.
    :return: (receiver_position_ecef, clock_bias_b, residuals_vector)
    """
    num_sats = sat_positions.shape[0]
    if num_sats < 4:
        raise ValueError(f"Multilateration requires >= 4 satellites, got {num_sats}.")

    rp = np.zeros(3)
    b = 0.0

    f = np.zeros(num_sats)
    jacobian = np.zeros((4, num_sats))

    for _ in range(50):
        for k in range(num_sats):
            diff_vec = sat_positions[k] - rp
            dist = np.linalg.norm(diff_vec)
            f[k] = dist - (corrected_pseudoranges[k] + b)

            jacobian[:3, k] = -diff_vec / dist
            jacobian[3, k] = -1.0

        delta = -f.dot(np.linalg.pinv(jacobian))
        rp += delta[:3]
        b += delta[3]

        if np.linalg.norm(delta) < 1e-4:
            break

    return rp, b, f


def ecef_to_wgs84(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """
    Converts Cartesian ECEF coordinates (x, y, z) to WGS-84 Geodetic coordinates
    (Latitude, Longitude, Altitude above ellipsoid).

    :return: (latitude_deg, longitude_deg, altitude_meters)
    """
    # Longitude
    lon_rad = math.atan2(y, x)
    lon_deg = math.degrees(lon_rad)

    # Iterative latitude & altitude solver
    p = math.sqrt(x ** 2 + y ** 2)
    lat_rad = math.atan2(z, p * (1.0 - WGS84_E2))

    for _ in range(10):
        n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * (math.sin(lat_rad) ** 2))
        h = p / math.cos(lat_rad) - n
        lat_rad = math.atan2(z, p * (1.0 - WGS84_E2 * (n / (n + h))))

    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg, h


def compute_dop(sat_positions: np.ndarray, receiver_pos: np.ndarray) -> Dict[str, float]:
    """Computes Dilution of Precision (GDOP, PDOP, HDOP, VDOP)."""
    num_sats = sat_positions.shape[0]
    g_matrix = np.zeros((num_sats, 4))

    for i in range(num_sats):
        diff = sat_positions[i] - receiver_pos
        dist = np.linalg.norm(diff)
        g_matrix[i, :3] = -diff / dist
        g_matrix[i, 3] = 1.0

    q_mat = np.linalg.inv(g_matrix.T @ g_matrix)
    gdop = math.sqrt(np.trace(q_mat))
    pdop = math.sqrt(q_mat[0, 0] + q_mat[1, 1] + q_mat[2, 2])
    hdop = math.sqrt(q_mat[0, 0] + q_mat[1, 1])
    vdop = math.sqrt(q_mat[2, 2])

    return {'GDOP': gdop, 'PDOP': pdop, 'HDOP': hdop, 'VDOP': vdop}


def solve_receiver_position(satellite_list: Optional[Union[List[int], np.ndarray]] = None) -> Tuple[float, float, float, Dict[str, float]]:
    """
    End-to-end WGS-84 receiver positioning solver.

    :param satellite_list: Optional list of satellite PRNs. Defaults to correct_sats.mat.
    :return: (latitude, longitude, altitude, dop_metrics)
    """
    if satellite_list is None:
        correct_file = DATA_DIR / "correct_sats.mat"
        data = sio.loadmat(str(correct_file))
        satellite_list = data['correct_sats'].flatten()

    sat_array = np.array(satellite_list)
    num_sats = len(sat_array)

    sat_positions = np.zeros((num_sats, 3))
    corrected_pseudoranges = np.zeros(num_sats)

    for idx, sat_prn in enumerate(sat_array):
        eph_file = DATA_DIR / f"ephemerisAndPseudorange{sat_prn:02d}.mat"
        aux = sio.loadmat(str(eph_file))
        rho = float(aux['pseudorange'][0][0])

        eph = Ephemeris()
        eph.load_from_mat(eph_file)

        t_tr = eph.t_tr

        # Iterative satellite clock bias determination
        delta_t_s = 0.0
        time_sat = t_tr - rho / C
        t = time_sat

        for _ in range(20):
            last_dt = delta_t_s
            e_k = calc_eccentric_anomaly(eph, t)
            delta_t_s = calc_satellite_clock_bias(eph, e_k, t)
            t = time_sat - delta_t_s
            if abs(delta_t_s - last_dt) < 1e-10:
                break

        # Corrected pseudorange
        rho_c = rho + delta_t_s * C
        corrected_pseudoranges[idx] = rho_c

        # Satellite position at transmission time in ECEF
        sat_pos_tx = calc_satellite_position(eph, t_tr - rho_c / C)

        # Sagnac Earth rotation compensation to align with ECEF(t_tr)
        sat_positions[idx, :] = rotate_z(sat_pos_tx, OMEGA_DOT_E * rho_c / C)

    # First solve with all satellites
    rp_ecef, b, residuals = solve_range_equations_newton(sat_positions, corrected_pseudoranges)

    # Refine using top 4 most consistent satellites (smallest residuals)
    best_indices = np.argsort(np.abs(residuals))[:4]
    rp_ecef, b, residuals = solve_range_equations_newton(sat_positions[best_indices, :], corrected_pseudoranges[best_indices])

    # Sagnac Earth rotation during receiver signal transit time (b / c)
    rp_ecef = rotate_z(rp_ecef, OMEGA_DOT_E * b / C)

    # Convert to WGS-84
    lat, lon, alt = ecef_to_wgs84(rp_ecef[0], rp_ecef[1], rp_ecef[2])
    dops = compute_dop(sat_positions[best_indices, :], rp_ecef)

    return lat, lon, alt, dops
