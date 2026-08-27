"""
GPS Signal Tracking and BPSK Bit Demodulation Module
====================================================
Implements bit boundary synchronization, carrier and code tracking loops
(FLL/PLL and DLL), coherent matched filter despreading, and BPSK hard-decision decoding.
"""

from typing import Tuple
import warnings
import numpy as np

from .config import TS, CODES_PER_BIT, SAMPLES_PER_BIT, SAMPLES_PER_CODE
from .ca_code import get_ca_code
from .signal_io import get_signal_data


def find_first_bit_boundary(sat_prn: int, doppler: float, tau: int) -> int:
    """
    Locates the start sample of the first complete navigation data bit by detecting
    the pi phase transition across consecutive C/A code periods (20 C/A periods per bit).

    :param sat_prn: Satellite PRN number.
    :param doppler: Estimated Doppler frequency in Hz.
    :param tau: Sample index of the first C/A code start.
    :return: Sample index corresponding to the start of the first complete bit.
    """
    p = get_ca_code(sat_prn, upsample=True)
    samples_per_code = p.size

    y = get_signal_data(tau, tau + samples_per_code - 1)
    t = np.arange(samples_per_code) * TS
    doppler_corr = np.exp(-1.0j * 2.0 * np.pi * doppler * t)
    p_corr = p * doppler_corr

    last_ip = np.dot(y, p_corr)
    found = False

    while not found:
        tau += samples_per_code
        y = get_signal_data(tau, tau + samples_per_code - 1)
        inner_prod = np.dot(y, p_corr)

        # Correct for phase continuity across blocks
        corrected_ip = inner_prod * np.exp(-1.0j * 2.0 * np.pi * doppler * samples_per_code * TS)

        # A phase difference near pi indicates a bit transition
        phase_diff = np.abs(np.angle(corrected_ip * np.conj(last_ip)))
        if phase_diff > 3.0 * np.pi / 4.0:
            found = True
        last_ip = inner_prod

    # Map tau modulo bit duration
    tau = int(tau % SAMPLES_PER_BIT)
    return tau


def adjust_tau(sat_prn: int, tau: int, doppler: float, delta_tau_range: np.ndarray = np.arange(-2, 3)) -> int:
    """Refines code phase delay tau by evaluating correlation around current estimate."""
    p = get_ca_code(sat_prn, upsample=True)
    bit_p = np.tile(p, CODES_PER_BIT)

    candidate_taus = tau + delta_tau_range
    y = get_signal_data(int(np.min(candidate_taus)), int(np.max(candidate_taus)) + bit_p.size - 1)
    yc = y * np.exp(-1.0j * 2.0 * np.pi * doppler * np.arange(y.size) * TS)

    inner_products = np.zeros(candidate_taus.size, dtype=complex)
    for i in range(candidate_taus.size):
        yd = yc[i: i + bit_p.size]
        inner_products[i] = np.dot(yd, bit_p)

    best_idx = np.argmax(np.abs(inner_products))
    return int(candidate_taus[best_idx])


def adjust_doppler(sat_prn: int, tau: int, doppler: float, f_corr: float = 20.0) -> float:
    """Tracks Doppler frequency drift using a 3-point parabolic polynomial fit."""
    p = get_ca_code(sat_prn, upsample=True)
    bit_p = np.tile(p, CODES_PER_BIT)

    doppler_candidates = doppler + np.array([-f_corr, 0.0, f_corr])
    inner_products = np.zeros(3, dtype=complex)

    y = get_signal_data(tau, tau + bit_p.size - 1)
    for i, fd in enumerate(doppler_candidates):
        d_corr = np.exp(-1.0j * 2.0 * np.pi * fd * np.arange(bit_p.size) * TS)
        inner_products[i] = np.dot(y * d_corr, bit_p)

    # Solve y = a*f^2 + b*f + c via least squares
    vander = np.column_stack([doppler_candidates ** 2, doppler_candidates, np.ones(3)])
    abc = np.linalg.lstsq(vander, np.abs(inner_products), rcond=None)[0]

    if abc[0] >= 0:
        return doppler  # Concave parabola not achieved; preserve current Doppler

    new_doppler = -abc[1] / (2.0 * abc[0])
    return float(new_doppler)


def track_and_demodulate(sat_prn: int, doppler: float, tau: int, max_bits: int = 1500) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tracks a satellite signal and demodulates navigation data bits.

    :param sat_prn: Satellite PRN number.
    :param doppler: Initial Doppler frequency.
    :param tau: Initial bit start sample index.
    :param max_bits: Maximum number of bits to demodulate.
    :return: (decoded_bits, bit_start_taus) where decoded_bits contains {-1, +1}.
    """
    first_bit_tau = find_first_bit_boundary(sat_prn, doppler, tau)
    tau = first_bit_tau

    p = get_ca_code(sat_prn, upsample=True)
    bit_p = np.tile(p, CODES_PER_BIT)

    d_corr = np.exp(-1.0j * 2.0 * np.pi * doppler * np.arange(bit_p.size) * TS)
    p_matched = bit_p * d_corr

    bit_taus = []
    inner_prods = []
    phi = 0.0

    current_doppler = doppler
    current_tau = tau

    for bit_idx in range(max_bits):
        try:
            y = get_signal_data(current_tau, current_tau + bit_p.size - 1)
        except EOFError:
            break

        ip = np.dot(y, p_matched) * np.exp(-1.0j * phi)
        inner_prods.append(ip)
        bit_taus.append(current_tau)

        phi += 2.0 * np.pi * current_doppler * bit_p.size * TS

        # Periodic tracking loop update every 5 bits
        if (bit_idx + 1) % 5 == 0:
            current_tau = adjust_tau(sat_prn, current_tau + bit_p.size, current_doppler)
            current_doppler = adjust_doppler(sat_prn, current_tau, current_doppler)
            d_corr = np.exp(-1.0j * 2.0 * np.pi * current_doppler * np.arange(bit_p.size) * TS)
            p_matched = bit_p * d_corr
        else:
            current_tau += bit_p.size

    ip_arr = np.array(inner_prods, dtype=complex)
    if ip_arr.size == 0:
        return np.array([]), np.array([])

    # BPSK Differential Decision
    decoded_bits = np.zeros(ip_arr.size, dtype=int)
    decoded_bits[0] = 1

    for k in range(1, ip_arr.size):
        diff = ip_arr[k] * np.conj(ip_arr[k - 1])
        if diff.real < 0:
            decoded_bits[k] = -decoded_bits[k - 1]
        else:
            decoded_bits[k] = decoded_bits[k - 1]

    return decoded_bits, np.array(bit_taus)
