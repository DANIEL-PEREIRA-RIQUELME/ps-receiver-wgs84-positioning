"""
GPS Satellite Signal Acquisition Module
=======================================
Implements 2D parallel code-phase and Doppler frequency search across
all 32 GPS PRN Gold codes, providing coarse and fine Doppler/delay estimation.
"""

from typing import Tuple, List, Dict
import numpy as np
from scipy.signal import correlate as corr

from .config import TS, MAX_DOPPLER, PI_GPS, SAMPLES_PER_CODE
from .ca_code import get_ca_code
from .signal_io import get_signal_data


def coarse_estimate(sat_prn: int, doppler_step: float = 500.0) -> Tuple[float, int, float]:
    """
    Perform 2D parallel search over Doppler and delay grid for a given satellite PRN.

    :param sat_prn: Satellite PRN number (1 to 32, or 0-indexed integer).
    :param doppler_step: Frequency grid step in Hz (default: 500 Hz).
    :return: (doppler_estim, tau_estim, peak_metric) where:
        - doppler_estim: coarse Doppler shift estimate in Hz.
        - tau_estim: 1-indexed sample position of the first complete C/A code.
        - peak_metric: maximum cross-correlation magnitude.
    """
    # Generate 4x upsampled C/A code (length = 4092 samples)
    p = get_ca_code(sat_prn, upsample=True)

    # Read twice the C/A code length minus 1 samples (8183 samples)
    y = get_signal_data(1, 2 * p.size - 1)

    coarse_doppler_grid = np.linspace(-MAX_DOPPLER / 2.0, MAX_DOPPLER / 2.0, int(MAX_DOPPLER / doppler_step) + 1)
    t = np.arange(y.size) * TS

    peak_metric = -float('inf')
    best_doppler = 0.0
    best_tau = 1

    for fd in coarse_doppler_grid:
        # Time-domain Doppler frequency derotation
        ydc = y * np.exp(-1.0j * 2.0 * PI_GPS * fd * t)
        # Fast Fourier Transform based valid cross-correlation
        r_corr = np.abs(corr(ydc, p, mode='valid', method='fft'))

        tau = np.argmax(r_corr)
        r_max = r_corr[tau]

        if r_max > peak_metric:
            peak_metric = r_max
            best_doppler = fd
            best_tau = tau + 1  # 1-indexed sample offset

    return best_doppler, best_tau, peak_metric


def fine_estimate(sat_prn: int, coarse_tau: int, coarse_doppler: float, doppler_step: float = 20.0) -> Tuple[float, float]:
    """
    Refines the Doppler frequency estimate using coherent inner products over 10 C/A code blocks.

    :param sat_prn: Satellite PRN number.
    :param coarse_tau: Coarse delay estimate from coarse_estimate.
    :param coarse_doppler: Coarse Doppler estimate in Hz.
    :param doppler_step: Fine frequency search resolution in Hz (default: 20 Hz).
    :return: (refined_doppler, peak_inner_product)
    """
    p = get_ca_code(sat_prn, upsample=True)
    samples_per_code = p.size

    # Coherent integration across N=10 C/A code periods (10 ms)
    n_codes = 10
    long_p = np.tile(p, n_codes)
    t = np.arange(samples_per_code * n_codes) * TS

    coarse_tau = int(coarse_tau)
    y1 = get_signal_data(coarse_tau, coarse_tau + samples_per_code * n_codes - 1)
    y2 = get_signal_data(coarse_tau + n_codes * samples_per_code, coarse_tau + samples_per_code * 2 * n_codes - 1)

    peak_ip = 0.0
    best_doppler = coarse_doppler
    doppler_set = doppler_step * np.linspace(-20, 20, 41) + coarse_doppler

    for fd in doppler_set:
        doppler_corr = np.exp(-1.0j * 2.0 * PI_GPS * fd * t)
        ydc1 = y1 * doppler_corr
        ydc2 = y2 * doppler_corr

        ip1 = np.abs(np.dot(long_p, np.conj(ydc1)))
        ip2 = np.abs(np.dot(long_p, np.conj(ydc2)))
        ip_max = max(ip1, ip2)

        if ip_max > peak_ip:
            peak_ip = ip_max
            best_doppler = fd

    return best_doppler, peak_ip


def acquire_visible_satellites(num_satellites_to_keep: int = 6, doppler_step_coarse: float = 500.0) -> List[Dict]:
    """
    Scan all 32 GPS satellite PRNs, identify visible satellites, and refine their parameters.

    :param num_satellites_to_keep: Number of strongest visible satellites to retain.
    :param doppler_step_coarse: Coarse grid resolution in Hz.
    :return: List of dictionaries with satellite PRN, Doppler, tau, and metric.
    """
    results = []

    print(f"Scanning 32 GPS PRNs for visible satellites...")
    for prn in range(1, 33):
        fd_coarse, tau_coarse, metric = coarse_estimate(prn, doppler_step=doppler_step_coarse)
        results.append({
            'prn': prn,
            'fd_coarse': fd_coarse,
            'tau_coarse': tau_coarse,
            'metric': metric
        })
        print(f"PRN {prn:02d}: R = {metric:7.2f}, Coarse fd = {fd_coarse:6.0f} Hz, tau = {tau_coarse:4d}")

    # Sort descending by correlation peak metric
    results.sort(key=lambda x: x['metric'], reverse=True)
    selected = results[:num_satellites_to_keep]

    print(f"\nTop {num_satellites_to_keep} Visible Satellites Selected: {[s['prn'] for s in selected]}")
    print("Refining Doppler frequency estimates...")

    refined_sats = []
    print("-" * 45)
    print(f"{'PRN':>4} | {'Peak Metric':>12} | {'Doppler (Hz)':>12} | {'Tau':>6}")
    print("-" * 45)

    for sat in selected:
        fd_fine, fine_ip = fine_estimate(sat['prn'], sat['tau_coarse'], sat['fd_coarse'], doppler_step=20.0)
        entry = {
            'prn': sat['prn'],
            'fd': fd_fine,
            'tau': sat['tau_coarse'],
            'metric': fine_ip
        }
        refined_sats.append(entry)
        print(f"{entry['prn']:4d} | {entry['metric']:12.1f} | {entry['fd']:12.2f} | {entry['tau']:6d}")
    print("-" * 45)

    return refined_sats
