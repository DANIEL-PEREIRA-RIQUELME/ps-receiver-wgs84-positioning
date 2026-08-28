#!/usr/bin/env python3
"""
GPS Receiver Visualization Suite: 2D Acquisition & Doppler Spectrum
===================================================================
Generates publication-quality figures illustrating:
1. 2D Code-Phase / Doppler Acquisition surface and Gold code correlation peak.
2. Doppler frequency spectrum, parabolic polynomial fit, and baseband FFT before/after derotation.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate as corr, welch

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

# Add root directory to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FS, TS, PI_GPS, SAMPLES_PER_CODE, DATA_DIR, DOCS_DIR
from src.ca_code import get_ca_code
from src.signal_io import get_signal_data


def generate_acquisition_plots(prn: int = 29):
    """Generates 2D acquisition surface and code-phase cross-correlation plot."""
    print(f"Generating 2D Acquisition & Correlation plot for PRN {prn:02d}...")

    # Load 4x upsampled C/A code
    p = get_ca_code(prn, upsample=True)
    y = get_signal_data(1, 2 * p.size - 1)

    # 2D search grid
    doppler_grid = np.linspace(-5000.0, 5000.0, 101) # 100 Hz steps
    tau_axis = np.arange(p.size)
    corr_matrix = np.zeros((len(doppler_grid), p.size))
    t = np.arange(y.size) * TS

    for idx, fd in enumerate(doppler_grid):
        ydc = y * np.exp(-1.0j * 2.0 * PI_GPS * fd * t)
        r_corr = np.abs(corr(ydc, p, mode='valid', method='fft'))
        corr_matrix[idx, :] = r_corr[:p.size]

    # Best peak
    max_idx = np.unravel_index(np.argmax(corr_matrix), corr_matrix.shape)
    best_fd = doppler_grid[max_idx[0]]
    best_tau = max_idx[1]
    peak_val = corr_matrix[max_idx]

    # Theoretical Gold code autocorrelation
    p_single = get_ca_code(prn, upsample=False)
    gold_auto = np.abs(corr(p_single, p_single, mode='same'))
    lags_chips = np.arange(-511, 512)

    # Create figure with 3 subplots
    fig = plt.figure(figsize=(14, 9), dpi=300)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0])

    # 1. 2D Acquisition Heatmap / Surface
    ax1 = fig.add_subplot(gs[0, :])
    im = ax1.imshow(corr_matrix, aspect='auto', extent=[0, p.size, doppler_grid[0], doppler_grid[-1]],
                    cmap='viridis', origin='lower')
    ax1.plot(best_tau, best_fd, 'r*', markersize=14, markeredgewidth=1.5, markeredgecolor='white',
             label=f'Peak Acquisition: $\\tau={best_tau}$ samples, $f_d={best_fd:.0f}$ Hz')
    ax1.set_title(f'GPS L1 C/A 2D Parallel Acquisition Search Surface (Satellite PRN {prn:02d})',
                  fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel('Code Phase Offset $\\tau$ [samples @ $f_s = 4.092$ MHz]', fontsize=11)
    ax1.set_ylabel('Doppler Shift $f_d$ [Hz]', fontsize=11)
    ax1.legend(loc='upper right', frameon=True, framealpha=0.9)
    cbar = fig.colorbar(im, ax=ax1, pad=0.02)
    cbar.set_label('Cross-Correlation Magnitude $|R(\\tau, f_d)|$', fontsize=10)

    # 2. 1D Cross-correlation profile at optimal Doppler
    ax2 = fig.add_subplot(gs[1, 0])
    slice_1d = corr_matrix[max_idx[0], :]
    noise_floor = np.mean(slice_1d)
    ax2.plot(tau_axis, slice_1d, color='#1f77b4', linewidth=1.2, label='Cross-Correlation $|R(\\tau)|$')
    ax2.axvline(best_tau, color='red', linestyle='--', linewidth=1.5,
                label=f'Code Delay $\\tau = {best_tau}$ samples')
    ax2.axhline(noise_floor, color='gray', linestyle=':', label=f'Mean Noise Floor ({noise_floor:.1f})')
    ax2.set_title(f'Cross-Correlation at Peak Doppler ($f_d = {best_fd:.0f}$ Hz)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Code Delay $\\tau$ [samples]', fontsize=10)
    ax2.set_ylabel('Correlation Amplitude', fontsize=10)
    ax2.set_xlim([0, p.size])
    ax2.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=9)

    # 3. Gold Code Autocorrelation Properties
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(lags_chips, gold_auto / 1023.0, color='#2ca02c', linewidth=1.2, label='Normalized Autocorrelation')
    ax3.axhline(65.0 / 1023.0, color='crimson', linestyle=':', linewidth=1.0, label='Gold Code Bound (+65/1023)')
    ax3.axhline(-65.0 / 1023.0, color='crimson', linestyle=':', linewidth=1.0)
    ax3.set_title(f'1023-Chip Gold Code Autocorrelation ($R_{{PRN}}(m)$)', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Chip Lag $m$ [chips]', fontsize=10)
    ax3.set_ylabel('Normalized $R(m)$', fontsize=10)
    ax3.set_xlim([-100, 100])
    ax3.set_ylim([-0.15, 1.05])
    ax3.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=9)

    plt.tight_layout()
    output_path = DOCS_DIR / "figures" / "gps_acquisition_2d_correlation.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_doppler_fft_plots(prn: int = 29):
    """Generates Doppler frequency estimation curve and Baseband FFT PSD plot."""
    print(f"Generating Doppler Spectrum & FFT plot for PRN {prn:02d}...")

    # Load signal and C/A code
    p = get_ca_code(prn, upsample=True)
    samples_per_code = p.size
    n_codes = 10
    long_p = np.tile(p, n_codes)
    t_10 = np.arange(samples_per_code * n_codes) * TS

    tau_best = 890
    y_raw = get_signal_data(tau_best, tau_best + samples_per_code * n_codes - 1)

    # Fine Doppler sweep
    doppler_sweep = np.linspace(-4000.0, -1500.0, 251)
    ip_vals = np.zeros(len(doppler_sweep))

    for idx, fd in enumerate(doppler_sweep):
        ydc = y_raw * np.exp(-1.0j * 2.0 * PI_GPS * fd * t_10)
        ip_vals[idx] = np.abs(np.dot(long_p, np.conj(ydc)))

    best_doppler_idx = np.argmax(ip_vals)
    peak_doppler = doppler_sweep[best_doppler_idx]

    # Parabolic fit around peak
    p_doppler_pts = doppler_sweep[best_doppler_idx - 10 : best_doppler_idx + 11]
    p_ip_pts = ip_vals[best_doppler_idx - 10 : best_doppler_idx + 11]
    poly_coeffs = np.polyfit(p_doppler_pts, p_ip_pts, 2)
    fit_x = np.linspace(p_doppler_pts[0], p_doppler_pts[-1], 100)
    fit_y = np.polyval(poly_coeffs, fit_x)
    exact_peak_fd = -poly_coeffs[1] / (2.0 * poly_coeffs[0])

    # Power Spectral Density (FFT) before and after Doppler compensation
    n_fft = 8192
    y_large = get_signal_data(1, 65536)
    t_large = np.arange(y_large.size) * TS
    y_derotated = y_large * np.exp(-1.0j * 2.0 * PI_GPS * exact_peak_fd * t_large)

    freqs, psd_raw = welch(y_large, fs=FS, nperseg=n_fft, return_onesided=False)
    _, psd_derot = welch(y_derotated, fs=FS, nperseg=n_fft, return_onesided=False)

    freqs_khz = np.fft.fftshift(freqs) / 1e3
    psd_raw_db = 10.0 * np.log10(np.fft.fftshift(psd_raw))
    psd_derot_db = 10.0 * np.log10(np.fft.fftshift(psd_derot))

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # 1. Doppler Peak & Parabolic Interpolation
    ax1.plot(doppler_sweep, ip_vals, color='#333333', linewidth=1.3, label='Coherent Metric $|\\langle y_c, p \\rangle|$')
    ax1.plot(p_doppler_pts, p_ip_pts, 'o', color='#1f77b4', markersize=5, label='Sampled Grid Points')
    ax1.plot(fit_x, fit_y, color='#d62728', linewidth=2.0, linestyle='--',
             label=f'Parabolic Fit: $f_d = {exact_peak_fd:.2f}$ Hz')
    ax1.axvline(exact_peak_fd, color='#d62728', linestyle=':', alpha=0.8)
    ax1.set_title(f'Carrier Frequency Offset (CFO) Doppler Peak Fitting (PRN {prn:02d})',
                  fontsize=11, fontweight='bold', pad=10)
    ax1.set_xlabel('Doppler Frequency $f_d$ [Hz]', fontsize=10)
    ax1.set_ylabel('Coherent Inner Product Magnitude', fontsize=10)
    ax1.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=9)

    # 2. Baseband Power Spectral Density (FFT)
    ax2.plot(freqs_khz, psd_raw_db, color='#ff7f0e', alpha=0.6, linewidth=1.0, label='Raw Baseband Spectrum')
    ax2.plot(freqs_khz, psd_derot_db, color='#1f77b4', linewidth=1.2,
             label=f'Doppler Compensated ($f_d={exact_peak_fd:.1f}$ Hz)')
    ax2.axvline(0, color='black', linestyle=':', alpha=0.7, label='Center DC ($0$ Hz)')
    ax2.axvline(-1023, color='gray', linestyle='--', alpha=0.7, label='L1 C/A Main Lobe ($\\pm 1.023$ MHz)')
    ax2.axvline(1023, color='gray', linestyle='--', alpha=0.7)
    ax2.set_title('Baseband Power Spectral Density (Welch PSD / FFT)', fontsize=11, fontweight='bold', pad=10)
    ax2.set_xlabel('Frequency offset relative to L1 Carrier [kHz]', fontsize=10)
    ax2.set_ylabel('Power Spectral Density [dB/Hz]', fontsize=10)
    ax2.set_xlim([-2046, 2046])
    ax2.legend(loc='lower center', frameon=True, framealpha=0.9, fontsize=8.5)

    plt.tight_layout()
    output_path = DOCS_DIR / "figures" / "gps_doppler_fft_spectrum.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    (DOCS_DIR / "figures").mkdir(parents=True, exist_ok=True)
    generate_acquisition_plots(prn=29)
    generate_doppler_fft_plots(prn=29)
    print("All acquisition and Doppler figures successfully generated!")
