#!/usr/bin/env python3
"""
GPS Receiver FFT & Spectral Analysis Plot Generator
===================================================
Generates high-resolution FFT plots comparing:
1. Baseband spectrum of the received GPS signal vs. theoretical C/A Gold code spectrum.
2. Doppler carrier despreading & frequency-domain carrier peak emergence.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FS, TS, PI_GPS, DOCS_DIR
from src.ca_code import get_ca_code
from src.signal_io import get_signal_data


def generate_fft_plot(prn: int = 29, fd: float = -2730.0, tau: int = 890):
    print("Generating comprehensive GPS FFT spectrum plot...")

    # Load 4x upsampled C/A code
    p = get_ca_code(prn, upsample=True)
    n_codes = 20  # 1 bit = 20 ms
    long_p = np.tile(p, n_codes)
    n_samples = long_p.size

    # Load matching raw signal
    y = get_signal_data(tau, tau + n_samples - 1)
    t = np.arange(n_samples) * TS

    # Doppler derotated signal
    y_derotated = y * np.exp(-1.0j * 2.0 * PI_GPS * fd * t)

    # Despread signal (pointwise multiplied by Gold code)
    y_despread = y_derotated * long_p

    # Compute FFTs / PSDs via Welch method for clean smooth spectra
    n_fft = 4096
    f_p, psd_p = welch(long_p, fs=FS, nperseg=n_fft, return_onesided=False)
    f_raw, psd_raw = welch(y, fs=FS, nperseg=n_fft, return_onesided=False)
    f_desp, psd_desp = welch(y_despread, fs=FS, nperseg=n_fft, return_onesided=False)

    # Shift frequencies to center at DC
    freqs_mhz = np.fft.fftshift(f_raw) / 1e6
    psd_p_db = 10.0 * np.log10(np.fft.fftshift(psd_p) + 1e-12)
    psd_raw_db = 10.0 * np.log10(np.fft.fftshift(psd_raw) + 1e-12)
    psd_desp_db = 10.0 * np.log10(np.fft.fftshift(psd_desp) + 1e-12)

    # Normalize for relative comparison
    psd_p_db -= np.max(psd_p_db)
    psd_raw_db -= np.max(psd_raw_db)
    psd_desp_db -= np.max(psd_desp_db)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Panel 1: Spread Spectrum vs Raw Noise Spectrum
    ax1.plot(freqs_mhz, psd_p_db, color='#1f77b4', linewidth=1.2, label='C/A Code Sinc² Envelope (Theoretical)')
    ax1.plot(freqs_mhz, psd_raw_db, color='#d62728', alpha=0.75, linewidth=1.1, label='Raw Antenna Input (Signal Below Noise)')
    ax1.axvline(0, color='black', linestyle=':', alpha=0.6, label='Carrier Center ($0$ MHz)')
    ax1.axvline(-1.023, color='gray', linestyle='--', alpha=0.6, label='First Nulls ($\\pm 1.023$ MHz)')
    ax1.axvline(1.023, color='gray', linestyle='--', alpha=0.6)
    ax1.set_title('GPS L1 C/A Baseband Power Spectral Density (FFT)', fontsize=11, fontweight='bold', pad=10)
    ax1.set_xlabel('Baseband Frequency [MHz]', fontsize=10)
    ax1.set_ylabel('Normalized Power Spectral Density [dB]', fontsize=10)
    ax1.set_xlim([-2.046, 2.046])
    ax1.set_ylim([-35, 5])
    ax1.legend(loc='lower center', frameon=True, framealpha=0.9, fontsize=8.5)

    # Panel 2: Despreading Effect & Carrier Peak Emergence
    ax2.plot(freqs_mhz, psd_raw_db, color='#d62728', alpha=0.6, linewidth=1.0, label='Spread Signal (Pre-Despreading)')
    ax2.plot(freqs_mhz, psd_desp_db, color='#2ca02c', linewidth=1.3, label=f'Despread Signal (Post-Correlation PRN {prn:02d})')
    ax2.axvline(0, color='darkgreen', linestyle=':', linewidth=1.2, label='Concentrated Carrier Energy')
    ax2.set_title('CDMA Despreading Gain in Frequency Domain', fontsize=11, fontweight='bold', pad=10)
    ax2.set_xlabel('Baseband Frequency [MHz]', fontsize=10)
    ax2.set_ylabel('Normalized Power Spectral Density [dB]', fontsize=10)
    ax2.set_xlim([-1.5, 1.5])
    ax2.set_ylim([-35, 5])
    ax2.legend(loc='lower center', frameon=True, framealpha=0.9, fontsize=8.5)

    plt.tight_layout()
    output_file = DOCS_DIR / "figures" / "gps_fft_spectral_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    generate_fft_plot()
