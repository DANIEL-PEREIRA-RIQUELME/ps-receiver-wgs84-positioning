#!/usr/bin/env python3
"""
GPS Receiver: 2D Satellite Signal Acquisition & Tracking
=========================================================
Scans all 32 GPS PRN Gold codes across code-phase (delay) and Doppler frequency grids,
identifies visible satellites with high cross-correlation metrics, refines their Doppler shifts,
synchronizes to bit boundaries, and demodulates the 50 bps BPSK navigation data.
"""

import sys
import time
from pathlib import Path
import numpy as np
import scipy.io as sio

# Ensure src package is in Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import DATA_DIR
from src.acquisition import acquire_visible_satellites
from src.tracking import track_and_demodulate


def main():
    print("=" * 65)
    print("  GPS SDR Receiver: Signal Acquisition & Tracking")
    print("=" * 65)

    raw_signal_file = DATA_DIR / "01.mat"
    if not raw_signal_file.exists():
        print(f"Error: Raw signal file {raw_signal_file} not found.")
        return

    t0 = time.perf_counter()

    # 1. 2D Acquisition of visible satellites
    visible_sats = acquire_visible_satellites(num_satellites_to_keep=6, doppler_step_coarse=500.0)

    print("\n--- Demodulating Navigation Bits from Visible Satellites ---")
    for sat in visible_sats:
        prn = sat['prn']
        fd = sat['fd']
        tau = sat['tau']

        print(f"\n[Tracking PRN {prn:02d}] Initial Doppler: {fd:8.2f} Hz, Initial Tau: {tau:5d}")
        bits, bit_taus = track_and_demodulate(prn, fd, tau, max_bits=100)

        print(f"  [✓] Successfully demodulated {bits.size} navigation bits.")
        if bits.size > 0:
            print(f"      First 20 bits: {bits[:20]}")

            # Save short bits capture
            sio.savemat(str(DATA_DIR / f"bits{prn:02d}-short.mat"), {
                'decodedBits': bits,
                'taus': bit_taus
            })

    elapsed = time.perf_counter() - t0
    print("\n" + "=" * 65)
    print(f"Acquisition and Demodulation completed in {elapsed:.2f} seconds.")
    print("=" * 65)


if __name__ == "__main__":
    main()
