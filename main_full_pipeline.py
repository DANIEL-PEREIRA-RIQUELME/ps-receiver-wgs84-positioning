#!/usr/bin/env python3
"""
GPS Receiver: Decoding and Positioning Pipeline
===============================================
Runs the navigation-message decoding and positioning stages on the recorded data:

1. Navigation decoding: preamble sync, parity check and ephemeris extraction from
   the demodulated bitstreams in ``data/bits*-long.mat``.
2. Positioning: orbit propagation, clock and Sagnac corrections and the
   Gauss-Newton WGS-84 solver.

Signal acquisition and tracking of the raw I/Q capture is a separate step
(``main_acquisition.py``), because demodulating 30 s of data takes minutes.
"""
import sys
import time
from pathlib import Path

# Ensure src package is in Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from main_ephemeris_decoding import main as decode_navigation_data
from src.pvt_solver import solve_receiver_position


def main():
    print("=" * 70)
    print("  GPS SDR Receiver: Decoding and Positioning Pipeline")
    print("=" * 70)

    total_t0 = time.perf_counter()

    # Step 1: decode the navigation messages (writes data/correct_sats.mat)
    print("\n[Step 1/2] Decoding navigation messages...")
    decode_navigation_data()

    # Step 2: 3D Positioning Solver
    print("\n[Step 2/2] Solving satellite orbits and receiver position via Gauss-Newton...")
    lat, lon, alt, dops = solve_receiver_position()

    total_time = time.perf_counter() - total_t0

    print("\n" + "=" * 70)
    print("  NAVIGATION & POSITIONING SOLUTION")
    print("=" * 70)
    print(f"  Geodetic System:   WGS-84 (World Geodetic System 1984)")
    print(f"  Latitude:          {lat:12.06f}° N")
    print(f"  Longitude:         {lon:12.06f}° E")
    print(f"  Ellipsoid Height:  {alt:12.02f} m")
    print("-" * 70)
    print("  Dilution of Precision (DOP):")
    print(f"    GDOP (Geometric):  {dops['GDOP']:.2f}")
    print(f"    PDOP (Position):   {dops['PDOP']:.2f}")
    print(f"    HDOP (Horizontal): {dops['HDOP']:.2f}")
    print(f"    VDOP (Vertical):   {dops['VDOP']:.2f}")
    print("-" * 70)
    print(f"  Google Maps Link:  https://www.google.com/maps?q={lat:.06f},{lon:.06f}")
    print(f"  Execution Time:    {total_time * 1000.0:.2f} ms")
    print("=" * 70)


if __name__ == "__main__":
    main()
