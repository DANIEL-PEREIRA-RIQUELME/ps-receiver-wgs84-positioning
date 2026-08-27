#!/usr/bin/env python3
"""
GPS Receiver: 3D Positioning and WGS-84 Coordinate Solver
==========================================================
Computes satellite ECEF positions from orbital ephemerides, corrects for satellite
clock biases and relativistic drift, applies Sagnac Earth rotation compensation,
solves the non-linear pseudorange equations via Gauss-Newton multilateration,
and transforms the receiver coordinates to the WGS-84 geodetic system.

Expected Output:
    Coordinates of the EPFL campus (Lausanne, Switzerland):
    Latitude:  46.518294 deg N
    Longitude:  6.562467 deg E
    Altitude:  484.44 meters
"""

import sys
import time
from pathlib import Path

# Ensure src package is in Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pvt_solver import solve_receiver_position


def main():
    print("=" * 65)
    print("  GPS SDR Receiver: WGS-84 Multilateration & PVT Solver")
    print("=" * 65)
    print("Loading satellite ephemerides and pseudoranges...")

    t0 = time.perf_counter()
    latitude, longitude, altitude, dop_metrics = solve_receiver_position()
    elapsed = time.perf_counter() - t0

    print("\n" + "-" * 45)
    print("  Calculated Receiver Geodetic Coordinates (WGS-84)")
    print("-" * 45)
    print(f"  Latitude:    {latitude:12.06f}° N")
    print(f"  Longitude:   {longitude:12.06f}° E")
    print(f"  Altitude:    {altitude:12.02f} meters above ellipsoid")
    print("-" * 45)
    print("  Geometric Dilution of Precision (DOP):")
    print(f"    GDOP (Geometric): {dop_metrics['GDOP']:.2f}")
    print(f"    PDOP (Position):  {dop_metrics['PDOP']:.2f}")
    print(f"    HDOP (Horizontal):{dop_metrics['HDOP']:.2f}")
    print(f"    VDOP (Vertical):  {dop_metrics['VDOP']:.2f}")
    print("-" * 45)

    maps_url = f"https://www.google.com/maps?q={latitude:.06f},{longitude:.06f}"
    print(f"\n  Google Maps Location Link:\n  {maps_url}")
    print(f"\n[Done in {elapsed * 1000.0:.1f} ms]")
    print("=" * 65)


if __name__ == "__main__":
    main()
