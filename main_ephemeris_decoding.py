#!/usr/bin/env python3
"""
GPS Receiver: Navigation Message, Ephemeris & Pseudorange Decoding
==================================================================
Decodes 50 bps GPS navigation bitstreams, aligns subframes via preamble (0x8B),
resolves BPSK sign ambiguity, verifies IS-GPS-200D extended Hamming (32, 26) parity,
extracts Keplerian orbital elements (e, sqrt(a), i0, Omega0, w, M0, etc.),
and calculates satellite pseudoranges.
"""

import sys
from pathlib import Path
import numpy as np
import scipy.io as sio

# Ensure src package is in Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import DATA_DIR
from src.navigation import remove_excess_bits, establish_parity, bits_to_subframes
from src.ephemeris import decode_ephemeris, compute_pseudorange


def main():
    print("=" * 65)
    print("  GPS Navigation Message & Ephemeris Decoder")
    print("=" * 65)

    # Satellite PRNs with long demodulated bitstreams available
    sat_list = [4, 21, 25, 26, 29, 31]
    tau_ref = 1

    decoded_ephemerides = {}
    valid_sats = []

    for prn in sat_list:
        mat_file = DATA_DIR / f"bits{prn:02d}-long.mat"
        if not mat_file.exists():
            print(f"Skipping PRN {prn:02d}: file {mat_file.name} not found.")
            continue

        print(f"\n--- Processing Satellite PRN {prn:02d} ---")
        mat_data = sio.loadmat(str(mat_file))
        raw_bits = mat_data['decodedBits'].flatten() if 'decodedBits' in mat_data else mat_data['bits'].flatten()

        try:
            # 1. Preamble sync & sign ambiguity removal
            clean_bits, first_subframe_idx = remove_excess_bits(raw_bits)

            # 2. Hamming parity validation
            parity_bits, parity_ok = establish_parity(clean_bits)
            if not parity_ok:
                print(f"  [!] Parity check failed for PRN {prn:02d}, but continuing.")
            else:
                print(f"  [✓] Parity check passed for PRN {prn:02d}.")

            # 3. Subframe segmentation
            subframes_matrix, sf_ids = bits_to_subframes(parity_bits)
            print(f"  [✓] Detected Subframe IDs: {sf_ids}")

            # Locate first group of subframes [1, 2, 3]
            page_idx = -1
            for idx in range(len(sf_ids) - 2):
                if sf_ids[idx:idx + 3] == [1, 2, 3]:
                    page_idx = idx
                    break

            if page_idx == -1:
                print(f"  [!] Could not locate complete [1, 2, 3] subframe sequence for PRN {prn:02d}.")
                continue

            page_123 = subframes_matrix[:, page_idx:page_idx + 3]

            # 4. Decode ephemeris parameters
            eph = decode_ephemeris(page_123, prn=prn)
            print(f"  [✓] Ephemeris decoded:")
            print(f"      Week Number:  {eph.wn}")
            print(f"      TOW (SF1-3):  {eph.TOW1} s, {eph.TOW2} s, {eph.TOW3} s")
            print(f"      Semi-major a: {(eph.sqrt_a ** 2) / 1e3:.2f} km")
            print(f"      Eccentricity: {eph.e:.6f}")
            print(f"      Inclination:  {np.degrees(eph.i_0):.2f}°")

            # 5. Compute Pseudorange
            if 'taus' in mat_data:
                taus = mat_data['taus'].flatten()
                rho = compute_pseudorange(taus, tau_ref, first_subframe_idx)
            else:
                # Use stored pseudorange reference
                aux = sio.loadmat(str(DATA_DIR / f"ephemerisAndPseudorange{prn:02d}.mat"))
                rho = float(aux['pseudorange'][0][0])

            print(f"      Pseudorange:  {rho / 1e3:.2f} km")

            decoded_ephemerides[prn] = eph
            valid_sats.append(prn)

        except Exception as err:
            print(f"  [!] Error processing PRN {prn:02d}: {err}")

    # Save validated satellites list
    sio.savemat(str(DATA_DIR / "correct_sats.mat"), {'correct_sats': np.array(valid_sats)})
    print("\n" + "=" * 65)
    print(f"Successfully decoded ephemerides for {len(valid_sats)} satellites: {valid_sats}")
    print("=" * 65)


if __name__ == "__main__":
    main()
