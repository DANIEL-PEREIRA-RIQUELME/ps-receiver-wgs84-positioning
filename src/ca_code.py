"""
GPS Coarse/Acquisition (C/A) Gold Code Generator
================================================
Implements 1023-chip Gold code sequence generation for all 32 GPS PRNs
using 10-stage Linear Feedback Shift Registers (LFSR) G1 and G2 per IS-GPS-200D.
"""

from typing import Optional, Union
import numpy as np
import scipy.io as sio

from .config import DATA_DIR, SAMPLES_PER_CHIP, CHIPS_PER_CODE

# G2 phase selection tap pairs for GPS PRN 1 through 32 (1-indexed)
PRN_G2_TAPS = {
    1: (2, 6), 2: (3, 7), 3: (4, 8), 4: (5, 9), 5: (1, 9),
    6: (2, 10), 7: (1, 8), 8: (2, 9), 9: (3, 10), 10: (2, 3),
    11: (3, 4), 12: (5, 6), 13: (6, 7), 14: (7, 8), 15: (8, 9),
    16: (9, 10), 17: (1, 4), 18: (2, 5), 19: (3, 6), 20: (4, 7),
    21: (5, 8), 22: (6, 9), 23: (1, 3), 24: (4, 6), 25: (5, 7),
    26: (6, 8), 27: (7, 9), 28: (8, 10), 29: (1, 6), 30: (2, 7),
    31: (3, 8), 32: (4, 9)
}

_CA_CODE_CACHE: Optional[np.ndarray] = None


def generate_ca_code(prn: int) -> np.ndarray:
    """
    Generate a 1023-chip BPSK Gold code sequence {-1, +1} for a given PRN.

    :param prn: Satellite PRN number (1 to 32).
    :return: 1D numpy array of 1023 chips in {-1, +1}.
    """
    if prn not in PRN_G2_TAPS:
        raise ValueError(f"Invalid PRN: {prn}. Must be between 1 and 32.")

    # LFSR polynomials (1-indexed tap positions):
    # G1: 1 + x^3 + x^10 -> feedback taps 3 and 10
    # G2: 1 + x^2 + x^3 + x^6 + x^8 + x^9 + x^10 -> feedback taps 2, 3, 6, 8, 9, 10
    g1 = np.ones(10, dtype=int)
    g2 = np.ones(10, dtype=int)

    tap1, tap2 = PRN_G2_TAPS[prn]
    code = np.zeros(CHIPS_PER_CODE, dtype=int)

    for i in range(CHIPS_PER_CODE):
        # Output is G1[10] XOR G2[tap1] XOR G2[tap2]
        g1_out = g1[9]
        g2_out = g2[tap1 - 1] ^ g2[tap2 - 1]
        code[i] = g1_out ^ g2_out

        # G1 feedback: tap 3 XOR tap 10
        g1_fb = g1[2] ^ g1[9]
        # G2 feedback: tap 2 XOR tap 3 XOR tap 6 XOR tap 8 XOR tap 9 XOR tap 10
        g2_fb = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]

        # Shift registers
        g1 = np.roll(g1, 1)
        g1[0] = g1_fb
        g2 = np.roll(g2, 1)
        g2[0] = g2_fb

    # Convert binary {0, 1} to BPSK {-1, +1}
    return 1 - 2 * code


def get_ca_code(prn: int, upsample: bool = False) -> np.ndarray:
    """
    Retrieve C/A code for a satellite, with optional upsampling to sampling rate.

    :param prn: Satellite PRN number (1 to 32, or 0-indexed integer if from matrix).
    :param upsample: If True, upsamples the code by SAMPLES_PER_CHIP (4x).
    :return: 1D numpy array with the C/A code.
    """
    global _CA_CODE_CACHE

    # Load pre-generated CA codes matrix if available
    if _CA_CODE_CACHE is None:
        ca_file = DATA_DIR / "CAcodes.mat"
        if ca_file.exists():
            data = sio.loadmat(str(ca_file))
            _CA_CODE_CACHE = data['satCAcodes']

    if _CA_CODE_CACHE is not None and prn < _CA_CODE_CACHE.shape[0]:
        c = _CA_CODE_CACHE[prn, :].astype(float)
    else:
        actual_prn = prn if prn in PRN_G2_TAPS else prn + 1
        c = generate_ca_code(actual_prn).astype(float)

    if upsample:
        c = np.kron(c, np.ones(SAMPLES_PER_CHIP))

    return c
