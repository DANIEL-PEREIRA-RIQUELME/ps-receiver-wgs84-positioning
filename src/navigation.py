"""
GPS Navigation Message Frame and Parity Decoding Module
========================================================
Implements preamble synchronization, BPSK sign ambiguity resolution,
and IS-GPS-200D extended Hamming (32, 26) word parity check algorithm.
"""

from typing import Tuple, List, Union
import math
import numpy as np
from scipy.signal import correlate as corr

from .config import PREAMBLE, BITS_PER_SUBFRAME, BITS_PER_WORD, DATA_BITS_PER_WORD, WORDS_PER_SUBFRAME

# Parity check matrix H (26 x 6) from IS-GPS-200D Table 20-XIV
PARITY_H = np.array([
    [1, 0, 1, 0, 1, 0],  # d1
    [1, 1, 0, 1, 0, 0],  # d2
    [1, 1, 1, 0, 1, 1],  # d3
    [0, 1, 1, 1, 0, 0],  # d4
    [1, 0, 1, 1, 1, 1],  # d5
    [1, 1, 0, 1, 1, 1],  # d6
    [0, 1, 1, 0, 1, 0],  # d7
    [0, 0, 1, 1, 0, 1],  # d8
    [0, 0, 0, 1, 1, 1],  # d9
    [1, 0, 0, 0, 1, 1],  # d10
    [1, 1, 0, 0, 0, 1],  # d11
    [1, 1, 1, 0, 0, 0],  # d12
    [1, 1, 1, 1, 0, 1],  # d13
    [1, 1, 1, 1, 1, 0],  # d14
    [0, 1, 1, 1, 1, 1],  # d15
    [0, 0, 1, 1, 1, 0],  # d16
    [1, 0, 0, 1, 1, 0],  # d17
    [1, 1, 0, 0, 1, 0],  # d18
    [0, 1, 1, 0, 0, 1],  # d19
    [1, 0, 1, 1, 0, 0],  # d20
    [0, 1, 0, 1, 1, 0],  # d21
    [0, 0, 1, 0, 1, 1],  # d22
    [1, 0, 0, 1, 0, 1],  # d23
    [0, 1, 0, 0, 1, 1],  # d24
    [1, 0, 1, 0, 0, 1],  # D29*
    [0, 1, 0, 1, 1, 0]   # D30*
], dtype=int)


def bi2de(b: np.ndarray, msbflag: str = 'right-msb') -> Union[int, np.ndarray]:
    """Converts binary vectors or matrices to decimal integers."""
    b_mat = np.atleast_2d(b)
    ncols = b_mat.shape[1]
    ptwo = 2 ** np.arange(ncols)

    if msbflag.lower() == 'left-msb':
        b_mat = np.fliplr(b_mat)

    res = b_mat.dot(ptwo)
    return int(res[0]) if res.size == 1 else res


def remove_excess_bits(bits: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Identifies subframe boundaries via preamble cross-correlation, resolves BPSK
    sign ambiguity, and strips incomplete leading/trailing bits.

    :param bits: 1D array of demodulated bits in {-1, +1}.
    :return: (clean_binary_bits, first_subframe_idx) where bits are in {0, 1}.
    """
    pa_bpsk = 1 - 2 * PREAMBLE
    num_subframes = math.floor(bits.size / BITS_PER_SUBFRAME) - 1

    if num_subframes < 2:
        raise ValueError(f"Insufficient bitstream length: {bits.size} bits.")

    # Replicate preamble every 300 bits
    pa_pattern = np.zeros(BITS_PER_SUBFRAME, dtype=int)
    pa_pattern[:pa_bpsk.size] = pa_bpsk
    pa_repeated = np.tile(pa_pattern, num_subframes)

    # Cross-correlation
    c = np.round(corr(bits, pa_repeated, mode='full'))
    c = c[pa_repeated.size - 1:]

    peak_idx = int(np.argmax(np.abs(c)))

    # Resolve sign ambiguity
    if c[peak_idx] < 0:
        binary_bits = np.asarray(bits > 0, dtype=int)
    else:
        binary_bits = np.asarray(bits < 0, dtype=int)

    # Strip partial subframes
    binary_bits = binary_bits[peak_idx:]
    excess_end = binary_bits.size % BITS_PER_SUBFRAME
    if excess_end > 0:
        binary_bits = binary_bits[:-excess_end]

    return binary_bits, peak_idx


def establish_parity(bitstream: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Evaluates and corrects word parity for a sequence of 300-bit subframes per Table 20-XIV.

    :param bitstream: 1D binary array of {0, 1} with a multiple of 300 bits.
    :return: (corrected_bits, parity_passed)
    """
    ws = np.copy(bitstream)
    num_subframes = ws.size // BITS_PER_SUBFRAME
    total_words = num_subframes * WORDS_PER_SUBFRAME

    d29_prev = 0
    d30_prev = 0
    all_passed = True

    for k in range(total_words):
        word = ws[k * BITS_PER_WORD: (k + 1) * BITS_PER_WORD]

        # If D30* of previous word is 1, invert first 24 data bits
        if d30_prev == 1:
            word[:DATA_BITS_PER_WORD] = 1 - word[:DATA_BITS_PER_WORD]

        # Compute syndrome
        check_vec = np.copy(word[:DATA_BITS_PER_WORD + 2])
        check_vec[-2] = d29_prev
        check_vec[-1] = d30_prev

        syndrome = (check_vec.dot(PARITY_H) + word[DATA_BITS_PER_WORD:]) % 2
        if np.any(syndrome != 0):
            all_passed = False

        d29_prev = int(word[-2])
        d30_prev = int(word[-1])
        ws[k * BITS_PER_WORD: (k + 1) * BITS_PER_WORD] = word

    return ws, all_passed


def bits_to_subframes(clean_bits: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """
    Organizes parity-validated bits into a (300 x N) subframe matrix and decodes Subframe IDs (SFID).

    :param clean_bits: Parity-validated 1D binary array.
    :return: (subframes_matrix, subframe_ids)
    """
    num_subframes = clean_bits.size // BITS_PER_SUBFRAME
    subframes_matrix = clean_bits.reshape((BITS_PER_SUBFRAME, num_subframes), order='F')

    # SFID is located in bits 50-52 of Handover Word (HOW)
    sfid_bits = subframes_matrix[49:52, :]
    sfid_list = [int(bi2de(sfid_bits[:, col], 'left-msb')) for col in range(num_subframes)]

    return subframes_matrix, sfid_list
