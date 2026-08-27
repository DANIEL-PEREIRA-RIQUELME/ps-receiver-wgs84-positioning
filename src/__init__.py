"""
GPS Software Defined Radio (SDR) Receiver & Positioning Package
==============================================================
Autonomous GPS L1 C/A SDR transceiver, navigation message decoder,
and WGS-84 multilateration positioning solver.
"""

from .config import (
    C, F_L1, FS, TS, CHIP_RATE, CHIPS_PER_CODE, BIT_RATE,
    DATA_DIR, DOCS_DIR
)
from .ca_code import generate_ca_code, get_ca_code
from .signal_io import SignalBuffer, get_signal_data
from .acquisition import coarse_estimate, fine_estimate, acquire_visible_satellites
from .tracking import track_and_demodulate, find_first_bit_boundary
from .navigation import remove_excess_bits, establish_parity, bits_to_subframes
from .ephemeris import Ephemeris, decode_ephemeris, compute_pseudorange
from .pvt_solver import (
    solve_receiver_position, calc_satellite_position,
    calc_eccentric_anomaly, calc_satellite_clock_bias, ecef_to_wgs84
)

__all__ = [
    "C", "F_L1", "FS", "TS", "CHIP_RATE", "CHIPS_PER_CODE", "BIT_RATE",
    "DATA_DIR", "DOCS_DIR",
    "generate_ca_code", "get_ca_code",
    "SignalBuffer", "get_signal_data",
    "coarse_estimate", "fine_estimate", "acquire_visible_satellites",
    "track_and_demodulate", "find_first_bit_boundary",
    "remove_excess_bits", "establish_parity", "bits_to_subframes",
    "Ephemeris", "decode_ephemeris", "compute_pseudorange",
    "solve_receiver_position", "calc_satellite_position",
    "calc_eccentric_anomaly", "calc_satellite_clock_bias", "ecef_to_wgs84"
]
