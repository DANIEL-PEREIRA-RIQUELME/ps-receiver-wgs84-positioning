"""
GPS Ephemeris and Pseudorange Extraction Module
===============================================
Parses Keplerian orbital parameters, satellite clock bias coefficients,
and computes raw pseudoranges per IS-GPS-200D Section 20.3.3.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path
import numpy as np
import scipy.io as sio

from .config import C, BIT_RATE, SAMPLES_PER_BIT, PI_GPS, DATA_DIR
from .navigation import bi2de


@dataclass
class Ephemeris:
    """Stores complete GPS satellite ephemeris, telemetry, and clock parameters."""
    prn: int = 0
    sfid: int = 0

    # Telemetry and Handover Word (HOW)
    TOW1: int = 0
    TOW2: int = 0
    TOW3: int = 0
    wn: int = 0          # GPS Week Number
    alert: int = 0
    antispoof: int = 0

    # Clock Correction Parameters (Subframe 1)
    IODC: int = 0        # Issue of Data, Clock
    t_oc: int = 0        # Clock reference time [s]
    T_GD: float = 0.0    # Group delay differential [s]
    a_f0: float = 0.0    # Clock bias polynomial coefficient [s]
    a_f1: float = 0.0    # Clock drift coefficient [s/s]
    a_f2: float = 0.0    # Clock acceleration coefficient [s/s^2]

    # Orbital Parameters (Subframes 2 and 3)
    IODE: int = 0        # Issue of Data, Ephemeris
    t_oe: int = 0        # Ephemeris reference time [s]
    M_0: float = 0.0     # Mean anomaly at reference time [rad]
    delta_n: float = 0.0 # Mean motion difference [rad/s]
    e: float = 0.0       # Orbit eccentricity
    sqrt_a: float = 0.0  # Square root of semi-major axis [sqrt(m)]
    i_0: float = 0.0     # Inclination angle at reference time [rad]
    w: float = 0.0       # Argument of perigee [rad]
    Omega_0: float = 0.0 # Longitude of ascending node at reference time [rad]

    # Harmonic Perturbations and Rates
    Omegadot: float = 0.0 # Rate of right ascension [rad/s]
    idot: float = 0.0     # Rate of inclination angle [rad/s]
    C_us: float = 0.0     # Latitude cosine harmonic [rad]
    C_uc: float = 0.0     # Latitude sine harmonic [rad]
    C_rs: float = 0.0     # Radius cosine harmonic [m]
    C_rc: float = 0.0     # Radius sine harmonic [m]
    C_is: float = 0.0     # Inclination cosine harmonic [rad]
    C_ic: float = 0.0     # Inclination sine harmonic [rad]

    # Auxiliary timing
    t_tr: int = 0        # Earliest transmission time (min TOW - 6s)

    def load_from_mat(self, filepath: Path) -> None:
        """Loads ephemeris fields from a .mat file."""
        mat = sio.loadmat(str(filepath))
        eph_dict = mat['ephemeris'].flatten()[0]
        fields = list(mat['ephemeris'].dtype.fields.keys())

        for idx, field in enumerate(fields):
            val = eph_dict[idx].flatten()[0]
            if hasattr(self, field):
                setattr(self, field, val)

    def to_dict(self) -> Dict[str, Any]:
        """Converts ephemeris fields to a Python dictionary."""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def _read_unsigned(page: np.ndarray, sf: int, start_bit: int, num_bits: int, scale_exp: int) -> float:
    """Extracts unsigned integer with power of 2 scaling."""
    bits = page[start_bit - 1: start_bit + num_bits - 1, sf - 1]
    return float(bi2de(bits, 'left-msb') * (2.0 ** scale_exp))


def _read_signed(page: np.ndarray, sf: int, start_bit: int, num_bits: int, scale_exp: int) -> float:
    """Extracts two's-complement signed integer with power of 2 scaling."""
    bits = np.copy(page[start_bit - 1: start_bit + num_bits - 1, sf - 1])
    if bits[0] == 0:
        return float(bi2de(bits, 'left-msb') * (2.0 ** scale_exp))
    else:
        inverted = 1 - bits
        return float(-1.0 * (bi2de(inverted, 'left-msb') + 1) * (2.0 ** scale_exp))


def _read_2part_unsigned(page: np.ndarray, sf: int, am: int, nm: int, al: int, nl: int, scale_exp: int) -> float:
    """Extracts split unsigned bits (MSB part + LSB part)."""
    bits_msb = page[am - 1: am + nm - 1, sf - 1]
    bits_lsb = page[al - 1: al + nl - 1, sf - 1]
    combined = np.concatenate((bits_msb, bits_lsb))
    return float(bi2de(combined, 'left-msb') * (2.0 ** scale_exp))


def _read_2part_signed(page: np.ndarray, sf: int, am: int, nm: int, al: int, nl: int, scale_exp: int) -> float:
    """Extracts split two's-complement signed bits."""
    bits_msb = page[am - 1: am + nm - 1, sf - 1]
    bits_lsb = page[al - 1: al + nl - 1, sf - 1]
    combined = np.concatenate((bits_msb, bits_lsb))
    if combined[0] == 0:
        return float(bi2de(combined, 'left-msb') * (2.0 ** scale_exp))
    else:
        inverted = 1 - combined
        return float(-1.0 * (bi2de(inverted, 'left-msb') + 1) * (2.0 ** scale_exp))


def decode_ephemeris(page: np.ndarray, prn: int = 0) -> Ephemeris:
    """
    Decodes ephemeris parameters from subframes 1, 2, and 3 per IS-GPS-200D.

    :param page: (300 x 3) array with subframes 1, 2, and 3 in columns 0, 1, 2.
    :param prn: Satellite PRN number.
    :return: Populated Ephemeris object.
    """
    eph = Ephemeris(prn=prn)

    # Subframe 1: Telemetry & Clock Corrections
    eph.sfid = int(_read_unsigned(page, 1, 50, 3, 0))
    eph.TOW1 = int(_read_unsigned(page, 1, 31, 17, 0) * 6)
    eph.TOW2 = int(_read_unsigned(page, 2, 31, 17, 0) * 6)
    eph.TOW3 = int(_read_unsigned(page, 3, 31, 17, 0) * 6)
    eph.wn = int(_read_unsigned(page, 1, 61, 10, 0))
    eph.alert = int(_read_unsigned(page, 1, 18, 1, 0))
    eph.antispoof = int(_read_unsigned(page, 1, 19, 1, 0))

    eph.IODC = int(_read_2part_unsigned(page, 1, 83, 2, 211, 8, 0))
    eph.t_oc = int(_read_unsigned(page, 1, 219, 16, 4))
    eph.T_GD = _read_signed(page, 1, 197, 8, -31)
    eph.a_f0 = _read_signed(page, 1, 271, 22, -31)
    eph.a_f1 = _read_signed(page, 1, 249, 16, -43)
    eph.a_f2 = _read_signed(page, 1, 241, 8, -55)

    # Subframe 2: Keplerian Orbit (Part 1)
    eph.IODE = int(_read_unsigned(page, 2, 61, 8, 0))
    eph.t_oe = int(_read_unsigned(page, 2, 271, 16, 4))
    eph.M_0 = _read_2part_signed(page, 2, 107, 8, 121, 24, -31) * PI_GPS
    eph.delta_n = _read_signed(page, 2, 91, 16, -43) * PI_GPS
    eph.e = _read_2part_unsigned(page, 2, 167, 8, 181, 24, -33)
    eph.sqrt_a = _read_2part_unsigned(page, 2, 227, 8, 241, 24, -19)
    eph.C_uc = _read_signed(page, 2, 151, 16, -29)
    eph.C_us = _read_signed(page, 2, 211, 16, -29)
    eph.C_rs = _read_signed(page, 2, 69, 16, -5)

    # Subframe 3: Keplerian Orbit (Part 2)
    eph.C_rc = _read_signed(page, 3, 181, 16, -5)
    eph.C_ic = _read_signed(page, 3, 61, 16, -29)
    eph.C_is = _read_signed(page, 3, 121, 16, -29)
    eph.i_0 = _read_2part_signed(page, 3, 137, 8, 151, 24, -31) * PI_GPS
    eph.w = _read_2part_signed(page, 3, 197, 8, 211, 24, -31) * PI_GPS
    eph.Omega_0 = _read_2part_signed(page, 3, 77, 8, 91, 24, -31) * PI_GPS
    eph.Omegadot = _read_signed(page, 3, 241, 24, -43) * PI_GPS
    eph.idot = _read_signed(page, 3, 279, 14, -43) * PI_GPS

    # Minimum TOW minus 6s transmission duration
    eph.t_tr = int(min(eph.TOW1, eph.TOW2, eph.TOW3) - 6)
    return eph


def compute_pseudorange(bit_start_taus: np.ndarray, tau_ref: int, first_subframe_bit_idx: int) -> float:
    """
    Computes raw pseudorange rho in meters.

    :param bit_start_taus: Array of sample indices where each decoded bit begins.
    :param tau_ref: Reference receiver sample timestamp.
    :param first_subframe_bit_idx: Bit index where the first subframe begins.
    :return: Pseudorange in meters.
    """
    tb = 1.0 / BIT_RATE
    fractional_bits = (bit_start_taus[first_subframe_bit_idx] - tau_ref) / float(SAMPLES_PER_BIT)
    return float(C * fractional_bits * tb)
