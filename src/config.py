"""
GPS Receiver Configuration and Physical Constants
=================================================
Defines standard GPS physical constants, modulation parameters,
and filesystem paths according to the IS-GPS-200D specification.
"""

from pathlib import Path
import numpy as np

# -------------------------------------------------------------------------
# Filesystem Paths
# -------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "docs"

# -------------------------------------------------------------------------
# Basic GPS Physical Constants (IS-GPS-200D)
# -------------------------------------------------------------------------
# Speed of light in vacuum [m/s]
C = 299792458.0

# GPS L1 Carrier Frequency [Hz] (154 * 10.23 MHz)
F_L1 = 1575.42e6

# Earth's universal gravitational parameter [m^3/s^2] (WGS-84)
MU_E = 3.986005e14

# WGS-84 Earth's rotational rate [rad/s]
OMEGA_DOT_E = 7.2921151467e-5

# Relativistic clock correction constant F [s / sqrt(m)]
F_REL = -2.0 * np.sqrt(MU_E) / (C ** 2)

# WGS-84 Reference Ellipsoid Parameters
WGS84_A = 6378137.0          # Semi-major axis [m]
WGS84_F = 1.0 / 298.257223563 # Flattening factor
WGS84_B = WGS84_A * (1.0 - WGS84_F) # Semi-minor axis [m]
WGS84_E2 = 2.0 * WGS84_F - WGS84_F ** 2 # First eccentricity squared

# Mathematical constant Pi as defined in GPS standard Table 20-IV
PI_GPS = 3.1415926535898

# -------------------------------------------------------------------------
# C/A Code and Signal Modulation Parameters
# -------------------------------------------------------------------------
# C/A Gold code chip rate [chips/s]
CHIP_RATE = 1.023e6

# C/A code length [chips]
CHIPS_PER_CODE = 1023

# C/A code period [s] (1 ms)
CODE_PERIOD = 1e-3

# Navigation message bit rate [bps]
BIT_RATE = 50.0

# Number of C/A code repetitions per navigation data bit
CODES_PER_BIT = 20

# Samples per chip (with standard ADC configuration)
SAMPLES_PER_CHIP = 4

# Total ADC sampling frequency [Hz] (4.092 MHz)
FS = SAMPLES_PER_CHIP * CHIPS_PER_CODE * CODES_PER_BIT * int(BIT_RATE)

# Sampling period Ts [s]
TS = 1.0 / FS

# Samples per C/A code period (4 * 1023 = 4092 samples)
SAMPLES_PER_CODE = SAMPLES_PER_CHIP * CHIPS_PER_CODE

# Samples per navigation bit (4092 * 20 = 81840 samples)
SAMPLES_PER_BIT = SAMPLES_PER_CODE * CODES_PER_BIT

# Maximum expected Doppler frequency shift [Hz] (+/- 10 kHz)
MAX_DOPPLER = 10e3

# -------------------------------------------------------------------------
# Navigation Frame & Telemetry Structure (IS-GPS-200D)
# -------------------------------------------------------------------------
# Bits per telemetry word
BITS_PER_WORD = 30

# Data payload bits per word (excluding 6 parity bits)
DATA_BITS_PER_WORD = 24

# Words per subframe
WORDS_PER_SUBFRAME = 10

# Total bits per subframe (300 bits = 6.0 seconds)
BITS_PER_SUBFRAME = BITS_PER_WORD * WORDS_PER_SUBFRAME

# Subframes per complete navigation page/frame (5 subframes = 30.0 seconds)
SUBFRAMES_PER_PAGE = 5

# GPS Subframe Preamble pattern (8 bits: 1 0 0 0 1 0 1 1 = 0x8B)
PREAMBLE = np.array([1, 0, 0, 0, 1, 0, 1, 1], dtype=int)

# Samples per raw capture file block
SAMPLES_PER_FILE = 8179460
