"""
Raw RF Signal I/O and Buffer Management
=======================================
Handles streaming, caching, and slicing of raw complex baseband I/Q samples
from MATLAB v7 / v7.3 HDF5 files or binary float32 datasets.
"""

from typing import Optional, Tuple
import math
from pathlib import Path
import numpy as np
import scipy.io as sio
import h5py

from .config import DATA_DIR, SAMPLES_PER_FILE


class SignalBuffer:
    """Manages cached streaming of large raw RF signal files."""

    def __init__(self, data_dir: Optional[Path] = None, samples_per_file: int = SAMPLES_PER_FILE):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.samples_per_file = samples_per_file

        self.cache_cur: np.ndarray = np.array([], dtype=complex)
        self.cache_prev: np.ndarray = np.array([], dtype=complex)
        self.cache_cur_chunk: int = 0
        self.cache_prev_chunk: int = 0
        self.start_cur: int = 0
        self.start_prev: int = 0
        self.final_block: int = 0

    def _get_file_path(self, chunk_num: int) -> Path:
        """Constructs filepath for a given chunk number."""
        mat_path = self.data_dir / f"{chunk_num:02d}.mat"
        if mat_path.exists():
            return mat_path
        dat_path = self.data_dir / f"signal-{chunk_num - 1:03d}.dat"
        if dat_path.exists():
            return dat_path
        return mat_path

    def load_chunk(self, filename: Path) -> np.ndarray:
        """Loads complex I/Q samples from a MATLAB or DAT file."""
        if not filename.exists():
            raise EOFError(f"End of data reached: {filename} does not exist.")

        if filename.suffix == ".mat":
            try:
                mat = sio.loadmat(str(filename))
                data = mat['toto']
            except Exception:
                # MATLAB v7.3 HDF5 format
                with h5py.File(str(filename), 'r') as h5f:
                    raw = np.array(h5f.get('toto')).flatten()
                    data = raw['real'] + 1.0j * raw['imag']
        elif filename.suffix == ".dat":
            raw_float = np.fromfile(str(filename), dtype=np.float32)
            data = raw_float[0::2] + 1.0j * raw_float[1::2]
        else:
            raise ValueError(f"Unsupported file extension: {filename.suffix}")

        return data.flatten()

    def get_data(self, start_idx: int, end_idx: int) -> np.ndarray:
        """
        Extracts a slice of baseband signal samples from index start_idx to end_idx (inclusive, 1-indexed).

        :param start_idx: 1-indexed starting sample position.
        :param end_idx: 1-indexed ending sample position.
        :return: 1D complex numpy array of samples.
        """
        if start_idx > end_idx:
            raise ValueError(f"Start index ({start_idx}) must be <= end index ({end_idx})")

        achunk = math.ceil(float(start_idx) / self.samples_per_file)
        bchunk = math.ceil(float(end_idx) / self.samples_per_file)

        if bchunk - achunk > 1:
            raise ValueError("Requested data range spans too many chunks (>2 chunks).")

        # Load end chunk if not in cache
        if bchunk != self.cache_cur_chunk and bchunk != self.cache_prev_chunk:
            if bchunk == self.cache_cur_chunk + 1:
                self.cache_prev_chunk = self.cache_cur_chunk
                self.cache_prev = self.cache_cur
                self.start_prev = self.start_cur
            else:
                self.cache_prev_chunk = 0
                self.cache_prev = np.array([], dtype=complex)
                self.start_prev = 0

            filepath = self._get_file_path(bchunk)
            data = self.load_chunk(filepath)

            self.final_block = 0
            if data.size != self.samples_per_file:
                next_fp = self._get_file_path(bchunk + 1)
                if not next_fp.exists():
                    self.final_block = data.size
                else:
                    raise IOError(f"Malformed data in {filepath}: expected {self.samples_per_file} samples.")

            self.cache_cur = data
            self.cache_cur_chunk = bchunk
            self.start_cur = (bchunk - 1) * self.samples_per_file + 1

        # Load start chunk if different from end chunk and not in previous cache
        if achunk != bchunk and achunk != self.cache_prev_chunk:
            filepath = self._get_file_path(achunk)
            data = self.load_chunk(filepath)
            self.cache_prev = data
            self.cache_prev_chunk = achunk
            self.start_prev = (achunk - 1) * self.samples_per_file + 1

        # Extract requested slices
        part_prev = np.array([], dtype=complex)
        part_cur = np.array([], dtype=complex)

        if start_idx < self.start_cur and self.cache_prev.size > 0:
            p_start = int(start_idx - self.start_prev)
            p_end = int(min(self.samples_per_file, end_idx - self.start_prev)) + 1
            part_prev = self.cache_prev[p_start:p_end]

        if end_idx >= self.start_cur and self.cache_cur.size > 0:
            c_start = int(max(0, start_idx - self.start_cur))
            c_end = int(end_idx - self.start_cur) + 1
            part_cur = self.cache_cur[c_start:c_end]

        if part_prev.size == 0:
            res = part_cur
        elif part_cur.size == 0:
            res = part_prev
        else:
            res = np.concatenate((part_prev, part_cur))

        return res.flatten()


# Default singleton instance for global access
_DEFAULT_BUFFER = SignalBuffer()


def get_signal_data(start_idx: int, end_idx: int) -> np.ndarray:
    """Global convenience accessor for default signal buffer."""
    return _DEFAULT_BUFFER.get_data(start_idx, end_idx)
