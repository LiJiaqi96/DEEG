"""
Main package for loading common data file types:
    1. csv
    2. Common MNE-supported file types (txt, mat, etc.)
to numpy array with dimension (p, m, e).
"""
import numpy as np
from mne.io import read_raw
from pathlib import Path
from utils import read_csv

supported = {
    ".edf": read_raw,
    ".bdf": read_raw,
    ".gdf": read_raw,
    ".vhdr": read_raw,
    ".fif": read_raw,
    ".fif.gz": read_raw,
    ".set": read_raw,
    ".cnt": read_raw,
    ".mff": read_raw,
    ".nxe": read_raw,
    ".hdr": read_raw,
    ".mat": read_raw,
    ".bin": read_raw,
    ".data": read_raw,
    ".sqd": read_raw,
    ".con": read_raw,
    ".ds": read_raw,
    ".txt": read_raw,
    ".csv": read_csv
}

def load_data(fname, *, preload=False, verbose=None, **kwargs):
    """
    :param fname: Files you are gonna read
    :param mne: Indicate if the file is mne-supported
    :return: numpy array of eeg data
    """
    ext = "".join(Path(fname).suffixes)
    if ext in supported:
        if ext == ".csv":
            return supported[ext](fname)
        else:
            raw = supported[ext](fname, preload=preload, verbose=verbose, **kwargs)
            data, time = raw[:]
            return data


