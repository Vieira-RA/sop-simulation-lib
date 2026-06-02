# src/stokes.py
import numpy as np

def normalize_stokes(S):
    """Normalize a Stokes vector so that S0 = 1."""
    if S.ndim == 1:
        return S / S[0]
    else:
        return S / S[0, :]