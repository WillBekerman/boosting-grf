"""
Public API surface for the boosting_grf package implementing Algorithm 1.
"""
from .model import Alg1GRFModel, fit_alg1_grf

__all__ = ["Alg1GRFModel", "fit_alg1_grf"]
