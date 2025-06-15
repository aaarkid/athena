"""
Athena - High-performance reinforcement learning library

This package provides Python bindings for the Athena RL library written in Rust.
"""

try:
    from athena_py import NeuralNetwork, DqnAgent, ReplayBuffer
    __all__ = ["NeuralNetwork", "DqnAgent", "ReplayBuffer"]
except ImportError as e:
    raise ImportError(
        "Failed to import Rust extension. "
        "Please ensure the package is properly installed with: "
        "pip install -e . or python setup.py develop"
    ) from e

__version__ = "0.1.0"