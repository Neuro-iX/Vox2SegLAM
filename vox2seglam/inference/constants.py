"""Inference defaults that a command line needs before anything heavy is loaded."""

__all__ = ["DEFAULT_N_PASSES"]

#: landmark refinement passes at inference; 1 runs the cascade once, as the paper does
DEFAULT_N_PASSES = 1
