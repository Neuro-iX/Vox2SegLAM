import os
import random

import numpy as np
import torch

from ..config import Config

__all__ = ["set_seed"]

# deterministic cuBLAS needs a fixed workspace; it is read when CUDA initialises, so it is
# set when this module is imported (before any model reaches the GPU)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


# seeds every RNG of the main process (workers: data/loaders.py). deterministic=True makes a
# run bit-exact on the same hardware and software (strict deterministic algorithms; an op
# without one raises); False lets cuDNN auto-tune, faster and not bit-exact.
def set_seed(cfg: Config) -> None:
    seed = int(cfg.repro.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if cfg.repro.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = bool(cfg.repro.cudnn_benchmark)
