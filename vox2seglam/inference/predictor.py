import gc
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..config import Config, load_config
from ..data.normalize import normalize_zscore_brain_masked, normalize_zscore_volume
from ..utils.device import deterministic_inference
from ..utils.io import to_zyx
from ..models import Vox2SegLAM, template_from_state_dict

# full-model passes per volume, shared by predict.py and score_split.py so reported test
# metrics describe what the CLI outputs; it lives in constants.py because argparse needs it
# before torch is loaded
from .constants import DEFAULT_N_PASSES

__all__ = ["Predictor", "Prediction", "DEFAULT_N_PASSES"]

# numeric precision of the network on a GPU (landmark refinement itself always runs in float32)
PRECISIONS = {"float32": None, "bfloat16": torch.bfloat16, "float16": torch.float16}

_OUT_OF_MEMORY = (
    "the GPU ran out of memory: a full 240 x 304 x 240 volume needs about 9 GB in float32 and "
    "6.5 GB in half precision. Free some memory, or run on the CPU with --device cpu (slower, "
    "no GPU needed)"
)


def _half_precision() -> str:
    """bfloat16 where the GPU has it natively, float16 otherwise."""
    native = torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False)
    return "bfloat16" if native else "float16"


# result of one prediction; the model works in ZYX voxel order, and coordinates_xyz() is
# the one place that flips it
@dataclass
class Prediction:
    segmentation: np.ndarray | None
    coordinates: np.ndarray
    landmark_ids: list[int]
    probabilities: np.ndarray | None = None
    affine: np.ndarray | None = None  # voxel -> world (mm) of the input NIfTI, when known
    precision: str = "float32"  # what the network ran in; half precision on a GPU low on memory

    def coordinates_xyz(self) -> np.ndarray:
        return self.coordinates[:, ::-1].copy()

    def coordinates_world_mm(self) -> np.ndarray | None:
        if self.affine is None:
            return None
        xyz = self.coordinates_xyz()
        return xyz @ self.affine[:3, :3].T + self.affine[:3, 3]

    # JSON-serialisable view, keyed by landmark id. Each coordinate system names its convention
    # (NIfTI affines are RAS+; predict.py writes LPS and RAS side by side), since an
    # unnamed one hides a left/right swap.
    def as_dict(self) -> dict[str, Any]:
        systems = {"voxel_zyx": self.coordinates, "voxel_xyz": self.coordinates_xyz()}
        world = self.coordinates_world_mm()
        if world is not None:
            systems["world_ras_mm"] = world
        return {
            "producer": "vox2seglam.inference.Predictor",
            "precision": self.precision,
            "landmarks": {
                str(lid): {name: [float(v) for v in arr[k]] for name, arr in systems.items()}
                for k, lid in enumerate(self.landmark_ids)
            },
        }


# A trained checkpoint applied to new volumes. It recovers its own config from beside the
# checkpoint, so architecture and preprocessing match the weights (serving a brain-masked
# checkpoint with whole-volume normalisation would be a silent error).
class Predictor:
    def __init__(
        self,
        model: Vox2SegLAM,
        cfg: Config,
        device: torch.device | None = None,
        n_passes: int = DEFAULT_N_PASSES,
        precision: str = "float32",
    ) -> None:
        if n_passes < 1:
            raise ValueError(f"n_passes must be >= 1, got {n_passes}")
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {sorted(PRECISIONS)}, got {precision!r}")
        # the same prediction twice gives the same file: without this, the reductions in the
        # graph steps move the landmarks by about 1e-4 voxel from one run to the next. Global
        # to torch, so it is set here, where every prediction passes, and nowhere in training
        # (a run chooses with repro.deterministic).
        deterministic_inference()
        self.precision = precision
        self.cfg = cfg
        self.device = device or torch.device("cpu")
        self.n_passes = int(n_passes)
        self.model = model.to(self.device).eval()
        self.landmark_ids = cfg.task.resolved_landmark_ids()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        config: str | Path | Config | None = None,
        device: str | torch.device | None = None,
        n_passes: int = DEFAULT_N_PASSES,
        precision: str = "float32",
    ) -> "Predictor":
        checkpoint = Path(checkpoint)
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)

        # config resolution: explicit argument > config.yaml beside the checkpoint > the experiment
        # name stored in the checkpoint > an error
        if isinstance(config, Config):
            cfg = config
        elif config is not None:
            cfg = load_config(config)
        else:
            sidecar = checkpoint.parent / "config.yaml"
            if sidecar.exists():
                cfg = load_config(sidecar)
            elif ckpt.get("config_name"):
                cfg = load_config(ckpt["config_name"])
            else:
                raise ValueError(
                    f"cannot determine the configuration for {checkpoint}. No "
                    f"config.yaml beside it and no 'config_name' inside it -- "
                    "pass config=... explicitly."
                )

        device = torch.device(device) if device is not None else _pick_device(cfg)
        state = ckpt.get("model", ckpt)
        # the template (mean coords, graph edges) comes from the checkpoint: no template_dir needed
        model = Vox2SegLAM.from_config(cfg, device=device, template=template_from_state_dict(state))

        model.load_state_dict(state, strict=True)

        predictor = cls(model=model, cfg=cfg, device=device, n_passes=n_passes, precision=precision)
        predictor.checkpoint_epoch = ckpt.get("epoch")
        return predictor

    # the preprocessing state this checkpoint expects (`predict.py --show-contract`); the model
    # resamples and normalises nothing beyond the z-score in normalize()
    def expected_input(self) -> dict[str, Any]:
        return {
            "shape_zyx": tuple(self.cfg.task.patch_size),
            "skull_stripped": self.cfg.preprocess.skull_stripped,
            "bias_field_corrected": self.cfg.preprocess.bias_field_corrected,
            "orientation": self.cfg.preprocess.orientation,
            "normalisation": (
                "brain-masked z-score"
                if self.cfg.preprocess.zscore_brain_masked
                else "whole-volume z-score"
            ),
            "note": "normalisation is applied by the Predictor; pass raw intensities.",
        }

    # (predicted, ground-truth) landmarks of every subject of a split; shared by
    # the fit-postprocess command
    @torch.no_grad()
    def landmark_pairs(self, split: str = "val"):
        from ..data import build_dataloaders

        loaders = build_dataloaders(self.cfg, splits=(split,))
        if split not in loaders:  # build_dataloaders skips an empty split
            raise ValueError(
                f"the {split} split is empty (paths.split_{split}); there is nothing to "
                "fit the post-processing on"
            )
        for batch in loaders[split]:
            x = batch["image"].to(self.device)
            x = x.unsqueeze(0) if x.ndim == 4 else x
            if x.shape[0] != 1:
                raise ValueError(
                    f"landmark_pairs handles one subject per batch, got {x.shape[0]}; "
                    "set data.eval_batch_size=1"
                )
            coords = self.model(x)["coords"][0].float().cpu().numpy()
            yield coords, batch["coords"].reshape(-1, 3).numpy()

    def normalize(self, volume: torch.Tensor) -> torch.Tensor:
        if self.cfg.preprocess.zscore_brain_masked:
            return normalize_zscore_brain_masked(volume, self.cfg.preprocess.zscore_eps)
        return normalize_zscore_volume(volume, self.cfg.preprocess.zscore_eps)

    @torch.no_grad()
    def predict_volume(
        self,
        volume_zyx: np.ndarray | torch.Tensor,
        return_probabilities: bool = False,
    ) -> Prediction:
        x = self._as_input(volume_zyx)
        if x.shape[0] != 1:
            # the outputs below index [0] and normalize() reduces over the whole tensor: a batch would
            # drop every volume but the first and mix their z-score statistics
            raise ValueError(
                f"predict_volume handles one volume at a time, got a batch of "
                f"{x.shape[0]}; loop over the volumes (the z-score is per volume)"
            )

        pred = None
        try:
            pred = self._forward(x)
        except torch.cuda.OutOfMemoryError as err:
            if self.precision != "float32":
                raise RuntimeError(_OUT_OF_MEMORY) from err
        if pred is None:
            # A full-resolution volume needs about 9 GB in float32 and about 6.5 GB in half
            # precision: try that before giving up (landmarks move by under 0.06 mm). Done
            # outside the except block, whose traceback still holds the failed activations.
            gc.collect()
            torch.cuda.empty_cache()
            self.precision = _half_precision()
            warnings.warn(
                f"the GPU ran out of memory in float32; continuing in {self.precision} "
                "(landmarks move by under 0.06 mm)",
                RuntimeWarning,
                stacklevel=2,
            )
            try:
                pred = self._forward(x)
            except torch.cuda.OutOfMemoryError as again:
                raise RuntimeError(_OUT_OF_MEMORY) from again

        segmentation = None
        probabilities = None
        if pred["seg"]:
            logits = pred["seg"][-1]  # last deep-supervision entry = full resolution
            segmentation = logits.argmax(dim=1)[0].cpu().numpy().astype(np.int16)
            if return_probabilities:
                probabilities = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()

        return Prediction(
            segmentation=segmentation,
            coordinates=pred["coords"][0].cpu().numpy().astype(np.float32),
            landmark_ids=list(self.landmark_ids),
            probabilities=probabilities,
            precision=self.precision if self.device.type == "cuda" else "float32",
        )

    def _forward(self, x: torch.Tensor) -> dict:
        # n_passes > 1 re-runs the whole model with the previous pass's final coordinates as the
        # new init, so the refiner's patch is centred closer to the true landmark (distinct from
        # model.lam.steps, the GNN steps within one pass)
        half = PRECISIONS[self.precision] if self.device.type == "cuda" else None
        with torch.autocast("cuda", dtype=half, enabled=half is not None):
            pred = self.model(x, coords_init_zyx=None)
            for _ in range(self.n_passes - 1):
                pred = self.model(x, coords_init_zyx=pred["coords"])
        return pred

    def predict_file(self, path: str | Path, **kwargs) -> Prediction:
        from ..utils.io import load_nifti

        import nibabel as nib

        arr_xyz, affine = load_nifti(path, dtype=np.float32)
        expected = self.cfg.preprocess.orientation
        if expected is not None:
            actual = "".join(nib.aff2axcodes(affine))
            if actual != expected:
                raise ValueError(
                    f"{path}: orientation {actual}, but the model was trained on {expected} "
                    "volumes; reorient the input first (e.g. nibabel.as_closest_canonical "
                    "then the training orientation)"
                )
        pred = self.predict_volume(to_zyx(arr_xyz), **kwargs)
        pred.affine = affine
        return pred

    def _as_input(self, volume) -> torch.Tensor:
        if isinstance(volume, np.ndarray):
            volume = torch.from_numpy(np.ascontiguousarray(volume))
        volume = volume.float()

        if volume.ndim == 3:
            volume = volume[None, None]
        elif volume.ndim == 4:
            volume = volume[None]
        elif volume.ndim != 5:
            raise ValueError(
                f"expected a 3D volume (Z,Y,X) or a batched 4D/5D tensor, "
                f"got shape {tuple(volume.shape)}"
            )

        expected = tuple(self.cfg.task.patch_size)
        actual = tuple(volume.shape[-3:])
        if actual != expected:
            raise ValueError(
                f"volume shape {actual} does not match the grid this checkpoint "
                f"was trained on, {expected}. Resample it onto the reference "
                "grid first -- the model has no resampling of its own."
            )

        return self.normalize(volume).to(self.device)


# honours cfg.train.device, falling back to CPU if CUDA is asked for and absent
def _pick_device(cfg: Config) -> torch.device:
    if cfg.train.device.startswith("cuda") and torch.cuda.is_available():
        return torch.device(cfg.train.device)
    return torch.device("cpu")
