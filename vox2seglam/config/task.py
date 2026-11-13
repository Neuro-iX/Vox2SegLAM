from dataclasses import dataclass


@dataclass(frozen=True)
class TaskCfg:
    n_classes: int = 17  # segmentation classes
    n_landmarks: int = 16  # K, the number of landmarks the model predicts
    landmark_ids: tuple[int, ...] | None = (
        None  # explicit landmark id subset/order; None = 1..n_landmarks
    )
    patch_size: tuple[int, int, int] = (240, 304, 240)  # reference grid, ZYX voxels
    voxel_spacing_mm: tuple[float, float, float] = (0.7, 0.7, 0.7)  # of that grid, ZYX
    n_input_channels: int = 1  # channels fed to the backbone; 1 = a single T1w volume

    # every metric reported in millimetres is a voxel count times this spacing, so a
    # checkpoint trained on another grid must carry its own: no metric defaults to ours
    def isotropic_spacing_mm(self) -> float:
        """The single voxel size, for the metrics that assume an isotropic grid."""
        if len(set(self.voxel_spacing_mm)) != 1:
            raise ValueError(
                f"task.voxel_spacing_mm={self.voxel_spacing_mm} is anisotropic, but this "
                "metric measures distances from a voxel-count map and needs one spacing."
            )
        return float(self.voxel_spacing_mm[0])

    # the actual landmark ids used, validating landmark_ids against n_landmarks when set
    def resolved_landmark_ids(self) -> list[int]:
        if self.landmark_ids is not None:
            if len(self.landmark_ids) != self.n_landmarks:
                raise ValueError(
                    f"landmark_ids has {len(self.landmark_ids)} entries but "
                    f"n_landmarks is {self.n_landmarks}"
                )
            return list(self.landmark_ids)
        return list(range(1, self.n_landmarks + 1))
