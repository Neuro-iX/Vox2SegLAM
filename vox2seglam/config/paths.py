import os
from dataclasses import dataclass
from pathlib import Path


PACKAGE_DIR = (
    Path(__file__).resolve().parents[1]
)  # .../vox2seglam/, used as the default anchor below
REPO_DIR = PACKAGE_DIR.parent  # repo root, used by loading.py to find configs/experiments/


@dataclass(frozen=True)
class PathsCfg:
    # the dataset is not redistributable (README): put it, or a symlink, at <repo>/data, or
    # set paths.data_root (relative paths are resolved against the repository root)
    data_root: str = "data"

    # subdirectories of data_root; the defaults are the 16-region / 16-landmark ground truth (the
    # protocol-governed subset of the original 26-region / 20-landmark annotations). The
    # skull-stripped and N4 caches next to image_dir come from data/prepare.
    image_dir: str = "MRI_Volumes"
    seg_dir: str = "Segmentation_Volumes_16"
    landmark_dir: str = "Landmarks_Volumes_16"

    # per-subject filename suffix for each directory above, appended to the subject id
    image_suffix: str = "_T1w.nii.gz"
    seg_suffix: str = ".nii.gz"
    landmark_suffix: str = "_landmark_16.nii.gz"

    # empty = the subject lists of the published runs, fetched once from the archive on the
    # first run that needs them (they name subjects of cohorts whose terms govern their
    # redistribution, so they are published with the data rather than with the code).
    # Set it to your own directory; a relative path is resolved against the repository root.
    splits_dir: str = ""
    split_train: str = "train_ids.txt"
    split_val: str = "val_ids.txt"
    split_test: str = "test_ids.txt"

    # landmark template (mean coords, per-landmark std, graph edges). "" = automatic: the
    # shipped template named after the training split and landmark suffix (e.g. train_landmark_16)
    # if it was fitted on exactly this split, otherwise one fitted at the first training run into
    # <exp_dir>/template (training/template_setup.py).
    template_dir: str = ""

    # FreeSurfer install, only for the FreeSurfer route to SynthStrip
    freesurfer_home: str = ""

    exp_dir: str = "./experiments/base"  # where this run's config.yaml, checkpoints and tb/ logs go

    def resolve_data_root(self) -> str:
        return resolve_repo_path(self.data_root)

    def resolve_image_dir(self) -> str:
        return _under(self.resolve_data_root(), self.image_dir)

    def resolve_seg_dir(self) -> str:
        return _under(self.resolve_data_root(), self.seg_dir)

    def resolve_landmark_dir(self) -> str:
        return _under(self.resolve_data_root(), self.landmark_dir)

    # a directory the config has no field for (the preparation caches, named after image_dir)
    def resolve_under_data_root(self, dir_name: str) -> str:
        return _under(self.resolve_data_root(), dir_name)

    # automatic template name: split file without "_ids.txt" + landmark suffix without
    # extension, e.g. ("train_ids.txt", "_landmark_16.nii.gz") -> train_landmark_16
    def template_name(self) -> str:
        split = Path(self.split_train).name.removesuffix(".txt").removesuffix("_ids")
        landmarks = self.landmark_suffix.removesuffix(".gz").removesuffix(".nii").strip("_")
        return f"{split}_{landmarks}"

    def resolve_template_dir(self) -> str:
        if self.template_dir:
            return resolve_repo_path(self.template_dir)
        return str(PACKAGE_DIR / "templates" / self.template_name())

    # maps a split name to its id-list file, resolved under splits_dir. With splits_dir empty
    # this is the published set, which the first call fetches (asking first, as for the weights).
    def resolve_split(self, name: str) -> str:
        mapping = {
            "train": self.split_train,
            "val": self.split_val,
            "test": self.split_test,
        }
        if name not in mapping:
            raise KeyError(f"unknown split {name!r}; expected one of {sorted(mapping)}")
        if not self.splits_dir:
            from .. import assets

            return str(assets.resolve("splits")["splits"] / mapping[name])
        return _under(resolve_repo_path(self.splits_dir), mapping[name])


# a relative path is taken from the repository root, not the working directory
def resolve_repo_path(path: str) -> str:
    return _under(str(REPO_DIR), os.path.normpath(path)) if path else path


# join root and path unless path is absolute: a field takes a bare name or a full path
def _under(root: str, path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(root, path)
