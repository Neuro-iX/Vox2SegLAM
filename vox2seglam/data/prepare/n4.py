from pathlib import Path


# N4 estimates the (smooth, multiplicative) bias field on a shrunk image for speed,
# then evaluates it back at full resolution and divides it out; the brain mask (already
# available since this runs after skull-stripping) keeps background out of the fit
def n4(src: Path, dst: Path, cfg, _script=None) -> None:
    import SimpleITK as sitk

    image = sitk.ReadImage(str(src), sitk.sitkFloat32)
    mask = sitk.Cast(image > 0, sitk.sitkUInt8)

    shrink = int(cfg.preprocess.n4_shrink_factor)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(list(cfg.preprocess.n4_iterations))
    corrector.Execute(sitk.Shrink(image, [shrink] * 3), sitk.Shrink(mask, [shrink] * 3))

    corrected = image / sitk.Exp(corrector.GetLogBiasFieldAsImage(image))
    sitk.WriteImage(sitk.Mask(corrected, mask), str(dst))


# best-effort: records which SimpleITK produced this output, for provenance only
def sitk_version() -> str | None:
    try:
        import SimpleITK as sitk

        return sitk.Version_VersionString()
    except Exception:
        return None
