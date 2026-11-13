from collections.abc import Sequence

import torch
import torch.nn as nn

from .blocks import ResidualUNetDecoder, ResidualUNetEncoder

__all__ = ["UNet"]


# residual U-Net returning encoder and decoder features (for the landmark graph and refiner)
# and segmentation logits from one forward pass; the seg head is optional
class UNet(nn.Module):
    def __init__(
        self,
        num_input_channels: int,
        down_channels: Sequence[int],
        up_channels: Sequence[int],
        n_classes: int | None = None,
        patch_shape: Sequence[int] | None = None,
        deep_supervision: bool = True,
        deep_supervision_levels: Sequence[int] = (1, 2),
        upsample_mode: str = "trilinear",
        p_dropout: float | None = None,
        normalize: str = "instance",
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()

        if len(up_channels) != len(down_channels) - 1:
            raise ValueError(
                f"encoder must have one more stage than the decoder: got "
                f"{len(down_channels)} encoder and {len(up_channels)} decoder stages"
            )

        self.encoder = ResidualUNetEncoder(
            input_channels=num_input_channels,
            encoder_channels=down_channels,
            p_dropout=p_dropout,
            normalize=normalize,
            use_checkpoint=use_checkpoint,
        )
        self.decoder = ResidualUNetDecoder(
            encoder=self.encoder,
            decoder_channels=up_channels,
            p_dropout=p_dropout,
            normalize=normalize,
            use_checkpoint=use_checkpoint,
        )

        # [encoder + decoder] channels, indexed by model.lam.aggregate_indices and refine_skip_indices
        self.skip_channels: list[int] = list(down_channels) + list(up_channels)

        self.n_classes = n_classes
        self.deep_supervision = bool(deep_supervision) and n_classes is not None
        self.deep_supervision_levels = tuple(int(i) for i in deep_supervision_levels)
        self.aux_layers = nn.ModuleList()
        self.final_layer = None

        if n_classes is None:
            return

        if patch_shape is None:
            raise ValueError("patch_shape is required when n_classes is given")

        channels = list(up_channels)

        for level in self.deep_supervision_levels:
            if not 0 <= level < len(channels):
                raise ValueError(
                    f"deep_supervision_levels contains {level}, outside the "
                    f"{len(channels)} decoder stages (0..{len(channels) - 1})"
                )

        # one 1x1x1 conv + upsample per deep-supervision level: a seg head on a lower-resolution
        # decoder stage, giving earlier stages their own gradient (Dou et al., "3D deeply supervised
        # network...", MedIA 41, 2017). Auxiliary logits are upsampled to full resolution and
        # compared with the full-resolution ground truth; default levels (1, 2) are D7 (1/4) and
        # D8 (1/2).
        if self.deep_supervision:
            for level in self.deep_supervision_levels:
                self.aux_layers.append(
                    nn.Sequential(
                        nn.Conv3d(channels[level], n_classes, kernel_size=1, bias=False),
                        nn.Upsample(
                            size=tuple(patch_shape),
                            mode=upsample_mode,
                            # torch only accepts align_corners for interpolating modes
                            align_corners=(
                                False
                                if upsample_mode in ("linear", "bilinear", "bicubic", "trilinear")
                                else None
                            ),
                        ),
                    )
                )

        # the full-resolution head; a zero bias is a mild prior towards balanced class logits
        self.final_layer = nn.Conv3d(channels[-1], n_classes, kernel_size=1, bias=True)
        nn.init.zeros_(self.final_layer.bias)

    @classmethod
    def from_config(cls, cfg) -> "UNet":
        bb, head = cfg.model.backbone, cfg.model.seg_head
        return cls(
            num_input_channels=cfg.task.n_input_channels,
            down_channels=bb.encoder_channels,
            up_channels=bb.decoder_channels,
            n_classes=cfg.task.n_classes if head.enabled else None,
            patch_shape=cfg.task.patch_size,
            deep_supervision=head.deep_supervision,
            deep_supervision_levels=head.deep_supervision_levels,
            upsample_mode=head.upsample_mode,
            p_dropout=bb.p_dropout,
            normalize=bb.norm,
            use_checkpoint=bb.grad_checkpoint,
        )

    # seg is [] without a seg head, else [aux_0, aux_1, ..., full_resolution]
    def forward(
        self, x: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        encoder_skips = self.encoder(x)
        decoder_skips = self.decoder(encoder_skips)

        seg: list[torch.Tensor] = []
        if self.final_layer is not None:
            # The auxiliary heads only feed deep_supervision_seg_losses on the training path; every other
            # reader takes seg[-1]. Each is a full-size (B, n_classes, Z, Y, X) map (1.19 GB on the
            # published grid), so computing them in eval() wasted ~2.4 GB per forward. compute_losses
            # therefore expects a model in train mode, as trainer.py provides.
            if self.training:
                for layer, level in zip(self.aux_layers, self.deep_supervision_levels):
                    seg.append(layer(decoder_skips[level]))
            seg.append(self.final_layer(decoder_skips[-1]))

        return encoder_skips, decoder_skips, seg
