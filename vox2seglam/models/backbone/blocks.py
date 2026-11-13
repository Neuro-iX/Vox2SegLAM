from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

__all__ = ["ResidualBlock", "ResidualUNetEncoder", "ResidualUNetDecoder"]


# conv-norm-relu twice plus a skip connection (a 1x1 projection only if the channel count
# changes); the building block of encoder and decoder
class ResidualBlock(nn.Module):
    def __init__(
        self,
        num_channels_in: int,
        num_channels_out: int,
        normalize: str = "instance",
        p_dropout: float | None = None,
    ) -> None:
        super().__init__()

        if normalize == "batch":
            NormLayer = nn.BatchNorm3d
        elif normalize == "instance":
            NormLayer = lambda c: nn.InstanceNorm3d(c, affine=True)
        elif normalize == "layer":
            NormLayer = lambda c: nn.GroupNorm(1, c)
        else:
            raise ValueError("Invalid normalization.")

        self.conv1 = nn.Conv3d(num_channels_in, num_channels_out, kernel_size=3, padding=1)
        self.norm1 = NormLayer(num_channels_out)

        if p_dropout is not None and p_dropout > 0:
            self.dropout = nn.Dropout3d(p_dropout)
        else:
            self.dropout = nn.Identity()

        self.conv2 = nn.Conv3d(num_channels_out, num_channels_out, kernel_size=3, padding=1)
        self.norm2 = NormLayer(num_channels_out)

        if num_channels_in != num_channels_out:
            self.adapt_skip = nn.Sequential(
                nn.Conv3d(num_channels_in, num_channels_out, kernel_size=1, bias=False),
                NormLayer(num_channels_out),
            )
        else:
            self.adapt_skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_out = self.conv1(x)
        x_out = F.relu(self.norm1(x_out), inplace=False)
        x_out = self.dropout(x_out)
        x_out = self.conv2(x_out)
        x_out = self.norm2(x_out)
        x_out = x_out + self.adapt_skip(x)
        return F.relu(x_out, inplace=False)


# stage 0 is a full-resolution stem; each later stage halves the resolution (stride-2 conv)
# then runs a ResidualBlock. forward() returns every stage's output: the skips
class ResidualUNetEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        encoder_channels: Sequence[int],
        p_dropout: float | None,
        normalize: str = "instance",
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()

        self.num_steps = len(encoder_channels)
        self.channels = list(encoder_channels)

        down_layers = [
            nn.Sequential(
                nn.Conv3d(input_channels, self.channels[0], kernel_size=3, padding=1),
                ResidualBlock(
                    self.channels[0],
                    self.channels[0],
                    normalize=normalize,
                    p_dropout=p_dropout,
                ),
            )
        ]

        for i in range(1, self.num_steps):
            down_layers.append(
                nn.Sequential(
                    nn.Conv3d(self.channels[i - 1], self.channels[i], kernel_size=2, stride=2),
                    ResidualBlock(
                        self.channels[i],
                        self.channels[i],
                        normalize=normalize,
                        p_dropout=p_dropout,
                    ),
                )
            )

        self.encoder = nn.ModuleList(down_layers)
        self.use_checkpoint = use_checkpoint

    # gradient checkpointing recomputes activations on backward to save VRAM; skipped in eval
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        skips: list[torch.Tensor] = []
        for layer in self.encoder:
            if self.use_checkpoint and torch.is_grad_enabled():
                x = grad_checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
            skips.append(x)
        return skips


# transposed-conv upsampling, concatenation of the matching encoder skip, then a
# ResidualBlock; one stage per encoder stage except the bottleneck, the starting point
class ResidualUNetDecoder(nn.Module):
    def __init__(
        self,
        encoder: ResidualUNetEncoder,
        decoder_channels: Sequence[int],
        p_dropout: float | None,
        normalize: str = "instance",
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()

        num_steps = encoder.num_steps - 1
        self.channels = list(decoder_channels)

        # the first up-step reads the bottleneck (channels[-1]) and the next-to-last encoder stage
        # (channels[-2]); each later step moves one stage shallower
        up_layers = []

        up_layers.append(
            nn.Sequential(
                nn.ConvTranspose3d(
                    encoder.channels[-1],
                    self.channels[0],
                    kernel_size=2,
                    stride=2,
                ),
                ResidualBlock(
                    encoder.channels[-2] + self.channels[0],
                    self.channels[0],
                    normalize=normalize,
                    p_dropout=p_dropout,
                ),
            )
        )

        for i in range(1, num_steps):
            up_layers.append(
                nn.Sequential(
                    nn.ConvTranspose3d(
                        self.channels[i - 1],
                        self.channels[i],
                        kernel_size=2,
                        stride=2,
                    ),
                    ResidualBlock(
                        encoder.channels[-i - 2] + self.channels[i],
                        self.channels[i],
                        normalize=normalize,
                        p_dropout=p_dropout,
                    ),
                )
            )

        self.decoder = nn.ModuleList(up_layers)
        self.use_checkpoint = use_checkpoint

    @staticmethod
    def _decoder_step(
        up_conv: nn.Module,
        res_block: nn.Module,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        x = up_conv(x)
        x = torch.cat((x, skip), dim=1)
        return res_block(x)

    # reverses the encoder skips (down_skips[0] is the bottleneck, down_skips[i+1] matches
    # up-step i); returns every decoder stage's output, used by deep supervision
    def forward(self, skips: list[torch.Tensor]) -> list[torch.Tensor]:
        down_skips = skips[::-1]
        x = down_skips[0]

        up_skips: list[torch.Tensor] = []

        for i, layer in enumerate(self.decoder):
            if self.use_checkpoint and torch.is_grad_enabled():
                x = grad_checkpoint(
                    self._decoder_step,
                    layer[0],
                    layer[1],
                    x,
                    down_skips[i + 1],
                    use_reentrant=False,
                )
            else:
                x = layer[0](x)
                x = torch.cat((x, down_skips[i + 1]), dim=1)
                x = layer[1](x)
            up_skips.append(x)

        return up_skips
