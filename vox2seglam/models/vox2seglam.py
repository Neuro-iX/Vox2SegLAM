from typing import Any

import torch
import torch.nn as nn

from ..utils.geometry import clamp_coords_zyx
from .backbone.unet import UNet
from .landmark.graph import GraphSpec, LandmarkGraphNet
from .landmark.refiner import RefinerHead, decode
from .landmark.template import GraphTemplate, load_graph_template
from .sampling import sample_patches_multiscale

__all__ = ["Vox2SegLAM"]


# UNet backbone -> optional segmentation head, and in parallel backbone features ->
# LandmarkGraphNet (coarse coords) -> optional RefinerHead (sub-voxel correction)
class Vox2SegLAM(nn.Module):
    def __init__(
        self,
        unet: UNet,
        landmark_graph: LandmarkGraphNet,
        refiner: RefinerHead | None = None,
        *,
        refine_skip_indices: tuple = (0, 8),
        refine_patch_size: int = 13,
        patch_shape: tuple = (240, 304, 240),
        refine_pitch: float = 2.0,
        detach_backbone: bool = False,
        detach_graph_for_refiner: bool = False,
    ) -> None:
        super().__init__()
        self.unet = unet
        self.landmark_graph = landmark_graph
        self.refiner = refiner

        self.K = int(landmark_graph.K)
        self.patch_shape_zyx = tuple(int(v) for v in patch_shape)
        self.refine_skip_indices = tuple(int(i) for i in refine_skip_indices)
        self.refine_patch_size = int(refine_patch_size)
        self.refine_pitch = float(refine_pitch)
        self.detach_backbone = bool(detach_backbone)
        self.detach_graph_for_refiner = bool(detach_graph_for_refiner)

    @classmethod
    def from_config(
        cls, cfg, device: torch.device | None = None, template: GraphTemplate | None = None
    ) -> "Vox2SegLAM":
        """Build the model; `template` (e.g. template_from_state_dict of a checkpoint)
        replaces reading template_dir, so a trained model needs no template files."""
        cfg.validate()
        lam_cfg, bb_cfg = cfg.model.lam, cfg.model.backbone
        device = device or torch.device("cpu")

        if template is None:
            edges, edge_file = lam_cfg.edge_source()
            template = load_graph_template(
                template_dir=cfg.paths.resolve_template_dir(),
                device=device,
                K_expected=cfg.task.n_landmarks,
                load_edge_index=True,
                edge_filename=edge_file,
                edges=edges,
            )
        else:
            template = GraphTemplate(
                template.mean_coords_zyx.to(device), template.edge_index.to(device)
            )

        spec = GraphSpec(
            K=cfg.task.n_landmarks,
            patch_size=tuple(cfg.task.patch_size),
            encoder_channels=tuple(bb_cfg.encoder_channels),
            decoder_channels=tuple(bb_cfg.decoder_channels),
            graph_channels=tuple(lam_cfg.graph_channels),
            steps=lam_cfg.steps,
            aggregate_indices=tuple(tuple(x) for x in lam_cfg.aggregate_indices),
            norm=lam_cfg.graph_norm,
            clamp_coords=lam_cfg.clamp_coords,
            n_residual_blocks=lam_cfg.n_residual_blocks,
            n_f2f_hidden_layer=lam_cfg.n_hidden_layers,
            detach_between_steps=lam_cfg.detach_between_steps,
        )

        refiner = None
        if lam_cfg.refine:
            # the refiner's input channels: the sum of every skip map it samples (refine_skip_indices)
            skip_ch = list(bb_cfg.encoder_channels) + list(bb_cfg.decoder_channels)
            in_ch = sum(skip_ch[i] for i in lam_cfg.refine_skip_indices)
            refiner = RefinerHead(
                in_channels=in_ch,
                n_landmarks=cfg.task.n_landmarks,
                gnn_dim=int(lam_cfg.graph_channels[-1]),
                hidden=lam_cfg.refine_hidden,
                n_conv_layers=lam_cfg.refine_n_conv_layers,
                embed_dim=lam_cfg.refine_embed_dim,
                cond_dim=lam_cfg.refine_cond_dim,
            )

        return cls(
            unet=UNet.from_config(cfg),
            landmark_graph=LandmarkGraphNet(cfg=spec, template=template),
            refiner=refiner,
            refine_skip_indices=tuple(lam_cfg.refine_skip_indices),
            refine_patch_size=lam_cfg.refine_patch_size,
            patch_shape=tuple(cfg.task.patch_size),
            refine_pitch=lam_cfg.refine_pitch,
            detach_backbone=lam_cfg.detach_backbone,
            detach_graph_for_refiner=lam_cfg.detach_graph_for_refiner,
        )

    def forward(
        self,
        x: torch.Tensor,
        coords_init_zyx: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        encoder_skips, decoder_skips, seg_out = self.unet(x)
        skips = [s.float() for s in list(encoder_skips) + list(decoder_skips)]
        # stop-gradient: the landmark branch must not disturb the segmentation-optimised backbone
        if self.detach_backbone:
            skips = [s.detach() for s in skips]

        if coords_init_zyx is None:
            coords_init = (
                self.landmark_graph.mean_coords_zyx.unsqueeze(0)
                .expand(x.shape[0], -1, -1)
                .to(x.device)
            )
        else:
            coords_init = coords_init_zyx.to(x.device)

        coords_graph_last, deltas, gnn_latent, coords_steps = self.landmark_graph(
            skips=skips, coords_init_zyx=coords_init
        )

        out: dict[str, Any] = {
            "coords": coords_graph_last,
            "seg": seg_out,
            "deltas": deltas,
            "coords_steps": coords_steps,  # after each graph step, clamped as the model moved them
            "coords_init_zyx_used": coords_init,
        }
        if self.refiner is None:
            return out

        # stop-gradient: the refiner is trained on its own local objective (trainer.py)
        patch_pos = coords_graph_last
        cond_latent = gnn_latent
        coords_for_output = coords_graph_last
        if self.detach_graph_for_refiner:
            patch_pos = patch_pos.detach()
            cond_latent = cond_latent.detach()
            coords_for_output = coords_for_output.detach()
        patches = sample_patches_multiscale(
            skips,
            patch_pos,
            self.refine_skip_indices,
            self.refine_patch_size,
            self.refine_pitch,
            self.patch_shape_zyx,
        )
        head = self.refiner(patches, cond_latent)
        delta_local, probs = decode(
            head["logits"], head["offsets"], self.refine_patch_size, self.refine_pitch
        )

        # the refiner's correction replaces the output coords (the graph output stays under "local")
        out["coords"] = clamp_coords_zyx(coords_for_output + delta_local, self.patch_shape_zyx)
        out["local"] = {
            "coords_graph_last": coords_graph_last,
            "delta_local": delta_local,
            "logits": head["logits"],
            "offsets": head["offsets"],
            "probs": probs,
            "patch_size": self.refine_patch_size,
            "pitch": self.refine_pitch,
        }
        return out

    def voxel_parameters(self):
        return list(self.unet.parameters())

    def landmark_parameters(self):
        params = list(self.landmark_graph.parameters())
        if self.refiner is not None:
            params += list(self.refiner.parameters())
        return params

    # partitions every parameter into exactly one named stage, each with its own optimizer,
    # scheduler and curriculum entry: "seg" (if enabled), one "gnnN" per graph step, "refine"
    def stage_parameters(self) -> dict[str, list]:
        stages: dict[str, list] = {}
        has_seg = self.unet.n_classes is not None
        if has_seg:
            stages["seg"] = list(self.unet.parameters())

        graph = self.landmark_graph
        for step in range(len(graph.f2v_layers)):
            params = list(graph.f2f_res_layers[step].parameters()) + list(
                graph.f2v_layers[step].parameters()
            )
            # with no seg head the backbone joins the first GNN stage, its first consumer
            if step == 0 and not has_seg:
                params = list(self.unet.parameters()) + params
            stages[f"gnn{step + 1}"] = params

        if self.refiner is not None:
            stages["refine"] = list(self.refiner.parameters())

        # every parameter must land in exactly one stage, or some tensor goes untrained or is
        # double-counted
        covered = sum(len(v) for v in stages.values())
        expected = len(self.voxel_parameters()) + len(self.landmark_parameters())
        if covered != expected:
            raise RuntimeError(
                f"stage_parameters covers {covered} tensors but the model has {expected}"
            )
        return stages
