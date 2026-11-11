"""
gsplat (nerfstudio, CUDA) backend — the deployed server + CUDA desktops.

v5.2: batched depth-as-color rendering (tested D_batched_sh implementation).
Each camera retains its own camera-space Z as degree-0 SH coefficients.
Decode on the GPU and download one float32 depth block per camera group.
"""

from __future__ import annotations

import numpy as np
import torch

from gsplat import rasterization

from .base import Renderer, NEAR_PLANE, FAR_PLANE, SH_C0


# Independent of render_cameras.RENDER_BATCH (the outer RGB/view batch).
# Eight cameras per depth call matches the measured D_batched_sh configuration.
DEPTH_BATCH = 8


class GsplatRenderer(Renderer):
    name = "gsplat"

    def __init__(self):
        if not torch.cuda.is_available():
            # gsplat's rasterization is a compiled CUDA kernel; it cannot run on CPU.
            raise RuntimeError(
                "gsplat backend requires CUDA. On Apple Silicon use renderer='gsplat-metal'."
            )
        self.device = torch.device("cuda")

    def prepare(self, arrays):
        # gsplat 1.5.3 rasterization() wants activated scales + opacities.
        g = self._to_device(arrays)
        g["scales"] = torch.exp(g["log_scales"])
        g["opacities"] = torch.sigmoid(g["logit_opacities"])
        return g

    def render_rgb(self, g, w2c, K, width, height):
        B = w2c.shape[0]
        K_batch = K.unsqueeze(0).expand(B, -1, -1).contiguous()
        rgb_out, _, _ = rasterization(
            means=g["means"], quats=g["quats"], scales=g["scales"],
            opacities=g["opacities"], colors=g["sh"],
            viewmats=w2c, Ks=K_batch,
            width=width, height=height,
            sh_degree=g["sh_degree"],
            near_plane=NEAR_PLANE, far_plane=FAR_PLANE,
        )
        # Quantize before download: same RGB uint8 values, one byte/channel.
        return rgb_out.clamp(0, 1).mul(255).to(torch.uint8).cpu().numpy()

    def render_depth(self, g, w2c, K, width, height):
        """Return ordered float32 depth maps using the v5 depth-as-color formula.

        GPU decoding can differ slightly from the former NumPy decoding due to
        floating-point rounding. This is not gsplat's native D/ED render mode.
        """
        B = w2c.shape[0]
        N = g["means"].shape[0]
        K_batch = K.unsqueeze(0).expand(B, -1, -1).contiguous()

        # Camera-space Z for every Gaussian across all B views in one bmm.
        means_h = torch.cat(
            [g["means"], torch.ones(N, 1, device=self.device)], dim=1
        ).contiguous()  # (N, 4)
        z_batch = torch.bmm(
            w2c, means_h.T.unsqueeze(0).expand(B, -1, -1).contiguous(),
        )[:, 2, :].clamp(min=NEAR_PLANE).contiguous()  # (B, N)

        depth = []
        for start in range(0, B, DEPTH_BATCH):
            stop = min(start + DEPTH_BATCH, B)
            # Per-camera SH0 colors: (C, N, 1, 3), supported by gsplat 1.5.3.
            # Sharing one camera's Z values across the batch would be incorrect.
            dc = z_batch[start:stop, :, None, None].expand(-1, -1, 1, 3).contiguous()
            dr = rasterization(
                means=g["means"], quats=g["quats"], scales=g["scales"],
                opacities=g["opacities"], colors=dc,
                viewmats=w2c[start:stop], Ks=K_batch[start:stop],
                width=width, height=height, sh_degree=0,
                near_plane=NEAR_PLANE, far_plane=FAR_PLANE,
            )[0]
            if tuple(dr.shape) != (stop - start, height, width, 3):
                raise RuntimeError('Unexpected raster output shape: ' + str(dr.shape))

            # Same eager GPU operations as D_batched_sh; retain float32 depth.
            block = ((dr[:, :, :, 0] - 0.5) / SH_C0).clamp(min=0.0)
            if block.dtype != torch.float32:
                raise RuntimeError('Depth must remain float32, got ' + str(block.dtype))
            del dr, dc

            # One blocking download per group; NumPy views keep the CPU storage
            # alive and preserve the original list-of-depth-maps interface.
            cpu = block.cpu().numpy()
            depth.extend(cpu[i] for i in range(stop - start))
            del block
        if len(depth) != B:
            raise RuntimeError('Depth frame count mismatch')
        return depth
