import torch
import nvdiffrast.torch as dr
from torch import nn


class MeshRenderer_UV(nn.Module):
    def __init__(self, rasterize_size=224):
        super(MeshRenderer_UV, self).__init__()
        self.rasterize_size = rasterize_size
        self.ctx = None

    def forward(self, vertex, tri, feat=None):
        """
        Return:
            mask               -- torch.tensor, size (B, 1, H, W)
            depth              -- torch.tensor, size (B, 1, H, W)
            features(optional) -- torch.tensor, size (B, C, H, W) if feat is not None
            None               -- unused visible-vertex result

        Parameters:
            vertex          -- torch.tensor, size (B, N, 3)
            tri             -- torch.tensor, size (M, 3), shared triangles
            feat(optional)  -- torch.tensor, size (B, N, C), features
        """
        device = vertex.device
        rsize = int(self.rasterize_size)
        # Homogeneous coordinates; raster y has the same direction as v.
        if vertex.shape[-1] == 3:
            vertex = torch.cat([vertex, torch.ones([*vertex.shape[:2], 1]).to(device)], dim=-1)
            vertex[..., 1] = -vertex[..., 1]

        if self.ctx is None:
            self.ctx = dr.RasterizeCudaContext(device=device)
            print("create cuda ctx on device cuda:%d" % device.index)

        tri = tri.type(torch.int32).contiguous()
        rast_out, _ = dr.rasterize(self.ctx, vertex.contiguous(), tri, resolution=[rsize, rsize])

        depth, _ = dr.interpolate(vertex.reshape([-1,4])[...,2].unsqueeze(1).contiguous(), rast_out, tri)
        depth = depth.permute(0, 3, 1, 2)
        mask = (rast_out[..., 3] > 0).float().unsqueeze(1)
        depth = mask * depth

        image = None
        if feat is not None:
            image, _ = dr.interpolate(feat, rast_out, tri)
            image = image.permute(0, 3, 1, 2)
            image = mask * image

        return mask, depth, image, None
