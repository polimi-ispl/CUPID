import torch
import numpy as np
from ..util.nv_diffrast import MeshRenderer_UV
from .networks import ReconNetWrapper

def batched_bilinear_interpolate(img, x, y):
    """
    Batched bilinear interpolation function.
    
    Parameters:
        img: (B, H, W, C) - batch of images
        x, y: (B, N) - batch of coordinates
        
    Returns:
        interpolated_values: (B, N, C) - interpolated pixel values
    """
    B, H, W, C = img.shape
    N = x.shape[1]
    
    x0 = torch.floor(x).long()
    x1 = x0 + 1
    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, W - 1)
    x1 = torch.clamp(x1, 0, W - 1)
    y0 = torch.clamp(y0, 0, H - 1)
    y1 = torch.clamp(y1, 0, H - 1)

    # Create batch indices for advanced indexing
    batch_idx = torch.arange(B, device=img.device).unsqueeze(1).expand(-1, N)

    # Index with batch dimension: img[batch_idx, y_coords, x_coords]
    i_a = img[batch_idx, y0, x0]  # Top-left (B, N, C)
    i_b = img[batch_idx, y1, x0]  # Bottom-left (B, N, C)
    i_c = img[batch_idx, y0, x1]  # Top-right (B, N, C)
    i_d = img[batch_idx, y1, x1]  # Bottom-right (B, N, C)

    # Calculate weights based on fractional distances
    wa = (x1 - x) * (y1 - y)  # Weight for top-left (B, N)
    wb = (x1 - x) * (y - y0)  # Weight for bottom-left (B, N)
    wc = (x - x0) * (y1 - y)  # Weight for top-right (B, N)
    wd = (x - x0) * (y - y0)  # Weight for bottom-right (B, N)

    # Expand weights to match color channels
    wa = wa.unsqueeze(-1)  # (B, N, 1)
    wb = wb.unsqueeze(-1)  # (B, N, 1)
    wc = wc.unsqueeze(-1)  # (B, N, 1)
    wd = wd.unsqueeze(-1)  # (B, N, 1)

    # Weighted sum of the four corner pixels
    return wa * i_a + wb * i_b + wc * i_c + wd * i_d

def process_uv(uv_coords, uv_h = 224, uv_w = 224):
    uv_coords[:,0] = uv_coords[:,0] * (uv_w - 1)
    uv_coords[:,1] = uv_coords[:,1] * (uv_h - 1)
    # uv_coords[:,1] = uv_h - uv_coords[:,1] - 1
    uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1)))) # add z
    return uv_coords

class uv_face_model:
    def __init__(self, device, texture_size=224, model_path=None, recon_path=None):

        if model_path is None or recon_path is None:
            raise ValueError(
                "uv_face_model requires explicit model_path (face_model.npy) and "
                "recon_path (net_recon.pth); use cupid.weights to resolve them."
            )
        self.device = device
        self.texture_size = texture_size

        model = np.load(model_path, allow_pickle=True).item()

        # mean shape, size (107127, 1)
        self.u = torch.tensor(model['u'], requires_grad=False, dtype=torch.float32, device=self.device)
        # face identity bases, size (107127, 80)
        self.id = torch.tensor(model['id'], requires_grad=False, dtype=torch.float32, device=self.device)
        # face expression bases, size (107127, 64)
        self.exp = torch.tensor(model['exp'], requires_grad=False, dtype=torch.float32, device=self.device)
        # triangle faces, size (70789, 3)
        self.tri = torch.tensor(model['tri'], requires_grad=False, dtype=torch.int64, device=self.device)

        # projection matrix for rendering
        self.persc_proj = torch.tensor([1015.0, 0, 112.0, 0, 1015.0, 112.0, 0, 0, 1], requires_grad=False, dtype=torch.float32, device=self.device).reshape([3, 3]).transpose(0,1)
        self.camera_distance = 10.0

        processed_uv_coords_numpy = process_uv(model['uv_coords'].copy(), texture_size, texture_size)
        self.processed_uv_coords = (torch.tensor(processed_uv_coords_numpy, requires_grad=False, dtype=torch.float32, device=self.device) / (texture_size - 1) - 0.5) * 2 

        self.uv_renderer = MeshRenderer_UV(rasterize_size=int(texture_size))
        
        # Load Reconstruction Network
        self.net_recon = ReconNetWrapper()

        self.net_recon.load_state_dict(torch.load(recon_path, map_location=torch.device('cpu'))['net_recon'])
        self.net_recon = self.net_recon.to(self.device)
        self.net_recon.eval()


    def split_alpha(self, alpha):
        """
        Return:
            alpha_dict     -- a dict of torch.tensors

        Parameters:
            alpha          -- torch.tensor, size (B, 257)
        """
        alpha_id = alpha[:, :80]
        alpha_exp = alpha[:, 80: 144]
        alpha_alb = alpha[:, 144: 224]
        alpha_a = alpha[:, 224: 227]
        alpha_sh = alpha[:, 227: 254]
        alpha_t = alpha[:, 254:]
        return {
            'id': alpha_id,
            'exp': alpha_exp,
            'alb': alpha_alb,
            'angle': alpha_a,
            'sh': alpha_sh,
            'trans': alpha_t
        }

    def compute_shape(self, alpha_id, alpha_exp):
        """
        Return:
            face_shape       -- torch.tensor, size (B, N, 3), face vertice without rotation or translation

        Parameters:
            alpha_id         -- torch.tensor, size (B, 80), identity parameter
            alpha_exp        -- torch.tensor, size (B, 64), expression parameter
        """
        batch_size = alpha_id.shape[0]
        face_shape = torch.einsum('ij,aj->ai', self.id, alpha_id) + torch.einsum('ij,aj->ai', self.exp, alpha_exp) + self.u.reshape([1, -1])
        return face_shape.reshape([batch_size, -1, 3])

    def compute_rotation(self, angles):
        """
        Return:
            rot              -- torch.tensor, size (B, 3, 3), pts @ trans_mat

        Parameters:
            angles           -- torch.tensor, size (B, 3), use radian
        """
        batch_size = angles.shape[0]
        ones = torch.ones([batch_size, 1]).to(self.device)
        zeros = torch.zeros([batch_size, 1]).to(self.device)
        x, y, z = angles[:, :1], angles[:, 1:2], angles[:, 2:],
        
        rot_x = torch.cat([
            ones, zeros, zeros,
            zeros, torch.cos(x), -torch.sin(x), 
            zeros, torch.sin(x), torch.cos(x)
        ], dim=1).reshape([batch_size, 3, 3])
        
        rot_y = torch.cat([
            torch.cos(y), zeros, torch.sin(y),
            zeros, ones, zeros,
            -torch.sin(y), zeros, torch.cos(y)
        ], dim=1).reshape([batch_size, 3, 3])

        rot_z = torch.cat([
            torch.cos(z), -torch.sin(z), zeros,
            torch.sin(z), torch.cos(z), zeros,
            zeros, zeros, ones
        ], dim=1).reshape([batch_size, 3, 3])

        rot = rot_z @ rot_y @ rot_x
        return rot.permute(0, 2, 1)

    def transform(self, face_shape, rot, trans):
        """
        Return:
            face_shape       -- torch.tensor, size (B, N, 3) pts @ rot + trans

        Parameters:
            face_shape       -- torch.tensor, size (B, N, 3)
            rot              -- torch.tensor, size (B, 3, 3)
            trans            -- torch.tensor, size (B, 3)
        """
        return face_shape @ rot + trans.unsqueeze(1)

    def to_camera(self, face_shape):
        face_shape[..., -1] = self.camera_distance - face_shape[..., -1]
        return face_shape

    def to_image(self, face_shape):
        """
        Return:
            face_proj        -- torch.tensor, size (B, N, 2)

        Parameters:
            face_shape       -- torch.tensor, size (B, N, 3)
        """
        # to image_plane
        face_proj = face_shape @ self.persc_proj
        face_proj = face_proj[..., :2] / face_proj[..., 2:]
        return face_proj


    def forward(self, input_img):
        batch_size = len(input_img)
        assert self.net_recon.training == False
        with torch.no_grad():
            alpha = self.net_recon(input_img)

        alpha_dict = self.split_alpha(alpha)
        face_shape = self.compute_shape(alpha_dict['id'], alpha_dict['exp'])
        rotation = self.compute_rotation(alpha_dict['angle'])
        face_shape_transformed = self.transform(face_shape, rotation, alpha_dict['trans'])

        # face vertice in 3d
        v3d = self.to_camera(face_shape_transformed)

        # face vertice in 2d image plane
        v2d = self.to_image(v3d)

        img_colors = batched_bilinear_interpolate(
            input_img.permute(0, 2, 3, 1), 
            v2d[:,:,0], 
            (self.texture_size-1) - v2d[:,:,1]
        )
        _, _, uv_color_img, _ = self.uv_renderer(self.processed_uv_coords.repeat(batch_size, 1, 1), self.tri, img_colors)

        return uv_color_img