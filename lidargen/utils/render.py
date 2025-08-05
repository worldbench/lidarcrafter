import kornia
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from kornia.geometry.conversions import axis_angle_to_rotation_matrix


def make_Rt(
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    device: torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    # rotation of point clouds
    zero = torch.zeros(1, device=device)
    roll = torch.full_like(zero, fill_value=roll, device=device)
    pitch = torch.full_like(zero, fill_value=pitch, device=device)
    yaw = torch.full_like(zero, fill_value=yaw, device=device)

    # extrinsic parameters: yaw -> pitch order
    R = axis_angle_to_rotation_matrix(torch.stack([zero, zero, yaw], dim=-1))
    R @= axis_angle_to_rotation_matrix(torch.stack([zero, pitch, zero], dim=-1))
    R @= axis_angle_to_rotation_matrix(torch.stack([roll, zero, zero], dim=-1))
    t = torch.tensor([[x, y, z]], device=device)
    return R, t


def render_point_clouds(
    points: torch.Tensor,
    colors: torch.Tensor | None = None,
    size: int = 800,
    R: torch.Tensor | None = None,
    t: torch.Tensor | None = None,
    focal_length=1.0,
) -> torch.Tensor:
    points = points.clone()
    points[..., 2] *= -1
    device = points.device

    if colors is None:
        B, N, _ = points.shape
        colors = torch.ones(B, N, 3).to(points)

    # extrinsic parameters
    if R is not None:
        assert R.shape[-2:] == (3, 3)
        points = points @ R
    if t is not None:
        assert t.shape[-1:] == (3,)
        points += t

    # intrinsic parameters
    K = torch.eye(3, device=device)
    K[0, 0] = focal_length  # fx
    K[1, 1] = focal_length  # fy
    K[0, 2] = 0.5  # cx, points in [-1,1]
    K[1, 2] = 0.5  # cy
    K = K[None]

    # project 3d points onto the image plane
    uv = kornia.geometry.project_points(points, K)

    uv = uv * size
    mask = (0 < uv) & (uv < size - 1)
    mask = torch.logical_and(mask[..., [0]], mask[..., [1]])

    colors = colors * mask

    # z-buffering
    uv = size - uv
    depth = torch.norm(points, p=2, dim=-1, keepdim=True)  # B,N,1
    weight = 1.0 / torch.exp(3.0 * depth)
    weight *= (depth > 1e-8).detach()
    bev = bilinear_rasterizer(uv, weight * colors, (size, size))
    bev /= bilinear_rasterizer(uv, weight, (size, size)) + 1e-8
    return bev


def bilinear_rasterizer(
    coords: torch.Tensor, values: torch.Tensor, out_shape: tuple[int, int]
):
    """
    https://github.com/VCL3D/SphericalViewSynthesis/blob/master/supervision/splatting.py
    """

    B, _, C = values.shape
    H, W = out_shape
    device = coords.device

    h = coords[..., [0]].expand(-1, -1, C)
    w = coords[..., [1]].expand(-1, -1, C)

    # Four adjacent pixels
    h_t = torch.floor(h)
    h_b = h_t + 1  # == torch.ceil(h)
    w_l = torch.floor(w)
    w_r = w_l + 1  # == torch.ceil(w)

    h_t_safe = torch.clamp(h_t, 0.0, H - 1)
    h_b_safe = torch.clamp(h_b, 0.0, H - 1)
    w_l_safe = torch.clamp(w_l, 0.0, W - 1)
    w_r_safe = torch.clamp(w_r, 0.0, W - 1)

    weight_h_t = (h_b - h) * (h_t == h_t_safe).detach().float()
    weight_h_b = (h - h_t) * (h_b == h_b_safe).detach().float()
    weight_w_l = (w_r - w) * (w_l == w_l_safe).detach().float()
    weight_w_r = (w - w_l) * (w_r == w_r_safe).detach().float()

    # Bilinear weights
    weight_tl = weight_h_t * weight_w_l
    weight_tr = weight_h_t * weight_w_r
    weight_bl = weight_h_b * weight_w_l
    weight_br = weight_h_b * weight_w_r

    # For stability
    weight_tl *= (weight_tl >= 1e-3).detach().float()
    weight_tr *= (weight_tr >= 1e-3).detach().float()
    weight_bl *= (weight_bl >= 1e-3).detach().float()
    weight_br *= (weight_br >= 1e-3).detach().float()

    values_tl = values * weight_tl  # (B,N,C)
    values_tr = values * weight_tr
    values_bl = values * weight_bl
    values_br = values * weight_br

    indices_tl = (w_l_safe + W * h_t_safe).long()
    indices_tr = (w_r_safe + W * h_t_safe).long()
    indices_bl = (w_l_safe + W * h_b_safe).long()
    indices_br = (w_r_safe + W * h_b_safe).long()

    render = torch.zeros(B, H * W, C, device=device)
    render.scatter_add_(dim=1, index=indices_tl, src=values_tl)
    render.scatter_add_(dim=1, index=indices_tr, src=values_tr)
    render.scatter_add_(dim=1, index=indices_bl, src=values_bl)
    render.scatter_add_(dim=1, index=indices_br, src=values_br)
    render = render.reshape(B, H, W, C).permute(0, 3, 1, 2)

    return render


def estimate_surface_normal(
    points: torch.Tensor, d: int = 2, mode: str = "closest"
) -> torch.Tensor:
    # estimate surface normal from coordinated point clouds
    # re-implemented the following codes with pytorch:
    # https://github.com/wkentaro/morefusion/blob/master/morefusion/geometry/estimate_pointcloud_normals.py
    # https://github.com/jmccormac/pySceneNetRGBD/blob/master/calculate_surface_normals.py

    assert points.dim() == 4, f"expected (B,3,H,W), but got {points.shape}"
    B, C, H, W = points.shape
    assert C == 3, f"expected C==3, but got {C}"
    device = points.device

    # points = F.pad(points, (0, 0, d, d), mode="constant", value=float("inf"))
    points = F.pad(points, (0, 0, d, d), mode="replicate")
    points = F.pad(points, (d, d, 0, 0), mode="circular")
    points = points.permute(0, 2, 3, 1)  # (B,H,W,3)

    # 8 adjacent offsets
    #  -----------
    # | 7 | 6 | 5 |
    #  -----------
    # | 0 |   | 4 |
    #  -----------
    # | 1 | 2 | 3 |
    #  -----------
    offsets = torch.tensor(
        [
            # (dh,dw)
            (-d, 0),  # 0
            (-d, d),  # 1
            (0, d),  # 2
            (d, d),  # 3
            (d, 0),  # 4
            (d, -d),  # 5
            (0, -d),  # 6
            (-d, -d),  # 7
        ],
        device=device,
    )

    # (B,H,W) indices
    b = torch.arange(B, device=device)[:, None, None]
    h = torch.arange(H, device=device)[None, :, None]
    w = torch.arange(W, device=device)[None, None, :]
    k = torch.arange(8, device=device)

    # anchor points
    b1 = b[:, None]  # (B,1,1,1)
    h1 = h[:, None] + d  # (1,1,H,1)
    w1 = w[:, None] + d  # (1,1,1,W)
    anchors = points[b1, h1, w1]  # (B,H,W,3) -> (B,1,H,W,3)

    # neighbor points
    offset = offsets[k]  # (8,2)
    b2 = b1
    h2 = h1 + offset[None, :, 0, None, None]  # (1,8,H,1)
    w2 = w1 + offset[None, :, 1, None, None]  # (1,8,1,W)
    points1 = points[b2, h2, w2]  # (B,8,H,W,3)

    # anothor neighbor points
    offset = offsets[(k + 2) % 8]
    b3 = b1
    h3 = h1 + offset[None, :, 0, None, None]
    w3 = w1 + offset[None, :, 1, None, None]
    points2 = points[b3, h3, w3]  # (B,8,H,W,3)

    if mode == "closest":
        # find the closest neighbor pair
        diff = torch.norm(points1 - anchors, dim=4)
        diff = diff + torch.norm(points2 - anchors, dim=4)
        i = torch.argmin(diff, dim=1)  # (B,H,W)
        # get normals by cross product
        anchors = anchors[b, 0, h, w]  # (B,H,W,3)
        points1 = points1[b, i, h, w]  # (B,H,W,3)
        points2 = points2[b, i, h, w]  # (B,H,W,3)
        vector1 = points1 - anchors
        vector2 = points2 - anchors
        normals = torch.cross(vector1, vector2, dim=-1)  # (B,H,W,3)
    elif mode == "mean":
        # get normals by cross product
        vector1 = points1 - anchors
        vector2 = points2 - anchors
        normals = torch.cross(vector1, vector2, dim=-1)  # (B,8,H,W,3)
        normals = normals.mean(dim=1)  # (B,H,W,3)
    else:
        raise NotImplementedError(mode)

    normals = normals / (torch.norm(normals, dim=3, keepdim=True) + 1e-8)
    normals = normals.permute(0, 3, 1, 2)  # (B,3,H,W)

    return normals


@torch.no_grad()
def colorize(tensor: torch.Tensor, cmap_fn=cm.turbo) -> torch.Tensor:
    colors = cmap_fn(np.linspace(0, 1, 256))[:, :3]
    colors = torch.from_numpy(colors).to(tensor)
    tensor = tensor.squeeze(1) if tensor.ndim == 4 else tensor
    ids = (tensor * 256).clamp(0, 255).long()
    tensor = F.embedding(ids, colors).permute(0, 3, 1, 2)
    tensor = tensor.mul(255).clamp(0, 255).byte()
    return tensor