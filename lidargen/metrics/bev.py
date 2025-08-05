import torch
from scipy.spatial.distance import jensenshannon


def point_cloud_to_histogram(
    point_cloud: torch.Tensor,
    field_size: float = 160.0,
    bins: int = 100,
    min_depth: float = 3.0,  # from lidargen
    max_depth: float = 70.0,  # from lidargen
) -> torch.Tensor:
    assert point_cloud.ndim == 2, "must be (N, 3)"
    assert bins % 2 == 0
    depth = point_cloud.norm(p=2, dim=1)
    mask = (depth > min_depth) & (depth < max_depth)
    bound = field_size / 2
    hist = torch.histogramdd(
        point_cloud[mask, 0:2].cpu(),  # xy
        bins=bins,
        range=[-bound, bound, -bound, bound],
    ).hist
    return hist


def cdist_rbf(p: torch.Tensor, q: torch.Tensor, sigma: float = 0.5) -> torch.Tensor:
    """RBF kernel"""
    assert p.ndim == q.ndim == 2
    dist = torch.cdist(p, q, p=2.0)
    gamma = 1 / (2 * sigma**2)
    dist = torch.exp(-gamma * dist**2)
    return dist


@torch.no_grad()
def compute_jsd_2d(hist1: torch.Tensor, hist2: torch.Tensor) -> float:
    """BEV-based Jensen-Shannon divergence (JSD)"""
    hist1 = hist1.flatten(1)  # (N, size**2)
    hist2 = hist2.flatten(1)  # (N, size**2)
    p = hist1.sum(dim=0) / hist1.sum()
    q = hist2.sum(dim=0) / hist2.sum()
    jsd = jensenshannon(p.cpu().numpy(), q.cpu().numpy())
    return jsd


@torch.no_grad()
def compute_mmd_2d(hist1: torch.Tensor, hist2: torch.Tensor) -> float:
    """BEV-based maximum mean discrepancy (MMD)"""
    hist1 = hist1.flatten(1)  # (N, size**2)
    hist2 = hist2.flatten(1)  # (N, size**2)
    p = hist1 / hist1.sum(dim=1, keepdim=True)
    q = hist2 / hist2.sum(dim=1, keepdim=True)
    mmd = cdist_rbf(p, p).mean() + cdist_rbf(q, q).mean() - 2 * cdist_rbf(p, q).mean()
    return mmd.item()
