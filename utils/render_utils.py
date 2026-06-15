from models import (
    PositionalEmbeddings,
    NeRF,
)

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

def volume_render(
    raw: torch.Tensor,
    z_vals: torch.Tensor,
    rays_d: torch.Tensor,
    raw_noise_std: float=0.,
    use_white_background: bool=False
) -> tuple[torch.Tensor, torch.Tensor]:
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([
        dists,
        torch.full_like(dists[..., :1], 1e10),
    ], dim=-1)
    dists = dists * rays_d.norm(dim=-1, keepdim=True)

    rgb = torch.sigmoid(raw[..., :3])
    sigma = raw[..., 3]

    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn_like(sigma) * raw_noise_std

    alpha = 1. - torch.exp(-F.relu(sigma + noise) * dists)
    T = torch.cumprod(
        torch.cat([torch.ones_like(alpha[..., :1]), 1. - alpha + 1e-10], dim=-1),
        dim=-1,
    )[..., :-1] 

    weights = alpha * T
    rgb_map = (weights[..., None] * rgb).sum(dim=-2)
    if use_white_background:
        acc_map = weights.sum(dim=-1)
        rgb_map = rgb_map + (1. - acc_map[..., None])
    return rgb_map, weights

def sample_pdf(
    bins: torch.Tensor,
    weights: torch.Tensor,
    n_importance: int,
    eps: float=1e-5,
    use_deterministic_sampling=False
) -> torch.Tensor:
    weights = weights + eps
    pdf = weights / weights.sum(dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1) 

    if use_deterministic_sampling:
        u = torch.linspace(0., 1., n_importance, device=bins.device)
        u = u.unsqueeze(0).expand(bins.shape[0], -1)
    else:
        u = torch.rand(bins.shape[0], n_importance, device=bins.device)
    u = u.contiguous()

    indices = torch.searchsorted(cdf, u, right=True)
    below = (indices - 1).clamp(min=0, max=bins.shape[-1] - 1)
    above = indices.clamp(min=0, max=bins.shape[-1] - 1)

    gathered_indices = torch.stack([below, above], dim=-1)
    gathered_cdf = torch.gather(cdf,  1, gathered_indices.reshape(cdf.shape[0],  -1)).reshape(*gathered_indices.shape)
    gathered_bins = torch.gather(bins, 1, gathered_indices.reshape(bins.shape[0], -1)).reshape(*gathered_indices.shape)

    denom = (gathered_cdf[..., 1] - gathered_cdf[..., 0]).clamp(min=1e-5)
    t = (u - gathered_cdf[..., 0]) / denom
    return gathered_bins[..., 0] + t * (gathered_bins[..., 1] - gathered_bins[..., 0])

def render(
    nerf_coarse: NeRF,
    nerf_fine: NeRF,
    embed_pts: PositionalEmbeddings,
    embed_viewdirs: PositionalEmbeddings,
    rays: torch.Tensor,
    n_samples: int,
    n_importance: int,
    perturb: bool=False,
    chunk: int=32768,
    near: float=2.0,
    far: float=6.0,
    raw_noise_std: float=0.0,
    use_white_background: bool=False,
) -> dict[str, torch.Tensor]:
    rays_o, rays_d = rays[0], rays[1]
    n_rays = rays_o.shape[0]

    t_vals = torch.linspace(0., 1., n_samples, device=rays_o.device)
    z_vals = near * (1. - t_vals) + far * t_vals
    z_vals = z_vals.unsqueeze(0).expand(n_rays, -1)

    if perturb:
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
        lower = torch.cat([z_vals[..., :1], mids],  dim=-1)
        t_rand = torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * t_rand
    
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    pts_flatten  = pts.reshape(-1, 3)
    pts_embed = embed_pts(pts_flatten)

    if embed_viewdirs is not None:
        dirs = rays_d[:, None].expand_as(pts)
        dirs_flat = F.normalize(dirs.reshape(-1, 3), dim=-1)
        dirs_embed = embed_viewdirs(dirs_flat)
    else:
        dirs_embed = None

    coarse_outputs = []
    for i in range(0, pts_embed.shape[0], chunk):
        p = pts_embed[i : i + chunk]
        d = dirs_embed[i : i + chunk] if dirs_embed is not None else None
        coarse_outputs.append(nerf_coarse(p, d))
    coarse_raw = torch.cat(coarse_outputs, dim=0)
    coarse_raw = coarse_raw.reshape(-1, n_samples, coarse_raw.shape[-1])
    rgb_map, weights = volume_render(coarse_raw.float(), z_vals.float(), rays_d.float(), raw_noise_std, use_white_background)
    output = {"rgb_map": rgb_map}

    if nerf_fine is not None and n_importance > 0:
        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_fine = sample_pdf(
            z_vals_mid,
            weights[..., 1: -1],
            n_importance,
            use_deterministic_sampling=(not perturb),
        ).detach()

        z_vals_fine, _ = torch.sort(torch.cat([z_vals, z_fine], dim=-1), dim=-1)

        pts_fine    = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_fine[..., :, None]
        pts_flatten_embed = embed_pts(pts_fine.reshape(-1, 3))

        if embed_viewdirs is not None:
            dirs_fine    = rays_d[:, None].expand_as(pts_fine)
            dirs_f_embed = embed_viewdirs(F.normalize(dirs_fine.reshape(-1, 3), dim=-1))
        else:
            dirs_f_embed = None

        fine_outputs = []
        for i in range(0, pts_flatten_embed.shape[0], chunk):
            p = pts_flatten_embed[i : i + chunk]
            d = dirs_f_embed[i : i + chunk] if dirs_f_embed is not None else None
            fine_outputs.append(nerf_fine(p, d))

        fine_raw = torch.cat(fine_outputs, dim=0)
        fine_raw = fine_raw.reshape(-1, z_vals_fine.shape[-1], fine_raw.shape[-1])

        rgb_map_fine, _ = volume_render(fine_raw.float(), z_vals_fine.float(), rays_d.float(), raw_noise_std, use_white_background)
        output["rgb_map_fine"] = rgb_map_fine
    return output

def get_rays(H: int, W: int, K: torch.Tensor, c2w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device=c2w.device),
        torch.arange(H, dtype=torch.float32, device=c2w.device),
        indexing="xy",
    )
    i = i.to(c2w.device)
    j = j.to(c2w.device)

    dirs = torch.stack([
         (i - K[0, 2]) / K[0, 0],
        -(j - K[1, 2]) / K[1, 1],
        -torch.ones_like(i, device=c2w.device),
    ], dim=-1).to(device=c2w.device)

    rays_d = (dirs[..., None, :] * c2w[:3, :3]).sum(dim=-1)
    rays_o = c2w[:3, 3].expand(rays_d.shape)
    return rays_o, rays_d

def render_path(
    nerf_coarse: NeRF,
    nerf_fine: NeRF,
    embed_pts: PositionalEmbeddings,
    embed_viewdirs: PositionalEmbeddings,
    render_poses: torch.Tensor,
    n_samples: int,
    n_importance: int,
    batch_size: int,
    hwf: tuple[int, int, int],
    K: torch.Tensor,
    chunk: int=32768,
    near: float=2.0,
    far: float=6.0,
    use_white_background: bool=False,
) -> np.ndarray:
    H, W, focal = hwf

    rgbs = []
    for idx, pose in enumerate(tqdm(render_poses)):
        pose = pose[:3, :4]
        rays_o, rays_d = get_rays(H, W, K, pose)
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)

        all_rgb = []
        with torch.no_grad():
            for i in range(0, rays_o.shape[0], batch_size):
                rays = torch.stack([
                    rays_o[i: i + batch_size],
                    rays_d[i: i + batch_size],
                ], dim=0)

                output = render(
                    nerf_coarse=nerf_coarse,
                    nerf_fine=nerf_fine,
                    embed_pts=embed_pts,
                    embed_viewdirs=embed_viewdirs,
                    rays=rays,
                    n_samples=n_samples,
                    n_importance=n_importance,
                    perturb=False,
                    chunk=chunk,
                    near=near,
                    far=far,
                    raw_noise_std=0.,
                    use_white_background=use_white_background,
                )
                rgb_chunk = output.get("rgb_map_fine", output["rgb_map"])
                all_rgb.append(rgb_chunk.cpu())
        rgb = torch.cat(all_rgb, dim=0)
        rgb = rgb.reshape(H, W, 3).numpy()
        rgbs.append(rgb)
    rgbs = np.stack(rgbs, axis=0)
    return rgbs