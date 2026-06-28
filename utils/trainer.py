from models import (
    NeRF,
    PositionalEmbeddings
)

import os

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from torchutils import (
    ArgumentParser,
    set_seed,
)

from .load_blender_data import load_blender_data
from .render_utils import *

def parse_args():
    parser = ArgumentParser("Simple training loop.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Pretrained training configuration.",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["blender"],
        help="Dataset type. Must be one of: 'blender'.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Path to cache directory.",
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="Use cache data."
    )
    parser.add_argument(
        "--force_cache",
        action="store_true",
        help="Recache data."
    )
    parser.add_argument(
        "--half_res",
        action="store_true",
        help="Resize training data to 400x400."
    )
    parser.add_argument(
        "--include_file_ext",
        action="store_true",
        help="Whether to include file extension in the file path when reading training data.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Data root directory.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=8,
        help="Number of layers in the network (default: 8).",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=256,
        help="Number of channels per layer (default: 256).",
    )
    parser.add_argument(
        "--skip_connections",
        type=int,
        nargs="+",
        default=[4],
        help="Index of skip connection (default: [4]).",
    )
    parser.add_argument(
        "--use_viewdirs",
        action="store_true",
        help="Use full 5D input instead of 3D.",
    )
    parser.add_argument(
        "--use_white_background",
        action="store_true",
        help="Assume white background.",
    )
    parser.add_argument(
        "--central_crop_warmup_steps",
        type=int,
        default=500,
        help="Number of iterations to train for on center crops (default: 500).",
    )
    parser.add_argument(
        "--central_crop_warmup_fraction",
        type=float,
        default=0.5,
        help="Fraction of the training image taken for a center crop (default: 0.5).",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=200000,
        help="Number of iterations to train for (default: 200000).",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=64,
        help="Number of coarse samples per ray (default: 64).",
    )
    parser.add_argument(
        "--n_random_rays",
        type=int,
        default=1024,
        help="Number of rays (default: 1024).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Learning rate (default: 5e-4).",
    )
    parser.add_argument(
        "--lr_decay",
        type=float,
        default=0.1,
        help="Learning rate decay (default: 0.1).",
    )
    parser.add_argument(
        "--lr_decay_rate",
        type=int,
        default=500,
        help="Learning rate decay period (default: 500).",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="Adam beta1 (default: 0.9).",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="Adam beta2 (default: 0.999).",
    )
    parser.add_argument(
        "--n_importance",
        type=int,
        default=128,
        help="Number of additional fine samples per ray (default: 128).",
    )
    parser.add_argument(
        "--multires",
        type=int,
        default=10,
        help="log2 of max freq for positional encoding (3D location) (default: 10).",
    )
    parser.add_argument(
        "--multires_views",
        type=int,
        default=4,
        help="log2 of max freq for positional encoding (2D direction) (default: 4).",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=32768,
        help="Number of rays processed in parallel (default: 32768).",
    )
    parser.add_argument(
        "--perturb",
        action="store_true",
        help="Set `True` for jitter.",
    )
    parser.add_argument(
        "--raw_noise_std",
        type=float,
        default=0.,
        help="Standard deviation of noise added to during volume rendering.",
    )
    parser.add_argument(
        "--n_camera_poses",
        type=int,
        default=40,
        help="Number for equidistant camera angles in final output (default: 40).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(".", "results"),
        help="Directory to write the final output to (default: ./results).",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=os.path.join(".", "checkpoints"),
        help="Directory to write the final checkpoint to (default: ./checkpoints).",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=4.,
        help="Radius of the camera orbit (default: 4.).",
    )
    parser.add_argument(
        "--phi",
        type=float,
        default=-30.,
        help="Azimuthal angle of the camera (default: -30.).",
    )
    parser.add_argument(
        "--seed",
        type=float,
        default=0,
        help="Random seed (default: 0).",
    )
    args = parser.parse_args()    
    return args

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    set_seed(args.seed)
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    if args.dataset_type == "blender":
        near = 2.
        far = 6.

        train_images, train_poses, train_render_poses, train_hwf = load_blender_data(
            root=args.root,
            n_camera_poses=args.n_camera_poses,
            radius=args.radius,
            phi=args.phi,
            cache_dir=args.cache_dir,
            use_cache=args.use_cache,
            force_cache=args.force_cache,
            split="train",
            half_res=args.half_res,
            include_file_ext=args.include_file_ext,
        )

        train_render_poses = train_render_poses.to(device)
        
    if args.use_white_background:
        train_images = train_images[...,:3]*train_images[...,-1:] + (1. - train_images[...,-1:])
    else:
        train_images = train_images[...,:3]
    
    train_H, train_W, train_focal = train_hwf
    train_H, train_W = int(train_H), int(train_W)
    train_hwf = [train_H, train_W, train_focal]

    K = np.array([
        [train_focal, 0          , 0.5*train_W],
        [0          , train_focal, 0.5*train_H],
        [0          , 0          , 1          ],
    ])
    K = torch.from_numpy(K).to(device)

    model_parameters = []

    embed_pts = PositionalEmbeddings(
        in_dims=3,
        max_freq=args.multires - 1,
        num_freqs=args.multires,
        include_input=True,
        use_log_sampling=True,
        periodic_functions=(torch.sin, torch.cos),
    )

    if args.use_viewdirs:
        embed_viewdirs = PositionalEmbeddings(
            in_dims=3,
            max_freq=args.multires_views - 1,
            num_freqs=args.multires_views,
            include_input=True,
            use_log_sampling=True,
            periodic_functions=(torch.sin, torch.cos),
        )
        in_channel_views = embed_viewdirs.out_dim
    else:
        in_channel_views = 0

    out_channels = 4
    in_channels = embed_pts.out_dim
    nerf_coarse = NeRF(
        depth=args.depth,
        W=args.W,
        in_channels=in_channels,
        in_channel_views=in_channel_views,
        out_channels=out_channels,
        skip_connections=args.skip_connections,
        use_viewdirs=args.use_viewdirs,
    )
    nerf_coarse = nerf_coarse.to(device)
    model_parameters.extend(nerf_coarse.parameters())

    nerf_fine = None
    if args.n_importance:
        nerf_fine = NeRF(
            depth=args.depth,
            W=args.W,
            in_channels=in_channels,
            in_channel_views=in_channel_views,
            out_channels=out_channels,
            skip_connections=args.skip_connections,
            use_viewdirs=args.use_viewdirs,
        )
        nerf_fine = nerf_fine.to(device)
        model_parameters.extend(nerf_fine.parameters())

    optimizer = optim.Adam(
        model_parameters,
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: args.lr_decay ** (step / args.lr_decay_rate)
    )

    use_amp = (device.type == "cuda")
    scaler = GradScaler(
        enabled=use_amp,
        init_scale=2**8,
        growth_interval=1000,
    )

    global_step = 0
    pbar = tqdm(range(args.n_steps))

    for _ in pbar:
        img_idx = np.random.randint(0, len(train_images))
        target = torch.from_numpy(train_images[img_idx]).to(device)
        pose = torch.from_numpy(train_poses[img_idx, :3, :4]).to(device)

        rays_o, rays_d = get_rays(train_H, train_W, K, pose)

        if args.n_random_rays > 0:
            if global_step < args.central_crop_warmup_steps:
                h_frac = int(train_H * args.central_crop_warmup_fraction)
                w_frac = int(train_W * args.central_crop_warmup_fraction)
                h_start = train_H // 2 - h_frac // 2
                w_start = train_W // 2 - w_frac // 2
                coords = torch.stack(
                    torch.meshgrid(
                        torch.arange(h_start, h_start + h_frac, device=device),
                        torch.arange(w_start, w_start + w_frac, device=device),
                        indexing="ij",
                    ), dim=-1,
                ).reshape(-1, 2)
            else:
                coords = torch.stack(
                    torch.meshgrid(
                        torch.arange(train_H, device=device),
                        torch.arange(train_W,  device=device),
                        indexing="ij",
                    ), dim=-1,
                ).reshape(-1, 2)
            selected_coords = coords[torch.randperm(len(coords))[:args.n_random_rays]]
            rays_o = rays_o[selected_coords[:, 0], selected_coords[:, 1]]
            rays_d = rays_d[selected_coords[:, 0], selected_coords[:, 1]]
            target = target[selected_coords[:, 0], selected_coords[:, 1]]
        rays = torch.stack([rays_o, rays_d], dim=0)

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, enabled=use_amp):
            output = render(
                nerf_coarse=nerf_coarse,
                nerf_fine=nerf_fine,
                embed_pts=embed_pts,
                embed_viewdirs=embed_viewdirs if args.use_viewdirs else None,
                rays=rays,
                n_samples=args.n_samples,
                n_importance=args.n_importance,
                perturb=args.perturb,
                chunk=args.chunk,
                near=near,
                far=far,
                raw_noise_std=args.raw_noise_std,
                use_white_background=args.use_white_background,
            )

            loss = F.mse_loss(output["rgb_map"], target)
            if "rgb_map_fine" in output:
                loss = loss + F.mse_loss(output["rgb_map_fine"], target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        psnr = -10. * torch.log10(loss.detach())
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "psnr": f"{psnr.item():.2f}",
        })
        global_step += 1

    train_poses = torch.from_numpy(train_poses).to(device)
    rgbs = render_path(
        nerf_coarse=nerf_coarse,
        nerf_fine=nerf_fine,
        embed_pts=embed_pts,
        embed_viewdirs=embed_viewdirs if args.use_viewdirs else None,
        render_poses=train_poses,
        n_samples=args.n_samples,
        n_importance=args.n_importance,
        batch_size=args.n_random_rays,
        hwf=train_hwf,
        K=K,
        chunk=args.chunk,
        near=near,
        far=far,
        use_white_background=args.use_white_background,
    )
    mse = np.mean((rgbs - train_images) ** 2)
    psnr = -10. * np.log10(mse)
    print(f"Mean PSNR: {psnr:.2f}")

    rgbs = render_path(
        nerf_coarse=nerf_coarse,
        nerf_fine=nerf_fine,
        embed_pts=embed_pts,
        embed_viewdirs=embed_viewdirs if args.use_viewdirs else None,
        render_poses=train_render_poses,
        n_samples=args.n_samples,
        n_importance=args.n_importance,
        batch_size=args.n_random_rays,
        hwf=train_hwf,
        K=K,
        chunk=args.chunk,
        near=near,
        far=far,
        use_white_background=args.use_white_background,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"{os.path.basename(args.root)}.gif")
    imageio.mimsave(
        output_file,
        (rgbs * 255).clip(0, 255).astype(np.uint8),
        fps=30,
        loop=0,
    )

    checkpoint_dir = os.path.join(args.checkpoint_dir, f"{os.path.basename(args.root)}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_file = os.path.join(checkpoint_dir, f"{os.path.basename(args.root)}_{args.n_steps}.pth")

    torch.save({
        "nerf_coarse": {
            "depth": nerf_coarse.depth,
            "W": nerf_coarse.W,
            "in_channels": nerf_coarse.in_channels,
            "in_channel_views": nerf_coarse.in_channel_views,
            "out_channels": nerf_coarse.out_channels,
            "skip_connections": nerf_coarse.skip_connections,
            "use_viewdirs": nerf_coarse.use_viewdirs,
            "weights": nerf_coarse.state_dict(),
        },
        "nerf_fine": {
            "depth": nerf_fine.depth,
            "W": nerf_fine.W,
            "in_channels": nerf_fine.in_channels,
            "in_channel_views": nerf_fine.in_channel_views,
            "out_channels": nerf_fine.out_channels,
            "skip_connections": nerf_fine.skip_connections,
            "use_viewdirs": nerf_fine.use_viewdirs,
            "weights": nerf_fine.state_dict(),
        } if nerf_fine is not None else None,
        "embed_pts": {
            "in_dims": embed_pts.in_dims,
            "max_freq": embed_pts.max_freq,
            "num_freqs": embed_pts.num_freqs,
            "include_input": embed_pts.include_input,
            "use_log_sampling": embed_pts.use_log_sampling,
        },
        "embed_viewdirs": {
            "in_dims": embed_viewdirs.in_dims,
            "max_freq": embed_viewdirs.max_freq,
            "num_freqs": embed_viewdirs.num_freqs,
            "include_input": embed_viewdirs.include_input,
            "use_log_sampling": embed_viewdirs.use_log_sampling,
        } if args.use_viewdirs else None,
        "render_poses": train_render_poses.cpu(),
        "n_samples": args.n_samples,
        "n_importance": args.n_importance,
        "batch_size": args.n_random_rays,
        "hwf": train_hwf,
        "K": K.cpu(),
        "chunk": args.chunk,
        "near": near,
        "far": far,
        "use_white_background": args.use_white_background,
    }, checkpoint_file)

if __name__ == "__main__":
    main()
