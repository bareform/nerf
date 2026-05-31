import json
import os
import pickle

import imageio
import numpy as np
import torch
import torch.nn.functional as F

trans_t = lambda t : torch.tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()

rot_phi = lambda phi : torch.tensor([
    [1,           0,            0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi),  np.cos(phi), 0],
    [0,           0,            0, 1]]).float()

rot_theta = lambda th : torch.tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0         , 1,           0, 0],
    [np.sin(th), 0,  np.cos(th), 0],
    [0         , 0,           0, 1]]).float()

to_nerf = lambda c2w : torch.tensor([
    [-1, 0, 0, 0],
    [ 0, 0, 1, 0],
    [ 0, 1, 0, 0],
    [ 0, 0, 0, 1],
]).float() @ c2w


def load_blender_data(
    root: str,
    n_camera_poses: int,
    radius: float,
    phi: float,
    cache_dir: str=None,
    use_cache: bool=True,
    force_cache: bool=False,
    split="train",
    half_res=False,
    include_file_ext=True,
):
    if use_cache:
        if not force_cache:
            if cache_dir is not None:
                cache_file = os.path.join(cache_dir, f"{os.path.basename(root)}_{split}.pkl")
                if os.path.exists(cache_file):
                    with open(cache_file, "rb") as fp:
                        data = pickle.load(fp)
                    return data["imgs"], data["poses"], data["render_poses"], [data["H"], data["W"], data["focal"]]

    with open(os.path.join(root, f"transforms_{split}.json"), "r") as fp:
        data_dict = json.load(fp)

    frames = data_dict["frames"]

    first = imageio.imread(os.path.join(root, frames[0]["file_path"] + (".png" if include_file_ext else "")))
    H, W, C = first.shape
    imgs  = np.empty((len(frames), H, W, C), dtype=np.float32)
    poses = np.empty((len(frames), 4, 4),    dtype=np.float32)

    imgs[0]  = first.astype(np.float32) / 255.0
    poses[0] = np.array(frames[0]["transform_matrix"])
    del first

    for i, frame in enumerate(frames[1:], start=1):
        frame_name = os.path.join(root, frame["file_path"] + (".png" if include_file_ext else ""))
        imgs[i]  = imageio.imread(frame_name).astype(np.float32) / 255.0
        poses[i] = np.array(frame["transform_matrix"])

    camera_angle_x = float(data_dict["camera_angle_x"])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    render_poses = []
    for theta in np.linspace(-180, 180, n_camera_poses + 1, endpoint=False):
        c2w = trans_t(radius)
        c2w = rot_phi(phi/180. * np.pi) @ c2w
        c2w = rot_theta(theta/180. * np.pi) @ c2w
        c2w = to_nerf(c2w)
        render_poses.append(c2w)
    render_poses = torch.stack(render_poses, dim=0)

    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.
        resized = np.empty((len(frames), H, W, C), dtype=np.float32)
        for i in range(len(frames)):
            t = torch.from_numpy(imgs[i]).permute(2, 0, 1).unsqueeze(0)
            t = F.interpolate(t, size=(H, W), mode="area")
            resized[i] = t.squeeze(0).permute(1, 2, 0).numpy()
        imgs = resized

    if use_cache:
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, f"{os.path.basename(root)}_{split}.pkl")
            with open(cache_file, "wb") as fp:
                pickle.dump({
                    "imgs": imgs,
                    "poses": poses,
                    "render_poses": render_poses,
                    "H": H,
                    "W": W,
                    "focal": focal,
                }, fp)

    return imgs, poses, render_poses, [H, W, focal]
