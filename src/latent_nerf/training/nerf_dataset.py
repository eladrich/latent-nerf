import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.latent_nerf.configs.render_config import RenderConfig
from src.latent_nerf.models.render_utils import get_rays, safe_normalize
from src.utils import get_view_direction


def visualize_poses(poses, size=0.1):
    # poses: [B, 4, 4]
    import trimesh
    axes = trimesh.creation.axis(axis_length=4)
    sphere = trimesh.creation.icosphere(radius=1)
    objects = [axes, sphere]

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show()


def rand_poses(size, device, radius_range=(1, 1.5), theta_range=(0, 150), phi_range=(0, 360),
               angle_overhead=30, angle_front=60, jitter=False):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, pi]
        phi_range: [min, max], should be in [0, 2 * pi]
    Return:
        poses: [size, 4, 4]
    '''

    theta_range = np.deg2rad(theta_range)
    phi_range = np.deg2rad(phi_range)
    angle_overhead = np.deg2rad(angle_overhead)
    angle_front = np.deg2rad(angle_front)

    radius = torch.rand(size, device=device) * (radius_range[1] - radius_range[0]) + radius_range[0]
    thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
    phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1)  # [B, 3]

    targets = 0

    # jitters
    if jitter:
        centers = centers + (torch.rand_like(centers) * 0.2 - 0.1)
        targets = targets + torch.randn_like(centers) * 0.2

    # lookat
    forward_vector = safe_normalize(targets - centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0).repeat(size, 1)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))

    if jitter:
        up_noise = torch.randn_like(up_vector) * 0.02
    else:
        up_noise = 0

    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1) + up_noise)

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    dirs = get_view_direction(thetas, phis, angle_overhead, angle_front)

    return poses, dirs


def circle_poses(device, radius=1.25, theta=60, phi=0, return_dirs=False, angle_overhead=30, angle_front=60):
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)
    angle_overhead = np.deg2rad(angle_overhead)
    angle_front = np.deg2rad(angle_front)

    thetas = torch.FloatTensor([theta]).to(device)
    phis = torch.FloatTensor([phi]).to(device)

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1)  # [B, 3]

    # lookat
    forward_vector = - safe_normalize(centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    if return_dirs:
        dirs = get_view_direction(thetas, phis, angle_overhead, angle_front)
    else:
        dirs = None

    return poses, dirs


class NeRFDataset:
    def __init__(self, cfg: RenderConfig, device, type='train', H=256, W=256, size=100):
        super().__init__()

        self.cfg = cfg
        self.device = device
        self.type = type  # train, val, test

        self.H = H
        self.W = W
        self.radius_range = cfg.radius_range
        self.fovy_range = cfg.fovy_range
        self.size = size

        self.training = self.type in ['train', 'all']

        self.cx = self.H / 2
        self.cy = self.W / 2

        # [debug] visualize poses
        # poses, dirs = rand_poses(100, self.device, return_dirs=self.cfg.dir_text, radius_range=self.radius_range)
        # visualize_poses(poses.detach().cpu().numpy())

    def collate(self, index):

        B = len(index)  # always 1
        fixed_viewpoint = False
        if self.training:
            # random pose on the fly
            poses, dirs = rand_poses(B, self.device, radius_range=self.radius_range,
                                     angle_overhead=self.cfg.angle_overhead, angle_front=self.cfg.angle_front,
                                     jitter=self.cfg.jitter_pose)

                # random focal
            fov = random.random() * (self.fovy_range[1] - self.fovy_range[0]) + self.fovy_range[0]
            focal = self.H / (2 * np.tan(np.deg2rad(fov) / 2))
            intrinsics = np.array([focal, focal, self.cx, self.cy])
        else:
            # circle pose
            phi = (index[0] / self.size) * 360
            theta = 80
            poses, dirs = circle_poses(self.device, radius=self.radius_range[1] * 1.2, theta=theta, phi=phi,
                 angle_overhead=self.cfg.angle_overhead,
                                       angle_front=self.cfg.angle_front)

            # fixed focal
            fov = (self.fovy_range[1] + self.fovy_range[0]) / 2
            focal = self.H / (2 * np.tan(np.deg2rad(fov) / 2))
            intrinsics = np.array([focal, focal, self.cx, self.cy])

        # sample a low-resolution but full image for CLIP
        rays = get_rays(poses, intrinsics, self.H, self.W, -1)

        data = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'dir': dirs,
            'fixed_viewpoint': fixed_viewpoint,
        }

        return data

    def dataloader(self):
        loader = DataLoader(list(range(self.size)), batch_size=1, collate_fn=self.collate, shuffle=self.training,
                            num_workers=0)
        loader._data = self  # an ugly fix... we need to access dataset in trainer.
        return loader
