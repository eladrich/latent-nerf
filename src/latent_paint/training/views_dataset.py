import numpy as np
import torch
from torch.utils.data import DataLoader

from src.latent_paint.configs.train_config import RenderConfig
from src.utils import get_view_direction


def rand_poses(size, device, radius_range=(1.0, 1.5), theta_range=(0.0, 150.0), phi_range=(0.0, 360.0),
               angle_overhead=30.0, angle_front=60.0):
    theta_range = np.deg2rad(theta_range)
    phi_range = np.deg2rad(phi_range)
    angle_overhead = np.deg2rad(angle_overhead)
    angle_front = np.deg2rad(angle_front)

    radius = torch.rand(size, device=device) * (radius_range[1] - radius_range[0]) + radius_range[0]
    thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
    phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]

    dirs = get_view_direction(thetas, phis, angle_overhead, angle_front)

    return dirs, thetas.item(), phis.item(), radius.item()


def circle_poses(device, radius=1.25, theta=60.0, phi=0.0, angle_overhead=30.0, angle_front=60.0):
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)
    angle_overhead = np.deg2rad(angle_overhead)
    angle_front = np.deg2rad(angle_front)

    thetas = torch.FloatTensor([theta]).to(device)
    phis = torch.FloatTensor([phi]).to(device)
    dirs = get_view_direction(thetas, phis, angle_overhead, angle_front)

    return dirs, thetas.item(), phis.item(), radius


class ViewsDataset:
    def __init__(self, cfg: RenderConfig, device, type='train', size=100):
        super().__init__()

        self.cfg = cfg
        self.device = device
        self.type = type  # train, val, test
        self.size = size

        self.training = self.type in ['train', 'all']

    def collate(self, index):

        B = len(index)  # always 1
        if self.training:
            # random pose on the fly
            dirs, thetas, phis, radius = rand_poses(B, self.device, radius_range=self.cfg.radius_range,
                                                    angle_overhead=self.cfg.angle_overhead,
                                                    angle_front=self.cfg.angle_front)

        else:
            # circle pose
            phi = (index[0] / self.size) * 360
            dirs, thetas, phis, radius = circle_poses(self.device, radius=self.cfg.radius_range[1] * 1.2, theta=60,
                                                      phi=phi,
                                                      angle_overhead=self.cfg.angle_overhead,
                                                      angle_front=self.cfg.angle_front)


        data = {
            'dir': dirs,
            'theta': thetas,
            'phi': phis,
            'radius': radius
        }

        return data

    def dataloader(self):
        loader = DataLoader(list(range(self.size)), batch_size=1, collate_fn=self.collate, shuffle=self.training,
                            num_workers=0)
        loader._data = self  # an ugly fix... we need to access dataset in trainer.
        return loader
