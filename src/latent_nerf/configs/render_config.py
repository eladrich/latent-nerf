from dataclasses import dataclass
from typing import Tuple

from src.latent_nerf.models.nerf_utils import NeRFType


@dataclass
class RenderConfig:
    """ Parameters for the NeRF Renderer """
    # Whether to use CUDA raymarching
    cuda_ray: bool = True
    # Maximal number of steps sampled per ray with cuda_ray
    max_steps: int = 1024
    # Number of steps sampled when rendering without cuda_ray
    num_steps: int = 128
    # Number of upsampled steps per ray
    upsample_steps: int = 0
    # Iterations between updates of extra status
    update_extra_interval: int = 16
    # batch size of rays at inference
    max_ray_batch: int = 4096
    # if positive, use a background model at sphere(bg_radius)
    bg_radius: float = 1.4
    # threshold for density grid to be occupied
    density_thresh: float = 10
    # Render width for training
    train_w: int = 64
    # Render height for training
    train_h: int = 64
    # Render width for inference
    eval_w: int = 128
    # Render height for inference
    eval_h: int = 128
    # Whether to randomly jitter sampled camera
    jitter_pose: bool = False
    # Assume the scene is bounded in box(-bound,bount)
    bound: float = 1
    # dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)
    dt_gamma: float = 0
    # minimum near distance for camera
    min_near: float = 0.1
    # training camera radius range
    radius_range: Tuple[float, float] = (1.0, 1.5)
    # training camera fovy range
    fovy_range: Tuple[float, float] = (40, 70)
    # Set [0,angle_overhead] as the overhead region
    angle_overhead: float = 30
    # Define the front angle region
    angle_front: float = 70
    # Which NeRF backbone to use
    backbone: str = 'grid'
    # Define the nerf output type
    nerf_type: NeRFType = NeRFType['latent']


