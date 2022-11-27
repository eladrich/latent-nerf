from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger

from .render_config import RenderConfig


@dataclass
class GuideConfig:
    """ Parameters defining the guidance """
    # Guiding text prompt
    text: str
    # Append direction to text prompts
    append_direction: bool = True
    # A Textual-Inversion concept to use
    concept_name: Optional[str] = None
    # A huggingface diffusion model to use
    diffusion_name: str = 'CompVis/stable-diffusion-v1-4'
    # Guiding mesh
    shape_path: Optional[str] = None
    # Scale of mesh in 1x1x1 cube
    mesh_scale: float = 0.7
    # Define the proximal distance allowed
    proximal_surface: float = 0.3


@dataclass
class OptimConfig:
    """ Parameters for the optimization process """
    # Loss scale for alpha entropy
    lambda_sparsity: float = 5e-4
    # Loss scale for mesh-guidance
    lambda_shape: float = 5e-6
    # Seed for experiment
    seed: int = 0
    # Total iters
    iters: int = 5000
    # Learning rate
    lr: float = 1e-3
    # use amp mixed precision training
    fp16: bool = True
    # Start shading at this iteration
    start_shading_iter: Optional[int] = None
    # Resume from checkpoint
    resume: bool = False
    # Load existing model
    ckpt: Optional[str] = None


@dataclass
class LogConfig:
    """ Parameters for logging and saving """
    # Experiment name
    exp_name: str
    # Experiment output dir
    exp_root: Path = Path('experiments/')
    # How many steps between save step
    save_interval: int = 100
    # Run only test
    eval_only: bool = False
    # Number of angles to sample for eval during training
    eval_size: int = 10
    # Number of angles to sample for eval after training
    full_eval_size: int = 100
    # Number of past checkpoints to keep
    max_keep_ckpts: int = 2
    # Skip decoding and vis only depth and normals
    skip_rgb: bool = False

    @property
    def exp_dir(self) -> Path:
        return self.exp_root / self.exp_name


@dataclass
class TrainConfig:
    """ The main configuration for the coach trainer """
    log: LogConfig = field(default_factory=LogConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    guide: GuideConfig = field(default_factory=GuideConfig)

    def __post_init__(self):
        if self.log.eval_only and (self.optim.ckpt is None and not self.optim.resume):
            logger.warning(
                'NOTICE! log.eval_only=True, but no checkpoint was chosen -> Manually setting optim.resume to True')
            self.optim.resume = True
