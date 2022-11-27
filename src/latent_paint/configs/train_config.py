from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple
from loguru import logger


@dataclass
class RenderConfig:
    """ Parameters for the Mesh Renderer """
    # Render width for training
    train_grid_size: int = 64
    # Render height for training
    eval_grid_size: int = 512
    # training camera radius range
    radius_range: Tuple[float, float] = (1.0, 1.5)
    # Set [0,angle_overhead] as the overhead region
    angle_overhead: float = 30
    # Define the front angle region
    angle_front: float = 70
    # Which NeRF backbone to use
    backbone: str = 'texture-mesh'

@dataclass
class GuideConfig:
    """ Parameters defining the guidance """
    # Guiding text prompt
    text: str
    # The mesh to paint
    shape_path: str
    # Append direction to text prompts
    append_direction: bool = True
    # A Textual-Inversion concept to use
    concept_name: Optional[str] = None
    # A huggingface diffusion model to use
    diffusion_name: str = 'CompVis/stable-diffusion-v1-4'
    # Scale of mesh in 1x1x1 cube
    shape_scale: float = 0.6
    # height of mesh
    dy: float = 0.25
    # texture image resolution
    texture_resolution=128
    # texture mapping interpolation mode from texture image, options: 'nearest', 'bilinear', 'bicubic'
    texture_interpolation_mode: str= 'nearest'


@dataclass
class OptimConfig:
    """ Parameters for the optimization process """
    # Seed for experiment
    seed: int = 0
    # Total iters
    iters: int = 5000
    # Learning rate
    lr: float = 1e-2
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
    # Export a mesh
    save_mesh: bool = True
    # Number of past checkpoints to keep
    max_keep_ckpts: int = 2

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
            logger.warning('NOTICE! log.eval_only=True, but no checkpoint was chosen -> Manually setting optim.resume to True')
            self.optim.resume = True

