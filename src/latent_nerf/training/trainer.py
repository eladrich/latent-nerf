import sys
from pathlib import Path
from typing import Tuple, Any, Dict, Callable, Union, List

import imageio
import numpy as np
import pyrallis
import torch
from PIL import Image
from loguru import logger
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import utils
from src.latent_nerf.configs.train_config import TrainConfig
from src.latent_nerf.models.renderer import NeRFRenderer
from src.latent_nerf.training.nerf_dataset import NeRFDataset
from src.stable_diffusion import StableDiffusion
from src.utils import make_path, tensor2numpy


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.train_step = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        utils.seed_everything(self.cfg.optim.seed)

        # Make dirs
        self.exp_path = make_path(self.cfg.log.exp_dir)
        self.ckpt_path = make_path(self.exp_path / 'checkpoints')
        self.train_renders_path = make_path(self.exp_path / 'vis' / 'train')
        self.eval_renders_path = make_path(self.exp_path / 'vis' / 'eval')
        self.final_renders_path = make_path(self.exp_path / 'results')

        self.init_logger()
        pyrallis.dump(self.cfg, (self.exp_path / 'config.yaml').open('w'))

        self.nerf = self.init_nerf()
        self.diffusion = self.init_diffusion()
        self.text_z = self.calc_text_embeddings()
        self.losses = self.init_losses()
        self.optimizer, self.scaler = self.init_optimizer()
        self.dataloaders = self.init_dataloaders()

        self.past_checkpoints = []
        if self.cfg.optim.resume:
            self.load_checkpoint(model_only=False)
        if self.cfg.optim.ckpt is not None:
            self.load_checkpoint(self.cfg.optim.ckpt, model_only=True)

        logger.info(f'Successfully initialized {self.cfg.log.exp_name}')

    def init_nerf(self) -> NeRFRenderer:
        if self.cfg.render.backbone == 'grid':
            from src.latent_nerf.models.network_grid import NeRFNetwork
        else:
            raise ValueError(f'{self.cfg.render.backbone} is not a valid backbone name')

        model = NeRFNetwork(self.cfg.render).to(self.device)
        logger.info(
            f'Loaded {self.cfg.render.backbone} NeRF, #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')
        logger.info(model)
        return model

    def init_diffusion(self) -> StableDiffusion:
        diffusion_model = StableDiffusion(self.device, model_name=self.cfg.guide.diffusion_name,
                                          concept_name=self.cfg.guide.concept_name,
                                          latent_mode=self.nerf.latent_mode)
        for p in diffusion_model.parameters():
            p.requires_grad = False
        return diffusion_model

    def calc_text_embeddings(self) -> Union[torch.Tensor, List[torch.Tensor]]:
        ref_text = self.cfg.guide.text
        if not self.cfg.guide.append_direction:
            text_z = self.diffusion.get_text_embeds([ref_text])
        else:
            text_z = []
            for d in ['front', 'side', 'back', 'side', 'overhead', 'bottom']:
                text = f"{ref_text}, {d} view"
                text_z.append(self.diffusion.get_text_embeds([text]))
        return text_z

    def init_optimizer(self) -> Tuple[Optimizer, Any]:
        optimizer = torch.optim.Adam(self.nerf.get_params(self.cfg.optim.lr), betas=(0.9, 0.99), eps=1e-15)
        scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.optim.fp16)
        return optimizer, scaler

    def init_dataloaders(self) -> Dict[str, DataLoader]:
        train_dataloader = NeRFDataset(self.cfg.render, device=self.device, type='train', H=self.cfg.render.train_h,
                                       W=self.cfg.render.train_w, size=100).dataloader()
        val_loader = NeRFDataset(self.cfg.render, device=self.device, type='val', H=self.cfg.render.eval_h,
                                 W=self.cfg.render.eval_w,
                                 size=self.cfg.log.eval_size).dataloader()
        # Will be used for creating the final video
        val_large_loader = NeRFDataset(self.cfg.render, device=self.device, type='val', H=self.cfg.render.eval_h,
                                       W=self.cfg.render.eval_w,
                                       size=self.cfg.log.full_eval_size).dataloader()
        dataloaders = {'train': train_dataloader, 'val': val_loader, 'val_large': val_large_loader}
        return dataloaders

    def init_losses(self) -> Dict[str, Callable]:
        losses = {}
        if self.cfg.optim.lambda_shape > 0 and self.cfg.guide.shape_path is not None:
            from src.latent_nerf.training.losses.shape_loss import ShapeLoss
            losses['shape_loss'] = ShapeLoss(self.cfg.guide)
        if self.cfg.optim.lambda_sparsity > 0:
            from src.latent_nerf.training.losses.sparsity_loss import sparsity_loss
            losses['sparsity_loss'] = sparsity_loss
        return losses

    def init_logger(self):
        logger.remove()  # Remove default logger
        log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{message}</level>"
        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, format=log_format)
        logger.add(self.exp_path / 'log.txt', colorize=False, format=log_format)

    def train(self):
        logger.info('Starting training ^_^')
        # Evaluate the initialization
        self.evaluate(self.dataloaders['val'], self.eval_renders_path)
        self.nerf.train()

        pbar = tqdm(total=self.cfg.optim.iters, initial=self.train_step,
                    bar_format='{desc}: {percentage:3.0f}% training step {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        while self.train_step < self.cfg.optim.iters:
            # Keep going over dataloader until finished the required number of iterations
            for data in self.dataloaders['train']:
                if self.nerf.cuda_ray and self.train_step % self.cfg.render.update_extra_interval == 0:
                    with torch.cuda.amp.autocast(enabled=self.cfg.optim.fp16):
                        self.nerf.update_extra_state()

                self.train_step += 1
                pbar.update(1)

                self.optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=self.cfg.optim.fp16):
                    pred_rgbs, pred_ws, loss = self.train_render(data)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                if self.train_step % self.cfg.log.save_interval == 0:
                    self.save_checkpoint(full=True)
                    self.evaluate(self.dataloaders['val'], self.eval_renders_path)
                    self.nerf.train()

                if np.random.uniform(0, 1) < 0.05:
                    # Randomly log rendered images throughout the training
                    self.log_train_renders(pred_rgbs)
        logger.info('Finished Training ^_^')
        logger.info('Evaluating the last model...')
        self.full_eval()
        logger.info('\tDone!')

    def evaluate(self, dataloader: DataLoader, save_path: Path, save_as_video: bool = False):
        logger.info(f'Evaluating and saving model, iteration #{self.train_step}...')
        self.nerf.eval()
        save_path.mkdir(exist_ok=True)

        if save_as_video:
            all_preds = []
            all_preds_normals = []
            all_preds_depth = []

        for i, data in enumerate(dataloader):
            with torch.cuda.amp.autocast(enabled=self.cfg.optim.fp16):
                preds, preds_depth, preds_normals = self.eval_render(data)

            pred, pred_depth, pred_normals = tensor2numpy(preds[0]), tensor2numpy(preds_depth[0]), tensor2numpy(
                preds_normals[0])

            if save_as_video:
                all_preds.append(pred)
                all_preds_normals.append(pred_normals)
                all_preds_depth.append(pred_depth)
            else:
                if not self.cfg.log.skip_rgb:
                    Image.fromarray(pred).save(save_path / f"{self.train_step}_{i:04d}_rgb.png")
                Image.fromarray(pred_normals).save(save_path / f"{self.train_step}_{i:04d}_normals.png")
                Image.fromarray(pred_depth).save(save_path / f"{self.train_step}_{i:04d}_depth.png")

        if save_as_video:
            all_preds = np.stack(all_preds, axis=0)
            all_preds_normals = np.stack(all_preds_normals, axis=0)
            all_preds_depth = np.stack(all_preds_depth, axis=0)

            dump_vid = lambda video, name: imageio.mimsave(save_path / f"{self.train_step}_{name}.mp4", video, fps=25,
                                                           quality=8, macro_block_size=1)

            if not self.cfg.log.skip_rgb:
                dump_vid(all_preds, 'rgb')
            dump_vid(all_preds_normals, 'normals')
            dump_vid(all_preds_depth, 'depth')
        logger.info('Done!')

    def full_eval(self):
        self.evaluate(self.dataloaders['val_large'], self.final_renders_path, save_as_video=True)

    def train_render(self, data: Dict[str, Any]):
        rays_o, rays_d = data['rays_o'], data['rays_d']  # [B, N, 3]

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        if self.cfg.optim.start_shading_iter is None or self.train_step < self.cfg.optim.start_shading_iter:
            shading = 'albedo'
            ambient_ratio = 1.0
        else:
            shading = 'lambertian'
            ambient_ratio = 0.1

        bg_color = torch.rand((B * N, 3), device=rays_o.device)  # Will be used if bg_radius <= 0
        outputs = self.nerf.render(rays_o, rays_d, staged=False, perturb=True, bg_color=bg_color,
                                   ambient_ratio=ambient_ratio, shading=shading, force_all_rays=True)
        pred_rgb = outputs['image'].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        pred_ws = outputs['weights_sum'].reshape(B, 1, H, W)

        # text embeddings
        if self.cfg.guide.append_direction:
            dirs = data['dir']  # [B,]
            text_z = self.text_z[dirs]
        else:
            text_z = self.text_z

        # Guidance loss
        loss_guidance = self.diffusion.train_step(text_z, pred_rgb)
        loss = loss_guidance

        # Sparsity loss
        if 'sparsity_loss' in self.losses:
            loss += self.cfg.optim.lambda_sparsity * self.losses['sparsity_loss'](pred_ws)

        # Shape loss
        if 'shape_loss' in self.losses:
            loss += self.cfg.optim.lambda_shape * self.losses['shape_loss'](outputs['xyzs'], outputs['sigmas'])

        return pred_rgb, pred_ws, loss

    def eval_render(self, data, bg_color=None, perturb=False):
        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        if bg_color is not None:
            bg_color = bg_color.to(rays_o.device)
        else:
            bg_color = torch.ones(3, device=rays_o.device)  # [3]

        shading = data['shading'] if 'shading' in data else 'albedo'
        ambient_ratio = data['ambient_ratio'] if 'ambient_ratio' in data else 1.0
        light_d = data['light_d'] if 'light_d' in data else None

        outputs = self.nerf.render(rays_o, rays_d, staged=True, perturb=perturb, light_d=light_d,
                                   ambient_ratio=ambient_ratio, shading=shading, force_all_rays=True, bg_color=bg_color)

        pred_depth = outputs['depth'].reshape(B, H, W)
        if self.nerf.latent_mode:
            pred_latent = outputs['image'].reshape(B, H, W, 3 + 1).permute(0, 3, 1, 2).contiguous()
            if self.cfg.log.skip_rgb:
                # When rendering in a size that is too large for decoding
                pred_rgb = torch.zeros(B, H, W, 3, device=pred_latent.device)
            else:
                pred_rgb = self.diffusion.decode_latents(pred_latent).permute(0, 2, 3, 1).contiguous()
        else:
            pred_rgb = outputs['image'].reshape(B, H, W, 3).contiguous().clamp(0, 1)

        pred_depth = pred_depth.unsqueeze(-1).repeat(1, 1, 1, 3)

        # Render again for normals
        shading = 'normal'
        outputs_normals = self.nerf.render(rays_o, rays_d, staged=True, perturb=perturb, light_d=light_d,
                                           ambient_ratio=ambient_ratio, shading=shading, force_all_rays=True,
                                           disable_background=True)
        pred_normals = outputs_normals['image'][:, :, :3].reshape(B, H, W, 3).contiguous()

        return pred_rgb, pred_depth, pred_normals

    def log_train_renders(self, pred_rgbs: torch.Tensor):
        if self.nerf.latent_mode:
            pred_rgb_vis = self.diffusion.decode_latents(pred_rgbs).permute(0, 2, 3,
                                                                            1).contiguous()  # [1, 3, H, W]
        else:
            pred_rgb_vis = pred_rgbs.permute(0, 2, 3,
                                             1).contiguous().clamp(0, 1)  #
        save_path = self.train_renders_path / f'step_{self.train_step:05d}.jpg'
        save_path.parent.mkdir(exist_ok=True)

        pred = tensor2numpy(pred_rgb_vis[0])

        Image.fromarray(pred).save(save_path)

    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(self.ckpt_path.glob('*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                logger.info(f"Latest checkpoint is {checkpoint}")
            else:
                logger.info("No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        if 'model' not in checkpoint_dict:
            self.nerf.load_state_dict(checkpoint_dict)
            logger.info("loaded model.")
            return

        missing_keys, unexpected_keys = self.nerf.load_state_dict(checkpoint_dict['model'], strict=False)
        logger.info("loaded model.")
        if len(missing_keys) > 0:
            logger.warning(f"missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            logger.warning(f"unexpected keys: {unexpected_keys}")

        if self.cfg.render.cuda_ray:
            if 'mean_count' in checkpoint_dict:
                self.nerf.mean_count = checkpoint_dict['mean_count']
            if 'mean_density' in checkpoint_dict:
                self.nerf.mean_density = checkpoint_dict['mean_density']

        if model_only:
            return

        self.past_checkpoints = checkpoint_dict['checkpoints']
        self.train_step = checkpoint_dict['train_step'] + 1
        logger.info(f"load at step {self.train_step}")

        if self.optimizer and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                logger.info("loaded optimizer.")
            except:
                logger.warning("Failed to load optimizer.")

        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                logger.info("loaded scaler.")
            except:
                logger.warning("Failed to load scaler.")

    def save_checkpoint(self, full=False):

        name = f'step_{self.train_step:06d}'

        state = {
            'train_step': self.train_step,
            'checkpoints': self.past_checkpoints,
        }

        if self.nerf.cuda_ray:
            state['mean_count'] = self.nerf.mean_count
            state['mean_density'] = self.nerf.mean_density

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['scaler'] = self.scaler.state_dict()

        state['model'] = self.nerf.state_dict()

        file_path = f"{name}.pth"

        self.past_checkpoints.append(file_path)

        if len(self.past_checkpoints) > self.cfg.log.max_keep_ckpts:
            old_ckpt = self.ckpt_path / self.past_checkpoints.pop(0)
            old_ckpt.unlink(missing_ok=True)

        torch.save(state, self.ckpt_path / file_path)
