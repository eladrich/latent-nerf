import os

import kaolin as kal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from PIL import Image

from .mesh import Mesh
from .render import Renderer
from src.latent_paint.configs.train_config import TrainConfig


class TexturedMeshModel(nn.Module):
    def __init__(self,
                 opt: TrainConfig,
                 render_grid_size=64,
                 latent_mode=True,
                 texture_resolution=128,
                 device=torch.device('cpu')):

        super().__init__()
        self.device = device
        self.opt = opt
        self.latent_mode = latent_mode
        self.dy = self.opt.guide.dy
        self.mesh_scale = self.opt.guide.shape_scale
        self.texture_resolution = texture_resolution

        # linear rgb estimator from latents
        # https://discuss.huggingface.co/t/decoding-latents-to-rgb-without-upscaling/23204
        self.linear_rgb_estimator = torch.tensor([
            #   R       G       B
            [0.298, 0.207, 0.208],  # L1
            [0.187, 0.286, 0.173],  # L2
            [-0.158, 0.189, 0.264],  # L3
            [-0.184, -0.271, -0.473],  # L4
        ]).to(self.device)

        self.renderer = Renderer(device=self.device, dim=(render_grid_size, render_grid_size),
                                 interpolation_mode=self.opt.guide.texture_interpolation_mode)
        self.env_sphere, self.mesh = self.init_meshes()
        self.background_sphere_colors, self.texture_img, self.texture_img_rgb_finetune = self.init_paint()
        self.vt, self.ft = self.init_texture_map()

        self.face_attributes = kal.ops.mesh.index_vertices_by_faces(
            self.vt.unsqueeze(0),
            self.ft.long()).detach()

    def init_meshes(self, env_sphere_path='shapes/env_sphere.obj'):
        env_sphere = Mesh(env_sphere_path, self.device)

        mesh = Mesh(self.opt.guide.shape_path, self.device)
        mesh.normalize_mesh(inplace=True, target_scale=self.mesh_scale, dy=self.dy)

        return env_sphere, mesh

    def init_paint(self, init_rgb_color=(1.0, 0.0, 0.0)):
        # random color face attributes for background sphere
        background_sphere_colors = nn.Parameter(torch.rand(1, self.env_sphere.faces.shape[0], 3, 4).cuda())

        # inverse linear approx to find latent
        A = self.linear_rgb_estimator.T
        regularizer = 1e-2
        init_color_in_latent = (torch.pinverse(A.T @ A + regularizer * torch.eye(4).cuda()) @ A.T) @ torch.tensor(
            list(init_rgb_color)).float().to(A.device)

        # init colors with target latent plus some noise
        texture_img = nn.Parameter(
            init_color_in_latent[None, :, None, None] * 0.3 + 0.4 * torch.randn(1, 4, self.texture_resolution,
                                                                                self.texture_resolution).cuda())

        # used only for latent-paint fine-tuning, values set when reading previous checkpoint statedict
        texture_img_rgb_finetune = nn.Parameter(torch.zeros(1, 3,
                                                            self.texture_resolution, self.texture_resolution).cuda())

        return background_sphere_colors, texture_img, texture_img_rgb_finetune

    def init_texture_map(self):
        cache_path = self.opt.log.exp_dir
        vt_cache, ft_cache = cache_path / 'vt.pth', cache_path / 'ft.pth'
        if self.mesh.vt is not None and self.mesh.ft is not None \
                and self.mesh.vt.shape[0] > 0 and self.mesh.ft.min() > -1:
            vt = self.mesh.vt.cuda()
            ft = self.mesh.ft.cuda()
        elif vt_cache.exists() and ft_cache.exists():
            vt = torch.load(vt_cache).cuda()
            ft = torch.load(ft_cache).cuda()
        else:
            logger.info('running xatlas to unwrap UVs for mesh: v={v_np.shape} f={f_np.shape}')
            # unwrap uvs
            import xatlas
            v_np = self.mesh.vertices.cpu().numpy()
            f_np = self.mesh.faces.int().cpu().numpy()
            atlas = xatlas.Atlas()
            atlas.add_mesh(v_np, f_np)
            chart_options = xatlas.ChartOptions()
            chart_options.max_iterations = 4
            atlas.generate(chart_options=chart_options)
            vmapping, ft_np, vt_np = atlas[0]  # [N], [M, 3], [N, 2]

            vt = torch.from_numpy(vt_np.astype(np.float32)).float().cuda()
            ft = torch.from_numpy(ft_np.astype(np.int64)).int().cuda()
            os.makedirs(cache_path, exist_ok=True)
            torch.save(vt.cpu(), vt_cache)
            torch.save(ft.cpu(), ft_cache)
        return vt, ft

    def forward(self, x):
        raise NotImplementedError

    def get_params(self):
        if self.latent_mode:
            return [self.background_sphere_colors, self.texture_img]
        else:
            return [self.background_sphere_colors, self.texture_img_rgb_finetune]

    @torch.no_grad()
    def export_mesh(self, path, guidance=None):
        v, f = self.mesh.vertices, self.mesh.faces.int()
        h0, w0 = 256, 256
        ssaa, name = 1, ''

        # v, f: torch Tensor
        v_np = v.cpu().numpy()  # [N, 3]
        f_np = f.cpu().numpy()  # [M, 3]

        if self.latent_mode:
            colors = guidance.decode_latents(self.texture_img).permute(0, 2, 3, 1).contiguous()
        else:
            colors = self.texture_img_rgb_finetune.permute(0, 2, 3, 1).contiguous()

        colors = colors[0].cpu().detach().numpy()
        colors = (colors * 255).astype(np.uint8)

        vt_np = self.vt.detach().cpu().numpy()
        ft_np = self.ft.detach().cpu().numpy()

        colors = Image.fromarray(colors)

        if ssaa > 1:
            colors = colors.resize((w0, h0), Image.LINEAR)

        colors.save(os.path.join(path, f'{name}albedo.png'))

        # save obj (v, vt, f /)
        obj_file = os.path.join(path, f'{name}mesh.obj')
        mtl_file = os.path.join(path, f'{name}mesh.mtl')

        logger.info('writing obj mesh to {obj_file}')
        with open(obj_file, "w") as fp:
            fp.write(f'mtllib {name}mesh.mtl \n')

            logger.info('writing vertices {v_np.shape}')
            for v in v_np:
                fp.write(f'v {v[0]} {v[1]} {v[2]} \n')

            logger.info('writing vertices texture coords {vt_np.shape}')
            for v in vt_np:
                # fp.write(f'vt {v[0]} {1 - v[1]} \n')
                fp.write(f'vt {v[0]} {v[1]} \n')

            logger.info('writing faces {f_np.shape}')
            fp.write(f'usemtl mat0 \n')
            for i in range(len(f_np)):
                fp.write(
                    f"f {f_np[i, 0] + 1}/{ft_np[i, 0] + 1} {f_np[i, 1] + 1}/{ft_np[i, 1] + 1} {f_np[i, 2] + 1}/{ft_np[i, 2] + 1} \n")

        with open(mtl_file, "w") as fp:
            fp.write(f'newmtl mat0 \n')
            fp.write(f'Ka 1.000000 1.000000 1.000000 \n')
            fp.write(f'Kd 1.000000 1.000000 1.000000 \n')
            fp.write(f'Ks 0.000000 0.000000 0.000000 \n')
            fp.write(f'Tr 1.000000 \n')
            fp.write(f'illum 1 \n')
            fp.write(f'Ns 0.000000 \n')
            fp.write(f'map_Kd {name}albedo.png \n')

    def render(self, theta, phi, radius, decode_func=None, test=False, dims=None):
        if test:
            return self.render_test(theta, phi, radius, decode_func, dims=dims)
        else:
            return self.render_train(theta, phi, radius)

    def render_train(self, theta, phi, radius):
        if self.latent_mode:
            texture_img = self.texture_img
            background_sphere_colors = self.background_sphere_colors
        else:
            texture_img = self.texture_img_rgb_finetune
            background_sphere_colors = self.background_sphere_colors @ self.linear_rgb_estimator

        pred_features, mask = self.renderer.render_single_view_texture(self.mesh.vertices,
                                                                       self.mesh.faces,
                                                                       self.face_attributes,
                                                                       texture_img,
                                                                       elev=theta,
                                                                       azim=phi,
                                                                       radius=radius,
                                                                       look_at_height=self.dy)

        pred_back, _ = self.renderer.render_single_view(self.env_sphere,
                                                        background_sphere_colors,
                                                        elev=theta,
                                                        azim=phi,
                                                        radius=radius,
                                                        look_at_height=self.dy)

        mask = mask.detach()
        pred_map = pred_back * (1 - mask) + pred_features * mask

        if self.latent_mode and mask.shape[-1] != 64:
            mask = F.interpolate(mask, (64, 64), mode='bicubic')
            pred_back = F.interpolate(pred_back, (64, 64), mode='bicubic')
            pred_features = F.interpolate(pred_features, (64, 64), mode='bicubic')
            pred_map = F.interpolate(pred_map, (64, 64), mode='bicubic')

        return {'image': pred_map, 'mask': mask, 'background': pred_back, 'foreground': pred_features}

    def render_test(self, theta, phi, radius, decode_func=None, dims=None):
        if self.latent_mode:
            assert decode_func is not None, 'decode function was not supplied to decode the latent texture image'
            texture_img = decode_func(self.texture_img)
        else:
            texture_img = self.texture_img_rgb_finetune

        pred_features, mask = self.renderer.render_single_view_texture(self.mesh.vertices,
                                                                       self.mesh.faces,
                                                                       self.face_attributes,
                                                                       texture_img,
                                                                       elev=theta,
                                                                       azim=phi,
                                                                       radius=radius,
                                                                       look_at_height=self.dy,
                                                                       dims=dims,
                                                                       white_background=True)

        return {'image': pred_features, 'texture_map': texture_img, 'mask': mask}
