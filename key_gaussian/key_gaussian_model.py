

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig
import torch
import torchvision
from einops import rearrange
import numpy as np
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.misc import torch_compile
from gsplat.rendering import rasterization
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.model_components import renderers
from gsplat.cuda_legacy._torch_impl import quat_to_rotmat

import os
@torch_compile()
def get_viewmat(optimized_camera_to_world):
    """
    function that converts c2w to gsplat world2camera matrix, using compile for some speed
    """
    R = optimized_camera_to_world[:, :3, :3]  # 3 x 3
    T = optimized_camera_to_world[:, :3, 3:4]  # 3 x 1
    # flip the z and y axes to align with gsplat conventions
    R = R * torch.tensor([[[1, -1, -1]]], device=R.device, dtype=R.dtype)
    # analytic matrix inverse to get world2camera matrix
    R_inv = R.transpose(1, 2)
    T_inv = -torch.bmm(R_inv, T)
    viewmat = torch.zeros(R.shape[0], 4, 4, device=R.device, dtype=R.dtype)
    viewmat[:, 3, 3] = 1.0  # homogenous
    viewmat[:, :3, :3] = R_inv
    viewmat[:, :3, 3:4] = T_inv
    return viewmat
@dataclass
class KeyGaussiansModelConfig(SplatfactoModelConfig):
    """BAD-Gaussians Model config"""
    
    _target: Type = field(default_factory=lambda: KeyGaussianModel)
    
    use_abs_grad: bool = True
    use_opacity_reg : bool = False
    opacity_reg : float =0.01
    use_min_axis_reg : bool = False
    min_axis_reg : float =0.01
    
class KeyGaussianModel(SplatfactoModel):

    config: KeyGaussiansModelConfig
    def __init__(self, config: KeyGaussiansModelConfig, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        self.eval_camera_optimizer  = None
   
    def get_outputs_optim_eval_pose(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a camera and returns a dictionary of outputs.

        Args:
            camera: The camera(s) for which output images are rendered. It should have
            all the needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}

        optimized_camera_to_world = self.eval_camera_optimizer.apply_to_camera(camera)
        # cropping
        # if self.crop_box is not None and not self.training:
        #     crop_ids = self.crop_box.within(self.means).squeeze()
        #     if crop_ids.sum() == 0:
        #         return self.get_empty_outputs(
        #             int(camera.width.item()), int(camera.height.item()), self.background_color
        #         )
        # else:
        #     crop_ids = None

        # if crop_ids is not None:
        #     opacities_crop = self.opacities[crop_ids]
        #     means_crop = self.means[crop_ids]
        #     features_dc_crop = self.features_dc[crop_ids]
        #     features_rest_crop = self.features_rest[crop_ids]
        #     scales_crop = self.scales[crop_ids]
        #     quats_crop = self.quats[crop_ids]
        # else:
        opacities_crop = self.opacities
        means_crop = self.means
        features_dc_crop = self.features_dc
        features_rest_crop = self.features_rest
        scales_crop = self.scales
        quats_crop = self.quats

        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)

        BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default
        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)
        viewmat = get_viewmat(optimized_camera_to_world)
        K = camera.get_intrinsics_matrices().cuda()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
        camera.rescale_output_resolution(camera_scale_fac)  # type: ignore

        # apply the compensation of screen space blurring to gaussians
        if self.config.rasterize_mode not in ["antialiased", "classic"]:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        if self.config.output_depth_during_training or not self.training:
            render_mode = "RGB+ED"
        else:
            render_mode = "RGB"

        if self.config.sh_degree > 0:
            sh_degree_to_use = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
        else:
            colors_crop = torch.sigmoid(colors_crop).squeeze(1)  # [N, 1, 3] -> [N, 3]
            sh_degree_to_use = None

        render, alpha, info = rasterization(
            means=means_crop,
            quats=quats_crop / quats_crop.norm(dim=-1, keepdim=True),
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=colors_crop,
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
            width=W,
            height=H,
            tile_size=BLOCK_WIDTH,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sh_degree=sh_degree_to_use,
            sparse_grad=False,
            absgrad=True,
            rasterize_mode=self.config.rasterize_mode,
            # set some threshold to disregrad small gaussians for faster rendering.
            # radius_clip=3.0,
        )
        # if self.training and info["means2d"].requires_grad:
        info["means2d"].retain_grad()
        self.xys = info["means2d"]  # [1, N, 2]
        self.radii = info["radii"][0]  # [N]
        alpha = alpha[:, ...]

        background = self._get_background_color()
        rgb = render[:, ..., :3] + (1 - alpha) * background
        rgb = torch.clamp(rgb, 0.0, 1.0)

        if render_mode == "RGB+ED":
            depth_im = render[:, ..., 3:4]
            depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).squeeze(0)
        else:
            depth_im = None

        if background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)
        
        return {
            "rgb": rgb.squeeze(0),  # type: ignore
            "depth": depth_im,  # type: ignore
            "accumulation": alpha.squeeze(0),  # type: ignore
            "background": background,  # type: ignore
        }  # type: ignore
    def _get_background_color(self):
        if self.config.background_color == "random":
            if self.training:
                background = torch.rand(3, device=self.device)
            else:
                background = self.background_color.to(self.device)
        elif self.config.background_color == "white":
            background = torch.ones(3, device=self.device)
        elif self.config.background_color == "black":
            background = torch.zeros(3, device=self.device)
        else:
            raise ValueError(f"Unknown background color {self.config.background_color}")
        return background
    def after_train(self, step: int):
        assert step == self.step
        # to save some training time, we no longer need to update those stats post refinement
        if self.step >= self.config.stop_split_at:
            return
        with torch.no_grad():
            # keep track of a moving average of grad norms
            visible_mask = (self.radii > 0).flatten()
            if self.config.use_abs_grad:
                grads = self.xys.absgrad[0][visible_mask].norm(dim=-1)
            else:
                grads = self.xys.grad[0][visible_mask].norm(dim=-1)
            # print(f"grad norm min {grads.min().item()} max {grads.max().item()} mean {grads.mean().item()} size {grads.shape}")
            if self.xys_grad_norm is None:
                self.xys_grad_norm = torch.zeros(self.num_points, device=self.device, dtype=torch.float32)
                self.vis_counts = torch.ones(self.num_points, device=self.device, dtype=torch.float32)
            assert self.vis_counts is not None
            self.vis_counts[visible_mask] += 1
            self.xys_grad_norm[visible_mask] += grads
    # def split_gaussians(self, split_mask, samps):
    #     """
    #     This function splits gaussians that are too large
    #     """
    #     n_splits = split_mask.sum().item()
    #     CONSOLE.log(f"Splitting {split_mask.sum().item()/self.num_points} gaussians: {n_splits}/{self.num_points}")
    #     centered_samples = torch.randn((samps * n_splits, 3), device=self.device)  # Nx3 of axis-aligned scales
    #     scaled_samples = (
    #         torch.exp(self.scales[split_mask].repeat(samps, 1)) * centered_samples
    #     )  # how these scales are rotated
    #     quats = self.quats[split_mask] / self.quats[split_mask].norm(dim=-1, keepdim=True)  # normalize them first
    #     rots = quat_to_rotmat(quats.repeat(samps, 1))  # how these scales are rotated
    #     rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze()
    #     new_means = rotated_samples + self.means[split_mask].repeat(samps, 1)
    #     # step 2, sample new colors
    #     new_features_dc = self.features_dc[split_mask].repeat(samps, 1)
    #     new_features_rest = self.features_rest[split_mask].repeat(samps, 1, 1)
    #     # step 3, sample new opacities
    #     new_opacities = self.opacities[split_mask].repeat(samps, 1)
    #     # # revise the opacities
    #     # new_opacities = torch.logit(torch.sigmoid(new_opacities) / samps)
        
        
    #     # using our own method to sample opacities
    #     new_opacities = torch.logit((1-torch.sqrt(1-torch.sigmoid(new_opacities)))/samps)
    #     self.opacities[split_mask]=torch.logit((1-torch.sqrt(1-torch.sigmoid(self.opacities[split_mask])))/ samps).detach()
        
        
    #     # step 4, sample new scales
    #     size_fac = 1.6
    #     new_scales = torch.log(torch.exp(self.scales[split_mask]) / size_fac).repeat(samps, 1)
    #     self.scales[split_mask] = torch.log(torch.exp(self.scales[split_mask]) / size_fac)
    #     # step 5, sample new quats
    #     new_quats = self.quats[split_mask].repeat(samps, 1)
    #     out = {
    #         "means": new_means,
    #         "features_dc": new_features_dc,
    #         "features_rest": new_features_rest,
    #         "opacities": new_opacities,
    #         "scales": new_scales,
    #         "quats": new_quats,
    #     }
    #     for name, param in self.gauss_params.items():
    #         if name not in out:
    #             out[name] = param[split_mask].repeat(samps, 1)
    #     return out        
    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        pred_img = outputs["rgb"]

        # Set masked part of both ground-truth and rendered image to black.
        # This is a little bit sketchy for the SSIM loss.
        if "mask" in batch:
            # batch["mask"] : [H, W, 1]
            mask = self._downscale_if_required(batch["mask"])
            mask = mask.to(self.device)
            assert mask.shape[:2] == gt_img.shape[:2] == pred_img.shape[:2]
            gt_img = gt_img * mask
            pred_img = pred_img * mask

        Ll1 = torch.abs(gt_img - pred_img).mean()
        simloss = 1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...])
        if self.config.use_scale_regularization and self.step % 10 == 0:
            scale_exp = torch.exp(self.scales)
            scale_reg = (
                torch.maximum(
                    scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                    torch.tensor(self.config.max_gauss_ratio),
                )
                - self.config.max_gauss_ratio
            )
            scale_reg = 0.1 * scale_reg.mean()
        else:
            scale_reg = torch.tensor(0.0).to(self.device)

        if self.config.use_opacity_reg:
            opacity_reg = self.config.opacity_reg * torch.sigmoid(self.opacities).abs().mean()
        else:
            opacity_reg = torch.tensor(0.0).to(self.device)
        
        
        if self.config.use_min_axis_reg:
            axis_reg = self.config.min_axis_reg * torch.exp(self.scales.amin(dim=-1)).mean()
        else:
            axis_reg = torch.tensor(0.0).to(self.device)
        
        loss_dict = {
            "main_loss": (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss,
            "scale_reg": scale_reg,
            "opacity_reg": opacity_reg,
            "axis_reg" : axis_reg,
        }

        if self.training and ((batch["image_idx"]==70) or self.step in [0,1,2,3,4,5,6,7,8,9,10,50,100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,15000,20000,25000,29999]):
            # save pred_img
            
            os.makedirs("train_temp", exist_ok=True)
            if batch["image_idx"]==5:
                
                torchvision.utils.save_image(pred_img.permute(2, 0, 1), f"train_temp/pred_img5_{self.step}.png")
            else:
                torchvision.utils.save_image(pred_img.permute(2, 0, 1), f"train_temp/pred_img_{self.step}.png")
        
        if self.training:
            # Add loss from camera optimizer
            self.camera_optimizer.get_loss_dict(loss_dict)

        return loss_dict
    
    
    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}

        optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)[0, ...]

        # get the background color
        if self.training:
            assert camera.shape[0] == 1, "Only one camera at a time"
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)

            if self.config.background_color == "random":
                background = torch.rand(3, device=self.device)
            elif self.config.background_color == "white":
                background = torch.ones(3, device=self.device)
            elif self.config.background_color == "black":
                background = torch.zeros(3, device=self.device)
            else:
                background = self.background_color.to(self.device)
        else:
            optimized_camera_to_world = camera.camera_to_worlds

            if renderers.BACKGROUND_COLOR_OVERRIDE is not None:
                background = renderers.BACKGROUND_COLOR_OVERRIDE.to(self.device)
            else:
                background = self.background_color.to(self.device)

        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                return self.get_empty_outputs(int(camera.width.item()), int(camera.height.item()), background)
        else:
            crop_ids = None
        camera_scale_fac = 1.0 / self._get_downscale_factor()
        viewmat = get_viewmat(optimized_camera_to_world)
        W, H = int(camera.width[0] * camera_scale_fac), int(camera.height[0] * camera_scale_fac)
        self.last_size = (H, W)

        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            features_dc_crop = self.features_dc[crop_ids]
            features_rest_crop = self.features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest
            scales_crop = self.scales
            quats_crop = self.quats

        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)
        BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default
        K = camera.get_intrinsics_matrices().cuda()
        K[:, :2, :] *= camera_scale_fac
        # apply the compensation of screen space blurring to gaussians
        if self.config.rasterize_mode not in ["antialiased", "classic"]:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        if self.config.output_depth_during_training or not self.training:
            render_mode = "RGB+ED"
        else:
            render_mode = "RGB"
        if self.config.rasterize_mode!="antialiased":
            if self.config.sh_degree > 0:
                sh_degree_to_use = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
            else:
                sh_degree_to_use = None
        else:
            if self.config.sh_degree > 0:
                step=max((self.step-5000),0)
                sh_degree_to_use = min(step // self.config.sh_degree_interval, self.config.sh_degree)
            else:
                sh_degree_to_use = None

        
        # if self.training == False:
        #     modify_sclae =torch.exp(scales_crop)
        # else:
        #     factor=W*H/(means_crop.size(0))
        #     print("H : {} , W : {} , factor : {} , N : {}".format(H,W,factor,means_crop.size(0)))
        #     modify_sclae = torch.exp(scales_crop)*factor
        # factor=min(W*H/(9*np.pi*means_crop.size(0)),0.3)
        
        if self.config.rasterize_mode=="antialiased":
            alpha=(self.step/30000-0.1)/0.4
            factor = 0.3+2.7*(1-np.clip(alpha,0,1))# decay from 3 to 0.3
        else:
            factor=0
        
        # factor=0
        render, alpha, info = rasterization(
            means=means_crop,
            quats=quats_crop / quats_crop.norm(dim=-1, keepdim=True),
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=colors_crop,
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
            width=W,
            height=H,
            tile_size=BLOCK_WIDTH,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sh_degree=sh_degree_to_use,
            sparse_grad=False,
            absgrad=True,
            rasterize_mode=self.config.rasterize_mode,
            eps2d=factor,
            # set some threshold to disregrad small gaussians for faster rendering.
            # radius_clip=3.0,
        )
        if self.training and info["means2d"].requires_grad:
            info["means2d"].retain_grad()
        self.xys = info["means2d"]  # [1, N, 2]
        self.radii = info["radii"][0]  # [N]

        alpha = alpha[:, ...]
        rgb = render[:, ..., :3] + (1 - alpha) * background
        rgb = torch.clamp(rgb, 0.0, 1.0)
        if render_mode == "RGB+ED":
            depth_im = render[:, ..., 3:4]
            depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).squeeze(0)
        else:
            depth_im = None
            
        return {"rgb": rgb.squeeze(0), "depth": depth_im, "accumulation": alpha.squeeze(0), "background": background}  # type: ignore
