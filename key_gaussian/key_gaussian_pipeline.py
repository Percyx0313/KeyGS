

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Type, Union, cast
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from torch.cuda.amp.grad_scaler import GradScaler
from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from time import time
from nerfstudio.engine.optimizers import Optimizers
import torchvision.utils as vutils
import torch
from nerfstudio.cameras.cameras import Cameras, CameraType
@dataclass
class KeyGSPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: KeyGSPipeline)
class KeyGSPipeline(VanillaPipeline):
    def __init__(
        self,
        config: VanillaPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super().__init__(config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, grad_scaler=grad_scaler)
    def get_average_eval_image_metrics(self, step: Optional[int] = None, output_path: Optional[Path] = None, get_std: bool = False):
        
        if self.training==True:
            return super().get_average_eval_image_metrics(step, output_path, get_std)
        else:
            # optimize the camera pose
            num_epochs= 200 
            image_prefix="eval"
            data_loader=self.datamanager.fixed_indices_eval_dataloader
            self.model.eval_camera_optimizer: CameraOptimizer = self.model.config.camera_optimizer.setup( # type: ignore
            num_cameras=len(data_loader), device="cpu"
            )
            
            optim=torch.optim.Adam(self.model.eval_camera_optimizer.parameters(), lr=1e-3)
            gamma=0.01**(1/num_epochs)
            scheduler=torch.optim.lr_scheduler.ExponentialLR(optim, gamma=gamma)
            
            self.model.train()
            self.model.eval_camera_optimizer.train()
            
            metrics_dict_list = []
            num_images = len(data_loader)
            if output_path is not None:
                output_path.mkdir(exist_ok=True, parents=True)
            with Progress(
                
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                MofNCompleteColumn(),
                transient=True,
            ) as progress:
                task = progress.add_task("[green]Evaluating all images...", total=num_images)
                idx = 0
                for epoch in range(num_epochs):
                    optim.zero_grad(set_to_none=True)
                    for idx,(camera, batch) in enumerate(data_loader):
                        # time this the following line
                        inner_start = time()
                        camera.metadata={}
                        camera.metadata["cam_idx"] = idx
                        outputs = self.model.get_outputs_optim_eval_pose(camera=camera)
                        predicted_rgb = outputs["rgb"]
                        # print(predicted_rgb.required_grad)
                        gt_img=self.model.composite_with_background(self.model.get_gt_img(batch["image"]), outputs["background"])
                        
                        # compuet loss
                        Ll1 = torch.abs(gt_img - predicted_rgb).mean()
                        simloss = 1 - self.model.ssim(gt_img.permute(2, 0, 1)[None, ...], predicted_rgb.permute(2, 0, 1)[None, ...])
                        loss= (1 - self.model.config.ssim_lambda) * Ll1 + self.model.config.ssim_lambda * simloss
                        loss.backward()
                        optim.step()
                    scheduler.step()
                
                    
                for idx,(camera, batch) in enumerate(data_loader):
                    outputs = self.model.get_outputs_optim_eval_pose(camera=camera)
                    metrics_dict, image_dict = self.model.get_image_metrics_and_images(outputs, batch)
                    if output_path is not None:
                        for key in image_dict.keys():
                            image = image_dict[key]  # [H, W, C] order
                            vutils.save_image(
                                image.permute(2, 0, 1).cpu(), output_path / f"{image_prefix}_{key}_{idx:04d}.png"
                            )

                    assert "num_rays_per_sec" not in metrics_dict
                    height, width = camera.height, camera.width
                    num_rays = height * width
                    metrics_dict["num_rays_per_sec"] = (num_rays / (time() - inner_start)).item()
                    fps_str = "fps"
                    assert fps_str not in metrics_dict
                    metrics_dict[fps_str] = (metrics_dict["num_rays_per_sec"] / (height * width)).item()
                    metrics_dict_list.append(metrics_dict)
                    print(f"{idx} , PSNR {metrics_dict['psnr']}")
                    progress.advance(task)
                    idx = idx + 1
                    
                    

            metrics_dict = {}
            for key in metrics_dict_list[0].keys():
                if get_std:
                    key_std, key_mean = torch.std_mean(
                        torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list])
                    )
                    metrics_dict[key] = float(key_mean)
                    metrics_dict[f"{key}_std"] = float(key_std)
                else:
                    metrics_dict[key] = float(
                        torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
                    )

            self.train()
            print(metrics_dict)
            return metrics_dict
                