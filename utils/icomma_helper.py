import torch
from LoFTR.src.loftr import LoFTR, default_cfg
from copy import deepcopy
import numpy as np
from typing import NamedTuple
from utils.lie_utils import so3_t3_to_SE3
from utils.pose_utils import compose_split
# Load the pre-trained LoFTR model. For more details, please refer to https://github.com/zju3dv/LoFTR.
def load_LoFTR(ckpt_path:str,temp_bug_fix:bool):
    _default_cfg = deepcopy(default_cfg)
    _default_cfg['coarse']['temp_bug_fix'] = temp_bug_fix  # set to False when using the old ckpt
   
    LoFTR_model = LoFTR(config=_default_cfg)
    LoFTR_model.load_state_dict(torch.load(ckpt_path)['state_dict'])
    LoFTR_model= LoFTR_model.eval().cuda()
    
    return LoFTR_model

rot_psi = lambda phi: np.array([
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1]])

rot_theta = lambda th: np.array([
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1]])

rot_phi = lambda psi: np.array([
        [np.cos(psi), -np.sin(psi), 0, 0],
        [np.sin(psi), np.cos(psi), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])

def trans_t_xyz(tx, ty, tz):
    T = np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ])
    return T

def combine_3dgs_rotation_translation(R_c2w, T_w2c):
    RT_w2c = np.eye(4)
    RT_w2c[:3, :3] = R_c2w.T
    RT_w2c[:3, 3] = T_w2c
    RT_c2w=np.linalg.inv(RT_w2c)
    return RT_c2w

def get_pose_estimation_input(obs_view,delta):
    gt_pose_c2w=torch.tensor(combine_3dgs_rotation_translation(obs_view.R,obs_view.T)).to(torch.double)
    random_r=(torch.rand(3)-0.5)*2*delta[0]/180.*np.pi
    random_t=(torch.rand(3)-0.5)*2*delta[1]
    noise_RT=so3_t3_to_SE3(random_r,random_t).to(torch.double)
    # noise_RT=torch.cat([noise_RT,pad],dim=0)
    
    # start_pose_c2w =  trans_t_xyz(random_t[0],random_t[1],random_t[2]) @ rot_phi(random_r[0]/180.*np.pi) @ rot_theta(random_r[1]/180.*np.pi) @ rot_psi(random_r[2]/180.*np.pi)  @ gt_pose_c2w
    start_pose_c2w = compose_split(gt_pose_c2w[:3,],noise_RT )
    
    
    pad=torch.tensor([0,0,0,1]).unsqueeze(0)
    start_pose_c2w=torch.cat([start_pose_c2w,pad],dim=0)
    
    
    icomma_info = iComMa_input_info(gt_pose_c2w=gt_pose_c2w,
        start_pose_w2c=torch.from_numpy(np.linalg.inv(start_pose_c2w)).float(),
        query_image= obs_view.original_image[0:3, :, :],
        FoVx=obs_view.FoVx,
        FoVy=obs_view.FoVy,
        image_width=obs_view.image_width,
        image_height=obs_view.image_height)
    
    return icomma_info

class iComMa_input_info(NamedTuple):
    start_pose_w2c:torch.tensor
    gt_pose_c2w:np.array
    query_image:torch.tensor
    FoVx:float
    FoVy:float
    image_width:int
    image_height:int

    