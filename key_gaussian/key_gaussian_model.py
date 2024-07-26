

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig

@dataclass
class KeyGaussiansModelConfig(SplatfactoModelConfig):
    """BAD-Gaussians Model config"""

    _target: Type = field(default_factory=lambda: KeyGaussianModel)
    
class KeyGaussianModel(SplatfactoModel):

    config: KeyGaussiansModelConfig
    def __init__(self, config: KeyGaussiansModelConfig, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
