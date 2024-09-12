

# Python Imports


# Package Imports
import torch

# Project Imports
from .base_model import BaseModel


class BasicModel(BaseModel):
    def __init__(self, model_configs, model_architecture_configs):
        super(BasicModel, self).__init__()
       	self.layer = torch.nn.Linear(1, 1)

    def forward(self, data):	
    	return torch.ones((1, )).to(self.layer.device)


    def get_submodels(self):
        return {"full_model": self}
