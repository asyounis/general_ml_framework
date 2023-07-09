
# Python Imports
import copy 
import sys
import re

# Package Imports
import yaml

# Project Imports
from ..utils import *


class BaseTrainer:
    def __init__(self, experiment_name, experiment_configs, save_dir, device, model):

        # Save in case we need it
        self.experiment_name = experiment_name
        self.experiment_configs = experiment_configs
        self.save_dir = save_dir
        self.device = device
        self.model = model

        # Extract the training parameters
        self.training_configs = experiment_configs["training_configs"]
        self.epochs = get_mandatory_config("epochs", self.training_configs, "self.training_configs")
        self.batch_sizes = get_mandatory_config_as_type("batch_sizes", self.training_configs, "self.training_configs", dict)





        # self.early_stopping_patience = self.training_params["early_stopping_patience"]
        # self.early_stopping_start_offset = self.training_params["early_stopping_start_offset"]


    def train(self):
        pass