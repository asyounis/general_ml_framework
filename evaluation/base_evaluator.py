
# Python Imports

# Package Imports
import yaml
import torch
from tqdm import tqdm

# Project Imports
from ..utils import *
from ..model_saver_loader import ModelSaverLoader

class BaseEvaluator:
    def __init__(self, experiment_name, experiment_configs, save_dir, logger, device, model, evaluation_dataset):

        # Save in case we need it
        self.experiment_name = experiment_name
        self.experiment_configs = experiment_configs
        self.save_dir = save_dir
        self.logger = logger
        self.device = device
        self.model = model
        self.evaluation_dataset = evaluation_dataset

        # Extract the mandatory training configs
        self.evaluation_configs = get_mandatory_config("evaluation_configs", experiment_configs, "experiment_configs")
        batch_sizes = get_mandatory_config_as_type("batch_sizes", self.evaluation_configs, "evaluation_configs", dict)
        self.quantitative_config = get_mandatory_config_as_type("quantitative_config", self.evaluation_configs, "evaluation_configs", dict)
        self.qualitative_config = get_mandatory_config_as_type("qualitative_config", self.evaluation_configs, "evaluation_configs", dict)

        # Extract the optional configs
        self.num_cpu_cores_for_dataloader = get_optional_config_with_default("num_cpu_cores_for_dataloader", self.evaluation_configs, "evaluation_configs", default_value=4)

        # get all the models
        if(isinstance(self.model, torch.nn.DataParallel)):
            self.all_models = self.model.module.get_submodels()
            self.all_models["full_model"] = self.model.module
        else:
            self.all_models = self.model.get_submodels()
            self.all_models["full_model"] = self.model

        # Create the model saver
        self.model_saver = ModelSaverLoader(self.all_models, self.save_dir)

        # Move the model to the correct device
        if(isinstance(self.model, torch.nn.DataParallel)):
            self.model = self.model.to(self.device[0])
        else:
            self.model = self.model.to(self.device)


    def evaluate(self):

        # Do everything in evaluation mode (aka with no gradients)
        with torch.no_grad():

            # Set the model into evaluation mode
            self.model.eval()

            # Do Quantitative evaluation
            self._do_quantitative_evaluation()

            # Do Qualitative evaluation
            self._do_qualitative_evaluation()




    def _do_quantitative_evaluation(self):

        # Check if we need to do this evaluation part
        if(self.quantitative_config["do_run"] == False):
            return

        # Get the configs
        batch_sizes = self.quantitative_config["batch_sizes"]
        num_cpu_cores_for_dataloader = self.quantitative_config["num_cpu_cores_for_dataloader"]

        # create the dataloaders
        evaluation_loader = self._create_data_loaders(batch_sizes, num_cpu_cores_for_dataloader, self.evaluation_dataset, "evaluation")



    def _do_qualitative_evaluation(self):

        # Check if we need to do this evaluation part
        if(self.qualitative_config["do_run"] == False):
            return






    def _create_data_loaders(self, batch_sizes, num_cpu_cores_for_dataloader, dataset, dataset_type):

        if(dataset is None):
            return None

        if(dataset_type not in batch_sizes):
            assert(False)

        # get the batch size
        batch_size = batch_sizes[dataset_type]

        # Check if the dataset has a custom collate function we should be using
        has_custom_collate_function = getattr(dataset, "get_collate_function", None)
        if callable(has_custom_collate_function):
            custom_collate_function = dataset.get_collate_function()
        else:
            custom_collate_function = None

        # Create the data-loader
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_cpu_cores_for_dataloader, pin_memory=True, persistent_workers=True, collate_fn=custom_collate_function)

        return dataloader

