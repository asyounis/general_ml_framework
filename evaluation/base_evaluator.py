
# Python Imports

# Package Imports
import yaml
import torch
from tqdm import tqdm

# Project Imports
from ..utils import *
from ..model_saver_loader import ModelSaverLoader
from .metric_pretty_printer import MetricPrettyPrinter

class BaseEvaluator:
    def __init__(self, experiment_name, experiment_configs, save_dir, logger, device, model, evaluation_dataset, metrics_classes):

        # Save in case we need it
        self.experiment_name = experiment_name
        self.experiment_configs = experiment_configs
        self.save_dir = save_dir
        self.logger = logger
        self.device = device
        self.model = model
        self.evaluation_dataset = evaluation_dataset
        self.metrics_classes = metrics_classes

        # Extract the mandatory training configs
        self.evaluation_configs = get_mandatory_config("evaluation_configs", experiment_configs, "experiment_configs")
        self.quantitative_config = get_mandatory_config_as_type("quantitative_config", self.evaluation_configs, "evaluation_configs", dict)
        self.qualitative_config = get_mandatory_config_as_type("qualitative_config", self.evaluation_configs, "evaluation_configs", dict)

        # Extract the optional configs
        self.num_cpu_cores_for_dataloader = get_optional_config_with_default("num_cpu_cores_for_dataloader", self.evaluation_configs, "evaluation_configs", default_value=4)

        # Create a qualitative and quantitative save directory
        self.quantitative_save_dir = "{}/quantitative/".format(self.save_dir)
        self.qualitative_save_dir = "{}/qualitative/".format(self.save_dir)
        ensure_directory_exists(self.quantitative_save_dir)
        ensure_directory_exists(self.qualitative_save_dir)

        # get all the models
        if(isinstance(self.model, torch.nn.DataParallel)):
            self.all_models = self.model.module.get_submodels()
            self.all_models["full_model"] = self.model.module
        else:
            self.all_models = self.model.get_submodels()
            self.all_models["full_model"] = self.model

        # Create the model saver
        self.model_saver = ModelSaverLoader(self.all_models, self.save_dir, self.logger)

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


    def do_forward_pass(self, data):
        raise NotImplemented


    def _do_quantitative_evaluation(self):

        # Check if we need to do this evaluation part
        if(self.quantitative_config["do_run"] == False):
            return

        # get the metrics configs and create the metrics we will be using
        metric_configs = get_mandatory_config_as_type("metric_configs", self.quantitative_config, "quantitative_config", dict)
        metrics = self._create_metrics(metric_configs, self.metrics_classes)

        # Get the configs
        batch_sizes = self.quantitative_config["batch_sizes"]

        # create the dataloaders
        evaluation_loader = self._create_data_loaders(batch_sizes, self.num_cpu_cores_for_dataloader, self.evaluation_dataset, "evaluation")

        # put the models into evaluation mode
        for model_name in self.all_models.keys():
            self.all_models[model_name].eval()
        self.model.eval()

        # Reset all the metrics
        for metric_name in metrics.keys():
            metrics[metric_name].reset()

        # Run through all the data and compute the outputs
        t = tqdm(iter(evaluation_loader), leave=False, total=len(evaluation_loader))
        for step, data in enumerate(t): 

            # Do the forward pass over the data and get the model output
            outputs = self.do_forward_pass(data)

            # Update all the metrics
            for metric_name in metrics.keys():
                metrics[metric_name].add_values(outputs, data)

        # Save the metric data
        metric_pretty_printer = MetricPrettyPrinter(self.logger, self.quantitative_save_dir)
        metric_pretty_printer.print_metrics(metrics)


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


    def _create_metrics(self, metric_configs, metrics_classes):

        # All the metrics that we will create
        metrics = dict()

        # Create the metrics one by one
        for metric_name in metric_configs.keys():

            # Get the specific configs for this metric
            metric_specific_configs = metric_configs[metric_name]

            # Get the metric type.  This is how we will create the metric
            metric_type = metric_specific_configs["type"]

            # Make sure we have that trainer
            if(metric_type not in metrics_classes):
                print("Unknown evaluation metric type \"{}\"".format(metric_type))
                assert(False)

            # Create the metric
            metric_cls = metrics_classes[metric_type]
            metric = metric_cls(metric_name, metric_specific_configs)

            # Add the metric 
            metrics[metric_name] = metric
            
        return metrics

    