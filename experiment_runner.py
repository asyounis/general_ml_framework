

# Python Imports
import sys

# Package Imports
import yaml

# Project Imports
from .utils import *
from .config_file_loader import ConfigFileLoader
from .device_selector import DeviceSelector

class ExperimentRunner:
    def __init__(self, config_file):
    
        # Load the config File
        config_file_loader = ConfigFileLoader(config_file)

        # Get the loaded configs
        self.model_architecture_configs = config_file_loader.get_model_architecture_configs()
        self.all_experiment_configs = config_file_loader.get_experiment_configs()

        # The list of datasets: Dataset name -> dataset class
        self.dataset_classes = dict()

        # The list of models: Model name -> model class
        self.model_classes = dict()

        # The list of trainers: trainer name -> trainer class
        self.trainer_classes = dict()

    def add_dataset(self, name, cls):
        self.dataset_classes[name] = cls

    def add_model(self, name, cls):
        self.model_classes[name] = cls

    def add_trainer(self, name, cls):
        self.trainer_classes[name] = cls

    def run(self):

        # Create the device selector that we will use to get the device to use 
        device_selector = DeviceSelector()

        # Run through each experiment and run it
        for experiment_configs in self.all_experiment_configs:

            # Get the experiment name.  We need a better way of doing this cause this is so dumb
            experiment_name = list(experiment_configs.keys())[0]

            # Extract the configs because we store it in a weird list/dict nested config 
            experiment_configs = experiment_configs[experiment_name]

            # Check if we should even run this experiment
            do_run = get_mandatory_config("do_run", experiment_configs, "experiment_configs")
            if(do_run == False):
                continue

            # Select the device to run on
            gpu_info_str = device_selector.get_gpu_info_str(indent="\t\t")
            device = get_mandatory_config("device", experiment_configs, "experiment_configs")
            device = device_selector.get_device(device)

            # Get the save directory and make sure that it exists 
            save_dir = get_mandatory_config("save_dir", experiment_configs, "experiment_configs")
            ensure_directory_exists(save_dir)


            # Make the model
            model = None

            # Print some info
            print("\n\n")
            print("----------------------------------------------------------------------")
            print("----------------------------------------------------------------------")
            print("Running experiment \"{}\"".format(experiment_name))
            print("----------------------------------------------------------------------")

            print("\tGPU Device Info:")
            print(gpu_info_str)
            print("")
            print("\tDevice         : {}".format(device))
            print("\tSave Directory : {}".format(save_dir))
            print("")

            # Detect if this is a training or evaluation and do the right thing
            experiment_type = get_mandatory_config("experiment_type", experiment_configs, "experiment_configs")
            if(experiment_type == "training"):

                # Run the training
                self._run_training(experiment_name, experiment_configs, save_dir, device, model)

            elif(experiment_type == "evaluation"):
                pass

            else:
                print("Unknown experiment type \"{}\"".format(experiment_type))
                assert(False)



    def _run_training(self, experiment_name, experiment_configs, save_dir, device, model):


        # Get training type
        training_type = get_mandatory_config("training_type", experiment_configs, "experiment_configs")

        # Make sure we have that trainer
        if(training_type not in self.trainer_classes):
            print("Unknown trainer type \"{}\"".format(training_type))
            assert(False)

        # Create the trainer
        trainer_cls = self.trainer_classes[training_type]
        trainer = trainer_cls(experiment_name, experiment_configs, save_dir, device, model)

        # train!!
        trainer.train()