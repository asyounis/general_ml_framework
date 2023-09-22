

# Python Imports
import sys
import copy 

# Package Imports
import yaml
import torch

# Project Imports
from .utils import *
from .config_file_loader import ConfigFileLoader
from .device_selector import DeviceSelector
from .model_saver_loader import ModelSaverLoader
from .logger import Logger

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

        # The list of evaluators: evaluator name -> evaluator class
        self.evaluator_classes = dict()

    def add_dataset(self, name, cls):
        self.dataset_classes[name] = cls

    def add_model(self, name, cls):
        self.model_classes[name] = cls

    def add_trainer(self, name, cls):
        self.trainer_classes[name] = cls

    def add_evaluator(self, name, cls):
        self.evaluator_classes[name] = cls

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

            # Get the number of times to run the experiment
            number_of_runs = get_optional_config_with_default("number_of_runs", experiment_configs, "experiment_configs", default_value=1)

            # Run multiple times
            for run_number in range(number_of_runs):


                # Make a run specific variable set
                variables = dict()
                variables["<framework_var_run_number>"] = "run_{:03d}".format(run_number)

                # Make a copy of the configs for this run
                experiment_configs_copy = copy.deepcopy(experiment_configs)

                # Resolve the variables
                experiment_configs_copy = ConfigFileLoader.resolve_variables(variables, experiment_configs_copy)                

                # Select the device to run on
                gpu_info_str = device_selector.get_gpu_info_str(indent="\t")
                device = get_mandatory_config("device", experiment_configs_copy, "experiment_configs_copy")
                device = device_selector.get_device(device)

                # Get the save directory and make sure that it exists 
                # Note we add the run number to the save dir
                save_dir = get_mandatory_config("save_dir", experiment_configs_copy, "experiment_configs_copy")
                save_dir = "{}/run_{:04d}/".format(save_dir, run_number)
                ensure_directory_exists(save_dir)

                # Create the Logger
                logger = Logger(save_dir)

                # Print some info
                logger.log("\n\n")
                logger.log("----------------------------------------------------------------------")
                logger.log("----------------------------------------------------------------------")
                logger.log("Running experiment \"{}\"".format(experiment_name))
                logger.log("----------------------------------------------------------------------")

                # Log some some important things
                logger.log("GPU Device Info:")
                logger.log(gpu_info_str)
                logger.log("")
                logger.log("Device         : {}".format(device))
                logger.log("Save Directory : {}".format(save_dir))
                logger.log("")

                # Make the model
                model = self._create_model(experiment_configs_copy)

                # Load the model!
                if("pretrained_models" in experiment_configs_copy):
                    pretrained_models_configs = get_mandatory_config("pretrained_models", experiment_configs_copy, "experiment_configs_copy")
                    ModelSaverLoader.load_models(model, pretrained_models_configs)

                # If we have more than 1 Device then we should be in parallel mode
                if(isinstance(device, list)):
                    assert(len(device) > 1)
                    model = torch.nn.DataParallel(model, device_ids=device)


                # Detect if this is a training or evaluation and do the right thing
                experiment_type = get_mandatory_config("experiment_type", experiment_configs_copy, "experiment_configs_copy")
                if(experiment_type == "training"):
                    self._run_training(experiment_name, experiment_configs_copy, save_dir, logger, device, model)

                elif(experiment_type == "evaluation"):
                    self._run_evaluation(experiment_name, experiment_configs, save_dir, logger, device, model)

                else:
                    print("Unknown experiment type \"{}\"".format(experiment_type))
                    assert(False)



    def _run_training(self, experiment_name, experiment_configs, save_dir, logger, device, model):

        # Get training type
        training_type = get_mandatory_config("training_type", experiment_configs, "experiment_configs")

        # Make sure we have that trainer
        if(training_type not in self.trainer_classes):
            print("Unknown trainer type \"{}\"".format(training_type))
            assert(False)

        # Create the datasets
        dataset_configs = get_mandatory_config("dataset_configs", experiment_configs, "experiment_configs")
        training_dataset = self._create_dataset(dataset_configs, "training")
        validation_dataset = self._create_dataset(dataset_configs, "validation")

        # Create the trainer
        trainer_cls = self.trainer_classes[training_type]
        trainer = trainer_cls(experiment_name, experiment_configs, save_dir, logger, device, model, training_dataset, validation_dataset)

        # train!!
        trainer.train()

    def _run_evaluation(self, experiment_name, experiment_configs, save_dir, logger, device, model):

        # Get evaluation type
        evaluation_type = get_mandatory_config("evaluation_type", experiment_configs, "experiment_configs")

        # Make sure we have that trainer
        if(evaluation_type not in self.evaluator_classes):
            print("Unknown trainer type \"{}\"".format(evaluation_type))
            assert(False)

        # Create the datasets
        dataset_configs = get_mandatory_config("dataset_configs", experiment_configs, "experiment_configs")
        evaluation = self._create_dataset(dataset_configs, "evaluation")

        # Create the trainer
        evaluator_cls = self.evaluator_classes[evaluation_type]
        evaluator = evaluator_cls(experiment_name, experiment_configs, save_dir, logger, device, model, evaluation, validation_dataset)

        # train!!
        evaluator.evaluate()


    def _create_dataset(self, dataset_configs, dataset_type):

        # get the name of the datasets
        dataset_name = get_mandatory_config("dataset_name", dataset_configs, "dataset_configs")

        # Make sure that the name is in the passed in dataset
        assert(dataset_name in self.dataset_classes)

        # create the dataset
        dataset_cls = self.dataset_classes[dataset_name]
        dataset = dataset_cls(dataset_configs, dataset_type)

        return dataset



    def _create_model(self, experiment_configs):

        # Extract the model configs
        model_configs = get_mandatory_config("model_configs", experiment_configs, "experiment_configs")

        # Check what the main model is
        main_model_name = get_mandatory_config("main_model_name", model_configs, "model_configs")

        # get the main model type
        main_model_config = get_mandatory_config(main_model_name, self.model_architecture_configs, "model_architecture_configs")
        main_model_type = get_mandatory_config("type", main_model_config, "main_model_config")

        # Make sure the model has been provided to us
        assert(main_model_type in self.model_classes)

        # Create the model
        model_cls = self.model_classes[main_model_type]
        model = model_cls(model_configs, self.model_architecture_configs)

        return model