

# Python Imports
import sys
import copy 
import argparse
import time 
import gc
import datetime


# Package Imports
import yaml
import torch
from prettytable import PrettyTable

# Project Imports
from .utils.config import *
from .utils.general import *
from .utils.yaml import *
from .utils.distributed import *
from .config_file_loader import ConfigFileLoader
from .device_selector import DeviceSelector
from .model_saver_loader import ModelSaverLoader
from .logger import Logger

class ExperimentRunner:
    def __init__(self):
            
        # Parse the command line arguments
        args = self._parse_cmd_arguments()

        # Unpack the arguments
        config_file = args.config_file
        self.number_of_runs = args.number_of_runs
        self.run_numbers = args.run_numbers
        self.load_from_checkpoint = args.load_from_checkpoint

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

        # The list of metrics: metric name -> metric class
        self.metric_classes = dict()

    def add_dataset(self, name, cls):
        self.dataset_classes[name] = cls

    def add_model(self, name, cls):
        self.model_classes[name] = cls

    def add_trainer(self, name, cls):
        self.trainer_classes[name] = cls

    def add_evaluator(self, name, cls):
        self.evaluator_classes[name] = cls

    def add_metric(self, name , cls):
        self.metric_classes[name] = cls

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
            number_of_runs = get_optional_config_with_default("number_of_runs", experiment_configs, "experiment_configs", default_value=self.number_of_runs)

            # Run multiple times
            for run_number in range(number_of_runs):

                # Check if we should run this experiment
                if(self.run_numbers is not None):
                    if(run_number not in self.run_numbers):
                        continue

                # Run the experiment
                self._run_experiment(experiment_name, experiment_configs, run_number, number_of_runs, device_selector)


                # Cleanup after running the experiment
                gc.collect()                
                torch.cuda.empty_cache()
                # print(torch.cuda.memory_summary())


    def _run_experiment(self, experiment_name, experiment_configs, run_number, number_of_runs, device_selector):

        # Make a run specific variable set
        variables = dict()
        variables["<framework_var_run_number>"] = "run_{:04d}".format(run_number)

        # Make a copy of the configs for this run
        experiment_configs_copy = copy.deepcopy(experiment_configs)

        # Resolve the variables
        experiment_configs_copy = ConfigFileLoader.resolve_variables(variables, experiment_configs_copy)                

        # Select the device to run on
        gpu_info_str = device_selector.get_gpu_info_str(indent="\t")
        device_configs = get_mandatory_config("device_configs", experiment_configs_copy, "experiment_configs_copy")
        device = device_selector.get_device(device_configs)

        # Get the save directory and make sure that it exists 
        # Note we add the run number to the save dir
        save_dir = get_mandatory_config("save_dir", experiment_configs_copy, "experiment_configs_copy")
        save_dir = "{}/run_{:04d}/".format(save_dir, run_number)
        ensure_directory_exists(save_dir)

        # Create the Logger and create a new file for it
        logger = Logger(save_dir)
        logger.create_new_log_file()

        # Print some info
        logger.log("\n\n")
        logger.log("----------------------------------------------------------------------")
        logger.log("----------------------------------------------------------------------")
        logger.log("Running experiment \"{}\". Run Number {:02d} out of {:02d}".format(experiment_name, run_number, number_of_runs))
        if(self.run_numbers is not None):
            logger.log("Note: Run numbers are: {}".format(str(self.run_numbers)))
        logger.log("----------------------------------------------------------------------")

        # Log some some important things
        logger.log("GPU Device Info:")
        logger.log(gpu_info_str)
        logger.log("")
        logger.log("Device         : {}".format(device))
        logger.log("Save Directory : {}".format(save_dir))
        logger.log("")

        # Start the timer
        start_time = time.perf_counter()

        # Detect if this is a training or evaluation and do the right thing
        experiment_type = get_mandatory_config("experiment_type", experiment_configs_copy, "experiment_configs_copy")
        if(experiment_type == "training"):
            self._run_training(experiment_name, experiment_configs_copy, save_dir, logger, device)

        elif(experiment_type == "evaluation"):
            self._run_evaluation(experiment_name, experiment_configs_copy, save_dir, logger, device)

        else:
            print("Unknown experiment type \"{}\"".format(experiment_type))
            assert(False)


        # End the timer and save it so we know how long this experiment ran for
        end_time = time.perf_counter()
        runtime_seconds = end_time - start_time
        runtime_minutes = runtime_seconds / 60.0
        runtime_hours = runtime_minutes / 60.0
        runtime_days = runtime_hours / 24.0
        runtime_human_readable = datetime.timedelta(seconds=runtime_seconds)

        # Log some some important things
        logger.log("")
        logger.log("Experiment Runtimes:")
        logger.log("\tRuntime (seconds)   : {:03f} seconds".format(runtime_seconds))
        logger.log("\tRuntime (minutes)   : {:03f} minutes".format(runtime_minutes))
        logger.log("\tRuntime (hours)     : {:03f} hours".format(runtime_hours))
        logger.log("\tRuntime (days)      : {:03f} days".format(runtime_days))
        logger.log("\tRuntime (Human Readable)   (DD days, HH:MM:SS.SS): {}".format(str(runtime_human_readable)))
        logger.log("")







    def _run_training_helper(self, rank, world_size, master_port, experiment_name, experiment_configs, save_dir, devices):

        if(world_size > 1):
            distributed_setup(rank, world_size, master_port)

        # Create a logger
        logger = Logger(save_dir, distributed_rank=rank)

        # Make the model
        model = self._create_model(experiment_configs, logger)

        # Print the model stats
        if(rank == 0):
            self._print_model_stats(model, logger)        

        # Select the device we will use for the training
        if(isinstance(devices, list)):
            device = devices[rank]
        else:
            device = devices

        # Move the model to the correct device
        model = model.to(device)

        # Wrap it in the DistributedDataParallel
        if(world_size > 1):
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True)

        # Get training type
        training_type = get_mandatory_config("training_type", experiment_configs, "experiment_configs")

        # Make sure we have that trainer
        if(training_type not in self.trainer_classes):
            print("Unknown trainer type \"{}\"".format(training_type))
            assert(False)

        # Create the distributed parameters
        if(world_size > 1):
            distributed_execution_parameters = dict()
            distributed_execution_parameters["is_using_distributed"] = True
            distributed_execution_parameters["rank"] = rank
            distributed_execution_parameters["world_size"] = world_size
        else:
            distributed_execution_parameters = dict()
            distributed_execution_parameters["is_using_distributed"] = False
            distributed_execution_parameters["rank"] = rank
            distributed_execution_parameters["world_size"] = world_size


        # Create the trainer
        trainer_cls = self.trainer_classes[training_type]
        trainer = trainer_cls(experiment_name, experiment_configs, save_dir, logger, device, model, self._create_dataset, self.load_from_checkpoint, distributed_execution_parameters)

        # train!!
        trainer.train()

        # Cleanup afterwards
        if(world_size > 1):
            distributed_cleanup()

    def _run_training(self, experiment_name, experiment_configs, save_dir, logger, devices):


        # The world size is the number of devices
        if(isinstance(devices, list)):
            world_size = len(devices)
        else:
            world_size = 1


        # Get a random unsused port for all the comms to go through
        master_port = distributed_get_open_port()
        logger.log("Distributed Master Port: {}".format(master_port))


        if(world_size > 1):
            # Spawn all the training jobs
            torch.multiprocessing.spawn(
                self._run_training_helper,
                args=(world_size, master_port, experiment_name, experiment_configs, save_dir, devices),
                nprocs=world_size
                )
        else:
            self._run_training_helper(0, world_size, master_port, experiment_name, experiment_configs, save_dir, devices)

    def _run_evaluation(self, experiment_name, experiment_configs, save_dir, logger, device):

        # Make the model
        model = self._create_model(experiment_configs, logger)

        # Print the model stats
        self._print_model_stats(model, logger)

        # Get evaluation type
        evaluation_type = get_mandatory_config("evaluation_type", experiment_configs, "experiment_configs")

        # Make sure we have that trainer
        if(evaluation_type not in self.evaluator_classes):
            print("Unknown evaluation type \"{}\"".format(evaluation_type))
            assert(False)

        # Create the trainer
        evaluator_cls = self.evaluator_classes[evaluation_type]
        evaluator = evaluator_cls(experiment_name, experiment_configs, save_dir, logger, device, model, self._create_dataset, self.metric_classes)

        # evaluate!!
        evaluator.evaluate()

    def _create_dataset(self, dataset_config_file, dataset_type):

        # Load the file
        dataset_configs = load_yaml_file(dataset_config_file)
        dataset_configs = dataset_configs["dataset_configs"]

        # get the name of the datasets
        dataset_name = get_mandatory_config("dataset_name", dataset_configs, "dataset_configs")

        # Make sure that the name is in the passed in dataset
        assert(dataset_name in self.dataset_classes)

        # create the dataset
        dataset_cls = self.dataset_classes[dataset_name]
        dataset = dataset_cls(dataset_configs, dataset_type)

        return dataset

    def _create_model(self, experiment_configs, logger):

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

        # Init the model
        model_init_configs = get_optional_config_with_default("model_init_configs", experiment_configs, "experiment_configs", default_value=None)
        model.init(model_init_configs)

        # Load the model!
        if("pretrained_models" in experiment_configs):
            pretrained_models_configs = get_mandatory_config("pretrained_models", experiment_configs, "experiment_configs")
            ModelSaverLoader.load_models(model, pretrained_models_configs, logger)


        return model

    def _parse_cmd_arguments(self):

        # Create the parser
        parser = argparse.ArgumentParser()

        # We always need a config file
        parser.add_argument("-c", "--config_file", dest="config_file", help="Specify the filepath to the config file to use", required=True, type=str, action="store")

        # We need an optional number of runs
        parser.add_argument("-n", "--number_of_runs", dest="number_of_runs", help="Specify the number of times to run an experiment", required=False, type=int, default=1)

        # We need an optional number of runs
        parser.add_argument("-r", "--run_numbers", dest="run_numbers", help="Specify a specific run to run", required=False, type=int, default=None, nargs="+")

        # We need an optional load from checkpoint
        parser.add_argument("-l", "--load_from_checkpoint", dest="load_from_checkpoint", help="Override if we should load from checkpoint or not", required=False, type=bool, default=None)

        # Parse!!
        args = parser.parse_args()

        return args

    def _print_model_stats(self, model, logger):

        # Get the total number of parameters
        total_num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
        # Get the number of parameters for each of the sub-models
        sub_model_dict = model.get_submodels()
        sub_model_num_params = dict()
        for sub_model_name in sub_model_dict:
            sub_model = sub_model_dict[sub_model_name]
            sub_model_num_params[sub_model_name] = sum(p.numel() for p in sub_model.parameters() if p.requires_grad)
        

        # Display them nicely
        table = PrettyTable()
        table.field_names = ["Model Name", "Number of Trainable Parameters"]

        # Add in the main model
        table.add_row(["full model", "{:,d}".format(total_num_params)])

        # Add in the sub models
        for sub_model_name in sub_model_num_params:
            table.add_row(["{:}".format(sub_model_name), "{:,d}".format(sub_model_num_params[sub_model_name])])

        # Add indent to the table and print it
        table_str = str(table)
        table_str = table_str.split("\n")
        table_str = ["\t{}".format(ts) for ts in table_str]
        table_str = "\n".join(table_str)

        # Print!
        logger.log("\n")
        logger.log("Model Stats:")
        logger.log(table_str)
        logger.log("\n")

