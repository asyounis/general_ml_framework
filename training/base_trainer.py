
# Python Imports
import copy 
import sys
import re
import time

# Package Imports
import yaml
import torch
from tqdm import tqdm

# Project Imports
from ..utils import *
from .data_plotter import DataPlotter
from .early_stopping import EarlyStopping
from ..model_saver_loader import ModelSaverLoader

class BaseTrainer:
    def __init__(self, experiment_name, experiment_configs, save_dir, logger, device, model,training_dataset, validation_dataset):

        # Save in case we need it
        self.experiment_name = experiment_name
        self.experiment_configs = experiment_configs
        self.save_dir = save_dir
        self.logger = logger
        self.device = device
        self.model = model
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset

        # Extract the mandatory training configs
        self.training_configs = get_mandatory_config("training_configs", experiment_configs, "experiment_configs")
        self.epochs = get_mandatory_config("epochs", self.training_configs, "training_configs")
        batch_sizes = get_mandatory_config_as_type("batch_sizes", self.training_configs, "training_configs", dict)
        optimizer_configs = get_mandatory_config_as_type("optimizer_configs", self.training_configs, "training_configs", dict)
        learning_rates = get_mandatory_config_as_type("learning_rates", self.training_configs, "training_configs", dict)
        early_stopping_configs = get_mandatory_config_as_type("early_stopping_configs", self.training_configs, "training_configs", dict)


        # Extract the optional configs
        self.num_cpu_cores_for_dataloader = get_optional_config_with_default("num_cpu_cores_for_dataloader", self.training_configs, "training_configs", default_value=4)
        self.accumulate_gradients_counter = get_optional_config_with_default("accumulate_gradients_counter", self.training_configs, "training_configs", default_value=1)
        self.gradient_clip_value = get_optional_config_with_default("gradient_clip_value", self.training_configs, "training_configs", default_value=None)
        self.load_from_checkpoint = get_optional_config_with_default("load_from_checkpoint", self.training_configs, "training_configs", default_value=False)

        # create the dataloaders
        self.training_loader = self._create_data_loaders(batch_sizes, self.training_dataset, "training")
        self.validation_loader = self._create_data_loaders(batch_sizes, self.validation_dataset, "validation")

        # get all the models
        if(isinstance(self.model, torch.nn.DataParallel)):
            self.all_models = self.model.module.get_submodels()
            self.all_models["full_model"] = self.model.module
        else:
            self.all_models = self.model.get_submodels()
            self.all_models["full_model"] = self.model

        # Create the optimizer
        self.optimizers = self._create_optimizers(optimizer_configs, learning_rates)

        # Construct the early stopping
        self.early_stopping = EarlyStopping(early_stopping_configs)

        # Create the data plotters
        self.data_plotters = self._create_data_plotters()

        # Keep track of the time averages
        self.timing_data = dict()
        self.timing_data["average_training_time"] = []
        self.timing_data["average_validation_time"] = []

        # Some things needed to keep track of for training
        self.last_finised_epoch = -1

        # Load from the checkpoint
        self._load_from_checkpoint()


        # Create the model saver
        self.model_saver = ModelSaverLoader(self.all_models, self.save_dir, self.logger)

        # Move the model to the correct device
        if(isinstance(self.model, torch.nn.DataParallel)):
            self.model = self.model.to(self.device[0])
        else:
            self.model = self.model.to(self.device)


    def train(self):

        # Keep track of the best validation loss
        best_validation_loss = None

        # Go through the epochs
        for epoch in tqdm(range((self.last_finised_epoch+1), self.epochs, 1)):

            # Do The training pass
            training_loss, average_training_time = self._do_training_epoch(epoch)
            self.data_plotters["training_epoch_loss"].add_value(training_loss)

            # Do the validation pass
            validation_loss, average_validation_time = self._do_validation_epoch(epoch)
            self.data_plotters["validation_epoch_loss"].add_value(validation_loss)

            # Log the validation and training times once after the first epoch so that 
            # The system has stabilized enough
            if(epoch == 1):
                self.logger.log("Average training time per batch: {:04f} seconds".format(average_training_time), print_to_terminal=False)
                self.logger.log("Average validation time per batch: {:04f} seconds".format(average_validation_time), print_to_terminal=False)

            if(epoch > 0):
                # We should keep track of the time averages and save it
                self.timing_data["average_training_time"].append(average_training_time)
                self.timing_data["average_validation_time"].append(average_validation_time)
                torch.save(self.timing_data, "{}/timing_data.pt".format(self.save_dir))

            # Make all plotters write!
            for data_plotter_name in self.data_plotters:
                self.data_plotters[data_plotter_name].plot_and_save()

            # See if the validation loss is better
            if((best_validation_loss is None) or (validation_loss < best_validation_loss)):
                best_validation_loss = validation_loss
                is_best = True
            else:
                is_best = False

            # Save the models
            self.model_saver.save_models(epoch, is_best)

            # We finished this epoch so record that we finished it
            self.last_finised_epoch = epoch

            # Update the early stopping but dont actually do any stopping because we need to checkpoint
            self.early_stopping(validation_loss)

            # Create the checkpoint for this epoch
            self._create_checkpoint(epoch)

            # Determine if we should early stop
            if(self.early_stopping.do_stop()):
                break


    def _do_training_epoch(self, epoch):


        # Freeze the models that we are not training. So if a model 
        # has a batchnorm layer, in eval mode the batchnorm will be frozen
        if("full_model" in self.optimizers.keys()):
            assert(len(self.optimizers.keys()) == 1)
            for model_name in self.all_models.keys():
                self.all_models[model_name].train()

        else:

            if(len(self.optimizers.keys()) != 0):
                self.model.train()

            for model_name in self.all_models.keys():
                if(model_name in self.optimizers.keys()):
                    self.all_models[model_name].train()
                else:
                    if(isinstance(self.model, torch.nn.DataParallel)):
                        if(self.all_models[model_name] != self.model.module):
                            self.all_models[model_name].eval()
                    else:
                        if(self.all_models[model_name] != self.model):
                            self.all_models[model_name].eval()

        # Keep track of stats needed to compute the average loss
        total_loss = 0
        number_of_losses_to_use_for_average_loss = 0

        # Keep track of some timing information
        total_time_taken_seconds = 0
        number_of_losses_to_use_for_average_time = 0

        # Create an iterator 
        training_loader_iter = iter(self.training_loader)

        # Do a training step so we can get the compiling out of the way so we can have accurate ETA info
        data = next(training_loader_iter)
        loss, batch_size = self._do_training_step(0, data)
        assert(loss is not None)

        # keep track of the average loss
        total_loss += loss.item() * batch_size
        number_of_losses_to_use_for_average_loss += batch_size
        self.data_plotters["training_iteration_loss"].add_value(loss.cpu().item())

        # Go through all the data once
        t = tqdm(training_loader_iter, leave=False, total=len(self.training_loader)-1, initial=1)
        for step_tmp, data in enumerate(t):

            # Add 1 to account for the step we already took
            step = step_tmp + 1

            # Start the timer
            start_time = time.time()

            # Do a training step
            loss, batch_size = self._do_training_step(step, data)
            if(loss is None):
                continue

            # Stop the timer
            end_time = time.time()

            # record the elapsed time but skip the first step since there is a JIT compile that may run 
            # and that can skew the time
            if(step != 0):
                total_time_taken_seconds += float(end_time - start_time)
                number_of_losses_to_use_for_average_time += 1


            # keep track of the average loss
            total_loss += loss.item() * batch_size
            number_of_losses_to_use_for_average_loss += batch_size

            # Add the loss for the batch so we can do step losses
            self.data_plotters["training_iteration_loss"].add_value(loss.cpu().item())

        # Compute the average loss
        average_loss = float(total_loss) / float(number_of_losses_to_use_for_average_loss)

        # Compute the average time
        average_time = float(total_time_taken_seconds) / float(number_of_losses_to_use_for_average_time)

        return average_loss, average_time


    def _do_training_step(self, step, data):
        
        # Zero out the gradients in prep for optimization
        for model_name in self.all_models.keys():
            self.all_models[model_name].zero_grad()

        # Do the forward pass over the data
        loss, batch_size = self.do_forward_pass(data)

        # If the loss is not valid then move on
        if(loss is None):
            return None, None

        # Compute the gradient
        loss.backward()

        # Compute the gradient norm for the models and add it to the data plotters
        for model_name in self.all_models.keys():

            # We want L2 Norm
            norm_type = 2

            # Compute the gradient norm
            norm = [torch.norm(p.grad.detach(), norm_type) for p in self.all_models[model_name].parameters() if p.grad is not None]

            # if there is no norm then we cant compute the norm
            if(len(norm) == 0):
                return None, None

            # Finish computing the norm 
            gradient_norm = torch.norm(torch.stack(norm) , norm_type)

            # Add it to the data plotter
            data_plotter_name = "gradient_norm_{}".format(model_name)
            self.data_plotters[data_plotter_name].add_value(gradient_norm.cpu().item())

        # Check if we are in a condition to take an optimization step and if so take the step
        if((((step+1) % self.accumulate_gradients_counter) == 0) or ((step+1) == len(self.training_loader))):

            # if we have a gradient clipping value then do the clipping
            if(self.gradient_clip_value is not None):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_value)

            # Take an optimization step
            for optimizer in self.optimizers.values():
                optimizer.step()

        return loss, batch_size


    def _do_validation_epoch(self, epoch):

        # Dont need the gradients for the evaluation epochs
        with torch.no_grad():

            # Put all the models in evaluation mode
            for model_name in self.all_models.keys():
                self.all_models[model_name].eval()

            # Keep track of stats needed to compute the average loss
            total_loss = 0
            number_of_losses_to_use_for_average_loss = 0

            # Keep track of some timing information
            total_time_taken_seconds = 0
            number_of_losses_to_use_for_average_time = 0

            # Go through all the data once
            t = tqdm(iter(self.validation_loader), leave=False, total=len(self.validation_loader))
            for step, data in enumerate(t):
        
                # Start the timer
                start_time = time.time()

                # Do the forward pass over the data
                loss, batch_size = self.do_forward_pass(data)

                # If the loss is not valid then move on
                if(loss is None):
                    continue

                # Stop the timer
                end_time = time.time()

                # record the elapsed time but skip the first step since there is a JIT compile that may run 
                # and that can skew the time
                if(step != 0):
                    total_time_taken_seconds += float(end_time - start_time)
                    number_of_losses_to_use_for_average_time += 1

                # Add the loss for the batch so we can do step losses
                self.data_plotters["validation_iteration_loss"].add_value(loss.cpu().item())

                # keep track of the average loss
                total_loss += loss.item() * batch_size
                number_of_losses_to_use_for_average_loss += batch_size


            # Compute the average loss
            average_loss = float(total_loss) / float(number_of_losses_to_use_for_average_loss)

            # Compute the average time
            average_time = float(total_time_taken_seconds) / float(number_of_losses_to_use_for_average_time)

            return average_loss, average_time


    def _create_data_loaders(self, batch_sizes, dataset, dataset_type):

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

        # Check if we should shuffle the data
        if(dataset_type == "training"):
            shuffle_data = True
        else:
            shuffle_data = False

        # shuffle_data = False
        # print("shuffle_data = False")
        # print("shuffle_data = False")
        # print("shuffle_data = False")
        # print("shuffle_data = False")
        # print("shuffle_data = False")
        # print("shuffle_data = False")
        # print("shuffle_data = False")
        # print("shuffle_data = False")
        # print("shuffle_data = False")


        # Create the data-loader
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle_data, num_workers=self.num_cpu_cores_for_dataloader, pin_memory=True, persistent_workers=True, collate_fn=custom_collate_function)

        return dataloader


    def _create_optimizers(self, optimizer_configs, learning_rates):

        # All the optimizers we create
        all_optimizers = dict()

        # Make sure we have some learning rates
        assert(len(learning_rates) > 0)

        # Do some checks. You either set the full model learning rate or you set all the other models individually, not both
        if("full_model" in learning_rates):
            assert(len(learning_rates) == 1)
            assert(learning_rates["full_model"] != "freeze")

        # Make sure not all the models are frozen
        has_non_frozen_count = False
        for model_name in learning_rates.keys():
            if(learning_rates[model_name] != "freeze"):
                has_non_frozen_count = True
                break
        assert(has_non_frozen_count)


        # for each model make an optimizer
        for model_name in learning_rates.keys():
            
            # Make sure the model exists
            if(model_name not in self.all_models):
                self.logger.log("WARNING: learning rate for model name \"{}\" was specified but model does not exist... Skipping...".format(model_name))
                continue

            # Extract the learning rate
            lr = learning_rates[model_name]

            # if this is a string then make it lower case so we can case agnostic
            if(isinstance(lr, str)):
                lr = lr.lower()

            # If the learning rate is frozen then no optimizer is needed
            if(lr == "freeze"):

                # # If the learning rate is frozen then we want to mark the params as not needing a gradient 
                # # and also mark them as being in eval mode so that things like batchnorm are also frozen
                # for params in model.parameters():
                #     params.requires_grad = False
                # self.all_models[model_name].eval()
                pass

            else:

                # Create the optimizer based on the type
                optimizer_type = get_mandatory_config("type", optimizer_configs, "optimizer_configs")
                if(optimizer_type == "Adam"):
                    weight_decay = get_mandatory_config("weight_decay", optimizer_configs, "optimizer_configs")
                    optimizer = torch.optim.Adam(self.all_models[model_name].parameters(),lr=lr, weight_decay=weight_decay)

                elif(optimizer_to_use == "AdamW"):
                    weight_decay = get_mandatory_config("weight_decay", optimizer_configs, "optimizer_configs")
                    optimizer = torch.optim.AdamW(self.all_models[model_name].parameters(),lr=lr, weight_decay=weight_decay)

                elif(optimizer_to_use == "NAdam"):
                    weight_decay = get_mandatory_config("weight_decay", optimizer_configs, "optimizer_configs")
                    optimizer = torch.optim.NAdam(self.all_models[model_name].parameters(),lr=lr, weight_decay=weight_decay)

                elif(optimizer_to_use == "RMSProp"):
                    weight_decay = get_mandatory_config("weight_decay", optimizer_configs, "optimizer_configs")
                    momentum = get_mandatory_config("momentum", optimizer_configs, "optimizer_configs")
                    eps = get_mandatory_config("eps", optimizer_configs, "optimizer_configs")
                    alpha = get_mandatory_config("alpha", optimizer_configs, "optimizer_configs")
                    optimizer = torch.optim.RMSprop(self.all_models[model_name].parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum, eps=eps, alpha=alpha)

                elif(optimizer_to_use == "SGD"):
                    weight_decay = get_mandatory_config("weight_decay", optimizer_configs, "optimizer_configs")
                    momentum = get_mandatory_config("momentum", optimizer_configs, "optimizer_configs")
                    dampening = get_mandatory_config("dampening", optimizer_configs, "optimizer_configs")
                    optimizer = torch.optim.SGD(self.all_models[model_name].parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum, dampening=dampening)

                else:
                    print("Unknown optimizer type \"{}\"".format(optimizer_type))
                    exit()

                all_optimizers["model_name"] = optimizer

        return all_optimizers


    def _create_data_plotters(self):

        all_data_plotters = dict()
        
        # Create the epoch training and validation plotters
        all_data_plotters["training_epoch_loss"] = DataPlotter("Training Loss per Epoch", "Epoch", "Training Loss", self.save_dir, "training_loss_epoch.png")
        all_data_plotters["validation_epoch_loss"] = DataPlotter("Validation Loss per Epoch", "Epoch", "Validation Loss", self.save_dir, "validation_loss_epoch.png")

        # Create the iteration training and validation plotters
        all_data_plotters["training_iteration_loss"] = DataPlotter("Training Loss per Iteration", "Iteration", "Training Loss", self.save_dir, "training_loss_iteration.png")
        all_data_plotters["validation_iteration_loss"] = DataPlotter("Validation Loss per Iteration", "Iteration", "Validation Loss", self.save_dir, "validation_loss_iteration.png")
        
        # Create the gradient norm data plotters
        gradient_norm_plot_save_dir = "{}/gradient_norms/".format(self.save_dir)
        for model_name in self.all_models.keys():
            data_plotter_name = "gradient_norm_{}".format(model_name)
            all_data_plotters[data_plotter_name] = DataPlotter("Gradient L2 Norm for {} (Pre Clipping)".format(model_name), "Iteration", "Gradient L2 Norm", gradient_norm_plot_save_dir, "{}.png".format(data_plotter_name))


        return all_data_plotters


    def _create_checkpoint(self, epoch):
        '''
            Checkpoint this trainer 
        '''

        # The dict we will use to save all the checkpointed data
        checkpoint_dict = dict()

        # Save the training configs
        checkpoint_dict["training_configs"] = self.training_configs

        # Save the data plotters
        data_plotters_save_dicts = dict()
        for data_plotter_name in self.data_plotters.keys():
            data_plotters_save_dicts[data_plotter_name] = self.data_plotters[data_plotter_name].get_save_dict()
        checkpoint_dict["data_plotters"] = data_plotters_save_dicts

        # Save the optimizers
        optimizers_save_dicts = dict()
        for optimizer_name in self.optimizers.keys():
            optimizers_save_dicts[optimizer_name] = self.optimizers[optimizer_name].state_dict()
        checkpoint_dict["optimizers"] = optimizers_save_dicts


        # Save the early stopping
        checkpoint_dict["early_stopping"] = self.early_stopping.get_save_dict()

        # Save the timing data
        checkpoint_dict["timing_data"] = self.timing_data
        
        # Save the additional training housekeeping data to the checkpoint
        checkpoint_dict["last_finised_epoch"] = self.last_finised_epoch
        
        # Save the whole model
        # We dont care about any internal models, just the whole model for checkpointing
        if(isinstance(self.model, torch.nn.DataParallel)):
            checkpoint_dict["model"] = self.model.module.state_dict()
        else:
            checkpoint_dict["model"] = self.model.state_dict()


        # Make sure the checkpoints directory exists
        checkpoint_dir = "{}/checkpoints".format(self.save_dir)
        ensure_directory_exists(checkpoint_dir)

        # Save the checkpoint
        checkpoint_file = "{}/checkpoint_epoch_{:04d}.pt".format(checkpoint_dir, epoch)
        torch.save(checkpoint_dict, checkpoint_file)

        # Log what we did!
        log_text = "Saved checkpoint to \"{}\"".format(checkpoint_file)
        self.logger.log(log_text)



    def _load_from_checkpoint(self):
        '''
            Load this trainer from a checkpoint if we can and are told to
        '''

        # First check if we should load from a checkpoint.
        if(self.load_from_checkpoint == False):
            return

        # Check to see if the checkpoint directory exists
        checkpoint_dir = "{}/checkpoints".format(self.save_dir)
        if(os.path.exists(checkpoint_dir) == False):
            # No checkpoints to load
            return

        # Get the latest checkpoint to load
        all_checkpoint_files = os.listdir(checkpoint_dir)
        if(len(all_checkpoint_files) == 0):
            return 

        # Convert it to ints so we can select the largest one
        checkpointed_epochs = [s.replace(".pt", "").replace("checkpoint_epoch_", "") for s in all_checkpoint_files]
        checkpointed_epochs = [int(s) for s in checkpointed_epochs]

        # Get the epoch checkpoint to load (aka the latest epoch)
        epoch_to_load = max(checkpointed_epochs)

        # Load that checkpoint file
        checkpoint_file = "{}/checkpoint_epoch_{:04d}.pt".format(checkpoint_dir, epoch_to_load)
        checkpoint_dict = torch.load(checkpoint_file, map_location="cpu")


        # Compare the configs dictionaries so we can print a warning if they are not equal
        current_configs = copy.deepcopy(self.training_configs)
        loaded_configs = copy.deepcopy(checkpoint_dict["training_configs"])

        # Delete the "load_from_checkpoint" since its ok if thats different
        try:
            current_configs.pop("load_from_checkpoint")
        except:
            pass
        try:
            loaded_configs.pop("load_from_checkpoint")
        except:
            pass

        # If they are not equal then there could be problems so warn the user
        if(loaded_configs != current_configs):
            log_text = "training config from file and checkpoint loaded training config do not match\n"
            log_text += "This could cause issues.  We will not crash the load but you should be careful"
            self.logger.log_warning(log_text)

        # Log since we are loading the state dict!
        log_text = "Loading checkpoint file: \"{}\"".format(checkpoint_file)
        self.logger.log(log_text)

        # Load the data plotters
        data_plotters_save_dicts = checkpoint_dict["data_plotters"]
        for data_plotter_name in self.data_plotters.keys():
            self.data_plotters.load_from_dict(data_plotters_save_dicts[data_plotter_name])

        # load the optimizers
        optimizers_save_dicts = checkpoint_dict["optimizers"]
        for optimizer_name in self.optimizers.keys():
            self.optimizers[optimizer_name].load_state_dict(optimizers_save_dicts[optimizer_name])

        # Load the early stopping
        self.early_stopping.load_from_dict(checkpoint_dict["early_stopping"])

        # Load the timing data
        self.timing_data = checkpoint_dict["timing_data"]

        # Save the additional training housekeeping data to the checkpoint
        self.last_finised_epoch = checkpoint_dict["last_finised_epoch"]

        # Load the model
        model_state_dict = checkpoint_dict["model"]
        if(isinstance(self.model, torch.nn.DataParallel)):
            self.model.module.load_state_dict(model_state_dict)
        else:
            self.model.load_model_state_dict(state_dict)



    def do_forward_pass(self, data):
        raise NotImplemented