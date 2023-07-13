
# Python Imports
import copy 
import sys
import re

# Package Imports
import yaml
import torch

# Project Imports
from ..utils import *
from .data_plotter import DataPlotter
from ..model_saver_loader import ModelSaverLoader

class BaseTrainer:
    def __init__(self, experiment_name, experiment_configs, save_dir, device, model,training_dataset, validation_dataset):

        # Save in case we need it
        self.experiment_name = experiment_name
        self.experiment_configs = experiment_configs
        self.save_dir = save_dir
        self.device = device
        self.model = model
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset

        # Extract the mandatory training configs
        self.training_configs = experiment_configs["training_configs"]
        self.epochs = get_mandatory_config("epochs", self.training_configs, "training_configs")
        batch_sizes = get_mandatory_config_as_type("batch_sizes", self.training_configs, "training_configs", dict)
        optimizer_configs = get_mandatory_config_as_type("optimizer_configs", self.training_configs, "training_configs", dict)
        learning_rates = get_mandatory_config_as_type("learning_rates", self.training_configs, "training_configs", dict)

        # Extract the optional configs
        self.num_cpu_cores_for_dataloader = get_optional_config_with_default("num_cpu_cores_for_dataloader", self.training_configs, "training_configs", default_value=4)
        self.accumulate_gradients_counter = get_optional_config_with_default("accumulate_gradients_counter", self.training_configs, "training_configs", default_value=1)
        self.gradient_clip_value = get_optional_config_with_default("gradient_clip_value", self.training_configs, "training_configs", default_value=None)

        # self.early_stopping_patience = self.training_params["early_stopping_patience"]
        # self.early_stopping_start_offset = self.training_params["early_stopping_start_offset"]

        # create the dataloaders
        self.training_loader = self._create_data_loaders(batch_sizes, self.training_dataset, "training")
        self.validation_loader = self._create_data_loaders(batch_sizes, self.validation_dataset, "validation")

        # get all the models
        self.all_models = self.model.get_submodels()
        self.all_models["full_model"] = self.model

        # Create the optimizer
        self.optimizers, self.models_to_train = self._create_optimizers(optimizer_configs, learning_rates)

        # Create the data plotters
        self.data_plotters = self._create_data_plotters()

        # Create the model saver
        self.model_saver = ModelSaverLoader(self.all_models, self.save_dir)

    def train(self):

        # Keep track of the best validation loss
        best_validation_loss = None

        # Go through the epochs
        for epoch in tqdm(range(self.epochs)):

            # Do The training pass
            training_loss = self._do_training_epoch(epoch)
            all_data_plotters["training_epoch_loss"].add_value(training_loss)

            # Do the validation pass
            validation_loss = self._do_validation_epoch(epoch)
            all_data_plotters["validation_epoch_loss"].add_value(validation_loss)

            # Make all plotters write!
            for data_plotter in self.data_plotters:
                data_plotter.plot_and_save()

            # See if the validation loss is better
            if((best_validation_loss is None) or (validation_loss < best_validation_loss)):
                best_validation_loss = validation_loss
                is_best = True
            else:
                is_best = False

            # Save the models
            self.model_saver.save_models(epoch, is_best)


    def _do_training_epoch(self, epoch):
        
        # Freeze the models that we are not training. So if a model 
        # has a batchnorm layer, in eval mode the batchnorm will be frozen
        if("full_model" in self.models_to_train):
            assert(len(self.models_to_train) == 1)
            for model_name in self.all_models.keys():
                self.all_models[model_name].train()
        else:
            for model_name in self.all_models.keys():
                if(model_name in self.models_to_train):
                    self.all_models[model_name].train()
                else:
                    self.all_models[model_name].eval()



        # Keep track of stats needed to compute the average loss
        total_loss = 0
        number_of_losses_to_use_for_average_loss = 0

        # Go through all the data once
        t = tqdm(iter(self.train_loader), leave=False, total=len(self.train_loader))
        for step, data in enumerate(t):

            # Zero out the gradients in prep for optimization
            for model_name in self.all_models.keys():
                self.all_models[model_name].zero_grad()

            # Do the forward pass over the data
            loss, batch_size = self.do_forward_pass(data)

            # If the loss is not valid then move on
            if(loss is None):
                continue

            # Compute the gradient
            loss.backward()

            # Compute the gradient norm for the models and add it to the data plotters
            for model_name in self.all_models.keys():

                # We want L2 Norm
                norm_type = 2

                # Compute the gradient norm
                norm = [torch.norm(p.grad.detach(), norm_type) for p in self.models[model_name][0].parameters() if p.grad is not None]

                # if there is no norm then we cant compute the norm
                if(len(norm) == 0):
                    continue

                # Finish computing the norm 
                gradient_norm = torch.norm(torch.stack(norm) , norm_type)

                # Add it to the data plotter
                data_plotter_name = "gradient_norm_{}".format(model_name)
                self.all_data_plotters[data_plotter_name].add_value(gradient_norm.cpu().item())

            # Check if we are in a condition to take an optimization step and if so take the step
            if((((step+1) % self.accumulate_gradients_counter) == 0) or ((step+1) == len(self.train_loader))):

                # if we have a gradient clipping value then do the clipping
                if(self.gradient_clip_value is not None):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_value)

                # Take an optimization step
                for optimizer in self.optimizers:
                    optimizer.step()

            # Add the loss for the batch so we can do step losses
            self.all_data_plotters["training_iteration_loss"].add_value(loss.cpu().item())

            # keep track of the average loss
            total_loss += loss.item() * batch_size
            number_of_losses_to_use_for_average_loss += batch_size

        # Compute the average loss
        average_loss = float(total_loss) / float(number_of_losses_to_use_for_average_loss)

        return average_loss

    def _do_validation_epoch(self, epoch):

        # Dont need the gradients for the evaluation epochs
        with torch.no_grad():

            # Put all the models in evaluation mode
            for model_name in self.all_models.keys():
                self.all_models[model_name].eval()

            # Keep track of stats needed to compute the average loss
            total_loss = 0
            number_of_losses_to_use_for_average_loss = 0

            # Go through all the data once
            t = tqdm(iter(self.validation_loader), leave=False, total=len(self.validation_loader))
            for step, data in enumerate(t):

                # Do the forward pass over the data
                loss, batch_size = self.do_forward_pass(data)

                # If the loss is not valid then move on
                if(loss is None):
                    continue

                # Add the loss for the batch so we can do step losses
                self.all_data_plotters["validation_iteration_loss"].add_value(loss.cpu().item())

                # keep track of the average loss
                total_loss += loss.item() * batch_size
                number_of_losses_to_use_for_average_loss += batch_size

            # Compute the average loss
            average_loss = float(total_loss) / float(number_of_losses_to_use_for_average_loss)

            return average_loss


    def _create_data_loaders(self, batch_sizes, dataset, dataset_type):

        if(dataset is None):
            return None

        if(dataset_type not in batch_sizes):
            assert(False)

        # get the batch size
        batch_size = batch_sizes[dataset_type]


        # Check if the dataset has a custom collate function we should be using
        has_custom_collate_function = getattr(self, "get_collate_function", None)
        if callable(has_custom_collate_function):
            custom_collate_function = dataset.get_collate_function()
        else:
            custom_collate_function = None

        # Check if we should shuffle the data
        if(dataset_type == "training"):
            shuffle_data = True
        else:
            shuffle_data = False

        # Create the dataloader
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle_data, num_workers=self.num_cpu_cores_for_dataloader, pin_memory=True, persistent_workers=True, collate_fn=custom_collate_function)

        return dataloader


    def _create_optimizers(self, optimizer_configs, learning_rates):

        # All the optimizers we create
        all_optimizers = []

        # Keep track of the models that we are setting as learnable
        all_models_to_train = set()

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
                has_non_frozen_count == True
                break
        assert(has_non_frozen_count)


        # for each model make an optimizer
        for model_name in learning_rates.keys():
            
            # Make sure the model exists
            assert(model_name in self.all_models)

            # Extract the learning rate
            lr = learning_rates[model_name]

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
                    weight_decay = get_mandatory_config(weight_decay, optimizer_configs, "optimizer_configs")
                    optimizer = torch.optim.Adam(self.all_models[model_name].parameters(),lr=lr, weight_decay=weight_decay)

                elif(optimizer_to_use == "AdamW"):
                    weight_decay = get_mandatory_config(weight_decay, optimizer_configs, "optimizer_configs")
                    optimizer = torch.optim.AdamW(self.all_models[model_name].parameters(),lr=lr, weight_decay=weight_decay)

                elif(optimizer_to_use == "NAdam"):
                    weight_decay = get_mandatory_config(weight_decay, optimizer_configs, "optimizer_configs")
                    optimizer = torch.optim.NAdam(self.all_models[model_name].parameters(),lr=lr, weight_decay=weight_decay)

                elif(optimizer_to_use == "RMSProp"):
                    weight_decay = get_mandatory_config(weight_decay, optimizer_configs, "optimizer_configs")
                    momentum = get_mandatory_config(momentum, optimizer_configs, "optimizer_configs")
                    eps = get_mandatory_config(eps, optimizer_configs, "optimizer_configs")
                    alpha = get_mandatory_config(alpha, optimizer_configs, "optimizer_configs")
                    optimizer = torch.optim.RMSprop(self.all_models[model_name].parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum, eps=eps, alpha=alpha)

                elif(optimizer_to_use == "SGD"):
                    weight_decay = get_mandatory_config(weight_decay, optimizer_configs, "optimizer_configs")
                    momentum = get_mandatory_config(momentum, optimizer_configs, "optimizer_configs")
                    dampening = get_mandatory_config(dampening, optimizer_configs, "optimizer_configs")
                    optimizer = torch.optim.SGD(self.all_models[model_name].parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum, dampening=dampening)

                else:
                    print("Unknown optimizer type \"{}\"".format(optimizer_type))
                    exit()

                all_optimizers.append(optimizer)
                all_models_to_train.add(model_name)

        return all_optimizers, all_models_to_train


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
            all_data_plotters[data_plotter_name] = DataPlotter("Gradient L2 Norm for {} (Pre Clipping)".format(model_name), "Iteration", "Gradient L2 Norm", self.save_dir, "model_name.png")


        return all_data_plotters