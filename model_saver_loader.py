
# Python Imports

# Package Imports
import torch

# Project Imports
from .utils import *


class ModelSaverLoader:
	def __init__(self, models_dict, save_dir):
			
		# Save for later
		self.models_dict = models_dict
		self.save_dir = save_dir

	def save_models(self, epoch, is_best):

		# Save the models
		model_save_dir = "{}/models/epoch_{:05d}/".format(self.save_dir, epoch)
		self._save_models_in_dir(model_save_dir)

		# Save the models if its the best
		if(is_best):
			model_save_dir = "{}/models/best/".format(self.save_dir, epoch)
			self._save_models_in_dir(model_save_dir)


	def _save_models_in_dir(self, directory):

		# Make sure the directory exists
		ensure_directory_exists(directory)

		# Save the models state dicts
		for model_name in self.models_dict.keys():

			# Create the model save file
			model_save_filepath = "{}/{}.pt".format(directory, model_name)

			# Save the state dict
			torch.save(self.models_dict[model_name].state_dict(), model_save_filepath)	