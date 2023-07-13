

# Python Imports

# Package Imports
import torch
import numpy as np
import matplotlib.pyplot as plt

# Project Imports
from ..utils import *


class DataPlotter:
    def __init__(self, title, x_axis_label, y_axis_label, save_dir, filename, moving_average_length=1, plot_modulo=1000, save_raw_data=False):

        # Save all the info we need to keep around
        self.title = title
        self.x_axis_label = x_axis_label
        self.y_axis_label = y_axis_label
        self.save_dir = save_dir
        self.filename = filename
        self.moving_average_length = moving_average_length
        self.plot_modulo = plot_modulo
        self.save_raw_data = save_raw_data

        # The data structs that we will use to keep track of the data
        self.raw_values = []
        self.averaged_values = []
        self.moving_aveage_buffer = []

        # Make sure the directory we are going to save into exists
        ensure_directory_exists(self.save_dir)

        # How many times we added before we plotted
        self.count_since_last_save = 0


    def add_value(self, value):

        # Keep track of the raw value 
        self.raw_values.append(value)

        # Add the values to the average
        self._add_value_to_average(value)

        # Added a value
        self.count_since_last_save += 1

        # Check if we should plot on this iteration
        if(self.count_since_last_save >= self.plot_modulo):
            self.plot_and_save()
            

    def _add_value_to_average(self, value):

        # Add the value to the averaging buffer and make sure that the averaging
        # buffer size is constant
        self.moving_aveage_buffer.append(value)

        # If we dont have enough samples then we dont have enough to add to average 
        if(len(self.moving_aveage_buffer) < self.moving_average_length):
            return

        # If we have too many samples then we need to pop some until we have the right number of samples
        while(len(self.moving_aveage_buffer) > self.moving_average_length):
            self.moving_aveage_buffer.pop(0)

        # compute and add the average value
        avg = sum(self.moving_aveage_buffer) / float(self.moving_average_length)
        self.averaged_values.append(avg)

    def plot_and_save(self):

        # We are plotting so clear this
        self.count_since_last_save = 0

        # check if there is something to plot 
        if((len(self.raw_values) == 0) or (len(self.averaged_values) == 0)):
            return


        # check if there is something to plot 
        if((len(dp_obj.raw_values) == 0) or (len(dp_obj.averaged_values) == 0)):
            return

        # Plot the losses.  This overrides the previous plots
        fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=False, figsize=(12, 6))
        ax1 = axes[0,0]

        # Plot the Training loss
        ax1.plot(dp_obj.averaged_values, marker="o", color="red")

        # Add labels every so often so we can read the data
        for i in range(0, len(dp_obj.averaged_values), 10):
            ax1.text(i, dp_obj.averaged_values[i], "{:.2f}".format(dp_obj.averaged_values[i]))

        # Plot the trend line if we have enough data
        if(len(dp_obj.averaged_values) > 5):
            x = np.arange(0, len(dp_obj.averaged_values), 1)
            y = np.asarray(dp_obj.averaged_values)
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax1.plot(x,p(x),"b--")


        ax1.set_xlabel(dp_obj.x_axis_label)
        ax1.set_ylabel(dp_obj.y_axis_label)
        ax1.set_title(dp_obj.title)
        # ax1.legend()
        ax1.get_yaxis().get_major_formatter().set_scientific(False)


        # Compute Stats to put in the table
        raw_values_array = np.asarray(dp_obj.raw_values)
        mean = np.mean(raw_values_array)
        std = np.std(raw_values_array)
        median = np.median(raw_values_array)
        max_value = np.max(raw_values_array)
        min_value = np.min(raw_values_array)
        number_of_maxs = np.sum(np.abs(raw_values_array - max_value) < 1e-3)
        percent_of_maxs = float(number_of_maxs) / raw_values_array.shape[0]

        q75, q25 = np.percentile(raw_values_array, [75.0 ,25.0])
        iqr = q75 - q25


        columns = ["Stat", "Value"]
        cell_text = list()
        cell_text.append(["Mean", mean])
        cell_text.append(["STD", std])
        cell_text.append(["Median", median])
        cell_text.append(["IQR", iqr])
        cell_text.append(["Max Value", max_value])
        cell_text.append(["Min Value", min_value])
        cell_text.append(["Num Max Values", number_of_maxs])
        cell_text.append(["Percent Max Values", percent_of_maxs])
        cell_text.append(["Total Steps", raw_values_array.shape[0]])

        # ax1.table(cellText=cell_text, colLabels=columns, loc='bottom', bbox=(1.1, .2, 0.5, 0.5))
        ax1.table(cellText=cell_text, colLabels=columns, bbox=(1.1, .2, 0.5, 0.5))

        fig.tight_layout()


        # Save into the save dir, overriding any previously saved plots
        plt.savefig("{}/{}".format(dp_obj.save_dir, dp_obj.filename))

        # Close the figure when we are done to stop matplotlub from complaining
        plt.close('all')


        # If we should save the raw data then lets save it
        if(dp_obj.save_raw_data):

            # Convert to a torch tensor for saving
            raw_values_torch = torch.FloatTensor(dp_obj.raw_values)

            # Save it
            torch.save(raw_values_torch, "{}/{}.pt".format(dp_obj.save_dir, dp_obj.filename))


