
# Python Imports
import csv

# Package Imports
import torch
from prettytable import PrettyTable


# Ali Package Import

# Project Imports


class MetricPrettyPrinter:
    def __init__(self, logger, save_dir):

        # Save for later
        self.logger = logger
        self.save_dir = save_dir



    def print_metrics(self, metrics):
        
        # Get all the metric dicts
        all_metric_data = dict()
        for metric_name in metrics.keys():
            all_metric_data[metric_name] = metrics[metric_name].get_aggregated_result()

        # Get a set of all unique metric sub-names
        unique_metric_subnames = set()
        for metric_name in all_metric_data.keys():
            for metric_subname in all_metric_data[metric_name].keys():
                unique_metric_subnames.add(metric_subname)

        # Convert to a list and sort it
        unique_metric_subnames = list(unique_metric_subnames)
        unique_metric_subnames.sort()

        # make a dict mapping the subname to where it is in the row
        subname_index_mapping = {unique_metric_subnames[i]:i for i in range(len(unique_metric_subnames))}

        # Figure out the row length
        # This is the number of metric subnames + 1 (for the name of the metric)
        row_length = len(unique_metric_subnames) + 1

        # Put them into rows
        rows = []
        for metric_name in all_metric_data.keys():

            # Make the blank row
            row = ["" for i in range(row_length)]

            # Add the metric name
            row[0] = metric_name

            # Add the metrics aggregated values
            metric_data = all_metric_data[metric_name]
            for metric_subname in metric_data.keys():
                subname_idx = subname_index_mapping[metric_subname] + 1
                row[subname_idx] = "{:0.4f}".format(metric_data[metric_subname].item())

            # Add the row to all the rows
            rows.append(row)

        # Print the data
        self._print_table(rows, unique_metric_subnames)
        self._print_csv(rows, unique_metric_subnames)

    def _print_table(self, rows, unique_metric_subnames):

        # Create the table
        table = PrettyTable()
        table.field_names = ["metric"] + unique_metric_subnames

        # Add all the rows
        for row in rows:
            table.add_row(row)

        # Log it
        self.logger.log("\n\n")
        self.logger.log("==================================================================")
        self.logger.log("Quantitative Results:")
        self.logger.log("==================================================================")
        self.logger.log(str(table))
        self.logger.log("\n\n")


    def _print_csv(self, rows, unique_metric_subnames):

        # Create the header
        header = ["metric"] + unique_metric_subnames

        # Save to the file 
        csv_output_file = "{}/metrics.csv".format(self.save_dir)
        with open(csv_output_file, 'w') as f:

            # Create the CSV write
            writer = csv.writer(f)

            # write the header
            writer.writerow(header)

            # write all the rows
            writer.writerows(rows)

