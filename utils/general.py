
# Python Imports
import os

# Module imports 
import yaml

def print_dict_pretty(data_dict):
    print(yaml.dump(data_dict, allow_unicode=True, default_flow_style=False))
