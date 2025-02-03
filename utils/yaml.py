

# Package Imports
import yaml

def load_yaml_file(file_path):
    '''
        Load the YAML file into a python dict

        Parameters:
            file_path: The file path to the YAML file

        Returns:
            The loaded dictionary
    '''

    # Read and parse the config file
    with open(file_path) as file:

        # Load the whole file into a dictionary and return
        return yaml.load(file, Loader=yaml.FullLoader)
