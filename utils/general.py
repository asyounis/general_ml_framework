
# Python Imports
import os
import zipfile

# Module imports 
import yaml


def ensure_directory_exists(directory):
    '''
        Makes sure a directory exists.  If it does not exist then the directory is created

        Parameters:
            directory: The directory that needs to exist

        Returns:
            None
    '''
    if(not os.path.exists(directory)):
        os.makedirs(directory)



def print_dict_pretty(data_dict):
    print(yaml.dump(data_dict, allow_unicode=True, default_flow_style=False))


def compress_file(file_path, delete_file_after_compression=False, compression_level=9):

    # Make sure the compression level is correct
    assert((compression_level >= 0) and (compression_level <= 9))
    
    # Make sure the file exists
    assert(os.path.exists(file_path))

    # Create the zip file name
    zip_file_name = "{}.zip".format(file_path)
    
    # Get the name of the file to put in the zip file
    # this will prevent the directory structure from being created
    arcname = os.path.split(file_path)[1]

    # Zip it
    with zipfile.ZipFile(zip_file_name, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=compression_level) as archive:
        archive.write(file_path, arcname)

    # Delete the origional file 
    if(delete_file_after_compression):
        os.remove(file_path)




def uncompress_file(file_path, destination_dir=None):
    
    # Make sure the file exists
    assert(os.path.exists(file_path))

    # If the destination dir was not specified then we just extract in place
    if(destination_dir is None):
        destination_dir = os.path.split(file_path)[0]

    # Make sure the destination directory exists
    ensure_directory_exists(destination_dir)
    
    with zipfile.ZipFile(file_path, 'r') as archive:
        archive.extractall(destination_dir)
