
# Python Imports
import os


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
