import os
import datetime

def _create_res_dir(postfix):
    """
    Create a directory for figures if it doesn't exist.
    Returns: str: The path to the directory.
    """
    dir_path = os.path.join(os.getcwd(), 'figs', 'TB', postfix)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

def generate_file_postfix(format='%m-%d_%H-%M-%S'):
    """ Generate a postfix for file names based on the current date and time.
        format:  Format needed to generate the filename friendly date and time string.
        Returns: str: The generated postfix.  
    """
    return datetime.datetime.now().strftime(format)

# Postfix for the figures and CSV files.
FILE_POSTFIX = generate_file_postfix()

# Path to the directory where the files will be saved.
RESULTS_DIRECTORY = _create_res_dir(FILE_POSTFIX)
