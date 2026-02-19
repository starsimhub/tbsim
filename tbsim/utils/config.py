import os
import datetime

def create_res_dir(base=None, postfix=None, append_date=True):
    """
    Create a directory for figures if it doesn't exist.
    Returns: str: The path to the directory.
    """
    if base == None:
        base = os.getcwd()

    date = None
    if append_date:
        format = '%m-%d_%H-%M'
        date = datetime.datetime.now().strftime(format)

    paths = (x for x in [base, 'results', postfix, date] if x is not None)

    dir_path = os.path.join(*paths)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path
