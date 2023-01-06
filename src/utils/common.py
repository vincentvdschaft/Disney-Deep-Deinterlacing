import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re


def remove_axis(ax):
    """
    Removes xticks, yticks, and axis line. Can be used for cleanly ploting an image.
    If ax is an ndarray of axes object the function calls itself recursively to
    remove the axis from every element.
    Parameters
    ----------
    ax : matplotlib axes object
        The axes to remove from
    """
    # Call the method recursively if ax is an array of axes
    if isinstance(ax, np.ndarray):
        for n in range(ax.shape[0]):
            remove_axis(ax[n])
        return

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Remove axes themselves
    plt.setp(ax.spines.values(), visible=False)


def create_unique_dir(parent_directory, name):
    """Creates a new directory with a unique id-number in the name.

    Args:
        parent_directory (str, Path): The directory in which the file should
        be created name (str, Path): The desired directory name

    Returns:
        The path (:obj:`Path`) of the newly created file
    """
    # Create any parent directories if necessary
    Path(parent_directory).mkdir(parents=True, exist_ok=True)
    # Find the new filename
    file_path = get_unique_filename(parent_directory, name)
    # Create the directory
    file_path.mkdir(parents=True)

    return file_path


def get_unique_filename(parent_directory, name, extension=''):
    """Finds a unique file with a unique id-number in the name. If no files
    are present the id number will be 0. If there are files present the id
    number will be one larger than the largest one present.

    Args:
        parent_directory (str, Path): The directory in which the file should be
        created name (str, Path): The desired filename including the file
        extension

    Returns:
        The path (:obj:`Path`) of the newly created file
    """
    stem = Path(name).stem

    glob_pattern = name + 6 * '[0-9]' + extension
    # Find existing files matching name
    existing = list(Path(parent_directory).glob(glob_pattern))

    if len(existing) == 0:
        id = 0
    else:
        highest_id = int(re.search('\\d+', str(existing[-1])).group(0))
        id = highest_id + 1

    return Path(parent_directory, stem + str(id).zfill(6) + extension)
