# coding=utf-8


import os
from twisted.protocols.ftp import FileNotFoundError

from tensorflow.python.platform import gfile


def load_content(filename):
    """
    return all content in filename
    :param filename:
    :return:
    """
    with gfile.Open(filename, 'rb') as gf:
        lines = gf.readlines()
        return lines


def maybe_download(filename, work_directory):
    """Download the data from source url, unless it's already here.

    Args:
        filename: string, name of the file in the directory.
        work_directory: string, path to working directory.
    Returns:
        Path to resulting file.
    """
    if not gfile.Exists(work_directory):
        gfile.MakeDirs(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not gfile.Exists(filepath):
        raise FileNotFoundError('%s not found.' % filepath)
    return filepath
