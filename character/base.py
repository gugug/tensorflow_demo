# coding=utf-8
# ==============================================================================

"""Base utilities for loading datasets."""

import collections
import csv
import os
from os import path
import tempfile
import numpy as np

from tensorflow.python.platform import gfile
from twisted.protocols.ftp import FileNotFoundError

Dataset = collections.namedtuple('Dataset', ['data', 'target'])
Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])


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


if __name__ == '__main__':
    base_path = '/media/gu/493ce6af-560c-45cb-9cc4-66119285f579/gu/PycharmProjects/character_data/'
    # extract_labels(os.path.join(base_path,'user_character_map'))
