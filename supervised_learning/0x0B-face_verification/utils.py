#!/usa/bin/env python3
"""Load Images """
import glob
import numpy as np
import csv
import cv2
import os


def load_images(images_path, as_array=True):
    """
        images_path is the path to a directory from
                which to load images
        as_array is a boolean indicating whether
                the images should be loaded as one numpy.ndarray
                If True, the images should be loaded as a numpy.ndarray
                        of shape (m, h, w, c) where:
                m is the number of images
                h, w, and c are the height, width, and number of channels
                        of all images, respectively
                If False, the images should be loaded as a list of1
                individual numpy.ndarrays

        All images should be loaded in RGB format
        The images should be loaded in alphabetical order by filename
        Returns: images, filenames
        images is either a list/numpy.ndarray of all images
        filenames is a list of the filenames associated with each image
            in images
    """
    image_paths = glob.glob(images_path + "/*")
    images_names = [path.split('/')[-1] for path in image_paths]
    idx = np.argsort(images_names)
    images_prev = [cv2.imread(img) for img in image_paths]
    images_prev = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images_prev]
    images = []
    filenames = []
    for i in idx:
        images.append(images_prev[i])
        filenames.append(images_names[i])

    if as_array:
        images = np.stack(images, axis=0)

    return images, filenames


def load_csv(csv_path, params={}):
    """ that loads the contents of a csv file as a list of lists:

    csv_path is the path to the csv to load
    params are the parameters to load the csv with
    Returns: a list of lists representing the contents found in csv_path
    """
    csv_list = []
    with open(csv_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, params)
        for row in csv_reader:
            csv_list.append(row)

    return csv_list

def save_images(path, images, filenames):
    """ saves images to a specific path:

        - path is the path to the directory in which the images
            should be saved
        - images is a list/numpy.ndarray of images to save
        - filenames is a list of filenames of the images to save

        Returns: True on success and False on failure
    """
    try:
    	os.chdir(path)
    	for name, img in zip(filenames, images):
        	cv2.imwrite(name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    	os.chdir('../')
    	return True

    except FileNotFoundError:
        return False

def generate_triplets(images, filenames, triplet_names):
    """ generates triplets:

        - images is a numpy.ndarray of shape (n, h, w, 3) containing
            the various images in the dataset
        - filenames is a list of length n containing the corresponding
            filenames for images
        - triplet_names is a list of lists where each sublist contains
            the filenames of an anchor, positive, and negative image,
            respectively

    Returns: a list [A, P, N]

        - A is a numpy.ndarray of shape (m, h, w, 3) containing the
            anchor images for all m triplets
        - P is a numpy.ndarray of shape (m, h, w, 3) containing the
            positive images for all m triplets
        - N is a numpy.ndarray of shape (m, h, w, 3) containing the
            negative images for all m triplet
    """

    filenames_clean = [name.split('.')[0] for name in filenames]
    for i in range(len(triplet_names)):
        a, p, n = triplet_names[i]
        idx_a = filenames_clean.index(a)
        idx_p = filenames_clean.index(p)
        idx_n = filenames_clean.index(n)

        np_a_tmp = images[idx_a][np.newaxis, ...]
        np_p_tmp = images[idx_p][np.newaxis, ...]
        np_n_tmp = images[idx_n][np.newaxis, ...]
        if i != 0:
            np_a = np.concatenate([np_a, np_a_tmp], axis=0)
            np_p = np.concatenate([np_p, np_p_tmp], axis=0)
            np_n = np.concatenate([np_n, np_n_tmp], axis=0)
        else:
            np_a = np_a_tmp
            np_p = np_p_tmp
            np_n = np_n_tmp

    return [np_a, np_p, np_n]
