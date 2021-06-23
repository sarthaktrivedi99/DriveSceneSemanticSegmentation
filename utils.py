import os
from os.path import join
from PIL import Image
import numpy as np
from typing import Tuple
from scipy.ndimage import rotate, zoom
import random


class ImageGenerator(object):

    def __init__(self, batch_size, path_to_x, path_to_y, match_string_x, match_string_y, aug_fn,
                 random_generation=True) -> object:
        """
        @param batch_size: Batch size
        @param path_to_x: Path to input images
        @param path_to_y: Path to segmented images
        @param match_string_x: String to match input image
        @param match_string_y: String to match segmented image
        @param aug_fn: Function to augment the images
        @param random_generation: Switch to generate random batches

        @return object
        """
        self.batch_size = batch_size
        self.match_string_x = match_string_x
        self.match_string_y = match_string_y
        self.path_x = path_to_x
        self.path_y = path_to_y
        self.aug_fn = aug_fn
        self.random_gen = random_generation
        self.curr_index = 0
        self.list_paths_x = self.get_paths(path_to_x, match_string_x)
        # self.list_paths_y = self.get_paths(path_to_y,match_string_y)

    def get_paths(self, path, match_string) -> list:
        """
        Returns list of paths to all the images in the dataset that can be loaded into memory by the generator.
        @rtype: list
        """
        list_paths = []
        for i in os.listdir(path):
            for j in os.listdir(str(join(path, i))):
                if (j.split('_')[-1] == match_string + '.png'):
                    list_paths.append(str(join(path, i, j)))

        return list_paths

    def __len__(self) -> int:
        """
        Returns the length of the generator
        @return: int
        """
        return len(self.list_paths) / self.batch_size

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns batch of images and their corresponding segmentation images
        @rtype: Tuple[np.ndarray,np.ndarray]
        """
        batch_x = []
        batch_y = []
        if (self.random_gen):
            idx = np.random.uniform(low=0, high=len(self.list_paths), size=(self.batch_size))
        else:
            idx = np.arange(start=self.curr_index, stop=self.curr_index + self.batch_size)
            self.curr_index += self.batch_size
        for i in idx:
            x_sample = np.asarray(Image.open(self.list_paths_x[i])) / 255
            y_path = self.get_path_y(self.list_paths_x[i])
            y_sample = np.asarray(Image.open(y_path)) / 255
            if random.choice([True, False]):
                x_sample, y_sample = self.aug_fn(x_sample, y_sample)
            batch_y.append(y_sample)
            batch_x.append(x_sample)
        return np.array(batch_x), np.array(batch_y)

    def get_path_y(self, path) -> str:
        """
        Returns path to label image for a given input image
        @param path: path to the input image
        @rtype: str
        """
        y_path = path.split('/')
        y_path[0] = self.path_y.split('/')[0]
        filename = y_path[-1].split('_')
        filename[-1] = self.match_string_y + '.png'
        y_path[-1] = '_'.join(filename)
        y_path = '/'.join(y_path)

        return y_path


def augmentation_fn(x, y, rotation=True, noise=True) -> Tuple[np.ndarray, np.ndarray]:
    """
    @param x: Input Image
    @param y: Segmentation Image
    @param rotation: Switch to enable rotation of image
    @param zoom_: Switch to enable Zoom
    @param noise: Switch to enable adding noise to Input Image
    @rtype: Tuple[np.ndarray,np.ndarray]
    """
    if rotation:
        ang = int(random.uniform(0, 360))
        x = rotate(x, angle=ang, reshape=False)
        y = rotate(y, angle=ang, reshape=False)
    if noise:
        noise = np.random.normal(loc=0.1, scale=0.01, size=x.shape)
        x = x + noise
    return x, y
