
import os
from os.path import join
from PIL import Image
import numpy as np

class ImageGenerator(object):

    def __init__(self,batch_size,path_to_x,path_to_y,match_string_x,match_string_y,aug_fn,random_generation=True) -> object:
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
        self.list_paths_x = self.get_paths(path_to_x,match_string_x)
        # self.list_paths_y = self.get_paths(path_to_y,match_string_y)

    def get_paths(self,path,match_string) -> list:
        """
        Returns list of paths to all the images in the dataset that can be loaded into memory by the generator.
        @rtype: list
        """
        list_paths = []
        for i in os.listdir(path):
            for j in os.listdir(os.listdir(join(path,i))):
                if (j.split('_')[-1]==match_string+'.png'):
                    list_paths.append(str(join(path,i,j)))

        return list_paths


    def __len__(self) -> int:
        """
        Returns the length of the generator
        @return: int
        """
        return len(self.list_paths)/self.batch_size

    def __next__(self) -> tuple[np.ndarray,np.ndarray]:
        """
        Returns batch of images and their corresponding segmentation images
        @rtype: tuple[np.ndarray,np.ndarray]
        """
        batch_x = []
        batch_y = []
        if(self.random_gen):
            idx = np.random.uniform(low=0,high=len(self.list_paths),size=(self.batch_size))
        else:
            idx = np.arange(start=self.curr_index,stop=self.curr_index+self.batch_size)
            self.curr_index+= self.batch_size
        for i in idx:
            batch_x.append(Image.open(self.list_paths_x[i]))
            y_path = self.get_path_y(self.list_paths_x[i])
            batch_y.append(Image.open(y_path))
        return np.array(batch_x),np.array(batch_y)

    def get_path_y(self,path) -> str:
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
        y_path = y_path.join('/')

        return y_path
