import os
from os.path import join
from PIL import Image
import numpy as np
from typing import Tuple
from scipy.ndimage import rotate
import random
import psutil
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
# from numba import jit

class ImageGenerator(object):

    def __init__(self, batch_size, path_to_x, path_to_y, match_string_x, match_string_y, aug_fn, splits,
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
        self.splits = splits
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
    def __iter__(self):
        return self

    def __len__(self) -> int:
        """
        Returns the length of the generator
        @return: int
        """
        return (len(self.list_paths_x)) // self.batch_size

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns batch of images and their corresponding segmentation images
        @rtype: Tuple[np.ndarray,np.ndarray]
        """
        batch_x = []
        batch_y = []
        if (self.random_gen):
            idx = np.random.uniform(low=0, high=len(self.list_paths_x), size=(self.batch_size))
        else:
            if self.curr_index==self.__len__():
                self.curr_index=0
            idx = np.arange(start=self.curr_index, stop=self.curr_index + self.batch_size)
            self.curr_index += self.batch_size
        for i in idx:
            x_sample = np.asarray(Image.open(self.list_paths_x[i]).convert('RGB')) / 255
            y_path = self.get_path_y(self.list_paths_x[i])
            y_sample = np.asarray(Image.open(y_path).convert('RGB'))
            if random.choice([True, False]):
                x_sample, y_sample = self.aug_fn(x_sample, y_sample)
            y_sample = mask_to_arr(y_sample)
            if self.splits>1:
                x_sample = split(x_sample,self.splits)
                y_sample = split(y_sample,self.splits)
                for i in range(self.splits):
                    batch_x.append(x_sample[i])
                    batch_y.append(y_sample[i])
            else:
                batch_y.append(y_sample)
                batch_x.append(x_sample)
        return np.asarray(batch_x), np.asarray(batch_y)


    def get_path_y(self,path) -> str:
        """
        Returns path to label image for a given input image
        @param path: path to the input image
        @rtype: str
        """
        if psutil.WINDOWS:
            path_split = '\\'
        else:
            path_split = '/'

        y_path = path.split(path_split)
        y_path[0] = self.path_y.split(path_split)[0]
        filename = y_path[-1].split('_')
        filename[-1] = self.match_string_y + '.png'
        y_path[-1] = '_'.join(filename)
        y_path = path_split.join(y_path)
        return y_path

# Labeling from original Cityscapes Scripts
# def get_label():
#     Label = namedtuple('Label', [
#         'name',
#         'id',
#         'trainId',
#         'category',
#         'categoryId',
#         'hasInstances',
#         'ignoreInEval',
#         'color',
#     ])
#
#
#     labels = [
#         #Label('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
#         #Label('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
#         #Label('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
#         #Label('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
#         #Label('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
#         #Label('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
#         #Label('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
#         Label('road', 1, 0, 'flat', 1, False, False, (128, 64, 128)),
#         Label('sidewalk', 2, 1, 'flat', 1, False, False, (244, 35, 232)),
#         #Label('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
#         #Label('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
#         Label('building', 3, 2, 'construction', 2, False, False, (70, 70, 70)),
#         Label('wall', 4, 3, 'construction', 2, False, False, (102, 102, 156)),
#         Label('fence', 5, 4, 'construction', 2, False, False, (190, 153, 153)),
#         # Label('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
#         # Label('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
#         # Label('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
#         Label('pole', 6, 5, 'object', 3, False, False, (153, 153, 153)),
#         # Label('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
#         Label('traffic light', 7, 6, 'object', 3, False, False, (250, 170, 30)),
#         Label('traffic sign', 8, 7, 'object', 3, False, False, (220, 220, 0)),
#         Label('vegetation', 9, 8, 'nature', 4, False, False, (107, 142, 35)),
#         Label('terrain', 10, 9, 'nature', 4, False, False, (152, 251, 152)),
#         Label('sky', 11, 10, 'sky', 5, False, False, (70, 130, 180)),
#         Label('person', 12, 11, 'human', 6, True, False, (220, 20, 60)),
#         Label('rider', 13, 12, 'human', 6, True, False, (255, 0, 0)),
#         Label('car', 14, 13, 'vehicle', 7, True, False, (0, 0, 142)),
#         Label('truck', 15, 14, 'vehicle', 7, True, False, (0, 0, 70)),
#         Label('bus', 16, 15, 'vehicle', 7, True, False, (0, 60, 100)),
#         # Label('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
#         # Label('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
#         Label('train', 17, 16, 'vehicle', 7, True, False, (0, 80, 100)),
#         Label('motorcycle', 18, 17, 'vehicle', 7, True, False, (0, 0, 230)),
#         Label('bicycle', 19, 18, 'vehicle', 7, True, False, (119, 11, 32)),
#         # Label('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
#     ]
#     return labels

def split(image,splits):
    patches = []
    # print(image.shape)
    patch_h = image.shape[0] //(splits+1)
    patch_w = image.shape[1] //(splits+1)
    for i in range(splits):
        # print(i*patch_h)
        patches.append(image[i*patch_h:(i+1)*patch_h,i*patch_w:(i+1)*patch_w,:])
    return  patches
def get_label() -> dict:
    label_dict = {
        (0,0,0):0,
        (128, 64, 128):1,
        (244, 35, 232):2,
        (70, 70, 70):3,
        (102, 102, 156):4,
        (190, 153, 153):5,
        (153, 153, 153):6,
        (250, 170, 30):7,
        (220, 220, 0):8,
        (107, 142, 35):9,
        (152, 251, 152):10,
        (70, 130, 180):11,
        (220, 20, 60):12,
        (255, 0, 0):13,
        (0, 0, 142):14,
        (0, 0, 70):15,
        (0, 60, 100):16,
        (0, 80, 100):17,
        (0, 0, 230):18,
        (119, 11, 32):19

    }
    label_dict_exception = defaultdict(lambda: 0,label_dict)
    return label_dict_exception


def get_color() -> dict:
    label_dict = get_label()
    color_dict = {value: key for (key, value) in label_dict.items()}
    return  color_dict


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


def arr_to_categorical(image):
    # cp = np.zeros()
    # print(image.dtype)

    cp = np.eye(len(get_label()))[image.astype(int)]
    # for i in range(image.shape[0]):
    #     for j in range(image.shape[1]):
    #         cp[i,j,int(image[i,j])] = 1
    return cp.reshape((image.shape[0],image.shape[1],len(get_label().items())))


def mask_to_arr(image):
    label_dict = get_label()
    arr = np.zeros((image.shape[0],image.shape[1],1))
    for i,j in label_dict.items():
        x,y= np.where(np.sum(image,axis=-1)==np.sum(i))[:2]
        arr[x,y] = j
    # x,y = np.where(np.argmax(arr,axis=-1)==0)
    # z = np.zeros_like(x)
    # arr[x,y,z] = 1
    # for i in range(image.shape[0]):
    #     for j in range(image.shape[1]):
    #         try:
    #             arr[i,j,label_dict[tuple(image[i,j,:])]-1] = 1
    #         except:
    #             continue
    return arr_to_categorical(arr)

if __name__ == '__main__':
    test_gen = ImageGenerator(1, join('leftImg8bit', 'train'), join('gtFine', 'train'), 'leftImg8bit', 'gtFine_color',
                              augmentation_fn, 3, False)
    for epoch in range(2):
        for i in tqdm(range(len(test_gen))):
            images, labels = next(test_gen)
    # plt.imshow(labels[1,:,:,14])
    # plt.show()
