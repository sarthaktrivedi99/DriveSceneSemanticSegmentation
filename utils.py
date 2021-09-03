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
import cv2
from tensorflow.keras import backend as K
import tensorflow as tf
from skimage.filters.edges import sobel

class ImageGenerator(object):

    def __init__(self, batch_size, path_to_x, path_to_y, match_string_x, match_string_y, aug_fn, splits,
                 random_generation=True,random_crop=None,resize_toggle=True) -> object:
        """
        Generator class for generating inout,label pairs
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
        # self.splits = splits
        self.resize_toggle = resize_toggle
        self.random_crop = random_crop
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
        batch_bmap = []
        if (self.random_gen):
            idx = np.random.uniform(low=0, high=len(self.list_paths_x), size=(self.batch_size)).astype(int)
        else:
            if self.curr_index==self.__len__():
                self.curr_index=0
            idx = np.arange(start=self.curr_index, stop=self.curr_index + self.batch_size)
            self.curr_index += self.batch_size
        for i in idx:
            x_sample = np.asarray(Image.open(self.list_paths_x[i]).convert('RGB')) / 255
            y_path = self.get_path_y(self.list_paths_x[i])
            y_sample = np.asarray(Image.open(y_path).convert('RGB'))
            if self.resize_toggle==True:
                x_sample,y_sample = self.resize(x_sample,y_sample)
            if self.aug_fn!=None:
                x_sample, y_sample = self.aug_fn(x_sample, y_sample,crop_size=self.random_crop)
            y_sample = mask_to_arr(y_sample)
            # if self.bmap:
            #     bmap_sample = self.seg2bmap(y_sample)
            #     bmap_sample[bmap_sample>0] = 1
            # if self.splits>1:
            #     x_sample = split(x_sample,self.splits)
            #     y_sample = split(y_sample,self.splits)
            #     for i in range(self.splits):
            #         batch_x.append(x_sample[i])
            #         batch_y.append(y_sample[i])
            # else:
            # if self.bmap:
            #     batch_y.append(y_sample)
            #     batch_x.append(x_sample)
            #     batch_bmap.append(bmap_sample)

            batch_y.append(y_sample)
            batch_x.append(x_sample)
        return np.asarray(batch_x), np.asarray(batch_y)

    # def seg2bmap(self,seg, width=None, height=None):
    #  sobel_arr = []
    #  for i in range(seg.shape[-1]):
    #     sobel_arr.append(sobel(seg[...,i]))
    #  return np.asarray(np.sum(sobel_arr,axis=0))

    def resize(self,x,y,shape=(1024,512))-> Tuple[np.ndarray,np.ndarray]:
        """
        Resize function to reduce the dimension of the image,label pair
        Enables training on smaller GPUs
        @param x: Image
        @param y: Label
        @param shape: Desired output shape tupple
        @return: resized image,label pair
        """
        x = cv2.resize(x,shape,interpolation=cv2.INTER_NEAREST)
        y = cv2.resize(y,shape,interpolation=cv2.INTER_NEAREST)
        return x,y

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

def get_label() -> dict:
    """
    Returns label dictonary for specified RGB values
    @return: label dictionary
    """
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
    """
    Returns inverted label dictionary
    @return: inverted label dictionary
    """
    label_dict = get_label()
    color_dict = {value: key for (key, value) in label_dict.items()}
    return  color_dict


def augmentation_fn(x, y, crop_size) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augments the image,label pair
    This augmentation helps in learning more robust features
    @param x: Input Image
    @param y: Segmentation Image
    @rtype: Tuple[np.ndarray,np.ndarray]
    """
    h,w = x.shape[0],x.shape[1]
    # Random cropping
    new_h,new_w = int(h*crop_size),int(w*crop_size)
    start_h,start_w = int(np.random.uniform(0,h-new_h)), int(np.random.uniform(0,w-new_w))
    x_new,y_new = x[start_h:(start_h+new_h),start_w:(start_w+new_w),:], y[start_h:(start_h+new_h),start_w:(start_w+new_w),:]
    i = np.random.choice([0,90,180,270])
    x_new = rotate(x_new,i)
    y_new = rotate(y_new,i)
    i = np.random.choice([True,False])
    if i:
        x_new = np.fliplr(x_new)
        y_new = np.fliplr(y_new)
    return x_new, y_new


def arr_to_categorical(image):
    # cp = np.zeros()
    # print(image.dtype)

    cp = np.eye(len(get_label()))[image.astype(int)]
    # for i in range(image.shape[0]):
    #     for j in range(image.shape[1]):
    #         cp[i,j,int(image[i,j])] = 1
    return cp.reshape((image.shape[0],image.shape[1],len(get_label().items())))


def mask_to_arr(image)-> np.ndarray:
    """
    Converts a RGB image to array containing class for each pixel in the RGB image.
    @param image: RGB image
    @return: categorical array
    """
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

def categorical_to_img(arr)-> np.ndarray:
    """
    Converts categorical array to image
    input: softmax array
    output: image
    @param arr: input softmax image from the model
    @return: RGB image
    """
    s,h,w = arr.shape[:-1]
    argmax_arr = np.argmax(arr,axis=-1)
    img = np.zeros(shape=(s,h,w,3))
    color_dict = get_color()
    for i in range(s):
        for j in color_dict.keys():
            img[i][argmax_arr[i]==j] = color_dict[j]
    return img


# if __name__ == '__main__':
#     test_gen = ImageGenerator(1, join('leftImg8bit', 'train'), join('gtFine', 'train'), 'leftImg8bit', 'gtFine_color',
#                               augmentation_fn, 1, True,random_crop=0.75,bmap=True)
#
#     images, labels= next(test_gen)
#     # arr = mask_to_arr(labels)
#     img = categorical_to_img(labels['output'])
#     # print(img[0])
#     for i in range(img.shape[0]):
#         plt.imshow(img[i]/255)
#         plt.show()
#         plt.imshow(labels['edge'][i,:,:],cmap='gray')
#         plt.show()
#         plt.clf()
#         # plt.imshow(images[i])
#         # plt.show()
#         # plt.clf()
#         # plt.imshow(labels[0,:,:,1])
#         # plt.show()
#     # print(images.shape)
#     # for epoch in range(2):
#     #     for i in tqdm(range(len(test_gen))):
#     #         images, labels = next(test_gen)
#
#     # plt.imshow(labels[1,:,:,14])
#     # plt.show()