from utils import categorical_to_img,ImageGenerator
from losses import custom_MeanIOU
from model import UNET,UNET_inception
from os.path import join
import matplotlib.pyplot as plt

unet_model = UNET_inception(input_shape=(None,None,3),num_layers=4,filters=32,num_classes=20)
unet_model.summary()

unet_model.load_weights('/home/sarthak/Desktop/DriveSceneSemanticSegmentation/logs/20210824-144652/checkpoint')

unet_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc',custom_MeanIOU(num_classes=20)])

test_gen = ImageGenerator(1, join('leftImg8bit', 'val'), join('gtFine', 'val'), 'leftImg8bit', 'gtFine_color',
                              None, 1,False)
# print(test_gen.list_paths_x)

for i in range(10):
    x_test, y_test = next(test_gen)
    arr = unet_model.predict(x_test)

    img = categorical_to_img(arr)

    img_ground = categorical_to_img(y_test)

    for i in range(img.shape[0]):
        plt.imshow(img[i]/255)
        plt.title('predicted')
        plt.show()
        plt.clf()
        plt.imshow(img_ground[i]/255)
        plt.title('ground truth')
        plt.show()
