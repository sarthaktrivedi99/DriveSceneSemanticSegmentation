from model import UNET
from utils import ImageGenerator,augmentation_fn
from os.path import join
import tensorflow as tf

unet_model = UNET(input_shape=(1024,2048,3),num_layers=3,filters=8,num_classes=20)
unet_model.summary()
unet_model.compile(optimizer='adam',loss=tf.keras.losses.CategoricalCrossentropy(),metrics=['acc'])

train_gen = ImageGenerator(3, join('leftImg8bit', 'train'), join('gtFine', 'train'), 'leftImg8bit', 'gtFine_color',
                              augmentation_fn, False)
val_gen = ImageGenerator(3, join('leftImg8bit', 'val'), join('gtFine', 'val'), 'leftImg8bit', 'gtFine_color',
                              augmentation_fn, False)

unet_model.fit(train_gen,steps_per_epoch=len(train_gen),epochs=10,validation_data=val_gen,validation_steps=len(val_gen))
