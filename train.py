from model import UNET
from utils import ImageGenerator,augmentation_fn
from os.path import join

unet_model = UNET(input_shape=(None,None,3),num_layers=3)
unet_model.summary()
unet_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])

train_gen = ImageGenerator(1, join('leftImg8bit', 'train'), join('gtFine', 'train'), 'leftImg8bit', 'gtFine_color',
                              augmentation_fn, False)
val_gen = ImageGenerator(1, join('leftImg8bit', 'val'), join('gtFine', 'val'), 'leftImg8bit', 'gtFine_color',
                              augmentation_fn, False)

unet_model.fit(train_gen,steps_per_epoch=len(train_gen),epochs=10,validation_data=val_gen,validation_steps=len(val_gen))