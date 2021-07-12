from model import UNET
from utils import ImageGenerator,augmentation_fn
from os.path import join
from datetime import datetime
import tensorflow as tf


unet_model = UNET(input_shape=(None,None,3),num_layers=3,filters=8,num_classes=20)
unet_model.summary()
logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
callbacks = [tf.keras.callbacks.TensorBoard(log_dir=logdir),tf.keras.callbacks.ModelCheckpoint(
    filepath=join(logdir,'checkpoint'),
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)]
unet_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc',tf.keras.metrics.MeanIoU(num_classes=20)])

train_gen = ImageGenerator(1, join('leftImg8bit', 'train'), join('gtFine', 'train'), 'leftImg8bit', 'gtFine_color',
                              augmentation_fn, 3,False)
val_gen = ImageGenerator(1, join('leftImg8bit', 'val'), join('gtFine', 'val'), 'leftImg8bit', 'gtFine_color',
                              augmentation_fn, 3,False)

unet_model.fit(train_gen,steps_per_epoch=len(train_gen),epochs=10,validation_data=val_gen,validation_steps=len(val_gen),callbacks=callbacks)