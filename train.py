from model import UNET,UNET_inception
from utils import ImageGenerator,augmentation_fn
from losses import custom_MeanIOU, combined_cce_miou
from os.path import join
from datetime import datetime
import tensorflow as tf
import shutil
import glob
import os

# tf.compat.v1.enable_eager_execution()
unet_model = UNET_inception(input_shape=(None,None,3),num_layers=5,filters=32,num_classes=20)
unet_model.summary()
logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(logdir,exist_ok=True)
callbacks = [tf.keras.callbacks.TensorBoard(log_dir=logdir),tf.keras.callbacks.ModelCheckpoint(
    filepath=join(logdir,'checkpoint'),
    save_weights_only=True,
    monitor='val_acc',
    mode='max',
    save_best_only=True), tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5,restore_best_weights=True)]
current_path = os.path.dirname(os.path.abspath(__file__))
for py_file in glob.glob(join(current_path, "*.py")):
    shutil.copyfile(py_file, join(logdir,py_file.split("/")[-1]))
unet_model.compile(optimizer='adam',loss=combined_cce_miou,metrics=['acc',custom_MeanIOU(num_classes=20)])

for crop in [0.5,0.75,None]:
    train_gen = ImageGenerator(1, join('leftImg8bit', 'train'), join('gtFine', 'train'), 'leftImg8bit', 'gtFine_color',
                                  None, 1,False,random_crop=crop)
    val_gen = ImageGenerator(1, join('leftImg8bit', 'val'), join('gtFine', 'val'), 'leftImg8bit', 'gtFine_color',
                                  None, 1,False)

    unet_model.fit(train_gen,steps_per_epoch=len(train_gen),epochs=25,validation_data=val_gen,validation_steps=len(val_gen),callbacks=callbacks)
