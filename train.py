from model import UNET,UNET_inception
from utils import ImageGenerator,augmentation_fn
from losses import combined_cce_miou,custom_iou
from os.path import join
from datetime import datetime
import tensorflow as tf
import shutil
import glob
import os
import argparse


parser = argparse.ArgumentParser(description='Runs a given model on the test set')
parser.add_argument('--model', type=str, default='unet_inception', help='type of base model to use - value in [unet,unet_inception]')
parser.add_argument('--layers', type=int, default=5, help='layers in the UNET')
parser.add_argument('--num_runs', type=int, default=10, help='number of epochs')
parser.add_argument('--filters', type=int, default=32, help='number of filters')
parser.add_argument('--batch_size', type=int, default=1, help='batch size for parallel runs')
parser.add_argument('--resize', type=bool, default=True, help='DO you want the output to be resized ?')
args = parser.parse_args()
# Initialize the model
if args.model=='unet_inception':
    unet_model = UNET_inception(input_shape=(None,None,3),num_layers=args.layers,filters=args.filters,num_classes=20)
else:
    unet_model = UNET(input_shape=(None,None,3),num_layers=args.layers,filters=args.filters,num_classes=20)
unet_model.summary()
# Setup the log directory
logdir = join("logs" , datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(logdir,exist_ok=True)
# Create Callbacks to save model weights, log everything to tensorboard and early stopping to stop the model from overfitting
callbacks = [tf.keras.callbacks.TensorBoard(log_dir=logdir),tf.keras.callbacks.ModelCheckpoint(
    filepath=join(logdir,'checkpoint'),
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True), tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,mode='min',restore_best_weights=True)]
# current_path = os.path.dirname(os.path.abspath(__file__))
# # Copy all the py files to log folder
# for py_file in glob.glob(join(current_path, "*.py")):
#     shutil.copyfile(py_file, join(logdir,py_file.split("/")[-1]))
# # Compiling the model
unet_model.compile(optimizer='adam',loss=combined_cce_miou,metrics=['acc',custom_iou])

# Run three fittings one after the other
for crop,epoch in zip([0.5,0.75,1],[args.num_runs,args.num_runs,args.num_runs]):
    train_gen = ImageGenerator(args.batch_size, join('leftImg8bit', 'train'), join('gtFine', 'train'), 'leftImg8bit', 'gtFine_color',
                                  augmentation_fn,False,random_crop=crop)
    val_gen = ImageGenerator(args.batch_size, join('leftImg8bit', 'val'), join('gtFine', 'val'), 'leftImg8bit', 'gtFine_color',
                                  None, False)

    unet_model.fit(train_gen,steps_per_epoch=len(train_gen),epochs=epoch,validation_data=val_gen,validation_steps=len(val_gen),callbacks=callbacks)
