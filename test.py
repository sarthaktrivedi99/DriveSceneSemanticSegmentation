from utils import categorical_to_img,ImageGenerator
from losses import custom_iou,combined_cce_miou
from model import UNET,UNET_inception
from os.path import join
import os
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import MeanIoU
import seaborn as sns
from sklearn.metrics import jaccard_score
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Runs a given model on the test set')
parser.add_argument('--log_dir', type=str, required=False, help='location to the log directory')
parser.add_argument('--model', type=str, default='unet_inception', help='type of base model to use - value in [unet,unet_inception]')
parser.add_argument('--layers', type=int, default=5, help='layers in the UNET')
parser.add_argument('--filters', type=int, default=32, help='number of filters')
parser.add_argument('--batch_size', type=int, default=1, help='batch size for parallel runs')
parser.add_argument('--resize', type=bool, default=True, help='DO you want the output to be resized ?')
args = parser.parse_args()
jacc = []
if args.model=='unet_inception':
    unet_model = UNET_inception(input_shape=(None,None,3),num_layers=args.layers,filters=args.filters,num_classes=20)
else:
    unet_model = UNET(input_shape=(None,None,3),num_layers=args.layers,filters=args.filters,num_classes=20)
unet_model.summary()
log = args.log_dir
unet_model.load_weights(join(log,'checkpoint'))

unet_model.compile(optimizer='adam',loss=combined_cce_miou,metrics=['acc',custom_iou])

test_gen = ImageGenerator(args.batch_size, join('leftImg8bit', 'test'), join('gtFine', 'test'), 'leftImg8bit', 'gtFine_color',
                              None,False,resize_toggle=args.resize)
acc = []
m = MeanIoU(num_classes=20)
for i in range(len(test_gen)):
    location = test_gen.get_path_y(test_gen.list_paths_x[i])
    folders = join(log,'\\'.join(location.split('\\')[:-1]))
    print(join(log,location))
    os.makedirs(folders, exist_ok=True)
    x_test, y_test = next(test_gen)
    arr = unet_model.predict(x_test)
    img = categorical_to_img(arr)
    a = np.ndarray.flatten(np.argmax(arr[0],axis=-1))
    b = np.ndarray.flatten(np.argmax(y_test[0],axis=-1))
    acc.append(np.count_nonzero(a[a==b])/np.size(arr[0,:,:,0]))
    jacc.append(jaccard_score(np.ndarray.flatten(np.argmax(arr[0],axis=-1)),np.ndarray.flatten(np.argmax(y_test[0],axis=-1)),average='weighted'))
    m.update_state(np.argmax(y_test,axis=-1),np.argmax(arr,axis=-1))
    plt.imsave(join(log,location),img[0]/255)
# Plot Confusion matrix
print(f'MeanIOU - {np.mean(jacc)}, Accuracy - {np.mean(acc)}')
conf_matrix = m.total_cm
conf_matrix = conf_matrix/np.sum(conf_matrix,axis=1)
fig, ax = plt.subplots(figsize=(40,40))
sns.heatmap(conf_matrix, annot=True,fmt='.3f' ,xticklabels=range(20), yticklabels=range(20))
plt.xlabel('Predictions', fontsize=30)
plt.ylabel('Actuals', fontsize=30)
plt.title('Confusion Matrix', fontsize=30)
plt.show(block=False)
