import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.metrics import jaccard_score


def custom_iou(y_true,y_pred):
    """
    Custom IOU class based on sklearns jaccard_score function.
    @param y_true: True label
    @param y_pred: Predicted label
    @return: IOU score
    """
    y_true = K.flatten(K.argmax(y_true,axis=-1))
    y_pred = K.flatten(K.argmax(y_pred,axis=-1))
    return tf.py_function(jaccard_score,[y_true,y_pred,tf.range(20),1,'weighted'],Tout=tf.float32)



def iou_loss(y_true, y_pred, smooth=1e-3):
    """
    IOU loss based on http://cs.umanitoba.ca/~ywang/papers/isvc16.pdf
    @param y_true: True label
    @param y_pred: Predicted label
    @param smooth: smoothing factor to help with zero divides
    @return: IOU score
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true,-1) + K.sum(y_pred,-1) - intersection
    iou = (intersection + smooth) / ( union + smooth)
    return iou

def mean_iou_loss(y_true,y_pred,smooth=1e-3,classes=20):
    """
    Calculates Mean IOU using iou_loss function for each class
    @param y_true: True label
    @param y_pred: Predicted label
    @param smooth: Smoothing factot to help with zero divide
    @param classes: Number of classes
    @return: Mean IOU score
    """
    iou_class = []
    for i in range(classes):
        iou_class.append(iou_loss(tf.slice(y_true,begin=[0,0,0,i],size=[-1,-1,-1,1]),tf.slice(y_pred,begin=[0,0,0,i],size=[-1,-1,-1,1]),smooth=smooth))
    return tf.math.reduce_mean(tf.stack(iou_class))

def combined_cce_miou(y_true,y_pred):
    """
    Returns combined Categorical and IOU loss
    @param y_true: True label
    @param y_pred: Predicted label
    @return: combined categorical nad iou loss
    """
    return K.categorical_crossentropy(target=y_true,output=y_pred)+1-custom_iou(y_true,y_pred)
