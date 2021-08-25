import tensorflow as tf
import tensorflow.keras.backend as K


class custom_MeanIOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)

def iou_loss(y_true, y_pred, smooth=1e-3):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true,-1) + K.sum(y_pred,-1) - intersection
    iou = (intersection + smooth) / ( union + smooth)
    return iou

def mean_iou_loss(y_true,y_pred,smooth=1e-3,classes=20):
    iou_class = []
    for i in range(classes):
        iou_class.append(iou_loss(tf.slice(y_true,begin=[0,0,0,i],size=[-1,-1,-1,1]),tf.slice(y_pred,begin=[0,0,0,i],size=[-1,-1,-1,1]),smooth=smooth))
    return tf.math.reduce_mean(tf.stack(iou_class))

def combined_cce_miou(y_true,y_pred):
    return K.categorical_crossentropy(y_true,y_pred)-mean_iou_loss(y_true,y_pred)
