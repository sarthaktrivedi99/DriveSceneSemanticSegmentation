from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Dropout, Conv2D, Conv2DTranspose, Input, BatchNormalization, \
    concatenate, MaxPooling2D, Cropping2D,add,multiply,Lambda,UpSampling2D
import tensorflow.keras.backend as K

def inception_block(layer_in, f1, f2_in, f2_out, f3_in, f3_out, f4_out):
    """Taken from - https://machinelearningmastery.com/how-to-implement-major-architecture-innovations-for-convolutional-neural-networks/"""
    # 1x1 conv
    conv1 = Conv2D(f1, (1, 1), padding='same', activation='relu')(layer_in)
    # 3x3 conv
    conv3 = Conv2D(f2_in, (1, 1), padding='same', activation='relu')(layer_in)
    conv3 = Conv2D(f2_out, (3, 3), padding='same', activation='relu')(conv3)
    # 5x5 conv
    conv5 = Conv2D(f3_in, (1, 1), padding='same', activation='relu')(layer_in)
    conv5 = Conv2D(f3_out, (5, 5), padding='same', activation='relu')(conv5)
    # 3x3 max pooling
    pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(layer_in)
    pool = Conv2D(f4_out, (1, 1), padding='same', activation='relu')(pool)
    # concatenate filters, assumes filters/channels last
    layer_out = concatenate([conv1, conv3, conv5, pool], axis=-1)
    return layer_out


def deconv_inception_block(input, concatenation_tensor, n_filters, padding='valid'):
    x = Conv2DTranspose(n_filters, kernel_size=(2, 2), strides=(2, 2), padding=padding)(input)
    x = concatenate([x, concatenation_tensor])
    x = inception_block(x, f1=n_filters//4, f2_in=n_filters//3, f2_out=n_filters,f3_in=n_filters//2,f3_out=n_filters//3,f4_out=n_filters//8)
    return x



def conv2d_block(input, n_filters, kernel_size=3, stride=1, padding='valid', batch_norm=True, activation='relu'):
    """
    @param input
    @param kernel_size
    @param stride
    @param batch_norm
    """
    x = Conv2D(filters=n_filters, kernel_size=kernel_size, strides=stride, kernel_initializer='he_normal',
               padding=padding, use_bias=not batch_norm)(input)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Dropout(0.5)(x)
    x = Conv2D(filters=n_filters, kernel_size=kernel_size, strides=stride, kernel_initializer='he_normal',
               padding=padding, use_bias=not batch_norm)(x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Dropout(0.5)(x)
    return x


def deconv2d_block(input, concatenation_tensor, n_filters, padding='valid'):
    x = Conv2DTranspose(n_filters, kernel_size=(2, 2), strides=(2, 2), padding=padding)(input)
    # ch, cw = get_crop_shape(int_shape(concatenation_tensor), int_shape(x))
    # concationation_tensor = Cropping2D(cropping=(ch, cw))(concatenation_tensor)
    # x = Attention()([x, concatenation_tensor])
    x = concatenate([x, concatenation_tensor])
    x = Dropout(0.5)(x)
    x = conv2d_block(input=x, n_filters=n_filters, batch_norm=True, padding=padding)
    return x


def UNET(
        input_shape,
        num_classes=1,
        filters=64,
        num_layers=4,
        output_activation='softmax'
):
    input = Input(input_shape)
    x = input
    down_layers = []
    for l in range(num_layers):
        x = conv2d_block(input=x, n_filters=filters, padding='same')
        down_layers.append(x)
        x = MaxPooling2D((2, 2))(x)
        filters = filters * 2

    x = conv2d_block(input=x, n_filters=filters, batch_norm=True, padding='same')

    for conv in reversed(down_layers):
        filters //= 2  # decreasing number of filters with each layer
        x = deconv2d_block(input=x, concatenation_tensor=conv, n_filters=filters, padding='same')
    outputs = Conv2D(num_classes, (1, 1), activation=output_activation)(x)

    model = Model(inputs=[input], outputs=[outputs])
    return model



def UNET_inception(
        input_shape,
        num_classes=1,
        filters=64,
        num_layers=4,
        output_activation='softmax'
):
    input = Input(input_shape)
    x = input
    down_layers = []
    for l in range(num_layers):
        x = inception_block(x, filters//4, filters//3, filters,filters//2,filters//3,filters//8)
        down_layers.append(x)
        x = MaxPooling2D((2, 2))(x)
        filters = filters * 2

    x = inception_block(x, filters//4, filters//3, filters,filters//2,filters//3,filters//8)

    for conv in reversed(down_layers):
        filters //= 2  # decreasing number of filters with each layer
        x = deconv_inception_block(input=x, concatenation_tensor=conv, n_filters=filters, padding='same')
    outputs = Conv2D(num_classes, (1, 1), activation=output_activation)(x)

    model = Model(inputs=[input], outputs=[outputs])
    return model

# if __name__ == '__main__':
#     import tensorflow as tf
#     model = UNET_inception(input_shape=(None, None, 3), num_classes=20)
#     tf.keras.utils.plot_model(model,to_file='model.png')
