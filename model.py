from tensorflow.keras.models import Model
from tensorflow.keras.backend import int_shape
from tensorflow.keras.layers import Activation, Dropout, Conv2D, Conv2DTranspose, Input, BatchNormalization, \
    concatenate, MaxPooling2D, Cropping2D,add,multiply,Lambda,UpSampling2D
import tensorflow.keras.backend as K

def attention_up_and_concate(down_layer, layer, data_format='channels_first'):
    if data_format == 'channels_first':
        in_channel = down_layer.get_shape().as_list()[1]
    else:
        in_channel = down_layer.get_shape().as_list()[3]

    # up = Conv2DTranspose(out_channel, [2, 2], strides=[2, 2])(down_layer)
    up = UpSampling2D(size=(2, 2))(down_layer)

    layer = attention_block_2d(x=layer, g=up, inter_channel=in_channel // 4, data_format=data_format)

    if data_format == 'channels_first':
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))
    else:
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))

    concate = my_concat([up, layer])
    return concate


def attention_block_2d(x, g, inter_channel, data_format='channels_first'):
    # theta_x(?,g_height,g_width,inter_channel)

    theta_x = Conv2D(inter_channel, [1, 1], strides=[1, 1])(x)

    # phi_g(?,g_height,g_width,inter_channel)

    phi_g = Conv2D(inter_channel, [1, 1], strides=[1, 1])(g)

    # f(?,g_height,g_width,inter_channel)

    f = Activation('relu')(add([theta_x, phi_g]))

    # psi_f(?,g_height,g_width,1)

    psi_f = Conv2D(1, [1, 1], strides=[1, 1])(f)

    rate = Activation('sigmoid')(psi_f)

    # rate(?,x_height,x_width)

    # att_x(?,x_height,x_width,x_channel)

    att_x = multiply([x, rate])

    return att_x

def inception_block(layer_in, f1, f2_in, f2_out, f3_in, f3_out, f4_out):
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
    # x = Attention()([x, concatenation_tensor])
    x = concatenate([x, concatenation_tensor])
    # x = Dropout(0.5)(x)
    x = inception_block(x, f1=n_filters//4, f2_in=n_filters//3, f2_out=n_filters,f3_in=n_filters//2,f3_out=n_filters//3,f4_out=n_filters//8)
    return x

def deconv_inception_block_att(input, concatenation_tensor, n_filters, padding='valid'):
    y = attention_up_and_concate(input,concatenation_tensor,n_filters)
    x = Conv2DTranspose(n_filters, kernel_size=(2, 2), strides=(2, 2), padding=padding)(input)
    x = concatenate([x, y])
    # x = Dropout(0.5)(x)
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

def UNET_inception_att(
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
        x = deconv_inception_block_att(input=x, concatenation_tensor=conv, n_filters=filters, padding='same')
    outputs = Conv2D(num_classes, (1, 1), activation=output_activation)(x)

    model = Model(inputs=[input], outputs=[outputs])
    return model


if __name__ == '__main__':
    UNET_inception_att(input_shape=(None, None, 3), num_classes=20).summary()
