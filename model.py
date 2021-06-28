from tensorflow.keras.models import Model
from tensorflow.keras.backend import int_shape
from tensorflow.keras.layers import Conv2D,Conv2DTranspose,Input,BatchNormalization,concatenate,MaxPooling2D,Cropping2D

def conv2d_block(input,n_filters,kernel_size=3,stride=1,padding='valid',batch_norm=True,activation='relu'):
    """
    @param input
    @param kernel_size
    @param stride
    @param batch_norm
    """
    x = Conv2D(filters=n_filters,kernel_size=kernel_size,strides=stride,kernel_initializer='he_normal',padding=padding,activation=activation,use_bias=not batch_norm)(input)
    if batch_norm:
        x= BatchNormalization()(x)
    # x = Activation('relu')(x)
    x = Conv2D(filters=n_filters, kernel_size=kernel_size, strides=stride, kernel_initializer='he_normal', padding=padding,activation=activation,use_bias= not batch_norm)(x)
    if batch_norm:
        x = BatchNormalization()(x)
    return x

def deconv2d_block(input,concatenation_tensor,n_filters,padding='valid'):
    x = Conv2DTranspose(n_filters, kernel_size=(2,2),strides=(2,2), padding=padding)(input)
    ch, cw = get_crop_shape(int_shape(concatenation_tensor), int_shape(x))
    concationation_tensor = Cropping2D(cropping=(ch, cw))(concatenation_tensor)
    x = concatenate([x, concatenation_tensor])
    x = conv2d_block(input=x, n_filters=n_filters, batch_norm=False, padding=padding)
    return x

def get_crop_shape(target, refer):
    # width, the 3rd dimension
    cw = target[2] - refer[2]
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw/2), int(cw/2) + 1
    else:
        cw1, cw2 = int(cw/2), int(cw/2)
    # height, the 2nd dimension
    ch = target[1] - refer[1]
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch/2), int(ch/2) + 1
    else:
        ch1, ch2 = int(ch/2), int(ch/2)

    return (ch1, ch2), (cw1, cw2)
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
        x = MaxPooling2D((2, 2), strides=2)(x)
        filters = filters * 2

    x = conv2d_block(input=x, n_filters=filters, batch_norm=False, padding='same')

    for conv in reversed(down_layers):
        filters //= 2  # decreasing number of filters with each layer
        x = deconv2d_block(input=x, concatenation_tensor=conv,n_filters=filters, padding='same')
    outputs = Conv2D(num_classes, (1, 1), activation=output_activation)(x)

    model = Model(inputs=[input], outputs=[outputs])
    return model

if __name__ == '__main__':
    UNET(input_shape=(1024,2048,3),num_classes=1).summary()
