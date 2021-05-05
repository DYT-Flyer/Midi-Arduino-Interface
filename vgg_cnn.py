from tensorflow.keras import layers
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D

def vgg_cnn():# define model input
    visible = Input(shape=(25, 25, 1))
    # add vgg module
    layer = vgg_block(visible, 64, 2)
    # add vgg module
    layer = vgg_block(layer, 128, 2)
    # add vgg module
    layer = vgg_block(layer, 256, 4)
    
    x = layers.Flatten()(layer)
    x = (layers.Dense(256, activation='relu'))(x)
    x = (layers.Dropout(0.5))(x)
    x = (layers.Dense(32, activation='relu'))(x)
    x = (layers.Dropout(0.5))(x)
    x = (layers.Dense(5, activation='softmax'))(x)
    
    # create model
    model = Model(inputs=visible, outputs=x)
    
    return model

# function for creating a vgg block
def vgg_block(layer_in, n_filters, n_conv):
     # add convolutional layers
     for _ in range(n_conv):
        layer_in = Conv2D(n_filters, (3,3), padding='same', activation='relu')(layer_in)
     # add max pooling layer
     layer_in = MaxPooling2D((2,2), strides=(2,2))(layer_in)
     return layer_in