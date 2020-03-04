from keras.layers import Conv2D,UpSampling2D,Concatenate,Dropout
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizer import Adam
from keras.layes import Input,Model
def build_discriminator():

        def d_layer(layer_input, filters, f_size=4, normalization=True):
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = InstanceNormalization()(d)
            return d

        img = Input(shape=(128,128,3))

        d1 = d_layer(img, 64, normalization=False)
        d2 = d_layer(d1, 64*2)
        d3 = d_layer(d2, 64*4)
        d4 = d_layer(d3, 64*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model(img, validity)