from keras.layers import Conv2D,UpSampling2D,Concatenate,Dropout
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizer import Adam
from keras.layes import Input,Model
def build_generator():

        def conv2d(layer_input, filters, f_size=4):
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            d = InstanceNormalization()(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = InstanceNormalization()(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=(128,128,3))

        # Downsampling
        d1 = conv2d(d0, 32)
        d2 = conv2d(d1, 32*2)
        d3 = conv2d(d2, 32*4)
        d4 = conv2d(d3, 32*8)

        # Upsampling
        u1 = deconv2d(d4, d3, 32*4)
        u2 = deconv2d(u1, d2, 32*2)
        u3 = deconv2d(u2, d1, 32)

        u4 = UpSampling2D(size=2)(u3)
        output_img = Conv2D(channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

        return Model(d0, output_img)