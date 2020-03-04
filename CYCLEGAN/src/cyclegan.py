import discriminator
import generator
from keras.optimizer import Adam
from keras.layes import Input,Model
class CycleGan():
    def __init__(self):
        self.height=128
        self.width=128
        self.channels=channels
        self.img_shape=(self.height,self.width,self.channels)
        optimizer = Adam(0.0002, 0.5)

        self.d_A = discriminator.build_discriminator()
        self.d_B = discriminator.build_discriminator()
        self.d_A.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.d_B.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        self.g_AB = generator.build_generator()
        self.g_BA = generator.build_generator()

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)
        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)

        self.d_A.trainable = False
        self.d_B.trainable = False

        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        self.combined = Model(inputs=[img_A, img_B],
                              outputs=[ valid_A, valid_B,
                                        reconstr_A, reconstr_B,
                                        img_A_id, img_B_id ])
        self.combined.compile(loss=['mse', 'mse',
                                    'mae', 'mae',
                                    'mae', 'mae'],
                            loss_weights=[  1, 1,
                                            10, 10,
                                            1, 1 ],
                            optimizer=optimizer)