from cyclegan import CycleGan
import numpy as np
import datetime
import dataset
from tqdm import tqdm
def train(self, epochs, batch_size=1, sample_interval=50):
        gan=CycleGan()
        start_time = datetime.datetime.now()
        batch_size=1
        valid = np.ones((batch_size,16,16,1))
        fake = np.zeros((batch_size,16,16,2))

        for epoch in tqdm(range(epochs):)
            for batch_i, (imgs_A, imgs_B) in enumerate(dataset.load_batch(batch_size)):

       
                fake_B = gan.g_AB.predict(imgs_A)
                fake_A = gan.g_BA.predict(imgs_B)

                dA_loss_real = gan.d_A.train_on_batch(imgs_A, valid)
                dA_loss_fake = gan.d_A.train_on_batch(fake_A, fake)
                dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

                dB_loss_real = gan.d_B.train_on_batch(imgs_B, valid)
                dB_loss_fake = gan.d_B.train_on_batch(fake_B, fake)
                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

                d_loss = 0.5 * np.add(dA_loss, dB_loss)



                g_loss = gan.combined.train_on_batch([imgs_A, imgs_B],
                                                        [valid, valid,
                                                        imgs_A, imgs_B,
                                                        imgs_A, imgs_B])

                elapsed_time = datetime.datetime.now() - start_time

                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s " \
                                                                        % ( epoch, epochs,
                                                                            batch_i, len(dataset.load_batch()),
                                                                            d_loss[0], 100*d_loss[1],
                                                                            g_loss[0],
                                                                            np.mean(g_loss[1:3]),
                                                                            np.mean(g_loss[3:5]),
                                                                            np.mean(g_loss[5:6]),
                                                                            elapsed_time))
