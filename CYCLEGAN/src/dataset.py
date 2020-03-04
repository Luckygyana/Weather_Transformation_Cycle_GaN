import albumentations
import numpy as np
from glob import glob
def load_batch(self, batch_size=1, is_testing=False):
    data_type = "train" if not is_testing else "test"
    path_A = glob('../input/summer2winter_yosemite/%sA/*' % ( data_type))
    path_B = glob('../input/summer2winter_yosemite/%sB/*' % ( data_type))

    n_batches = int(min(len(path_A), len(path_B)) / batch_size)
    total_samples = n_batches * batch_size
    if not is_testing:
        aug=albumentations.Compose([
            albumentations.Normalize(always_apply=True),
            albumentations.Rotate(50,p=0.5),
            albumentations.augmentations.transforms(p=0.5)
        ])
    else:
        aug=albumentations.Compose([albumentations.Normalize(always_apply=True)]

    path_A = np.random.choice(path_A, total_samples, replace=False)
    path_B = np.random.choice(path_B, total_samples, replace=False)

    for i in range(n_batches-1):
        batch_A = path_A[i*batch_size:(i+1)*batch_size]
        batch_B = path_B[i*batch_size:(i+1)*batch_size]
        imgs_A, imgs_B = [], []
        for img_A, img_B in zip(batch_A, batch_B):
                    img_A=aug(image=np.array(img_A))['image']
                    imgs_B=aug(image=np.array(img_B))['image']

            imgs_A.append(img_A)
            imgs_B.append(img_B)

        yield imgs_A, imgs_B