# Weather_Transformation_Cycle_GaN

## What Is Cycle GAN
Image Translation Using Cycle GAN
The CycleGAN model was described by Jun-Yan Zhu, et al. in their 2017 paper titled “Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks.”

The model architecture is comprised of two generator models: one generator (Generator-A) for generating images for the first domain (Domain-A) and the second generator (Generator-B) for generating images for the second domain (Domain-B).

Generator-A -> Domain-A
Generator-B -> Domain-B
The generator models perform image translation, meaning that the image generation process is conditional on an input image, specifically an image from the other domain. Generator-A takes an image from Domain-B as input and Generator-B takes an image from Domain-A as input.

Domain-B -> Generator-A -> Domain-A
Domain-A -> Generator-B -> Domain-B
Each generator has a corresponding discriminator model.

The first discriminator model (Discriminator-A) takes real images from Domain-A and generated images from Generator-A and predicts whether they are real or fake. The second discriminator model (Discriminator-B) takes real images from Domain-B and generated images from Generator-B and predicts whether they are real or fake.

Domain-A -> Discriminator-A -> [Real/Fake]
Domain-B -> Generator-A -> Discriminator-A -> [Real/Fake]
Domain-B -> Discriminator-B -> [Real/Fake]
Domain-A -> Generator-B -> Discriminator-B -> [Real/Fake]
The discriminator and generator models are trained in an adversarial zero-sum process, like normal GAN models.

The generators learn to better fool the discriminators and the discriminators learn to better detect fake images. Together, the models find an equilibrium during the training process.

Additionally, the generator models are regularized not just to create new images in the target domain, but instead create translated versions of the input images from the source domain. This is achieved by using generated images as input to the corresponding generator model and comparing the output image to the original images.

Passing an image through both generators is called a cycle. Together, each pair of generator models are trained to better reproduce the original source image, referred to as cycle consistency.

Domain-B -> Generator-A -> Domain-A -> Generator-B -> Domain-B
Domain-A -> Generator-B -> Domain-B -> Generator-A -> Domain-A
There is one further element to the architecture referred to as the identity mapping.

This is where a generator is provided with images as input from the target domain and is expected to generate the same image without change. This addition to the architecture is optional, although it results in a better matching of the color profile of the input image.

Domain-A -> Generator-A -> Domain-A
Domain-B -> Generator-B -> Domain-B
Now that we are familiar with the model architecture, we can take a closer look at each model in turn and how they can be implemented.

The paper provides a good description of the models and training process, although the official Torch implementation was used as the definitive description for each model and training process and provides the basis for the the model implementations described below.


## DATASET
***https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/summer2winter_yosemite.zip***
