# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for generating an image using pre-trained StyleGAN generator."""

import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config

import cv2
from time import sleep

def main():
    # Initialize TensorFlow.
    tflib.init_tf()

    # Load pre-trained network.
    url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
    with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
        _G, _D, Gs = pickle.load(f)
        # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
        # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
        # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.

    # Print network details.
    Gs.print_layers()

    # Pick latent vector.
    rnd = np.random.RandomState()
    latents = rnd.randn(1, Gs.input_shape[1])
    count = 0

    while True:
        key = cv2.waitKey(1)
        if key == 27: # when ESC key is pressed break
            break

        count += 1
        if count == 1:
            # Generate image.
            rnd = np.random.RandomState()
            latents = rnd.randn(1, Gs.input_shape[1])
            fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
            images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)

            # Converte image
            img_bgr = cv2.resize(images[0], (300, 300))
            img_rgb = img_bgr[:,:,::-1]
            cv2.imshow('style gan demo', img_rgb)

            count = 0

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
