import sys

# Allow import from git submodules
sys.path.append("./submodules/stylegan2-ada/")

from moviepy.editor import *
import scipy
import dnnlib
import dnnlib.tflib as tflib
from datetime import datetime
import pickle
import numpy as np
import PIL.Image
from training.misc import create_image_grid

# Memory cache for already loaded pkls
networks_cache = {}


# Loads pre-trained network from pkl file or URL
# Result is cached for future loading the same pkl
# @todo Remote loading using HTTP(S)
def load_network(pkl):
    if pkl in networks_cache.keys():
        # Return network from cache
        return networks_cache[pkl]
    else:
        # Load network from pkl file and store to cache
        with open(pkl, 'rb') as stream:
            print("Loading neurals: {}".format(pkl))
            networks_cache[pkl] = pickle.load(stream, encoding='latin1')
            return networks_cache[pkl]


def generate_image(pkl: str, seed: int = 42, psi: float = None, randomize_noise: bool = False) -> PIL.Image:
    """ Generate single image and returns PIL.Image """

    tflib.init_tf()
    _G, _D, Gs = load_network(pkl)  # Loading neurals
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = randomize_noise
    if psi:
        Gs_kwargs.truncation_psi = psi

    print('Generating image (seed=%d, psi=%f)' % (seed, psi))
    rnd = np.random.RandomState(seed)
    z = rnd.randn(1, *Gs.input_shape[1:])  # [minibatch, component]
    tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars})  # [height, width]
    images = Gs.run(z, None, **Gs_kwargs)  # [minibatch, height, width, channel]
    image_pil = PIL.Image.fromarray(images[0], 'RGB')
    return image_pil
