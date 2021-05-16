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


#
# Generates unified video filename based on opional parameters
#
def generate_video_filename(dataset=None, timestamp=True, name="video", seed=None, duration=None, trunc=None):
    file_name = ""
    if dataset:
        file_name += dataset.replace("/", "-")
    if timestamp:
        file_name += datetime.now().strftime(" - %Y-%m-%d %H:%M")
    if name:
        file_name += " - " + name.replace("/", "-")
    if seed:
        file_name += " - seed={}".format(seed)
    if duration:
        file_name += " - {}sec".format(duration)
    if trunc:
        file_name += " - trunc={:03d}".format(int(100 * trunc))
    file_name += ".mp4"  # Append extension

    return file_name


def generate_image(pkl: str, seed: int = 42, trunc: float = None, randomize_noise: bool = False) -> PIL.Image:
    """ Generate single image and returns PIL.Image """

    tflib.init_tf()
    _G, _D, Gs = load_network(pkl)  # Loading neurals
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = randomize_noise
    if trunc and trunc != 1:
        Gs_kwargs.truncation_psi = trunc

    print(f'Generating image (seed={seed}, trunc={trunc})')
    rnd = np.random.RandomState(seed)
    z = rnd.randn(1, *Gs.input_shape[1:])  # [minibatch, component]
    tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars})  # [height, width]
    images = Gs.run(z, None, **Gs_kwargs)  # [minibatch, height, width, channel]
    image_pil = PIL.Image.fromarray(images[0], 'RGB')
    return image_pil


def generate_images(pkl, seeds=None, trunc=None, output_dir=None, ext="jpg"):
    if seeds is None:
        seeds = [1, 2, 3, 4, 5]
    for seed in seeds:
        img = generate_image(pkl=pkl, seed=seed, trunc=trunc)
        img.save(f"{output_dir}/{seed}.{ext}")
