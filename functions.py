# Allow import from git submodules
import sys
sys.path.append("./submodules/stylegan2-ada/")

import time
from moviepy.editor import *
import dnnlib
import dnnlib.tflib as tflib
from datetime import datetime
import pickle
import numpy as np
import PIL.Image
import progressbar

# Memory cache for already loaded pkls
networks_Gs_cache = {}


# Loads pre-trained network from pkl file or URL
# Result is stored in memory cache for the future loading of the same pkl
# @todo Remote loading using HTTP(S)
def load_network_Gs(pkl):
    if pkl in networks_Gs_cache.keys():
        return networks_Gs_cache[pkl]  # Return network from cache
    else:
        # Load network from pkl file and store to cache
        with open(pkl, 'rb') as stream:
            print("Loading network: {}".format(pkl))
            _G, _D, Gs = pickle.load(stream, encoding='latin1')
            networks_Gs_cache[pkl] = Gs
            return Gs

#
# Generates unified video filename based on opional parameters
#
def generate_video_filename(dir=None, dataset=None, timestamp=False, name="video", seed=None, duration=None, trunc=None, pkl=None):
    if dir:
        file_name = f"/{dir}/"
    else:
        file_name = ""

    if dataset:
        file_name += dataset.replace("/", "-")
    if pkl:
        file_name += time.strftime(' - %Y-%m-%d', time.localtime(os.path.getmtime(pkl)))
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

# from https://colab.research.google.com/drive/1ShgW6wohEFQtqs_znMna3dzrcVoABKIH
def generate_zs_from_seeds(seeds, Gs):
    zs = []
    for seed_idx, seed in enumerate(seeds):
        rnd = np.random.RandomState(seed)
        z = rnd.randn(1, *Gs.input_shape[1:])  # [minibatch, component]
        zs.append(z)
    return zs


def line_interpolate(zs, steps):
    out = []
    for i in range(len(zs) - 1):
        for index in range(steps):
            fraction = index / float(steps)
            out.append(zs[i + 1] * fraction + zs[i] * (1 - fraction))
    return out


def generate_image(pkl: str, seed: int = 42, trunc: float = None, randomize_noise: bool = False) -> PIL.Image:
    """ Generate single image and returns PIL.Image """

    tflib.init_tf()
    Gs = load_network_Gs(pkl)  # Loading neurals
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = randomize_noise
    if trunc is not None:
        Gs_kwargs.truncation_psi = trunc

    rnd = np.random.RandomState(seed)
    z = rnd.randn(1, *Gs.input_shape[1:])  # [minibatch, component]
    tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars})  # [height, width]
    images = Gs.run(z, None, **Gs_kwargs)  # [minibatch, height, width, channel]
    image_pil = PIL.Image.fromarray(images[0], 'RGB')
    return image_pil


def generate_images(pkl, seeds=None, trunc=None, output_dir=None, ext="jpg"):
    os.makedirs(output_dir, exist_ok=True)
    if seeds is None:
        seeds = [1, 2, 3, 4, 5]
    for seed in progressbar.progressbar(seeds, redirect_stdout=True):
        print(f'Generating image (seed={seed}, trunc={trunc})')
        img = generate_image(pkl=pkl, seed=seed, trunc=trunc)
        img.save(f"{output_dir}/{seed}.{ext}")
