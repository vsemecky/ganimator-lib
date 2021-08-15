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
from PIL import Image, ImageFont, ImageDraw
import progressbar
from matplotlib import font_manager
from moviepy.editor import *


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
# Get font path by font-family name
#
def font_by_name(family='sans-serif', weight='normal'):
    return font_manager.findfont(
        font_manager.FontProperties(family=family, weight=weight)
    )


#
# Draw textbox on PIL Image
#
def draw_text(draw: ImageDraw, image: Image, font, text="Text example", gravity="South", fill=(0, 0, 0), padding=5, margin=10):
    text_width, text_height = draw.textsize(text, font=font)
    gravity = gravity.lower()

    if gravity == 'south':
        x = (image.width - text_width) // 2
        y = image.height - text_height - margin - padding
    elif gravity == 'north':
        x = (image.width - text_width) // 2
        y = margin + padding
    elif gravity == 'center':
        x = (image.width - text_width) // 2
        y = (image.height - text_height) // 2
    elif gravity == 'southwest':
        x = margin + padding
        y = image.height - text_height - margin - padding
    elif gravity == 'southeast':
        x = image.width - margin - padding - text_width
        y = image.height - text_height - margin - padding
    elif gravity == 'northwest':
        x = y = margin + padding
    elif gravity == 'northeast':
        x = image.width - margin - padding - text_width
        y = margin + padding
    else:
        x = y = 0

    draw.rectangle((x - padding, y - padding, x + text_width + padding, y + text_height + padding), fill=fill)
    draw.text((x, y), text=text, font=font)


#
# Get PIL ImageFont by options
#
def get_image_font(family='sans-serif', weight='normal', size=12):
    font_path = font_manager.findfont(
        font_manager.FontProperties(family='sans-serif', weight='normal')
    )
    return ImageFont.truetype(font_path, size=size)


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


def generate_image(pkl: str, seed: int = 42, trunc: float = None, randomize_noise: bool = False) -> Image:
    """ Generate single image and returns PIL Image """

    tflib.init_tf()
    Gs = load_network_Gs(pkl)  # Loading neurals
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.conv, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = randomize_noise
    if trunc is not None:
        Gs_kwargs.truncation_psi = trunc

    rnd = np.random.RandomState(seed)
    z = rnd.randn(1, *Gs.input_shape[1:])  # [minibatch, component]
    tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars})  # [height, width]
    images = Gs.run(z, None, **Gs_kwargs)  # [minibatch, height, width, channel]
    image_pil = Image.fromarray(images[0], 'RGB')
    return image_pil


def generate_images(pkl, seeds=None, trunc=None, output_dir=None, ext="jpg"):
    os.makedirs(output_dir, exist_ok=True)
    if seeds is None:
        seeds = [1, 2, 3, 4, 5]
    for seed in progressbar.progressbar(seeds, redirect_stdout=True):
        print(f'Generating image (seed={seed}, trunc={trunc})')
        img = generate_image(pkl=pkl, seed=seed, trunc=trunc)
        img.save(f"{output_dir}/{seed}.{ext}")


def style_mixing_grid(pkl, row_seeds, col_seeds, truncation_psi, col_styles, outdir, minibatch_size=4):
    tflib.init_tf()
    Gs = load_network_Gs(pkl)  # Loading neurals

    w_avg = Gs.get_var('dlatent_avg') # [component]
    Gs_syn_kwargs = {
        'output_transform': dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
        'randomize_noise': False,
        'minibatch_size': minibatch_size
    }

    print('Generating W vectors...')
    all_seeds = list(set(row_seeds + col_seeds))
    all_z = np.stack([np.random.RandomState(seed).randn(*Gs.input_shape[1:]) for seed in all_seeds]) # [minibatch, component]
    all_w = Gs.components.mapping.run(all_z, None) # [minibatch, layer, component]
    all_w = w_avg + (all_w - w_avg) * truncation_psi # [minibatch, layer, component]
    w_dict = {seed: w for seed, w in zip(all_seeds, list(all_w))} # [layer, component]

    print('Generating images...')
    all_images = Gs.components.synthesis.run(all_w, **Gs_syn_kwargs) # [minibatch, height, width, channel]
    image_dict = {(seed, seed): image for seed, image in zip(all_seeds, list(all_images))}

    print('Generating style-mixed images...')
    for row_seed in row_seeds:
        for col_seed in col_seeds:
            w = w_dict[row_seed].copy()
            w[col_styles] = w_dict[col_seed][col_styles]
            image = Gs.components.synthesis.run(w[np.newaxis], **Gs_syn_kwargs)[0]
            image_dict[(row_seed, col_seed)] = image

    print('Saving image grid...')
    _N, _C, H, W = Gs.output_shape
    canvas = Image.new('RGB', (W * (len(col_seeds) + 1), H * (len(row_seeds) + 1)), 'black')
    for row_idx, row_seed in enumerate([None] + row_seeds):
        for col_idx, col_seed in enumerate([None] + col_seeds):
            if row_seed is None and col_seed is None:
                continue
            key = (row_seed, col_seed)
            if row_seed is None:
                key = (col_seed, col_seed)
            if col_seed is None:
                key = (row_seed, row_seed)
            canvas.paste(Image.fromarray(image_dict[key], 'RGB'), (W * col_idx, H * row_idx))

    string_styles = [str(i) for i in col_styles]
    styles_str = "".join(string_styles)
    canvas.save(f'{outdir}/grid-{styles_str}.png')
