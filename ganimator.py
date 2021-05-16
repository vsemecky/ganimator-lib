import sys

# Allow import from git submodules
sys.path.append("./submodules/stylegan2-ada/")

from functions import *
from moviepy.editor import *
import numpy as np


class StaticImageClip(ImageClip):
    """ Single static image generated from GAN. This clip will simply display the same image at all times."""

    def __init__(
            self,
            pkl: str,
            duration: int = 30,
            seed: int = 42,
            trunc: float = None,
            randomize_noise: bool = False
    ):
        pil_image = generate_image(pkl=pkl, seed=seed, trunc=trunc, randomize_noise=randomize_noise)
        super().__init__(np.array(pil_image), duration=duration)


class LatentWalkClip(VideoClip):
    """ Generates Random latent walk clip """

    def __init__(
            self,
            pkl: str,
            duration: int = 30,
            seed: int = 42,
            trunc: float = None,
            randomize_noise: bool = False,
            smoothing_sec: float = 1.0,
            mp4_fps: int = 30
    ):
        # Nepouzivane parametry z puvodni funkce
        grid_size = [1, 1]
        # image_shrink = 1
        # image_zoom = 1

        tflib.init_tf()
        Gs = load_network_Gs(pkl)
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)

        num_frames = int(np.rint(duration * mp4_fps))
        random_state = np.random.RandomState(seed)

        # Generating latent vectors
        shape = [num_frames, np.prod(grid_size)] + Gs.input_shape[1:]  # [frame, image, channel, component]
        all_latents = random_state.randn(*shape).astype(np.float32)
        all_latents = scipy.ndimage.gaussian_filter(all_latents, [smoothing_sec * mp4_fps] + [0] * len(Gs.input_shape), mode='wrap')
        all_latents /= np.sqrt(np.mean(np.square(all_latents)))

        def make_frame(t):
            """ Frame generation func for MoviePy """
            frame_idx = int(np.clip(np.round(t * mp4_fps), 0, num_frames - 1))
            latents = all_latents[frame_idx]
            images = Gs.run(latents, None, truncation_psi=trunc, randomize_noise=randomize_noise, output_transform=fmt)
            return images[0]

            # labels = np.zeros([latents.shape[0], 0], np.float32)
            # images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            # grid = create_image_grid(images, grid_size).transpose(1, 2, 0)  # HWC
            # Grayscale support
            # if grid.shape[2] == 1:
            #     grid = grid.repeat(3, 2)  # grayscale => RGB
            # return grid

        # Create VideoClip
        super().__init__(make_frame=make_frame, duration=duration)


class InterpolationClip(VideoClip):
    """ Generates interpolation video between seeds """

    def __init__(
            self,
            pkl: str,
            duration: int = 30,
            seeds: list = [1, 2, 3],
            trunc: float = None,
            randomize_noise: bool = False,
            smoothing_sec: float = 1.0,
            mp4_fps: int = 30
    ):
        tflib.init_tf()
        # Loading neurals
        Gs = load_network_Gs(pkl)

        noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

        zs = generate_zs_from_seeds(seeds, Gs)

        num_frames = int(np.rint(duration * mp4_fps))
        number_of_steps = int(num_frames / (len(zs) - 1)) +1  # todo Prejmenovat na num_steps nebo steps_count/frames_count
        points = line_interpolate(zs, number_of_steps)

        # Generate_latent_images()
        Gs_kwargs = dnnlib.EasyDict()
        Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        Gs_kwargs.randomize_noise = False
        if trunc is not None:
            Gs_kwargs.truncation_psi = trunc

        # Frame generation func for moviepy
        def make_frame(t):
            frame_idx = int(np.clip(np.round(t * mp4_fps), 0, num_frames - 1))

            # TIP: Oddebugovat run_generator,  co mu sem leze, zejmena len(zx)
            z_idx = frame_idx
            z = points[z_idx]

            # Puvodni loop
            if isinstance(z, list):
                z = np.array(z).reshape(1, 512)
            elif isinstance(z, np.ndarray):
                z.reshape(1, 512)

            Gs_kwargs.truncation_psi = psi  # todo presunout nahoru mimo make_frame
            noise_rnd = np.random.RandomState(1)  # fix noise
            tflib.set_vars({var: noise_rnd.randn(*var.shape.as_list()) for var in noise_vars})  # [height, width]
            images = Gs.run(z, None, **Gs_kwargs)  # [minibatch, height, width, channel]

            # todo Zbavit se gridu, kdyz potrebujeme jen jeden obrazek
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            grid = create_image_grid(images, [1, 1]).transpose(1, 2, 0)  # HWC
            # if grid.shape[2] == 1:
            #     grid = grid.repeat(3, 2)  # grayscale => RGB
            return grid

        # Create VideoClip
        super().__init__(make_frame=make_frame, duration=duration)


class ArrayClip(CompositeVideoClip):
    """
    Object-oriented implementation of MoviePy function array_clip()

    rows_widths
      widths of the different rows in pixels. If None, is set automatically.

    cols_widths
      widths of the different colums in pixels. If None, is set automatically.

    bg_color
       Fill color for the masked and unfilled regions. Set to None for these
       regions to be transparent (will be slower).

    """

    def __init__(self, array, rows_widths=None, cols_widths=None, bg_color=None):
        array = np.array(array)
        sizes_array = np.array([[c.size for c in line] for line in array])

        # find row width and col_widths automatically if not provided
        if rows_widths is None:
            rows_widths = sizes_array[:, :, 1].max(axis=1)
        if cols_widths is None:
            cols_widths = sizes_array[:, :, 0].max(axis=0)

        xx = np.cumsum([0] + list(cols_widths))
        yy = np.cumsum([0] + list(rows_widths))

        for j, (x, cw) in enumerate(zip(xx[:-1], cols_widths)):
            for i, (y, rw) in enumerate(zip(yy[:-1], rows_widths)):
                clip = array[i, j]
                w, h = clip.size
                if (w < cw) or (h < rw):
                    clip = (CompositeVideoClip([clip.set_position('center')],
                                               size=(cw, rw),
                                               bg_color=bg_color).
                            set_duration(clip.duration))

                array[i, j] = clip.set_position((x, y))

        super().__init__(array.flatten(), size=(xx[-1], yy[-1]), bg_color=bg_color)


class TruncComparisonClip(ArrayClip):
    """  """
    def __init__(
        self,
        pkl=None,
        grid=None, # (width, height)
        trunc_range=(0.2, 1),
        rows=3,
        cols=3,
        mp4_fps=30,
        duration=30,  # Duration in seconds
        smoothing_sec=1.0,
        randomize_noise=False,
        seed=420 # Starting seed of the first image
    ):
        clips = [[0 for col in range(cols)] for row in range(rows)]

        count = cols * rows
        step = (trunc_range[1] - trunc_range[0]) / (count -1)
        i = 0
        for row in range(0, rows):
            for col in range(0, cols):
                trunc = i * step
                clips[row][col] = LatentWalkClip(pkl=pkl, seed=seed, trunc=trunc, duration=duration, randomize_noise=randomize_noise, smoothing_sec=smoothing_sec)
                i += 1

        # Arrange clips into ArrayClip (parent class)
        super().__init__(clips)
