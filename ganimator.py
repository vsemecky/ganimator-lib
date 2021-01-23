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
            psi: float = None,
            randomize_noise: bool = False
    ):
        pil_image = generate_image(pkl=pkl, seed=seed, psi=psi, randomize_noise=randomize_noise)
        super().__init__(np.array(pil_image), duration=duration)


class LatentWalkClip(VideoClip):
    """ Generates Random latent walk clip """

    def __init__(
            self,
            pkl: str,
            duration: int = 30,
            seed: int = 42,
            psi: float = None,
            randomize_noise: bool = False,
            smoothing_sec: float = 1.0,
            mp4_fps: int = 30
    ):
        # Nepouzivane parametry z puvodni funkce
        grid_size = [1, 1]
        # image_shrink = 1
        # image_zoom = 1

        tflib.init_tf()
        _G, _D, Gs = load_network(pkl)
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
            images = Gs.run(latents, None, truncation_psi=psi, randomize_noise=randomize_noise, output_transform=fmt)
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
