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
            psi: float = 0.5,
            randomize_noise: bool = False
    ):
        pil_image = generate_image(pkl=pkl, seed=seed, psi=psi, randomize_noise=randomize_noise)
        super().__init__(np.array(pil_image), duration=duration)
