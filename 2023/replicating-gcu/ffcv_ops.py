# Modified code from:
# FFCV - Apache License 2.0 - Copyright (c) 2022 FFCV Team


import numpy as np
from numpy.random import rand
from typing import Callable, Optional, Tuple
from dataclasses import replace

from ffcv.pipeline.compiler import Compiler
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.state import State

class Cutout(Operation):
    """Cutout data augmentation (https://arxiv.org/abs/1708.04552).

    Parameters
    ----------
    crop_size : int
        Size of the random square to cut out.
    fill : Tuple[int, int, int], optional
        An RGB color ((0, 0, 0) by default) to fill the cutout square with.
        Useful for when a normalization layer follows cutout, in which case
        you can set the fill such that the square is zero
        post-normalization.
    """
    def __init__(self, prob: float, crop_size: int, fill: Tuple[int, int, int] = (0, 0, 0)):
        super().__init__()
        self.crop_size = crop_size
        self.fill = np.array(fill)
        self.prob = prob

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()
        crop_size = self.crop_size
        fill = self.fill
        prob = self.prob

        def cutout_square(images, *_):
            should_cutout = rand(images.shape[0]) < prob
            for i in my_range(images.shape[0]):
                if should_cutout[i]:
                    # Generate random origin
                    coord = (
                        np.random.randint(images.shape[1] - crop_size + 1),
                        np.random.randint(images.shape[2] - crop_size + 1),
                    )
                    # Black out image in-place
                    images[i, coord[0]:coord[0] + crop_size, coord[1]:coord[1] + crop_size] = fill
            return images

        cutout_square.is_parallel = True
        return cutout_square

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        return replace(previous_state, jit_mode=True), None



class RandomTranslate(Operation):
    """Translate each image randomly in vertical and horizontal directions
    up to specified number of pixels.

    Parameters
    ----------
    padding : int
        Max number of pixels to translate in any direction.
    fill : tuple
        An RGB color ((0, 0, 0) by default) to fill the area outside the shifted image.
    """

    def __init__(self, prob: float, padding: int, fill: Tuple[int, int, int] = (0, 0, 0)):
        super().__init__()
        self.padding = padding
        self.fill = np.array(fill)
        self.prob = prob

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()
        pad = self.padding
        fill = self.fill
        prob = self.prob

        def translate(images, dst):
            n, h, w, _ = images.shape
            dst[:] = fill
            dst[:, pad:pad+h, pad:pad+w] = images
            should_translate = rand(images.shape[0]) < prob
            for i in my_range(n):
                if should_translate[i]:
                    y_coord = np.random.randint(low=0, high=2 * pad + 1)
                    x_coord = np.random.randint(low=0, high=2 * pad + 1)
                    images[i] = dst[i, y_coord:y_coord+h, x_coord:x_coord+w]

            return images

        translate.is_parallel = True
        return translate

    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        h, w, c = previous_state.shape
        return (replace(previous_state, jit_mode=True), \
                AllocationQuery((h + 2 * self.padding, w + 2 * self.padding, c), previous_state.dtype))