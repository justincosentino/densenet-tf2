"""
Build for the DenseNet-C model [1].

[1] Huang, Gao, et al. "Densely connected convolutional networks." Proceedings of the
IEEE conference on computer vision and pattern recognition. 2017.
"""

from typing import Text
from typing import Tuple

import tensorflow as tf

from . import registry as registry


@registry.register("densenet_c")
def build(
    input_dim: Tuple[int, int, int], scope: Text = "densenet_c"
) -> tf.keras.Model:
    raise NotImplementedError("densenet_c is not yet implemented.")
