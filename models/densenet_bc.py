"""
Build for the DenseNet-BC model [1].

[1] Huang, Gao, et al. "Densely connected convolutional networks." Proceedings of the
IEEE conference on computer vision and pattern recognition. 2017.
"""

from typing import Text
from typing import Tuple

import tensorflow as tf

from . import registry as registry


@registry.register("densenet_bc")
def build(
    input_dim: Tuple[int, int, int], scope: Text = "densenet_bc"
) -> tf.keras.Model:
    raise NotImplementedError("densenet_bc is not yet implemented.")
