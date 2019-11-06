"""Basic registry for densenet model builders."""

from typing import Callable
from typing import Text

import tensorflow as tf

_MODELS = dict()


def register(model: Text) -> Callable:
    """Registers a new densenet model builder under the given model name."""

    def add_to_dict(func: Callable) -> Callable:
        _MODELS[model] = func
        return func

    return add_to_dict


def load_model(model: Text) -> tf.keras.Model:
    """Fetches and invokes the model builder associated with the given model name.

    Args:
        model: The registered name of the densenet model loader.

    Returns:
        A built instance of the model.

    Raises:
        ValueError: if an unknown model name is specified.
    """
    if model not in _MODELS:
        raise Exception("Invalid densenet model builder: {}".format(model))
    return _MODELS[model]()
