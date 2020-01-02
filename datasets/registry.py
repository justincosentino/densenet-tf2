"""Basic registry for dataset loaders."""

from typing import Callable
from typing import Text
from typing import Tuple

import tensorflow as tf
import tensorflow_datasets as tfds

_DATASETS = dict()


def register(dataset: Text) -> Callable:
    """Registers a new dataset loader under the given dataset name."""

    def add_to_dict(func: Callable) -> Callable:
        _DATASETS[dataset] = func
        return func

    return add_to_dict


def load_dataset(
    dataset: Text, batch_size: int, train_size: int, val_size: int
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tfds.core.DatasetInfo]:
    """Fetches and invokes the dataset loader associated with the given dataset name.

    Args:
        dataset: The registered name of the dataset loader.

    Returns:
        The train, validation, and test datasets.

    Raises:
        ValueError: if an unknown dataset name is specified.
    """
    if dataset not in _DATASETS:
        raise Exception("Invalid dataset loader: {}".format(dataset))
    return _DATASETS[dataset](
        batch_size=batch_size, train_size=train_size, val_size=val_size
    )
