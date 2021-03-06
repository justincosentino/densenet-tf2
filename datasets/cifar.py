"""Utilities for loading the cifar10 and cifar100 datasets."""

from typing import Callable
from typing import Text
from typing import Tuple

import tensorflow as tf
import tensorflow_datasets as tfds

import datasets.registry as registry


def _normalize_image_wrapper(
    mean: Tuple[float, float, float], stddev: Tuple[float, float, float]
) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:

    mean = tf.reshape(tf.constant(mean, dtype=tf.float64), [1, 1, 3])
    stddev = tf.reshape(tf.constant(stddev, dtype=tf.float64), [1, 1, 3])

    # TODO: move this back out of the load dataset method. Done here so that we can
    # predeclare mean / stddev without redeclaring for each image. This is placed within
    # this function's scope so that mean / stddev aren't declared when importing the
    # `datasets` library, which prevents us from changing the list of visible devices.
    def _normalize_image(
        image: tf.Tensor, label: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Normalizes an image and returns a given supervised training pair."""
        image = tf.cast(image, tf.float64)
        image = tf.math.subtract(image, mean)
        image = tf.math.divide(image, stddev)
        return image, label

    return _normalize_image


def _pad_image(
    image: tf.Tensor, label: tf.Tensor, padding: int = 4
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Pads an image and returns a given supervised training pair."""
    image = tf.pad(
        image, [[padding, padding], [padding, padding], [0, 0]], mode="CONSTANT"
    )
    return image, label


def _crop_image(
    image: tf.Tensor, label: tf.Tensor, out_size: Tuple[int, int, int] = None
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Randomly crops an image and returns a given supervised training pair."""
    if out_size is None:
        out_size = (32, 32, 3)
    image = tf.image.random_crop(image, out_size)
    return image, label


def _random_flip(
    image: tf.Tensor, label: tf.Tensor, prob=0.5
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Randomly flips an image and returns a given supervised training pair."""
    if tf.random.uniform(()) > 1 - prob:
        image = tf.image.flip_left_right(image)
    return image, label


def _load_dataset(
    dataset_name: Text,
    train_size: int,
    val_size: int,
    mean: Tuple[float, float, float],
    stddev: Tuple[float, float, float],
    batch_size: int = 64,
    shuffle_buffer_size: int = 1000,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tfds.core.DatasetInfo]:
    """Loads and preprocesses a given dataset using the tf2 tensorflow_datasets S3 API.

    Args:
        dataset_name: the dataset's S3 name.
        train_size: the number of training examples.
        val_size: the number of validation examples.
        batch_size: the batch size.
        shuffle_buffer_size: the size of the buffer used when shuffling data.

    Returns:
        The train, val, and test datasets and dataset info.
    """
    (train, val, test), info = tfds.load(
        dataset_name,
        as_supervised=True,
        split=[
            "train[0:{}]".format(train_size),
            "train[{}:{}]".format(train_size, train_size + val_size),
            "test",
        ],
        with_info=True,
    )

    train = (
        train.map(
            _normalize_image_wrapper(mean, stddev),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        .map(_pad_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .cache()
        .shuffle(shuffle_buffer_size)
        .map(_crop_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .map(_random_flip, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .batch(batch_size)
        .repeat()
    )

    val = (
        val.map(
            _normalize_image_wrapper(mean, stddev),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        # .shuffle(shuffle_buffer_size)
        .batch(batch_size).cache()
    )

    test = (
        test.map(
            _normalize_image_wrapper(mean, stddev),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        # .shuffle(shuffle_buffer_size)
        .batch(batch_size).cache()
    )

    return train, val, test, info


@registry.register("cifar10")
def load_cifar10(
    batch_size: int = 64, train_size: int = 45000, val_size: int = 5000
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tfds.core.DatasetInfo]:
    """Loads and preprocesses the cifar10 dataset.

    Args:
        batch_size: the batch size.
        train_size: the number of training examples.
        val_size: the number of validation examples.

    Returns:
        The train, val, and test datasets.
    """
    return _load_dataset(
        "cifar10:3.*.*",
        mean=(125.3, 123.0, 113.9),
        stddev=(63.0, 62.1, 66.7),
        train_size=train_size,
        val_size=val_size,
        batch_size=batch_size,
    )


@registry.register("cifar100")
def load_cifar100(
    batch_size: int = 64, train_size: int = 45000, val_size: int = 5000
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tfds.core.DatasetInfo]:
    """Loads and preprocesses the cifar100 dataset.

    Args:
        batch_size: the batch size.
        train_size: the number of training examples.
        val_size: the number of validation examples.

    Returns:
        The train, val, and test datasets.
    """
    return _load_dataset(
        "cifar100:3.*.*",
        mean=(129.3, 124.1, 112.4),
        stddev=(68.2, 65.4, 70.4),
        train_size=train_size,
        val_size=val_size,
        batch_size=batch_size,
    )
