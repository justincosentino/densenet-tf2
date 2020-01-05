"""Utilities used for updating device state (e.g., gpu memory)."""

from typing import List

import tensorflow as tf


def set_active_devices(devices: List[str]) -> None:
    """Sets the list of active devices and creates a distributed training strategy."""

    devices_to_use = set([int(gpu) for gpu in devices])

    physical_devices = [
        physical_device
        for i, physical_device in enumerate(
            tf.config.experimental.list_physical_devices("GPU")
        )
        if i in devices_to_use
    ]

    tf.config.experimental.set_visible_devices(physical_devices, "GPU")
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    return tf.distribute.MirroredStrategy()
