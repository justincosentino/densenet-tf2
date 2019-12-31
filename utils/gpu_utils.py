"""Utilities used for updating device state (e.g., gpu memory)."""

from typing import List

import tensorflow as tf


def set_active_devices(devices: List[str]) -> None:
    """Sets the list of active devices and enables memory growth limits."""
    device_ids = set([int(gpu) for gpu in devices])
    gpu_ids = [
        gpu_id
        for i, gpu_id in enumerate(tf.config.experimental.list_physical_devices("GPU"))
        if i in device_ids
    ]

    if gpu_ids:
        tf.config.experimental.set_visible_devices(gpu_ids, "GPU")
        for gpu_id in gpu_ids:
            tf.config.experimental.set_memory_growth(gpu_id, True)
