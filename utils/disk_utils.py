"""Shared utilities for writing or reading from disk."""

import io
import os
from typing import Text

import tensorflow as tf
from absl import flags
from absl import logging


def wrapped_log(string: Text, char: Text = "+", width: int = 88):
    """Logs a string wrapped by lines of the given char and width."""
    logging.info("\n".join(["", char * width, string.strip(), char * width, ""]))


def write_flags(
    flags: flags._flagvalues.FlagValues, output_dir: Text, file_name: Text = "flags.txt"
):
    """Saves a copy of the absl flags to the specified dir and file."""
    print(type(flags))
    flag_str = flags.flags_into_string()
    wrapped_log(flag_str)
    with open(os.path.join(output_dir, file_name), "wt") as f:
        f.write(flag_str)


def write_model_summary(
    model: tf.keras.Model, output_dir: Text, file_name: Text = "model_summary.txt"
) -> Text:
    """Captures, writes, and returns the model's summary.

    Args:
        model: the model.
        output_dir: the path to the output directory.

    Returns:
        The model summary.

    Raises:
        ValueError: if the output dir does not exist.
    """
    if not os.path.exists(output_dir):
        raise ValueError("output dir does not exist: {}".format(output_dir))

    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + "\n"))
    summary_string = stream.getvalue()
    stream.close()

    with open(os.path.join(output_dir, file_name), "wt") as f:
        f.write(summary_string)

    return summary_string
