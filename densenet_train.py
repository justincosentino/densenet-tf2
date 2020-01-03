"""Trains a DenseNet model. Logs and checkpoints are written to the given output dir."""

import os
from typing import Any
from typing import Text

import tensorflow as tf
from absl import app
from absl import flags
from absl import logging

import datasets.registry as datasets_registry
import models.registry as models_registry
from utils import disk_utils
from utils import gpu_utils

FLAGS = flags.FLAGS

flags.DEFINE_list("devices", "0,1,2,3", "a comma-separated list of gpu device ids.")
flags.DEFINE_integer("epochs", 300, "the number of training epochs.", lower_bound=1)
flags.DEFINE_integer("batch_size", 64, "the training batch size.", lower_bound=1)
flags.DEFINE_integer("train_size", 45000, "the training set size.", lower_bound=1)
flags.DEFINE_integer("val_size", 5000, "the validation set size.", lower_bound=0)
flags.DEFINE_enum(
    "dataset", "cifar10", ["cifar10", "cifar100", "svhn", "imagenet"], "the dataset."
)
flags.DEFINE_enum(
    "model",
    "densenet_32x32",
    ["densenet_32x32", "densenet121", "densenet161", "densenet169", "densenet201"],
    "the densenet model.",
)
flags.DEFINE_float(
    "lr",
    0.1,
    "the intial learning rate; divided by 10 twice (epoch # depends on dataset)",
    lower_bound=0,
)
flags.DEFINE_string(
    "output_dir", None, "output directory containing checkpoints and results."
)
flags.DEFINE_boolean("debug", False, "display debug output.")

flags.mark_flag_as_required("output_dir")


def run_train_loop(base_dir: Text, checkpoint_dir: Text, logging_dir: Text) -> None:
    """Run the overall training loop.

    Initializes metrics, summaries, checkpoints, etc.
    """

    # Set active gpus and allow for memory growth
    strategy = gpu_utils.set_active_devices(FLAGS.devices)

    # Init data, model, and optimizer.
    train_data, val_data, test_data, info = datasets_registry.load_dataset(
        FLAGS.dataset, FLAGS.batch_size, FLAGS.train_size, FLAGS.val_size
    )

    with strategy.scope():

        model = models_registry.load_model(FLAGS.model, {})

        optimizer = tf.keras.optimizers.SGD(
            learning_rate=FLAGS.lr, momentum=0.9, nesterov=True
        )

        # Specify the training configuration (optimizer, loss, metrics)
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )

    def scheduler(epoch: int) -> float:
        if "cifar10" == FLAGS.dataset or "cifar100" == FLAGS.dataset:
            if epoch < FLAGS.epochs * 0.5:
                return FLAGS.lr
            if epoch >= FLAGS.epochs * 0.5 and epoch < FLAGS.epochs * 0.75:
                return FLAGS.lr / 10.0
            return FLAGS.lr / 10.0 / 10.0

        if "imagenet" == FLAGS.dataset:
            if epoch < 30:
                return FLAGS.lr
            if epoch >= 30 and epoch < 60:
                return FLAGS.lr / 10.0
            return FLAGS.lr / 10.0 / 10.0

        raise ValueError("bad dataset")

    lr_sched_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=logging_dir, profile_batch=0
    )

    # TODO: consider adding early stopping?
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(
            checkpoint_dir, "checkpoint-weights-{epoch:02d}-{val_loss:.2f}.ckpt"
        ),
        save_weights_only=True,
        verbose=1,
        save_best_only=True,
    )

    # Write model summary to output dir.
    disk_utils.write_model_summary(model, base_dir)

    _ = model.fit(
        x=train_data,
        # TODO: consider passing in # epochs to the dataloader and removing this:
        steps_per_epoch=FLAGS.train_size // FLAGS.batch_size,
        epochs=FLAGS.epochs,
        callbacks=[lr_sched_callback, tensorboard_callback, checkpoint_callback],
        validation_data=val_data,
        verbose=2,
    )

    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    results = model.evaluate(test_data)
    print("\ntest loss {}, test acc: {}".format(results[0], results[1]))


def main(argv: Any):
    del argv  # Unused

    # Build model output dir.
    base_dir = os.path.join(FLAGS.output_dir, FLAGS.model)
    if os.path.exists(base_dir):
        raise ValueError("output_dir already exists: {}.".format(base_dir))
    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    logging_dir = os.path.join(base_dir, "logging")
    os.makedirs(base_dir)
    os.mkdir(checkpoint_dir)
    os.mkdir(logging_dir)

    # Write flags to output dir.
    disk_utils.write_flags(FLAGS, base_dir, "flags_train.txt")

    if FLAGS.debug:
        logging.set_verbosity(logging.DEBUG)
    else:
        logging.set_verbosity(logging.WARNING)

    run_train_loop(base_dir, checkpoint_dir, logging_dir)


if __name__ == "__main__":
    app.run(main)
