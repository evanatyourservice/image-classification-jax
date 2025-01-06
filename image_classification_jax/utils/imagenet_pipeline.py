# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ImageNet input pipeline.

Modified from original flax version."""
from typing import Union

import jax
import tensorflow as tf
import tensorflow_datasets as tfds

from image_classification_jax.utils.tf_preprocessing_tools import random_erasing


IMAGE_SIZE = 224
MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]


def _resize(image):
    return tf.image.resize([image], [IMAGE_SIZE, IMAGE_SIZE])[0]


def _decode_and_random_crop(image_bytes):
    shape = tf.io.extract_jpeg_shape(image_bytes)
    begin, crop_size, _ = tf.image.sample_distorted_bounding_box(
        shape,
        tf.zeros([0, 0, 4], tf.float32),
        area_range=(0.08, 1.0),
        aspect_ratio_range=(3.0/4, 4.0/3.0),
        min_object_covered=0,  # Don't enforce a minimum area
        use_image_if_no_bounding_boxes=True)

    # Crop the image to the specified bounding box
    offset_y, offset_x, _ = tf.unstack(begin)
    target_height, target_width, _ = tf.unstack(crop_size)
    crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
    image = tf.io.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
    return _resize(image)


def _decode_and_center_crop(image_bytes):
    shape = tf.io.extract_jpeg_shape(image_bytes)
    image_height = shape[0]
    image_width = shape[1]

    center_crop_size = tf.minimum(image_height, image_width)

    offset_height = (image_height - center_crop_size) // 2
    offset_width = (image_width - center_crop_size) // 2
    crop_window = tf.stack(
        [offset_height, offset_width, center_crop_size, center_crop_size]
    )
    image = tf.io.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
    return _resize(image)


def normalize_image(image):
    image -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=image.dtype)
    image /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=image.dtype)
    return image


def preprocess_for_train(image_bytes, dtype=tf.float32):
    image = _decode_and_random_crop(image_bytes)

    image = tf.image.random_flip_left_right(image)

    # a little extra augmentation
    image = random_erasing(image, probability=0.5, min_area=0.02, max_area=0.1)

    image = normalize_image(image)
    image = tf.image.convert_image_dtype(image, dtype=dtype)
    return image


def preprocess_for_eval(image_bytes, dtype=tf.float32):
    image = _decode_and_center_crop(image_bytes)
    image = normalize_image(image)
    image = tf.image.convert_image_dtype(image, dtype=dtype)
    return image


def _add_tpu_host_options(data):
    options = tf.data.Options()
    options.threading.private_threadpool_size = 48
    options.threading.max_intra_op_parallelism = 1

    # Stop a whole bunch of magic stuff that eats up all RAM:
    options.experimental_optimization.inject_prefetch = False

    return data.with_options(options)


def create_split(
    dataset_builder: Union[str, tfds.core.DatasetBuilder],
    batch_size,
    train,
    platform,
    dtype=tf.float32,
    shuffle_buffer_size=2_000,
    prefetch=2,
    cache=False,
):
    """Creates a split from the ImageNet dataset using TensorFlow Datasets.

    Args:
      dataset_builder: TFDS dataset builder or str of gcs path.
      batch_size: the batch size returned by the data pipeline.
      train: Whether to load the train or evaluation split.
      platform: The jax device platform program is running on.
      dtype: data type of the image.
      shuffle_buffer_size: Size of the shuffle buffer.
      prefetch: Number of items to prefetch in the dataset.
      cache: Whether to cache the dataset.
    Returns:
      A `tf.data.Dataset`.
    """
    if isinstance(dataset_builder, str):
        feature_description = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.int64),
            "synset": tf.io.FixedLenFeature([], tf.string),
            "class_name": tf.io.FixedLenFeature([], tf.string),
        }

        def parse_tfrecord(example_proto):
            parsed = tf.io.parse_single_example(example_proto, feature_description)
            return {"image": parsed["image"], "label": parsed["label"]}

        split = "train" if train else "val"
        file_pattern = f"{dataset_builder}/{split}/images*.tfrecord"
        filenames = tf.io.gfile.glob(file_pattern)
        filenames.sort()
        if not filenames:
            raise ValueError(f"No TFRecord files found at {file_pattern}")
        filenames = filenames[jax.process_index()::jax.process_count()]
        print(f"Process {jax.process_index()}: Using {len(filenames)} files from {split} split")
        ds = tf.data.TFRecordDataset(
            filenames, num_parallel_reads=16 if train else 4
        )
        ds = ds.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        if train:
            train_examples = dataset_builder.info.splits["train"].num_examples
            split_size = train_examples // jax.process_count()
            start = jax.process_index() * split_size
            split = f"train[{start}:{start + split_size}]"
        else:
            validate_examples = dataset_builder.info.splits["validation"].num_examples
            split_size = validate_examples // jax.process_count()
            start = jax.process_index() * split_size
            split = f"validation[{start}:{start + split_size}]"

        ds = dataset_builder.as_dataset(
            split=split, decoders={"image": tfds.decode.SkipDecoding()}
        )

    if platform == "tpu":
        ds = _add_tpu_host_options(ds)

    if platform != "cpu" and cache:
        ds = ds.cache()

    ds = ds.repeat()
    ds = ds.shuffle(shuffle_buffer_size)

    def decode_example(example):
        if train:
            image = preprocess_for_train(example["image"], dtype)
        else:
            image = preprocess_for_eval(example["image"], dtype)
        return {"image": image, "label": example["label"]}

    ds = ds.map(
        decode_example,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    ds = ds.batch(
        batch_size,
        drop_remainder=True,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    ds = ds.map(
        split_batch,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    if prefetch > 0:
        ds = ds.prefetch(prefetch)

    return ds.as_numpy_iterator()


def split_batch(batch):
    return tf.nest.map_structure(
        lambda x: tf.reshape(
            x,
            (jax.local_device_count(), x.shape[0] // jax.local_device_count())
            + x.shape[1:],
        ),
        batch,
    )
