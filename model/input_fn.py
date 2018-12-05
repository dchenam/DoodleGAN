"""Create the input data pipeline using `tf.data`"""

import tensorflow as tf
import functools


# TODO: Write a appropriate pre-processing functions. For more details goto: https://cs230-stanford.github.io/tensorflow-input-data.html
# def input_fn(config, is_training, filenames, labels):
#     """Input function for dataset
#     Args:
#         config: (Params) contains hyperparameters of the model and training configurations
#         is_training (bool) whether to use the train or test pipeline.
#                     At training, we shuffle the data and have multiple epochs
#         filenames: (list) filenames of the images
#         labels: (list) corresponding list of labels
#     Return:
#         Output of the input pipeline can be output of `tf.data` or a placeholder
#     """
#     num_samples = len(filenames)
#
#     # creates a Dataset object using `tf.data` (optimized input pipeline - faster than loop/generator)
#     if is_training:
#         dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
#                    .shuffle(num_samples)
#                    .map(example_train_preprocess, num_parallel_calls=4)
#                    .batch(config.batch_size)
#                    .prefetch(1)  # prefetch always ensures there is one batch ready to serve
#                    )
#     else:
#         dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
#                    .map(example_test_preprocess)
#                    .batch(config.batch_size)
#                    .prefetch(1)
#                    )
#
#     # create one-shot iterator from dataset (or use a re-initializable iterator from dataset)
#     images, labels = dataset.make_one_shot_iterator().get_next()
#
#
# def example_train_preprocess(image, label):
#     """Image pre-processing for training.
#     Apply the following operations:
#         - Horizontally flip the image with probability of 1/2
#         ...
#     """
#     image = tf.image.random_flip_left_right(image)
#
#     return image, label
#
#
# def example_test_preprocess(image, label):
#     """Image pre-processing for test.
#     Apply the following operations:
#         - Horizontally flip the image with probability of 1/2
#         ...
#     """
#     image = tf.image.random_flip_left_right(image)
#
#     return image, label
#

# ===========================================================================================#
"""
    TODO:
    1) need to parse dataset to TFRecord using create_dataset.py (from TF tutorial)
        - will get This will store the data in 10 shards of TFRecord files with 10000 items 
          per class for the training data and 1000 items per class as eval data.
    2) check if data is parsed correctly.
    3) tfrecord_pattern directory
    
"""

# get_input_fn version

# ===========================================================================================#
#
# def get_input_fn(is_training, tfrecord_pattern, batch_size):
#     """Creates an input_fn that stores all the data in memory.
#   Args:
#    mode: one of tf.contrib.learn.ModeKeys.{TRAIN, INFER, EVAL} ---> CHANGED TO (is_training)
#    tfrecord_pattern: path to a TF record file created using create_dataset.py.
#    batch_size: the batch size to output.
#   Returns:
#     A valid input_fn for the model estimator.
#   """
#
#     def _parse_tfexample_fn(example_proto, is_training):
#         """Parse a single record which is expected to be a tensorflow.Example."""
#         feature_to_type = {
#             'ink': tf.VarLenFeature(dtype=tf.float32),
#             "shape": tf.FixedLenFeature([2], dtype=tf.int64)
#         }
#         # if is_training != tf.estimator.ModeKeys.PREDICT:
#         # The labels won't be available at inference time, so don't add them
#         # to the list of feature_columns to be read.
#         feature_to_type["class_index"] = tf.FixedLenFeature([1], dtype=tf.int64)
#
#         parsed_features = tf.parse_single_example(example_proto, feature_to_type)
#         labels = None
#
#         # if mode != tf.estimator.ModeKeys.PREDICT:
#         labels = parsed_features["class_index"]
#         parsed_features["ink"] = tf.sparse_tensor_to_dense(parsed_features["ink"])
#
#         return parsed_features, labels
#
#     def _input_fn():
#         """Estimator `input_fn`.
#             Returns:
#             A tuple of:
#             - Dictionary of string feature name to `Tensor`.
#             - `Tensor` of target labels.
#         """
#         dataset = tf.data.TFRecordDataset.list_files(tfrecord_pattern)
#         if is_training:
#             dataset = dataset.shuffle(buffer_size=10)
#         dataset = dataset.repeat()
#         # Preprocesses 10 files concurrently and interleaves records from each file.
#
#         dataset = dataset.interleave(
#             tf.data.TFRecordDataset,
#             cycle_length=10,
#             block_length=1)
#
#         if is_training:
#             dataset = dataset.map(
#                 functools.partial(_parse_tfexample_fn, is_training=is_training),
#                 num_parallel_calls=10)
#         else:
#             dataset = dataset.map(
#                 functools.partial(_parse_tfexample_fn, is_training=is_training),
#                 )
#
#         # dataset = dataset.map(
#         #     functools.partial(_parse_tfexample_fn, is_training=is_training),
#         #     num_parallel_calls=10)
#
#         dataset = dataset.prefetsch(10000)
#         if is_training:
#             dataset = dataset.shuffle(buffer_size=1000000)
#         # Our inputs are variable length, so pad them.
#         dataset = dataset.padded_batch(
#             batch_size, padded_shapes=dataset.output_shapes)
#         features, labels = dataset.make_one_shot_iterator().get_next()
#
#         return features, labels
#
#     return _input_fn
#
#
#


# ===========================================================================================#

#def get_input_fn(is_training, tfrecord_pattern, batch_size):

def input_fn(config, is_training, filenames):
    """Estimator `input_fn`.
        Returns:
        A tuple of:
        - Dictionary of string feature name to `Tensor`.
        - `Tensor` of target labels.
    """
    dataset = tf.data.TFRecordDataset.list_files(filenames)
    if is_training:
        dataset = dataset.shuffle(buffer_size=10)
    dataset = dataset.repeat()
    # Preprocesses 10 files concurrently and interleaves records from each file.

    dataset = dataset.interleave(
        tf.data.TFRecordDataset,
        cycle_length=10,
        block_length=1)

    if is_training:
        dataset = dataset.map(
            functools.partial(_parse_tfexample_fn, is_training=is_training),
            num_parallel_calls=10)
    else:
        dataset = dataset.map(
            functools.partial(_parse_tfexample_fn, is_training=is_training),
            )

    dataset = dataset.prefetsch(10000)
    if is_training:
        dataset = dataset.shuffle(buffer_size=1000000)
    # Our inputs are variable length, so pad them.
    dataset = dataset.padded_batch(
        config.batch_size, padded_shapes=dataset.output_shapes)
    features, labels = dataset.make_one_shot_iterator().get_next()

    return features, labels


def _parse_tfexample_fn(example_proto, is_training):
    """Parse a single record which is expected to be a tensorflow.Example."""

    features = {"doodle": tf.FixedLenFeature((), dtype=tf.int64),
                "class_index": tf.FixedLenFeature((), tf.int64, default_value=0)}

    parsed_features = tf.parse_single_example(example_proto, features)
    parsed_features["doodle"] = tf.sparse_tensor_to_dense(parsed_features["doodle"])
    labels = parsed_features["class_index"]

    return parsed_features, labels
