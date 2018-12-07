import tensorflow as tf
import functools


def input_fn(config, filenames):
    """Estimator `input_fn`.
    """
    # Preprocess 10 files concurrently and interleaves records from each file.
    dataset = tf.data.TFRecordDataset.list_files(filenames)
    dataset = dataset.shuffle(buffer_size=10)
    dataset = dataset.repeat()

    dataset = dataset.interleave(
        tf.data.TFRecordDataset,
        cycle_length=10,
        block_length=1)

    dataset = (dataset
               .map(functools.partial(parse_fn), num_parallel_calls=10)
               .shuffle(buffer_size=1000000)
               .repeat()
               .batch(config.batch_size)
               .prefetch(1)
               )
    features, labels = dataset.make_one_shot_iterator().get_next()

    return features, labels


def parse_fn(drawit_proto):
    """Parse a single record which is expected to be a tensorflow.Example."""
	# return 28*28 2d bitmap as feature, and a one-hot label

    num_classes = 345

    features = {"doodle": tf.FixedLenFeature((28 * 28), dtype=tf.int64),
                "class_index": tf.FixedLenFeature((), tf.int64, default_value=0)}

    parsed_features = tf.parse_single_example(drawit_proto, features)

    labels = parsed_features["class_index"]
    labels = tf.one_hot(labels, num_classes)

    features = parsed_features['doodle']
    features = tf.cast(features, tf.float32)

    return features, labels