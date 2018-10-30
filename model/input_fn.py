"""Create the input data pipeline using `tf.data`"""

import tensorflow as tf


# TODO: Write a appropriate pre-processing functions. For more details goto: https://cs230-stanford.github.io/tensorflow-input-data.html
def input_fn(config, is_training, filenames, labels):
    """Input function for dataset
    Args:
        config: (Params) contains hyperparameters of the model and training configurations
        is_training (bool) whether to use the train or test pipeline.
                    At training, we shuffle the data and have multiple epochs
        filenames: (list) filenames of the images
        labels: (list) corresponding list of labels
    Return:
        Output of the input pipeline can be output of `tf.data` or a placeholder
    """
    num_samples = len(filenames)

    # creates a Dataset object using `tf.data` (optimized input pipeline - faster than loop/generator)
    if is_training:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
                   .shuffle(num_samples)
                   .map(example_train_preprocess, num_parallel_calls=4)
                   .batch(config.batch_size)
                   .prefetch(1)  # prefetch always ensures there is one batch ready to serve
                   )
    else:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
                   .map(example_test_preprocess)
                   .batch(config.batch_size)
                   .prefetch(1)
                   )

    # create one-shot iterator from dataset (or use a re-initializable iterator from dataset)
    images, labels = dataset.make_one_shot_iterator().get_next()


def example_train_preprocess(image, label):
    """Image pre-processing for training.
    Apply the following operations:
        - Horizontally flip the image with probability of 1/2
        ...
    """
    image = tf.image.random_flip_left_right(image)

    return image, label


def example_test_preprocess(image, label):
    """Image pre-processing for test.
    Apply the following operations:
        - Horizontally flip the image with probability of 1/2
        ...
    """
    image = tf.image.random_flip_left_right(image)

    return image, label
