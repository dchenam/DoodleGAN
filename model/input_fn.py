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
               .prefetch(config.batch_size)
               )
    features, labels = dataset.make_one_shot_iterator().get_next()

    return (features, labels)


def parse_fn(drawit_proto):
    """Parse a single record which is expected to be a tensorflow.Example."""
    num_classes = 345

    features = {"doodle": tf.FixedLenFeature((28 * 28), dtype=tf.int64),
                "class_index": tf.FixedLenFeature((), tf.int64, default_value=0)}

    parsed_features = tf.parse_single_example(drawit_proto, features)

    labels = parsed_features["class_index"]
    labels = tf.one_hot(labels, num_classes)

    features = parsed_features['doodle']

    features = tf.reshape(features, [28, 28, 1])
    features = tf.cast(features, tf.uint8)

    # convert from 0 - 255 to 0 - 1
    features = tf.image.convert_image_dtype(features, tf.float32)

    # normalize images from 0 - 1 to between -1 and 1
    features = tf.multiply(features, 2)
    features = tf.subtract(features, 1)

    return features, labels


def mnist_input(config):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


    def train_preprocess(image, label):
        image = (image / 127.5) - 1
        label = tf.one_hot(label, 345)
        return image, label


    def create_mnist_dataset(data, labels, batch_size):
      def gen():
        for image, label in zip(data, labels):
            yield image, label
      ds = tf.data.Dataset.from_generator(gen, (tf.float32, tf.int32), ((28,28 ), ()))

      return ds.map(train_preprocess).repeat().batch(batch_size)

    #train and validation dataset with different batch size
    train_dataset = create_mnist_dataset(x_train, y_train, config.batch_size)
    valid_dataset = create_mnist_dataset(x_test, y_test, config.batch_size)
    
    image, label = train_dataset.make_one_shot_iterator().get_next()
    
    return (image, label)