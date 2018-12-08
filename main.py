import os
import logging
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from shutil import copy

from model.input_fn import input_fn
from model.models import GAN
from model.trainer import Trainer
from utils import *


# TODO: add distribution strategy support (mirrored, all-reduce, or parameter-server)
# TODO: possibly change API to estimators or tf.contrib.gan estimators

def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.sample_dir, config.checkpoint_dir])
    copy(args.config, config.model_dir)

    # create tensorflow session
    sess = tf.Session()

    is_training = True

    # filenames should be type of file pattern
    filenames = 'training_data/training.tfrecord-?????-of-?????'

    # create your data input pipeline
    # in the features - class_index & doodle
    input = input_fn(config, filenames)

    feat, label = input
    feature = sess.run(feat)
    print(feature.shape)
    print(np.max(feature))
    print(np.min(feature))

    # create instance of the model you want
    model = GAN(config, input)

    # create tensorboard & terminal logger
    logger = Logger(sess, config)
    logger.set_logger(log_path=os.path.join(
        config.model_dir, args.mode + '.log'))

    # enter training or testing mode
    if is_training:
        logging.info(config.exp_description)
        logging.info("creating trainer...")

        # create trainer and path all previous components to it
        trainer = Trainer(sess, model, config, logger)

        # here you train your model
        trainer.train()

    else:
        # load latest checkpoint
        model.load(sess)


if __name__ == '__main__':
    main()
