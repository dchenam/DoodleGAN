import os
import logging
import tensorflow as tf
from shutil import copy

from model.input_fn import input_fn
from model.example_model import ExampleModel
from model.example_trainer import ExampleTrainer
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
    create_dirs([config.summary_dir, config.checkpoint_dir])
    copy(args.config, config.model_dir)

    # create tensorflow session
    sess = tf.Session()

    # check training or testing
    is_training = True if args.mode == 'train' else False

    # create your data input pipeline
    inputs = input_fn(config, is_training)

    # create instance of the model you want
    model = ExampleModel(config, is_training, inputs)

    # create tensorboard & terminal logger
    logger = Logger(sess, config)
    logger.set_logger(log_path=os.path.join(
        config.model_dir, args.mode + '.log'))

    # enter training or testing mode
    if is_training:
        logging.info(config.exp_description)
        logging.info("creating trainer...")

        # create trainer and path all previous components to it
        trainer = ExampleTrainer(sess, model, inputs, config, logger)

        # here you train your model
        trainer.train()

    else:
        # load latest checkpoint
        model.load(sess)


if __name__ == '__main__':
    main()
