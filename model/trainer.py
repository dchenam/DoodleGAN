from base import BaseTrain
from utils import denorm

from tqdm import trange
import tensorflow as tf
import numpy as np
import skimage.io
import logging
import time
import os


# TODO: change tensorboard summaries to appropriate tags, generate sample of image and write to disk every _ epochs
class Trainer(BaseTrain):
    def __init__(self, sess, model, config, logger):
        super(Trainer, self).__init__(sess, model, config, logger)

    def train(self):
        """overrode default base function for custom function
             - logs total duration of training in seconds
         """
        tik = time.time()
        for it in trange(self.config.num_iter):
            g_loss, d_loss, d_loss1, d_loss2, d_loss3 = self.train_step()
            if it % self.config.save_iter == 0:
                self.model.save(self.sess)
            if it % self.config.sample_iter == 0:
                images = self.sess.run([self.model.sample_image])

                # tf image summary for cloud servers
                # summaries_dict = {}
                # summaries_dict['sample_image'] = np.array(image).reshape([1, 28, 28, 1])
                # self.logger.summarize(it, summaries_dict=summaries_dict)

                for i, image in enumerate(images[0]):
                    image = denorm(np.squeeze(image))
                    sample_path = os.path.join(self.config.sample_dir, '{}-{}-sample.jpg'.format(i, it))
                    skimage.io.imsave(sample_path, image)

            if it % 100 == 0:
                summaries_dict = {}
                summaries_dict['g_loss'] = g_loss
                summaries_dict['d_loss'] = d_loss
                summaries_dict['d_real_loss'] = d_loss1
                summaries_dict['d_wrong_loss'] = d_loss2
                summaries_dict['d_fake_loss'] = d_loss3
                self.logger.summarize(it, summaries_dict=summaries_dict)

        tok = time.time()
        logging.info('Duration: {} seconds'.format(tok - tik))

    def train_step(self):
        """using `tf.data` API, so no feed-dict required"""
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            _, d_loss, d_loss1, d_loss2, d_loss3 = self.sess.run(
                [self.model.d_optim, self.model.d_loss, self.model.d_loss1, self.model.d_loss2, self.model.d_loss3])
            _, g_loss = self.sess.run([self.model.g_optim, self.model.g_loss])
        return g_loss, d_loss, d_loss1, d_loss2, d_loss3
