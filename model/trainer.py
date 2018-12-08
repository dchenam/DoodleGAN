from base import BaseTrain
from tqdm import trange
import numpy as np
import skimage.io
import logging
import time
import os
from utils import denorm


# TODO: change tensorboard summaries to appropriate tags, generate sample of image and write to disk every _ epochs
class Trainer(BaseTrain):
    def __init__(self, sess, model, config, logger):
        super(Trainer, self).__init__(sess, model, config, logger)

    def train(self):
        """overrode default base function for custom function
            - logs total duration of training in seconds
        """
        tik = time.time()
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs + 1, 1):
            self.train_epoch()
            self.sess.run(self.model.increment_cur_epoch_tensor)
        tok = time.time()
        logging.info('Duration: {} seconds'.format(tok - tik))

    def train_epoch(self):
        """logging summary to tensorboard and executing training steps per epoch"""
        g_losses = []
        d_losses = []
        for it in trange(self.config.num_iter_per_epoch):
            cur_it = self.model.global_step_tensor.eval(self.sess)
            g_loss, d_loss = self.train_step()
            g_losses.append(g_loss)
            d_losses.append(d_loss)
            if cur_it % self.config.save_iter == 0:
                self.model.save(self.sess)
            if cur_it % self.config.sample_iter == 0:
                image = self.sess.run([self.model.sample_image])
                image = denorm(np.squeeze(image))
                sample_path = os.path.join(self.config.sample_dir, '{}-sample.jpg'.format(cur_it))
                skimage.io.imsave(sample_path, image)
            if cur_it % 100 == 0:
                summaries_dict = {}
                summaries_dict['g_loss'] = g_loss
                summaries_dict['d_loss'] = d_loss
                self.logger.summarize(cur_it, summaries_dict=summaries_dict)

        g_loss = np.mean(g_losses)
        d_loss = np.mean(d_losses)

        cur_it = self.model.global_step_tensor.eval(self.sess)

        summaries_dict = {}
        summaries_dict['average_epoch_g_loss'] = g_loss
        summaries_dict['average_epoch_d_loss'] = d_loss
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)

    def train_step(self):
        """using `tf.data` API, so no feed-dict required"""
        _, d_loss = self.sess.run([self.model.d_optim, self.model.d_loss])
        _, g_loss = self.sess.run([self.model.g_optim, self.model.g_loss])
        return g_loss, d_loss
