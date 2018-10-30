from base import BaseTrain
from tqdm import trange
import numpy as np
import logging
import time


# TODO: change tensorboard summaries to appropriate tags, generate sample of image and write to disk every _ epochs
class ExampleTrainer(BaseTrain):
    def __init__(self, sess, model, inputs, config, logger):
        super(ExampleTrainer, self).__init__(sess, model, inputs, config, logger)

    def train(self):
        """overrode default base function for custom function
            - logs total duration of training in seconds
        """
        tik = time.time()
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs + 1, 1):
            self.train_epoch()
            if cur_epoch % self.config.save_iter == 0:
                self.model.save(self.sess)
            self.sess.run(self.model.increment_cur_epoch_tensor)
        tok = time.time()
        logging.info('Duration: {} seconds'.format(tok - tik))

    def train_epoch(self):
        """logging summary to tensorboard and executing training steps per epoch"""
        losses = []
        accs = []
        for it in trange(self.config.num_iter_per_epoch):
            loss, acc = self.train_step()
            losses.append(loss)
            accs.append(acc)
        loss = np.mean(losses)
        acc = np.mean(accs)

        cur_it = self.model.global_step_tensor.eval(self.sess)

        summaries_dict = {}
        summaries_dict['loss'] = loss
        summaries_dict['acc'] = acc
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)

    def train_step(self):
        """using `tf.data` API, so no feed-dict required"""
        _, loss, acc = self.sess.run([self.model.train_step, self.model.cross_entropy, self.model.accuracy])
        return loss, acc
