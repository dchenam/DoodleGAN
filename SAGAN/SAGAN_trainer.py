from base import BaseTrain
from tqdm import trange
import numpy as np
import logging
import time

# TODO: change tensorboard summaries to appropriate tags, generate sample of image and write to disk every _ epochs
class SAGANTrainer(BaseTrain):
    def __init__(self, sess, model, inputs, config, logger):
        super(SAGANTrainer, self).__init__(sess, model, inputs, config, logger)

    def train(self):
        """overrode default base function for custom function
            - logs total duration of training in seconds
        """
        counter = 1
        past_g_loss = -1.
        tik = time.time()
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs + 1, 1):
            self.train_epoch(counter, past_g_loss)
            if cur_epoch % self.config.save_iter == 0:
                self.model.save(self.sess)
            self.sess.run(self.model.increment_cur_epoch_tensor)
        tok = time.time()
        logging.info('Duration: {} seconds'.format(tok - tik))

    def train_epoch(self, counter, past_g_loss):
        """logging summary to tensorboard and executing training steps per epoch"""
        g_losses = []
        d_losses =[]
        g_accs = []
        d_accs = []
        for it in trange(self.config.num_iter_per_epoch):
            d_acc, d_loss, g_acc, g_loss= self.train_step(counter, past_g_loss)
            g_losses.append(g_loss)
            g_accs.append(g_acc)
            d_losses.append(d_loss)
            d_accs.append(d_acc)
        g_loss = np.mean(g_losses)
        g_acc = np.mean(g_accs)
        d_loss = np.mean(d_losses)
        d_acc = np.mean(d_accs)

        cur_it = self.model.global_step_tensor.eval(self.sess)

        summaries_dict = {}
        summaries_dict['g_loss'] = g_loss
        summaries_dict['g_acc'] = g_acc
        summaries_dict['d_loss'] = d_loss
        summaries_dict['d_acc'] = d_acc
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)

    def train_step(self, counter, past_g_loss):
        """using `tf.data` API, so no feed-dict required"""
        # update D network
        d_acc, d_loss = self.sess.run([self.d_optim, self.d_sum, self.d_loss])

        # update G network
        g_loss = None
        if (counter - 1) % self.config.n_critic == 0:
            g_acc, g_loss = self.sess.run([self.g_optim, self.g_sum, self.g_loss])
            self.writer.add_summary(summary_str, counter)
            past_g_loss = g_loss

        # display training status
        counter += 1
        if g_loss == None:
            g_loss = past_g_loss

        return d_acc, d_loss, g_acc, g_loss
    