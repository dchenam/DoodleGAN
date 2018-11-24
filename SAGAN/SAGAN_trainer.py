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
        # update D network
        _, summary_str, d_loss = self.sess.run([self.d_optim, self.d_sum, self.d_loss], feed_dict=train_feed_dict)

        # update G network
        g_loss = None
        if (counter - 1) % self.n_critic == 0:
            _, summary_str, g_loss = self.sess.run([self.g_optim, self.g_sum, self.g_loss], feed_dict=train_feed_dict)
            self.writer.add_summary(summary_str, counter)
            past_g_loss = g_loss

        # display training status
        counter += 1
        if g_loss == None:
            g_loss = past_g_loss
        print("Epoch: [%2d] [%5d/%5d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
              % (epoch, idx, self.iteration, time.time() - start_time, d_loss, g_loss))
        return loss, acc