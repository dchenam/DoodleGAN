from base import BaseModel
import tensorflow as tf
from ops import *
from utils import *
from tensorflow.contrib.data import prefetch_to_device, shuffle_and_repeat, map_and_batch

# TODO: Write a Generator and a Discriminator Model
class SAGANModel(BaseModel):
    def __init__(self, config, is_training, inputs):
        """Create Model Class
        Args:
            config: (parameters) model hyperparameters
            is_training: (bool) whether we are training or not
            inputs: (dict) contrains the inputs of the graph (features, labels...)
                    this can be 'tf.placeholder' or outputs of 'tf.data'
        """
        super(SAGANModel, self).__init__(config)
        self.inputs = inputs
        self.build_model()
        self.init_saver()

    def generator(self, z, is_training=True, reuse=False):
        with tf.variable_scope("generator", reuse=reuse):
            ch = 1024
            x = deconv(z, channels=ch, kernel=4, stride=1, padding='VALID', use_bias=False, sn=self.sn, scope='deconv')
            x = batch_norm(x, is_training, scope='batch_norm')
            x = relu(x)

            for i in range(self.layer_num // 2):
                if self.up_sample:
                    x = up_sample(x, scale_factor=2)
                    x = conv(x, channels=ch // 2, kernel=3, stride=1, pad=1, sn=self.sn, scope='up_conv_' + str(i))
                    x = batch_norm(x, is_training, scope='batch_norm_' + str(i))
                    x = relu(x)

                else:
                    x = deconv(x, channels=ch // 2, kernel=4, stride=2, use_bias=False, sn=self.sn,
                               scope='deconv_' + str(i))
                    x = batch_norm(x, is_training, scope='batch_norm_' + str(i))
                    x = relu(x)

                ch = ch // 2

            # Self Attention
            x = self.attention(x, ch, sn=self.sn, scope="attention", reuse=reuse)

            for i in range(self.layer_num // 2, self.layer_num):
                if self.up_sample:
                    x = up_sample(x, scale_factor=2)
                    x = conv(x, channels=ch // 2, kernel=3, stride=1, pad=1, sn=self.sn, scope='up_conv_' + str(i))
                    x = batch_norm(x, is_training, scope='batch_norm_' + str(i))
                    x = relu(x)

                else:
                    x = deconv(x, channels=ch // 2, kernel=4, stride=2, use_bias=False, sn=self.sn,
                               scope='deconv_' + str(i))
                    x = batch_norm(x, is_training, scope='batch_norm_' + str(i))
                    x = relu(x)

                ch = ch // 2

            if self.up_sample:
                x = up_sample(x, scale_factor=2)
                x = conv(x, channels=self.c_dim, kernel=3, stride=1, pad=1, sn=self.sn, scope='G_conv_logit')
                x = tanh(x)

            else:
                x = deconv(x, channels=self.c_dim, kernel=4, stride=2, use_bias=False, sn=self.sn,
                           scope='G_deconv_logit')
                x = tanh(x)

            return x

    def discriminator(self, x, is_training=True, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse):
            ch = 64
            x = conv(x, channels=ch, kernel=4, stride=2, pad=1, sn=self.sn, use_bias=False, scope='conv')
            x = lrelu(x, 0.2)

            for i in range(self.layer_num // 2):
                x = conv(x, channels=ch * 2, kernel=4, stride=2, pad=1, sn=self.sn, use_bias=False,
                         scope='conv_' + str(i))
                x = batch_norm(x, is_training, scope='batch_norm' + str(i))
                x = lrelu(x, 0.2)

                ch = ch * 2

            # Self Attention
            x = self.attention(x, ch, sn=self.sn, scope="attention", reuse=reuse)

            for i in range(self.layer_num // 2, self.layer_num):
                x = conv(x, channels=ch * 2, kernel=4, stride=2, pad=1, sn=self.sn, use_bias=False,
                         scope='conv_' + str(i))
                x = batch_norm(x, is_training, scope='batch_norm' + str(i))
                x = lrelu(x, 0.2)

                ch = ch * 2

            x = conv(x, channels=4, stride=1, sn=self.sn, use_bias=False, scope='D_logit')

            return x

    def attention(self, x, ch, sn=False, scope='attention', reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            f = conv(x, ch // 8, kernel=1, stride=1, sn=sn, scope='f_conv')  # [bs, h, w, c']
            g = conv(x, ch // 8, kernel=1, stride=1, sn=sn, scope='g_conv')  # [bs, h, w, c']
            h = conv(x, ch, kernel=1, stride=1, sn=sn, scope='h_conv')  # [bs, h, w, c]

            # N = h * w
            s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

            beta = tf.nn.softmax(s, axis=-1)  # attention map

            o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
            gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

            o = tf.reshape(o, shape=x.shape)  # [bs, h, w, C]
            x = gamma * o + x

        return x

    def build_model(self):
        #Generator
        noise = tf.random_normal([0, 100])
        label = self.inputs[1]
        img = self.inputs[0]

        # noises
        z = tf.placeholder(tf.float32, [self.batch_size, 1, 1, self.z_dim], name='z')

        # output of D for real images
        real_logits = self.discriminator(img)

        # output of D for fake images
        fake_images = self.generator(z)
        fake_logits = self.discriminator(fake_images, reuse=True)

        # get loss for discriminator
        d_loss = discriminator_loss(real=real_logits, fake=fake_logits)

        # get loss for generator
        g_loss = generator_loss(fake=fake_logits)

        # Optimizers
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'discriminator' in var.name]
        g_vars = [var for var in t_vars if 'generator' in var.name]
        self.d_optim = tf.train.AdamOptimizer(self.d_learning_rate, beta1=self.beta1, beta2=self.beta2).minimize(
            self.d_loss, var_list=d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.g_learning_rate, beta1=self.beta1, beta2=self.beta2).minimize(
            self.g_loss, var_list=g_vars)


def init_saver(self):
    with tf.name_scope("saver"):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

