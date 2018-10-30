from base import BaseModel
import tensorflow as tf

# TODO: Write a Generator and a Discriminator Model
class ExampleModel(BaseModel):
    def __init__(self, config, is_training, inputs):
        """Create Model Class
        Args:
            config: (parameters) model hyperparameters
            is_training: (bool) whether we are training or not
            inputs: (dict) contrains the inputs of the graph (features, labels...)
                    this can be 'tf.placeholder' or outputs of 'tf.data'
        """
        super(ExampleModel, self).__init__(config)
        self.inputs = inputs
        self.build_model()
        self.init_saver()

    def build_model(self):
        # network_architecture
        d1 = tf.layers.dense(self.inputs, 512, activation=tf.nn.relu, name="densee2")
        d2 = tf.layers.dense(d1, 10)

        with tf.name_scope("loss"):
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=d2))
            self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.cross_entropy,
                                                                                         global_step=self.global_step_tensor)
            correct_prediction = tf.equal(tf.argmax(d2, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def init_saver(self):
        with tf.name_scope("saver"):
            self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
