from base import BaseModel
import tensorflow as tf

# TODO: Write a Generator and a Discriminator Model
class ExampleModel(BaseModel):
    def __init__(self, config, is_training, inputs):
        """Create Model Class
        Args:
            config: (parameters) model hyperparameters
            is_training: (bool) whether we are training or not
            inputs: (dict) contains the inputs of the graph (features, labels...)
                    this can be 'tf.placeholder' or outputs of 'tf.data'
        """
        super(ExampleModel, self).__init__(config)		#configs/example.json
        self.inputs = inputs
        self.build_model()
        self.init_saver()

    def build_model(self):
        # network_architecture
        #d1 = tf.layers.dense(self.inputs, 512, activation=tf.nn.relu, name="densee2")
        #d2 = tf.layers.dense(d1, 10)
		#d_final = tf.layers.dense(d1, 345)	#there are 345 categories of Doodles 
        
		
		###############
		# Generator   #
		###############
		
		noise = tf.random_normal([,100])		
		label = self.inputs[1]		#to confirm: get label from input_fn.py; assumed one-hotted with; also assumed 345 downcasted into 128-d; image returned from input not used 
		img = self.inputs[0]	#assumed 28*28 flattened?
		concated = tf.concat([label, noise],0)		#to confirm: only concat one-hot and noise? concat by rows; dimension after concatenation = 100+128=228
		
		d1 = tf.layers.dense(label, 128, activation=tf.nn.relu)		#to confirm: output size as 128*1*1?
		d2 = tf.layers.conv2d_transpose(d1, 512, 2,'same','channels_last','leaky_relu')		#filter=100, kernel_size=2*1, stride=1, padding='same'
		d3 = tf.layers.conv2d_transpose(d2, 256, 2,'same','channels_last','leaky_relu')		
		d4 = tf.layers.conv2d_transpose(d3, 128, 2,'same','channels_last','leaky_relu')		
		d5 = tf.layers.conv2d_transpose(d4, 3, 2,'same','channels_last','leaky_relu')
		d_final = tf.layers.dense(d5, 784)	#generate a 28*28*1 fake image


		#################
		# Descriminator #
		#################
		
		c1 = tf.nn.convolution(img, 64, 5, 2, 'same','channels_last','leaky_relu')
		c2 = tf.nn.convolution(c1, 128, 5, 2, 'same','channels_last','leaky_relu')
		c3 = tf.nn.convolution(c2, 256, 5, 2, 'same','channels_last','leaky_relu')
		c4 = tf.nn.convolution(c3, 512, 5, 2, 'same','channels_last','leaky_relu')
		c5 = tf.nn.convolution(c4, 28, 5, 2, 'same','channels_last','leaky_relu')
		#so, no pooling? 
		
		concated2 = tf.concat([c5, d1],0)
		#c_final = tf.nn.convolution(c4, 28, 5, 2, 'same','channels_last','leaky_relu')

		
        with tf.name_scope("loss"):
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=d2))
            self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.cross_entropy,
                                                                                         global_step=self.global_step_tensor)
            correct_prediction = tf.equal(tf.argmax(d2, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		
		
		
			
			
			
    def init_saver(self):
        with tf.name_scope("saver"):
            self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
