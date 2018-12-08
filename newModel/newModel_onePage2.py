%matplotlib inline

import pickle as pkl
import numpy as np
import tensorflow as tf
import matplotlib.pyplot
from input_fn import input_fn
import config


filenames = 'training_data/training.tfrecord-?????-of-?????'
config = config.process_config('GAN.json')
features, labels = input_fn(config, filenames)

def model_inputs(real_dim, z_dim):
	inputs_real = tf.placeholder(tf.float32, (None, real_dim), name='input_real') 
	inputs_z = tf.placeholder(tf.float32, (None, z_dim), name='input_z')
	
	return inputs_real, inputs_z
	
def generator(z, out_dim, n_units=128, reuse=False, alpha=0.01):
	with tf.variable_scope('generator', reuse=reuse):
		d1 = tf.layers.dense(label, 128, activation=tf.nn.relu)		#to confirm: output size as 128*1*1?
		d2 = tf.layers.conv2d_transpose(d1, 512, 2,'same','channels_last','leaky_relu')		#filter=100, kernel_size=2*1, stride=1, padding='same'
		d3 = tf.layers.conv2d_transpose(d2, 256, 2,'same','channels_last','leaky_relu')		
		d4 = tf.layers.conv2d_transpose(d3, 128, 2,'same','channels_last','leaky_relu')		
		d5 = tf.layers.conv2d_transpose(d4, 3, 2,'same','channels_last','leaky_relu')
		d_logit = tf.layers.dense(d5, 784, activation=None) #generate a 28*28*1 fake image		
		out = tf.tanh(d_logit)
		return out
		
		'''
		# Hidden layer
		h1 = tf.layers.dense(z, n_units, activation=None)
		# Leaky ReLU
		h1 = tf.maximum(alpha * h1, h1)
		
		# Logits and tanh output
		logits = tf.layers.dense(h1, out_dim, activation=None)
		out = tf.tanh(logits)
		
		return out
		'''
		
def discriminator(x, n_units=128, reuse=False, alpha=0.01):
	with tf.variable_scope('discriminator', reuse=reuse):
		c1 = tf.nn.convolution(img, 64, 5, 2, 'same','channels_last','leaky_relu')
		c2 = tf.nn.convolution(c1, 128, 5, 2, 'same','channels_last','leaky_relu')
		c3 = tf.nn.convolution(c2, 256, 5, 2, 'same','channels_last','leaky_relu')
		c4 = tf.nn.convolution(c3, 512, 5, 2, 'same','channels_last','leaky_relu')
		c5 = tf.nn.convolution(c4, 28, 5, 2, 'same','channels_last','leaky_relu')		

		logits = tf.layers.dense(c5, 1, activation=None)
		out = tf.sigmoid(logits)
		
		return out, logits		
		
		'''
		# Hidden layer
		h1 = tf.layers.dense(x, n_units, activation=None)
		# Leaky ReLU
		h1 = tf.maximum(alpha * h1, h1)
		
		logits = tf.layers.dense(h1, 1, activation=None)
		out = tf.sigmoid(logits)
		
		return out, logits
		'''

# Size of input image to discriminator
input_size = 784
#input_size = 65536
# Size of latent vector to generator
z_size = 443
# Sizes of hidden layers in generator and discriminator
g_hidden_size = 128
d_hidden_size = 128
# Leak factor for leaky ReLU
alpha = 0.01
# Smoothing 
smooth = 0.1


tf.reset_default_graph()
# Create our input placeholders
input_real, input_z = model_inputs(input_size, z_size)

# Build the model
g_model = generator(input_z, input_size, n_units=g_hidden_size, alpha=alpha)
# g_model is the generator output

d_model_real, d_logits_real = discriminator(input_real, n_units=d_hidden_size, alpha=alpha)
d_model_fake, d_logits_fake = discriminator(g_model, reuse=True, n_units=d_hidden_size, alpha=alpha)

# Calculate losses
d_loss_real = tf.reduce_mean(
				  tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, 
														  labels=tf.ones_like(d_logits_real) * (1 - smooth)))
d_loss_fake = tf.reduce_mean(
				  tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, 
														  labels=tf.zeros_like(d_logits_real)))
d_loss = d_loss_real + d_loss_fake

g_loss = tf.reduce_mean(
			 tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
													 labels=tf.ones_like(d_logits_fake)))

# Optimizers
learning_rate = 0.002

# Get the trainable_variables, split into G and D parts
t_vars = tf.trainable_variables()
g_vars = [var for var in t_vars if var.name.startswith('generator')]
d_vars = [var for var in t_vars if var.name.startswith('discriminator')]

d_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=d_vars)
g_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=g_vars)

batch_size = 16
epochs = 100
samples = []
losses = []
# Only save generator variables
saver = tf.train.Saver(var_list=g_vars)
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for e in range(epochs):
		features_batch, labels_batch = sess.run([features, labels])
		batch_images = features_batch.reshape((batch_size, 784))
		batch_images = batch_images*2 - 1

		# Sample random noise for G
		batch_z_pre = np.random.uniform(-1, 1, size=(batch_size, z_size))
		labels_batch_pre = labels_batch.eval(sess)

		batch_z = np.empty([batch_size,z_size])
		
		for x,y in zip(batch_z_pre, labels_batch_pre):
			a1 = np.concatenate(x,y,axis=1)
			batch_z.append(a1, axis=0)

		# Run optimizers
		_ = sess.run(d_train_opt, feed_dict={input_real: batch_images, input_z: batch_z})
		_ = sess.run(g_train_opt, feed_dict={input_z: batch_z})

		# At the end of each epoch, get the losses and print them out
		train_loss_d = sess.run(d_loss, {input_z: batch_z, input_real: batch_images})
		train_loss_g = g_loss.eval({input_z: batch_z})
			
		print("Epoch {}/{}...".format(e+1, epochs),
			  "Discriminator Loss: {:.4f}...".format(train_loss_d),
			  "Generator Loss: {:.4f}".format(train_loss_g))	
		# Save losses to view after training
		losses.append((train_loss_d, train_loss_g))

		# Sample from generator as we're training for viewing afterwards
		sample_z_pre = np.random.uniform(-1, 1, size=(batch_size, z_size))

		sample_z = np.empty([batch_size,z_size])

		for x,y in zip(batch_z_pre, labels_batch_pre):
			a1 = np.concatenate(x,y,axis=1)
			sample_z.append(a1, axis=0)		

		gen_samples = sess.run(
					   generator(input_z, input_size, n_units=g_hidden_size, reuse=True, alpha=alpha),
					   feed_dict={input_z: sample_z})
		samples.append(gen_samples)
		saver.save(sess, './checkpoints/generator.ckpt')

# Save training generator samples
with open('train_samples.pkl', 'wb') as f:
	pkl.dump(samples, f)

# see the new samples generated 
saver = tf.train.Saver(var_list=g_vars)
with tf.Session() as sess:
	saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
	sample_z = np.random.uniform(-1, 1, size=(16, z_size))
	gen_samples = sess.run(
				   generator(input_z, input_size, n_units=g_hidden_size, reuse=True, alpha=alpha),
				   feed_dict={input_z: sample_z})
_ = view_samples(0, [gen_samples])

