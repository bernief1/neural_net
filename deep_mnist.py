# https://gist.github.com/saitodev/c4c7a8c83f5aa4a00e93084dd3f848c5
import tensorflow as tf
import numpy as np
import time

from keras import backend as K

##############################
### USER ADJUSTABLE PARAMS ###
##############################
normalize_inputs = False # must be false
normalize_outputs = False # must be false
mnist_data_scale = 1 # scale the number of training and testing entries .. useful to speed things up
learning_rate = 0.02
batch_size = 200
epochs = 50
num_hidden = 10 # not used
activation = 'relu'
random_seed = 41
shuffle = True
use_gpu = True
use_conv = True # must be true
input_as_images = False and use_conv # must be false
##############################
##############################

assert(use_conv)
assert(not input_as_images)

print("tensorflow version =", tf.__version__, "\n")
tf.set_random_seed(random_seed)
np.set_printoptions(formatter={'float': '{: 2.6f}'.format})

def one_hot(arrays):
	def one_hot_(array, m):
		n = len(array)
		b = np.zeros((n, m))
		b[np.arange(n), array] = 1
		return np.array(b).astype(int)
	m = np.amax([np.amax(a) for a in arrays]) + 1
	return (one_hot_(a, m) for a in arrays)

def shuffle_unison(arrays):
	n = len(arrays[0])
	for _, a in enumerate(arrays, start=1):
		assert n == len(a)
	perm = np.random.permutation(n)
	return (a[perm] for a in arrays)

def get_time_string(seconds):
	if seconds < 0.1:
		return "{:.4f} ms".format(1000 * seconds)
	elif seconds < 100:
		return "{:.4f} seconds".format(seconds)
	else:
		return "{:.4f} minutes".format(seconds / 60)

from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
if mnist_data_scale < 1:
	x_train = x_train[:-int(x_train.shape[0]*(1 - mnist_data_scale))]
	y_train = y_train[:-int(y_train.shape[0]*(1 - mnist_data_scale))]
	x_test = x_test[:-int(x_test.shape[0]*(1 - mnist_data_scale))]
	y_test = y_test[:-int(y_test.shape[0]*(1 - mnist_data_scale))]
[train_entries, image_width, image_height] = x_train.shape
[test_entries, image_width_, image_height_] = x_test.shape
assert (image_width, image_height) == (image_width_, image_height_)
image_size = image_width * image_height
if input_as_images:
	if K.image_data_format() == 'channels_first':
		x_train = x_train.reshape(train_entries, 1, image_width, image_height)
		x_test = x_test.reshape(test_entries, 1, image_width, image_height)
		input_shape = (1, image_width, image_height)
	else:
		x_train = x_train.reshape(train_entries, image_width, image_height, 1)
		x_test = x_test.reshape(test_entries, image_width, image_height, 1)
		input_shape = (image_width, image_height, 1)
else:
	x_train = x_train.reshape(train_entries, image_size)
	x_test = x_test.reshape(test_entries, image_size)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
num_inputs = x_train.shape[1]
num_classes = max(np.amax(y_train), np.amax(y_test)) + 1

y_train, y_test = one_hot((y_train, y_test))

# using smaller filters because my laptop has only 1.66GB available GPU memory ..
import psutil
if psutil.virtual_memory()[0] > 16*1024*1024*1024: # >16GB, assume it's my desktop
	filter0_size = 5
	filter0_depth = 32
	filter1_size = 5
	filter1_depth = 64
else:
	filter0_size = 3
	filter0_depth = 16
	filter1_size = 3
	filter1_depth = 32
flattened_size = (image_width // 4) * (image_height // 4)
num_dense_nodes = 1024
num_outputs = num_classes

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding='SAME')

def activation_node(x):
	if activation == 'relu':
		return tf.nn.relu(x)
	else:
		return tf.sigmoid(x)

X = tf.placeholder(tf.float32, [None, image_size])
Y = tf.placeholder(tf.float32, [None, num_outputs])

# Convolutional layer 1
W_conv1 = tf.Variable(tf.glorot_uniform_initializer()(shape=[filter0_size, filter0_size, 1, filter0_depth]))
b_conv1 = tf.Variable(tf.zeros(shape=[filter0_depth]))
h_conv1 = activation_node(conv2d(tf.reshape(X, [-1, image_width, image_height, 1]), W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Convolutional layer 2
W_conv2 = tf.Variable(tf.glorot_uniform_initializer()(shape=[filter1_size, filter1_size, filter0_depth, filter1_depth]))
b_conv2 = tf.Variable(tf.zeros(shape=[filter1_depth]))
h_conv2 = activation_node(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Fully connected layer 1
W_fc1 = tf.Variable(tf.glorot_uniform_initializer()(shape=[flattened_size * filter1_depth, num_dense_nodes]))
b_fc1 = tf.Variable(tf.zeros(shape=[num_dense_nodes]))
h_fc1 = activation_node(tf.matmul(tf.reshape(h_pool2, [-1, flattened_size * filter1_depth]), W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Fully connected layer 2 (Output layer)
W_fc2 = tf.Variable(tf.glorot_uniform_initializer()(shape=[num_dense_nodes, num_outputs]))
b_fc2 = tf.Variable(tf.zeros(shape=[num_outputs]))
output = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=output))
cost_batch = cost * tf.cast(tf.shape(Y)[0], tf.float32)
predict = tf.argmax(output, axis=1)
correct = tf.equal(predict, tf.argmax(Y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# Training algorithm
#train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)
update = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
config = tf.ConfigProto(device_count = {'GPU': 1 if use_gpu else 0})
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
#config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
	init = tf.global_variables_initializer()
	sess.run(init, options=options)

	# train the network
	start_time = time.perf_counter()
	epoch_time = start_time
	for epoch in range(epochs):
		def get_accuracy_string():
			acc = sess.run(accuracy, feed_dict={X: x_test, Y: y_test, keep_prob: 1.0}, options=options)
			return "{:.4f}%".format(100*acc)
		if shuffle:
			x_train, y_train = shuffle_unison((x_train, y_train))
			if num_classes == 1:
				y_train = y_train.flatten()
		if 0 < batch_size < train_entries:
			cost_sum = 0
			for i in range(0, train_entries, batch_size):
				j = min(i + batch_size, train_entries)
				cost_out, _ = sess.run([cost_batch, update], feed_dict={X: x_train[i:j], Y: y_train[i:j], keep_prob: 0.5}, options=options)
				cost_sum += cost_out
			cost_out = cost_sum / train_entries
		else:
			cost_out, _ = sess.run([cost, update], feed_dict={X: x_train, Y: y_train, keep_prob: 0.5}, options=options)
		if epoch < 10 or (epoch % 10) == 0:
			if epoch < 10:
				time1 = time.perf_counter()
				time_str = ", time = {}".format(get_time_string(time1 - epoch_time))
				epoch_time = time1
			else:
				time_str = ""
			if num_classes > 1:
				print("epoch {}: accuracy = {}, cost = {:.6f}{}".format(epoch, get_accuracy_string(), cost_out, time_str))
			else:
				if normalize_outputs:
					cost_out *= (y_max - y_min) ** 2
				print("epoch {}: cost = {:.6f}{}".format(epoch, cost_out, time_str))
	duration = time.perf_counter() - start_time
	print("epoch {}: accuracy = {}, total time = {} ({}/epoch)".format(epochs, get_accuracy_string(), get_time_string(duration), get_time_string(duration / epochs)))
