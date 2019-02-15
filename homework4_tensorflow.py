import tensorflow as tf
import numpy as np
import time

from keras import backend as K

# pick 1 ...
#source = "Admissions.csv" # binary (2 classes)
#source = "Advertising.csv" # numerical
#source = "IRIS" # categorical (3 classes)
source = "MNIST" # categorical (10 classes)

##############################
### USER ADJUSTABLE PARAMS ###
##############################
normalize_inputs = True
normalize_outputs = True
mnist_data_scale = 1 # scale the number of training and testing entries .. useful to speed things up
learning_rate = 0.02
batch_size = 20
epochs = 500
num_hidden = 10
activation = 'relu'
random_seed = 41
shuffle = True
use_gpu = True
use_conv = False # not supported yet
input_as_images = False and use_conv
##############################
##############################

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

# read in data
num_classes = 0 # will be assigned later
split = True
merge = False
if source == "MNIST":
	from keras.datasets import mnist
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	if mnist_data_scale < 1:
		x_train = x_train[:-int(x_train.shape[0] * (1 - mnist_data_scale))]
		y_train = y_train[:-int(y_train.shape[0] * (1 - mnist_data_scale))]
		x_test = x_test[:-int(x_test.shape[0] * (1 - mnist_data_scale))]
		y_test = y_test[:-int(y_test.shape[0] * (1 - mnist_data_scale))]
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
	normalize_inputs = False
	split = False
	merge = True
	batch_size = 200 # MNIST has a large training set - help it to be faster
	epochs = 50
elif source == "IRIS":
	from sklearn import datasets
	iris = datasets.load_iris()
	x_data = iris['data']
	y_data = iris['target']
	num_inputs = x_data.shape[1]
else:
	import pandas as pd
	if source == "Admissions.csv":
		output_column = 'admit'
	elif source == "Advertising.csv":
		output_column = 'Sales'
		num_classes = 1 # numerical
	df = pd.read_csv(source)
	y_data = df.ix[:, output_column].values
	df = df.drop(output_column, axis=1)
	if source == "Advertising.csv":
		df = df.drop('Unnamed: 0', axis=1) # index column is useless
	for i, name in enumerate(df.columns.values):
		print("column {}: '{}'".format(i, name))
	x_data = np.array(df.values)
	num_inputs = df.shape[1]

# split data into training and testing sets (unless it was loaded in separate arrays)
if split:
	from sklearn.model_selection import train_test_split
	x_train, x_test, y_train, y_test = train_test_split(
		x_data,
		y_data,
		test_size=0.30,
		random_state=random_seed)
	train_entries = x_train.shape[0]
	test_entries = x_test.shape[0]
elif merge:
	x_data = np.concatenate((x_train, x_test), axis=0)
	y_data = np.concatenate((y_train, y_test), axis=0)
if num_classes == 0:
	num_classes = np.amax(y_data, axis=0) + 1
print("num inputs = {}, num classes = {}".format(num_inputs, num_classes))
print("training entries = {}, testing entries = {}".format(train_entries, test_entries))

# normalize inputs (features) and outputs (labels), and convert to one-hot if categorical
if normalize_inputs: # scale input values to 0..1
	x_min = x_data.min(axis=0)
	x_max = x_data.max(axis=0)
	x_scale = 1 / (x_max - x_min)
	x_train = (x_train - x_min) * x_scale
	x_test = (x_test - x_min) * x_scale
if num_classes > 1:
	y_train, y_test = one_hot((y_train, y_test))
	normalize_outputs = False
elif normalize_outputs: # scale output values to 0..1
	y_min = np.amin(y_data)
	y_max = np.amax(y_data)
	y_scale = 1 / (y_max - y_min)
	y_train = (y_train - y_min) * y_scale
	y_test = (y_test - y_min) * y_scale

# set up tensorflow graph
num_outputs = num_classes
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
W1 = tf.Variable(tf.glorot_uniform_initializer()((num_inputs, num_hidden)))
b1 = tf.Variable(tf.zeros([num_hidden]))
H1 = tf.matmul(X, W1) + b1
if activation == 'relu':
	H1 = tf.nn.relu(H1)
else:
	H1 = tf.sigmoid(H1)
W2 = tf.Variable(tf.glorot_uniform_initializer()((num_hidden, num_outputs)))
b2 = tf.Variable(tf.zeros([num_outputs]))
H2 = tf.matmul(H1, W2) + b2
output = H2
if num_classes > 1:
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=output))
	predict = tf.argmax(output, axis=1)
	correct = tf.equal(predict, tf.argmax(Y, axis=1))
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
else:
	cost = tf.reduce_mean((output - Y) ** 2) # MSE
cost_batch = cost * tf.cast(tf.shape(Y)[0], tf.float32)
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
			acc = sess.run(accuracy, feed_dict={X: x_test, Y: y_test}, options=options)
			return "{:.4f}%".format(100*acc)
		if shuffle:
			x_train, y_train = shuffle_unison((x_train, y_train))
			if num_classes == 1:
				y_train = y_train.flatten()
		if 0 < batch_size < train_entries:
			cost_sum = 0
			for i in range(0, train_entries, batch_size):
				j = min(i + batch_size, train_entries)
				cost_out, _ = sess.run([cost_batch, update], feed_dict={X: x_train[i:j], Y: y_train[i:j]}, options=options)
				cost_sum += cost_out
			cost_out = cost_sum / train_entries
		else:
			cost_out, _ = sess.run([cost, update], feed_dict={X: x_train, Y: y_train}, options=options)
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

	# print out weights and biases
	W1_out, b1_out, W2_out, b2_out = sess.run([W1, b1, W2, b2], options=options)
	print("\nneural network state:")
	print("W1 = {}".format(W1_out))
	print("b1 = {}".format(b1_out))
	print("W2 = {}".format(W2_out))
	print("b2 = {}".format(b2_out))

	# print out predictions
	show_inputs = num_inputs <= 5
	if num_classes == 1:
		def show_predictions(x, y, s):
			p, cost_out = sess.run([output, cost], feed_dict={X: x, Y: y}, options=options)
			p = p.flatten()
			n, m = np.shape(x)
			print("\n{} data:".format(s))
			for i in range(n):
				if i >= 10:
					print("...")
					break
				x_ = x[i]
				y_ = y[i]
				p_ = p[i]
				if normalize_inputs:
					x_ = x_min + x_ * (x_max - x_min)
				if normalize_outputs:
					y_ = y_min + y_ * (y_max - y_min)
					p_ = y_min + p_ * (y_max - y_min)
				label = y_
				prediction = p_
				if show_inputs:
					input_str = "inputs={} ".format(x[i])
				print("{}label = {:.4f}, prediction = {:.4f}, difference = {:.4f}".format(input_str, label, prediction, prediction - label))
			y_range = [np.amin(y), np.mean(y), np.amax(y)]
			p_range = [np.amin(p), np.mean(p), np.amax(p)]
			if normalize_outputs:
				for i in range(3):
					y_range[i] = y_min + y_range[i] * (y_max - y_min)
					p_range[i] = y_min + p_range[i] * (y_max - y_min)
				cost_out *= (y_max - y_min) ** 2
			return (s, cost_out, y_range, p_range)
		def report_accuracy(results):
			labels_info = "labels min/mean/max = {:.4f}/{:.4f}/{:.4f}".format(results[2][0], results[2][1], results[2][2])
			pred_info = "predictions = {:.4f}/{:.4f}/{:.4f}".format(results[3][0], results[3][1], results[3][2])
			print("{} data cost = {:.8f}, {}, {}".format(results[0], results[1], labels_info, pred_info))
	else:
		def show_predictions(x, y, s):
			p, cost_out = sess.run([predict, cost], feed_dict={X: x, Y: y}, options=options)
			n, m = np.shape(x)
			correct = n
			print("\n{} data:".format(s))
			input_str = ""
			suppress = False
			for i in range(n):
				if i >= 10 and not suppress:
					print("...")
					suppress = True
				x_ = x[i]
				if normalize_inputs:
					x_ = x_min + x_ * (x_max - x_min)
				label = np.argmax(y[i])
				prediction = p[i]
				if label != prediction:
					scold = " INCORRECT"
					correct -= 1
				else:
					scold = ""
				if not suppress:
					if show_inputs:
						input_str = "inputs={} ".format(x_)
					print("{}label = {}, prediction = {}{}".format(input_str, label, prediction, scold))
			return (s, cost_out, correct, n)
		def report_accuracy(results):
			correct = results[2]
			total = results[3]
			print("{} data cost = {:.8f}, {} out of {} correct ({:.4f}%)".format(results[0], results[1], correct, total, 100 * correct / total))
	results1 = show_predictions(x_train, y_train, "training")
	results2 = show_predictions(x_test, y_test, "testing")
	print("")
	report_accuracy(results1)
	report_accuracy(results2)
