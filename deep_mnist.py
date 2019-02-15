import tensorflow as tf
import random as rn
import numpy as np
import time
import os

##############################
### USER ADJUSTABLE PARAMS ###
##############################
epochs = 50
batch_size = 200
learning_rate = 0.02
shuffle = True
use_gpu = True # else use CPU
use_keras = False # else use raw tensorflow codepath
use_conv2d = True # else use simple 3-layer dense network
num_hidden = 10 if not use_conv2d else 0
mnist_data_scale = 1 if use_conv2d else 1 # scale the number of training and testing entries .. useful to speed things up
force_deterministic = False
random_seed = 41
##############################

normalize_inputs = False # must be false
normalize_outputs = False # must be false

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

# [cost, accuracy] for first 3 epochs over 5 runs ..
# note that deterministic runs are consistent in score for the first 6 or 7 decimal places
# non-deterministic runs vary widely
#
# deterministic:
# run 1: [2.1783223117828370, 0.2306], [2.0429167129516603, 0.3325], [1.9148488014221192, 0.4039]
# run 2: [2.1783223091125490, 0.2306], [2.0429167083740234, 0.3325], [1.9148488075256347, 0.4039]
# run 3: [2.1783223079681395, 0.2306], [2.0429167053222654, 0.3325], [1.9148488021850585, 0.4039]
# run 4: [2.1783223079681395, 0.2306], [2.0429167022705080, 0.3325], [1.9148488048553467, 0.4039]
# run 5: [2.1783223094940185, 0.2306], [2.0429167053222654, 0.3325], [1.9148488014221192, 0.4039]
#
# non-deterministic
# run 1: [2.2574144844055177, 0.2013], [2.1387517375946046, 0.3052], [2.0254593414306640, 0.3522]
# run 2: [2.1411928066253663, 0.2347], [1.9809556648254394, 0.3550], [1.8324221063613892, 0.4329]
# run 3: [2.1372428092956540, 0.2429], [2.0103229276657104, 0.3567], [1.8791631507873534, 0.4492]
# run 4: [2.2036154617309570, 0.2268], [2.1070711654663086, 0.2813], [2.0114383541107177, 0.3299]
# run 5: [2.1748092433929442, 0.1945], [2.0556745847702027, 0.3033], [1.9350524660110473, 0.4072]

parallelism_threads = 0 # default
if force_deterministic:
    # The below is necessary in Python 3.2.3 onwards to
    # have reproducible behavior for certain hash-based operations.
    # See these references for further details:
    # https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
    # https://github.com/fchollet/keras/issues/2280#issuecomment-306959926
    os.environ['PYTHONHASHSEED'] = '0'

    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    rn.seed(random_seed)

    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.
    np.random.seed(random_seed)

    # The below tf.set_random_seed() will make random number generation
    # in the TensorFlow backend have a well-defined initial state.
    # For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed
    tf.set_random_seed(random_seed)

    # Force TensorFlow to use single thread.
    # Multiple threads are a potential source of non-reproducible results.
    # For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res
    parallelism_threads = 1

options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
config = tf.ConfigProto(device_count={'GPU': 1 if use_gpu else 0},
                        inter_op_parallelism_threads=parallelism_threads,
                        intra_op_parallelism_threads=parallelism_threads)
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
#config.gpu_options.allow_growth = True

from keras import backend as K

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
if use_keras and use_conv2d:
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
num_inputs = image_size
num_classes = max(np.amax(y_train), np.amax(y_test)) + 1
num_outputs = num_classes
y_train, y_test = one_hot((y_train, y_test))
print("num inputs = {}, num classes = {}".format(num_inputs, num_classes))
print("training entries = {}, testing entries = {}".format(train_entries, test_entries))

if use_conv2d:
    print("image_width = {}, image_height = {}".format(image_width, image_height))
    import psutil
    if psutil.virtual_memory()[0] > 16*1024*1024*1024: # >16GB, assume it's my desktop PC ..
        conv1_size = 5
        conv1_depth = 32
        conv2_size = 5
        conv2_depth = 64
    else: # .. assume it's my laptop, with a fraction of the GPU memory - make smaller filters
        conv1_size = 3
        conv1_depth = 16
        conv2_size = 3
        conv2_depth = 32
    num_dense_nodes = 1024
    dropout_prob = 0.5

if use_keras:
    from keras.layers.core import Dense, Activation, Dropout
    from keras.layers import Conv2D, MaxPooling2D, Flatten
    from keras.models import Sequential
    from keras.optimizers import SGD, Adam
    from keras.callbacks import LambdaCallback
    model = Sequential()
    if use_conv2d:
        if False: # https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
            model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
            model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.5))
        else: # deep mnist - see https://gist.github.com/saitodev/c4c7a8c83f5aa4a00e93084dd3f848c5
            model.add(Conv2D(conv1_depth, kernel_size=conv1_size, activation='relu', input_shape=input_shape))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(conv2_depth, kernel_size=conv2_size, activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Flatten())
            model.add(Dense(num_dense_nodes, activation='relu'))
            if dropout_prob > 0:
                model.add(Dropout(dropout_prob))
    else:
        model.add(Dense(num_hidden, input_shape=(image_size, ), activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer=SGD(lr=learning_rate))

    def get_accuracy_string():
        score = model.evaluate(x_test, y_test, verbose=0)
        return "{:.4f}%".format(100 * score[1])
    def get_accuracy_cost_strings():
        score = model.evaluate(x_test, y_test, verbose=0)
        cost_out = score[0]
        if normalize_outputs:
            cost_out *= (y_max - y_min) ** 2
        return ("{:.4f}%".format(100 * score[1]), "{:.9f}".format(cost_out))
    class CB_Master(object):
        def __init__(self, model):
            self.model = model
            self.epoch_time = time.perf_counter()
        def on_epoch_end(self, epoch, logs):
            if epoch <= 10 or (epoch % 10) == 0:
                if epoch <= 10:
                    time1 = time.perf_counter()
                    time_str = ", time = {}".format(get_time_string(time1 - self.epoch_time))
                    self.epoch_time = time1
                else:
                    time_str = ""
                acc_str, cst_str = get_accuracy_cost_strings()
                print("epoch {}: accuracy = {}, cost = {}{}".format(epoch, acc_str, cst_str, time_str))
    cb = CB_Master(model)
    callbacks = []
    callbacks.append(LambdaCallback(on_epoch_end=cb.on_epoch_end))
    K.tensorflow_backend.set_session(tf.Session(config=config))
    start_time = time.perf_counter()
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=shuffle, callbacks=callbacks, verbose=0)
    duration = time.perf_counter() - start_time
    accuracy_str = " accuracy = {},".format(get_accuracy_string())
    time_str = " total time = {} ({}/epoch)".format(get_time_string(duration),
                                                    get_time_string(duration / epochs))
    print("epoch {}:{}{}".format(epochs, accuracy_str, time_str))
    print("")
    for i, layer in enumerate(model.layers): # dump weights and biases for all layers
        try:
            weights, biases = layer.get_weights()
            print("layer {} weights:\n{}".format(i, weights))
            print("layer {} biases:\n{}".format(i, biases))
        except ValueError:
            print("layer {} has no weights/biases\n".format(i))
            pass
else:
    X = tf.placeholder(tf.float32, [None, num_inputs])
    Y = tf.placeholder(tf.float32, [None, num_outputs])

    if use_conv2d:
        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        def max_pool_2x2(x):
            return tf.nn.max_pool(x, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding='SAME')

        # Convolutional layer 1
        W_conv1 = tf.Variable(tf.glorot_uniform_initializer()(shape=[conv1_size, conv1_size, 1, conv1_depth]))
        b_conv1 = tf.Variable(tf.zeros(shape=[conv1_depth]))
        h_conv1 = tf.nn.relu(conv2d(tf.reshape(X, [-1, image_width, image_height, 1]), W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        # Convolutional layer 2
        W_conv2 = tf.Variable(tf.glorot_uniform_initializer()(shape=[conv2_size, conv2_size, conv1_depth, conv2_depth]))
        b_conv2 = tf.Variable(tf.zeros(shape=[conv2_depth]))
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        # Fully connected layer 1
        flattened_size = (image_width // 4) * (image_height // 4)
        W_fc1 = tf.Variable(tf.glorot_uniform_initializer()(shape=[flattened_size * conv2_depth, num_dense_nodes]))
        b_fc1 = tf.Variable(tf.zeros(shape=[num_dense_nodes]))
        h_fc1 = tf.nn.relu(tf.matmul(tf.reshape(h_pool2, [-1, flattened_size * conv2_depth]), W_fc1) + b_fc1)

        # Dropout
        keep_prob = tf.placeholder(tf.float32)
        if dropout_prob > 0:
            h_fc1 = tf.nn.dropout(h_fc1, keep_prob)

        # Fully connected layer 2 (Output layer)
        W_fc2 = tf.Variable(tf.glorot_uniform_initializer()(shape=[num_dense_nodes, num_outputs]))
        b_fc2 = tf.Variable(tf.zeros(shape=[num_outputs]))
        h_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2
        output = h_fc2
    else:
        W_layer1 = tf.Variable(tf.glorot_uniform_initializer()(shape=[num_inputs, num_hidden]))
        b_layer1 = tf.Variable(tf.zeros(shape=[num_hidden]))
        h_layer1 = tf.nn.relu(tf.matmul(X, W_layer1) + b_layer1)
        W_layer2 = tf.Variable(tf.glorot_uniform_initializer()(shape=[num_hidden, num_outputs]))
        b_layer2 = tf.Variable(tf.zeros(shape=[num_outputs]))
        h_layer2 = tf.matmul(h_layer1, W_layer2) + b_layer2
        output = h_layer2

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=output))
    cost_batch = cost * tf.cast(tf.shape(Y)[0], tf.float32)
    predict = tf.argmax(output, axis=1)
    correct = tf.equal(predict, tf.argmax(Y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    update = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer(), options=options)
        start_time = time.perf_counter()
        epoch_time = start_time
        for epoch in range(epochs):
            def get_accuracy_string():
                acc = sess.run(accuracy, feed_dict={X: x_test, Y: y_test, keep_prob: 1}, options=options)
                return "{:.4f}%".format(100 * acc)
            if shuffle:
                x_train, y_train = shuffle_unison((x_train, y_train))
                if num_classes == 1:
                    y_train = y_train.flatten()
            if 0 < batch_size < train_entries:
                cost_sum = 0
                for i in range(0, train_entries, batch_size):
                    j = min(i + batch_size, train_entries)
                    cost_out, _ = sess.run([cost_batch, update],
                                           feed_dict={X: x_train[i:j],
                                                      Y: y_train[i:j],
                                                      keep_prob: 1 - dropout_prob},
                                           options=options)
                    cost_sum += cost_out
                cost_out = cost_sum / train_entries
            else:
                cost_out, _ = sess.run([cost, update],
                                       feed_dict={X: x_train,
                                                  Y: y_train,
                                                  keep_prob: 1 - dropout_prob},
                                       options=options)
            if epoch <= 10 or (epoch % 10) == 0:
                if epoch <= 10:
                    time1 = time.perf_counter()
                    duration = time1 - epoch_time
                    time_str = ", time = {}".format(get_time_string(time1 - epoch_time))
                    epoch_time = time1
                else:
                    time_str = ""
                if num_classes > 1:
                    accuracy_str = " accuracy = {},".format(get_accuracy_string())
                else:
                    accuracy_str = ""
                if normalize_outputs:
                    cost_out *= (y_max - y_min) ** 2
                print("epoch {}:{} cost = {:.9f}{}".format(epoch, accuracy_str, cost_out, time_str))
        duration = time.perf_counter() - start_time
        accuracy_str = " accuracy = {},".format(get_accuracy_string())
        time_str = " total time = {} ({}/epoch)".format(get_time_string(duration),
                                                        get_time_string(duration / epochs))
        print("epoch {}:{}{}".format(epochs, accuracy_str, time_str))
