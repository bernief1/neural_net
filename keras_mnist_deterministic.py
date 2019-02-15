import tensorflow as tf
import random as rn
import numpy as np
import time
import os

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
num_hidden = 10
activation = 'relu'
random_seed = 41
shuffle = True
use_gpu = True
use_conv = True
input_as_images = True and use_conv
force_deterministic = False
##############################

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

from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.callbacks import LambdaCallback

from keras.datasets import mnist
from keras.utils import np_utils
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
num_classes = max(np.amax(y_train), np.amax(y_test)) + 1
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
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
print("image_width = {}, image_height = {}, num_classes = {}".format(image_width, image_height, num_classes))

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
num_dense_nodes = 1024
dropout_prob = 0.5
num_outputs = num_classes

model = Sequential()
if use_conv:
    if False: # https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
        model.add(Conv2D(32, kernel_size=(3, 3), activation=activation, input_shape=input_shape))
        model.add(Conv2D(64, kernel_size=(3, 3), activation=activation))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation=activation))
        model.add(Dropout(0.5))
    else: # deep mnist - see https://gist.github.com/saitodev/c4c7a8c83f5aa4a00e93084dd3f848c5
        model.add(Conv2D(filter0_depth, kernel_size=filter0_size, activation=activation, input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filter1_depth, kernel_size=filter1_size, activation=activation))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(num_dense_nodes, activation=activation))
        if dropout_prob > 0:
            model.add(Dropout(dropout_prob))
else:
    model.add(Dense(num_hidden, input_shape=(image_size, ), activation=activation))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer=SGD(lr=learning_rate))

class CB_Master(object):
    def __init__(self, model):
        self.model = model
        self.epoch_time = time.perf_counter()
    def on_epoch_end(self, epoch, logs):
        if epoch < 10 or (epoch % 10) == 0:
            if epoch < 10:
                time1 = time.perf_counter()
                time_str = ", time = {}".format(get_time_string(time1 - self.epoch_time))
                self.epoch_time = time1
            else:
                time_str = ""
            score = self.model.evaluate(x_test, y_test, verbose=0)
            print("epoch {}: accuracy = {:.4f}%, cost = {:.6f}{}".format(epoch, 100*score[1], score[0], time_str))
cb = CB_Master(model)
callbacks = []
callbacks.append(LambdaCallback(on_epoch_end=cb.on_epoch_end))

config = tf.ConfigProto(device_count={'GPU': 1 if use_gpu else 0},
                        inter_op_parallelism_threads=parallelism_threads,
                        intra_op_parallelism_threads=parallelism_threads)
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
#config.gpu_options.allow_growth = True
K.tensorflow_backend.set_session(tf.Session(config=config))

start_time = time.perf_counter()
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          shuffle=shuffle,
          callbacks=callbacks,
          verbose=0)
duration = time.perf_counter() - start_time
score = model.evaluate(x_test, y_test, verbose=0)
print("epoch {}: accuracy = {:.4f}%, total time = {} ({}/epoch)".format(epochs, 100*score[1], get_time_string(duration), get_time_string(duration / epochs)))

print("")
for i, layer in enumerate(model.layers):
    try:
        weights, biases = layer.get_weights()
        print("layer {} weights:\n{}".format(i, weights))
        print("layer {} biases:\n{}".format(i, biases))
    except ValueError:
        print("layer {} has no weights/biases".format(i))
        pass
