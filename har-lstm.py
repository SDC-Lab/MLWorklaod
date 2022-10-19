#!/usr/bin/env python3

import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd 
import boto3
import random
from datetime import datetime
import time
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, LSTM
# from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow import keras

import socket

# s3_resource = boto3.resource('s3')


def read_data(data_path, split = "train"):
	""" Read data """
	# Fixed params
	n_class = 6
	n_steps = 128

	# Paths
	path_ = os.path.join(data_path, split)
	path_signals = os.path.join(path_, "Inertial_Signals")

	# Read labels and one-hot encode
	label_path = os.path.join(path_, "y_" + split + ".txt")
	labels = pd.read_csv(label_path, header = None)

	# Read time-series data
	channel_files = os.listdir(path_signals)
	channel_files.sort()
	n_channels = len(channel_files)
	posix = len(split) + 5

	# Initiate array
	list_of_channels = []
	X = np.zeros((len(labels), n_steps, n_channels))

	i_ch = 0
	for fil_ch in channel_files:
		channel_name = fil_ch[:-posix]
		dat_ = pd.read_csv(os.path.join(path_signals,fil_ch), delim_whitespace = True, header = None)
		X[:,:,i_ch] = dat_.to_numpy()
        
		# Record names
		list_of_channels.append(channel_name)

		# iterate
		i_ch += 1

	# Return 
	return X, labels[0].values, list_of_channels

def standardize(train, test):
	""" Standardize data """

	# Standardize train and test
	X_train = (train - np.mean(train, axis=0)[None,:,:]) / np.std(train, axis=0)[None,:,:]
	X_test = (test - np.mean(test, axis=0)[None,:,:]) / np.std(test, axis=0)[None,:,:]

	return X_train, X_test


def standardize_test(test):
	X_test = (test - np.mean(test, axis=0)[None,:,:]) / np.std(test, axis=0)[None,:,:]
	return X_test


def one_hot(labels, n_class = 6):
	""" One-hot encoding """
	expansion = np.eye(n_class)
	y = expansion[:, labels-1].T
	assert y.shape[1] == n_class, "Wrong number of labels!"

	return y

def get_batches(X, y, batch_size = 100):
	""" Return a generator for batches """
	n_batches = len(X) // batch_size
	X, y = X[:n_batches*batch_size], y[:n_batches*batch_size]

	# Loop over batches and yield
	for b in range(0, len(X), batch_size):
		yield X[b:b+batch_size], y[b:b+batch_size]


def uploadS3(data_path, bucketname):
    path_ = os.listdir(data_path + '/')

    for fn in path_:
        s3_resource.Object(bucketname, fn).upload_file(data_path + '/' + fn)


def downloadS3(data_path, bucketname):
	if (os.path.exists(data_path) == False):
		os.mkdir(data_path)
    
	my_bucket = s3_resource.Bucket(bucketname)

	for s3_object in my_bucket.objects.all():
		path_, filename = os.path.split(s3_object.key)
		my_bucket.download_file(s3_object.key, data_path + '/' + filename)

def getLabel(output_label):
	return (np.argmax(output_label) + 1)


def numpy_to_bytes(arr: np.array) -> str:
    arr_dtype = bytearray(str(arr.dtype), 'utf-8')
    arr_shape = bytearray(','.join([str(a) for a in arr.shape]), 'utf-8')
    sep = bytearray('|', 'utf-8')
    arr_bytes = arr.ravel().tobytes()
    to_return = arr_dtype + sep + arr_shape + sep + arr_bytes
    return to_return


def bytes_to_numpy(serialized_arr: str) -> np.array:
    sep = '|'.encode('utf-8')
    i_0 = serialized_arr.find(sep)
    i_1 = serialized_arr.find(sep, i_0 + 1)
    arr_dtype = serialized_arr[:i_0].decode('utf-8')
    arr_shape = tuple([int(a) for a in serialized_arr[i_0 + 1:i_1].decode('utf-8').split(',')])
    arr_str = serialized_arr[i_1 + 1:]
    arr = np.frombuffer(arr_str, dtype = arr_dtype).reshape(arr_shape)
    return arr


# BASE_DIR = '/tmp/data'
# MODEL_DIR = '/tmp/har'
BASE_DIR = '/Users/lap089/Workspace/Serverless/HAR_serverless_20220120/data'
MODEL_DIR = '/Users/lap089/Workspace/Serverless/HAR_serverless_20220120/tmp'




# ---------  kubeless
def start_training_process(event, context):
    return harTrain()

def start_inference_process(event, context):
    return harPredict()



# ---------  openfaas
def handle(req):
    return str(socket.gethostname()) + ' - ' +  str(harPredict())




# ---------  fission
def main():
    return str(socket.gethostname()) + ' - ' +  str(harPredict())


def harTrain():
    print('Start Training: %s' % datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
    start_time = time.time()
    X_train, labels_train, list_ch_train = read_data(data_path=BASE_DIR+'/', split="train") # train
    X_test, labels_test, list_ch_test = read_data(data_path=BASE_DIR+'/', split="test") # test
    assert list_ch_train == list_ch_test, "Mistmatch in channels!"
    X_train, X_test = standardize(X_train, X_test)
    data_dir = MODEL_DIR
    bucket_name = 'har-cnn-model'
    #
    X_tr, X_vld, lab_tr, lab_vld = train_test_split(X_train, labels_train, stratify = labels_train, random_state = 123)
    #
    y_tr = one_hot(lab_tr)
    y_vld = one_hot(lab_vld)
    y_test = one_hot(labels_test)
    #
    lstm_size = 27         # 3 times the amount of channels
    #lstm_layers = 2        # Number of layers
    batch_size = 600       # Batch size
    seq_len = 128          # Number of steps
    learning_rate = 0.0001  # Learning rate (default is 0.001)
    epochs = 15
    #
    n_classes = 6
    n_channels = 9
    #
    classifier = Sequential()
    classifier.add(LSTM(lstm_size, input_shape=(seq_len, n_channels), return_sequences=True))
    classifier.add(Dropout(0.2))
    classifier.add(LSTM(lstm_size))
    classifier.add(Dense(n_classes, activation='relu'))
    classifier.add(Dropout(0.2))
    #classifier.summary()
    #
    classifier.compile(loss='mean_squared_error',
              optimizer=Adam(learning_rate=learning_rate, decay=1e-6),
              metrics=['accuracy'] )
    #classifier.fit(X_tr, y_tr, epochs=epochs, validation_data=(X_vld, y_vld), batch_size=batch_size)
    classifier.fit(X_tr, y_tr, epochs=epochs, validation_data=(X_vld, y_vld)) #for faster training 
    #
    if (os.path.exists(data_dir) == False):
        os.mkdir(f'./{data_dir}')
    #
    test_loss, test_acc = classifier.evaluate(X_test, y_test)
    print('Test Loss: {}'.format(test_loss))
    print('Test Accuracy: {}'.format(test_acc))
    #
    classifier.save(f'{data_dir}/lstm_model.h5')
    #
    print('End Training: %s' % datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
    print("Total Time: %s" % (time.time() - start_time))
    # uploadS3(data_dir, bucket_name) 
    return "training complete."


def harPredict():
    start_time = time.time()
    X_test, labels_test, list_ch_test = read_data(data_path=BASE_DIR+'/', split="test") # test
    X_test = standardize_test(X_test)
    #pick a random test sample
    x = random.randint(0, (len(X_test)-1))
    X_test_inst = X_test[[x],:]
    data_dir = MODEL_DIR
    bucket_name = 'har-cnn-model'
    #
    if (os.path.exists(data_dir) == False):
        print("Download Model...")
        downloadS3(data_dir, bucket_name)
    #
    classifier = keras.models.load_model(f'{data_dir}/lstm_model.h5')
    yhat = classifier.predict(np.array( X_test_inst))
    print("Total Time: %s" % (time.time() - start_time))
    #
    return str(getLabel(yhat))


print(harTrain())
# print(harPredict())

# 166.649296  162.3688 160.6639