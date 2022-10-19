#!/usr/bin/env python3

import numpy as np
import os
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
import pandas as pd 
import boto3
import random
from datetime import datetime
import time
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

s3_resource = boto3.resource('s3')


def read_data(data_path, split = "train"):
	""" Read data """
	# Fixed params
	n_class = 6
	n_steps = 128
    #
	# Paths
	path_ = os.path.join(data_path, split)
	path_signals = os.path.join(path_, "Inertial_Signals")
    # 
	# Read labels and one-hot encode
	label_path = os.path.join(path_, "y_" + split + ".txt")
	labels = pd.read_csv(label_path, header = None)
    #
	# Read time-series data
	channel_files = os.listdir(path_signals)
	channel_files.sort()
	n_channels = len(channel_files)
	posix = len(split) + 5
    #
	# Initiate array
	list_of_channels = []
	X = np.zeros((len(labels), n_steps, n_channels))
    #
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
    harTrain()
    return req




# ---------  fission
def main():
    harTrain()
    return "Training Finished"





def harTrain():
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
    batch_size = 600       # Batch size
    seq_len = 128          # Number of steps
    learning_rate = 0.0001
    epochs = 1000
    #
    n_classes = 6
    n_channels = 9
    #
    graph = tf.Graph()
    # Construct placeholders
    with graph.as_default():
        inputs_ = tf.placeholder(tf.float32, [None, seq_len, n_channels], name = 'inputs')
        labels_ = tf.placeholder(tf.float32, [None, n_classes], name = 'labels')
        keep_prob_ = tf.placeholder(tf.float32, name = 'keep')
        learning_rate_ = tf.placeholder(tf.float32, name = 'learning_rate')
    #
    with graph.as_default():
        # (batch, 128, 9) --> (batch, 64, 18)
        conv1 = tf.layers.conv1d(inputs=inputs_, filters=18, kernel_size=2, strides=1, 
                                padding='same', activation = tf.nn.relu)
        max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2, padding='same')
        # (batch, 64, 18) --> (batch, 32, 36)
        conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=36, kernel_size=2, strides=1, 
                                padding='same', activation = tf.nn.relu)
        max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2, padding='same')
        # (batch, 32, 36) --> (batch, 16, 72)
        conv3 = tf.layers.conv1d(inputs=max_pool_2, filters=72, kernel_size=2, strides=1, 
                                padding='same', activation = tf.nn.relu)
        max_pool_3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2, padding='same')
        # (batch, 16, 72) --> (batch, 8, 144)
        conv4 = tf.layers.conv1d(inputs=max_pool_3, filters=144, kernel_size=2, strides=1, 
                                padding='same', activation = tf.nn.relu)
        max_pool_4 = tf.layers.max_pooling1d(inputs=conv4, pool_size=2, strides=2, padding='same')
    #
    #
    with graph.as_default():
        # Flatten and add dropout
        flat = tf.reshape(max_pool_4, (-1, 8*144))
        flat = tf.nn.dropout(flat, keep_prob=keep_prob_)
        # Predictions
        logits = tf.layers.dense(flat, n_classes, name='logits')
        # Cost function and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_))
        optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost)
        # Accuracy
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
    if (os.path.exists(data_dir) == False):
        os.makedirs(f'{data_dir}')
    #
    validation_acc = []
    validation_loss = []
    #
    train_acc = []
    train_loss = []
    #
    with graph.as_default():
        saver = tf.train.Saver()
    #
    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        iteration = 1 
        # Loop over epochs
        for e in range(epochs):
            #
            # Loop over batches
            for x,y in get_batches(X_tr, y_tr, batch_size):
                #
                # Feed dictionary
                feed = {inputs_ : x, labels_ : y, keep_prob_ : 0.5, learning_rate_ : learning_rate}
                #
                # Loss
                loss, _ , acc = sess.run([cost, optimizer, accuracy], feed_dict = feed)
                train_acc.append(acc)
                train_loss.append(loss)
                #
                # Print at each 5 iters
                if (iteration % 5 == 0):
                    now = datetime.now().time() # time object
                    print(now, "Epoch: {}/{}".format(e, epochs),
                        "Iteration: {:d}".format(iteration),
                        "Train loss: {:6f}".format(loss),
                        "Train acc: {:.6f}".format(acc))
                #
                # Compute validation loss at every 10 iterations
                if (iteration%10 == 0):                
                    val_acc_ = []
                    val_loss_ = []
                    #
                    for x_v, y_v in get_batches(X_vld, y_vld, batch_size):
                        # Feed
                        feed = {inputs_ : x_v, labels_ : y_v, keep_prob_ : 1.0}  
                        #
                        # Loss
                        loss_v, acc_v = sess.run([cost, accuracy], feed_dict = feed)                    
                        val_acc_.append(acc_v)
                        val_loss_.append(loss_v)
                    #
                    # Print info
                    print("Epoch: {}/{}".format(e, epochs),
                        "Iteration: {:d}".format(iteration),
                        "Validation loss: {:6f}".format(np.mean(val_loss_)),
                        "Validation acc: {:.6f}".format(np.mean(val_acc_)))
                    #
                    # Store
                    validation_acc.append(np.mean(val_acc_))
                    validation_loss.append(np.mean(val_loss_))
                #
                # Iterate 
                iteration += 1
        #
        saver.save(sess, f'{data_dir}/har.ckpt')
    print("Total Time: %s" % (time.time() - start_time))
    return "training complete."
    #uploadS3(data_dir, bucket_name)

    # # Plot training and test loss
    # t = np.arange(iteration-1)

    # plt.figure(figsize = (6,6))
    # plt.plot(t, np.array(train_loss), 'r-', t[t % 10 == 0], np.array(validation_loss), 'b*')
    # plt.xlabel("iteration")
    # plt.ylabel("Loss")
    # plt.legend(['train', 'validation'], loc='upper right')
    # plt.show()

    # # Plot Accuracies
    # plt.figure(figsize = (6,6))

    # plt.plot(t, np.array(train_acc), 'r-', t[t % 10 == 0], validation_acc, 'b*')
    # plt.xlabel("iteration")
    # plt.ylabel("Accuray")
    # plt.legend(['train', 'validation'], loc='upper right')
    # plt.show()
    


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
    with tf.Session() as sess:
        # Restore
        new_saver = tf.train.import_meta_graph(f'{data_dir}/har.ckpt.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint(data_dir))
        #
        graph = tf.get_default_graph()
        #
        inputs_ = graph.get_tensor_by_name("inputs:0")
        #labels_ = graph.get_tensor_by_name("labels:0")
        keep_prob_ = graph.get_tensor_by_name("keep:0")
        #
        logits = graph.get_tensor_by_name("logits/BiasAdd:0") 
        #accuracy = graph.get_tensor_by_name("accuracy:0")
        # feed = {inputs_: X_test_inst, labels_: y_test_inst, keep_prob_: 1}
        # output_label = sess.run(accuracy, feed_dict=feed)
        # print(output_label)
        output_label = sess.run(logits, feed_dict={inputs_: X_test_inst, keep_prob_: 1})
        print("Total Time: %s" % (time.time() - start_time))
        return str(getLabel(output_label))


print(harTrain())
# 429.1725 444.1327