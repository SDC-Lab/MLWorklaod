import time
import logging
import os
from functools import partial
from multiprocessing.pool import Pool
import csv
from funcx.sdk.client import FuncXClient
fxc = FuncXClient(asynchronous=False)
fxc.throttling_enabled = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('requests').setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)

def funcx_start_inference():
    import numpy as np
    import os
    import pandas as pd 
    import boto3
    import random
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior() 

    s3_resource = boto3.resource('s3')

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

    def standardize_test(test):
        X_test = (test - np.mean(test, axis=0)[None,:,:]) / np.std(test, axis=0)[None,:,:]

        return X_test

    def downloadS3(data_path, bucketname):
        if (os.path.exists(data_path) == False):
            os.mkdir(data_path)
        
        my_bucket = s3_resource.Bucket(bucketname)

        for s3_object in my_bucket.objects.all():
            path_, filename = os.path.split(s3_object.key)
            my_bucket.download_file(s3_object.key, data_path + '/' + filename)

    def getLabel(output_label):
        return (np.argmax(output_label) + 1)


    BASE_DIR = '/tmp/data'
    MODEL_DIR = '/tmp/har'
    # BASE_DIR = '/Users/lap089/Workspace/Serverless/HAR_serverless_20220120/data'
    # MODEL_DIR = '/Users/lap089/Workspace/Serverless/HAR_serverless_20220120/tmp'


    def harPredict():
        X_test, labels_test, list_ch_test = read_data(data_path=BASE_DIR+'/', split="test") # test
        X_test = standardize_test(X_test)
        #pick a random test sample
        x = random.randint(0, (len(X_test)-1))
        X_test_inst = X_test[[x],:]
        data_dir = MODEL_DIR
        bucket_name = 'har-cnn-model'
        
        if (os.path.exists(data_dir) == False):
            print("Download Model...")
            downloadS3(data_dir, bucket_name)

        with tf.Session() as sess:
            # Restore
            new_saver = tf.train.import_meta_graph(f'{data_dir}/har.ckpt.meta')
            new_saver.restore(sess, tf.train.latest_checkpoint(data_dir))

            graph = tf.get_default_graph()

            inputs_ = graph.get_tensor_by_name("inputs:0")
            #labels_ = graph.get_tensor_by_name("labels:0")
            keep_prob_ = graph.get_tensor_by_name("keep:0")

            logits = graph.get_tensor_by_name("logits/BiasAdd:0") 
            #accuracy = graph.get_tensor_by_name("accuracy:0")

            # feed = {inputs_: X_test_inst, labels_: y_test_inst, keep_prob_: 1}
            # output_label = sess.run(accuracy, feed_dict=feed)
            # print(output_label)

            output_label = sess.run(logits, feed_dict={inputs_: X_test_inst, keep_prob_: 1})

            return str(getLabel(output_label))
    return harPredict()




def run_test(val):
  hello_function = fxc.register_function(funcx_start_inference)
  # hello_function = "d29fde73-22cd-4728-bb41-e4ab18262cb1"
  endpoint_id = "a52bacff-fe38-436a-9f6c-61026bd47894" # tqlap089@gmail.com - k8s
  # endpoint_id = "174c2042-ffb4-4d37-ba1c-9fbf1d211b73" # tqlap@apcs.vn - node 3
  # endpoint_id = '4b116d3c-1703-4f8f-9f6f-39921e5864df' # Public tutorial endpoint
  try:
    start = time.time()
    res = fxc.run(endpoint_id=endpoint_id, function_id=hello_function)
    result = fxc.get(f"tasks/{res}")
    while result['status'] != 'success':
        time.sleep(1)
        result = fxc.get(f"tasks/{res}")
    completion_time = result['completion_t']
    exec_time = (float(completion_time) - start) * 1000
    return exec_time
  except Exception as e:
    logger.error("EXCEPTION - %s" % e)
