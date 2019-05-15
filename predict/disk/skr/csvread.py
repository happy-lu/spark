import numpy as np
import tensorflow as tf
import os, sys, time
import math,copy
from  multiprocessing import Process

class disktrain(object):
    def __init__(self,diskdir):
        self.diskdir = diskdir
        self.diskfile_list = self.os_csv_path(self.diskdir)
        
    def os_csv_path(self, rootdir):#tell a dir path ,get all the files' name and put in a list
        file_list = []
        list_dir = os.listdir(rootdir)
        for i in range(0,len(list_dir)):
            path = os.path.join(rootdir,list_dir[i])
            if os.path.isfile(path) and (".csv" in path):
                file_list.append(path)
            if os.path.isdir(path):
                ret_list = self.os_csv_path(path)
                file_list = file_list + ret_list
        return file_list


    def csv_2_tensor(self):
        filename_queue = tf.train.string_input_producer(self.diskfile_list)
        reader = tf.TextLineReader()
        key, value = reader.read(filename_queue)
        print(key, value)
        record_defaults = [[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]]
        col1, col2, col3, col4, col5, col6 = tf.decode_csv(value, record_defaults=record_defaults)
        features = tf.reshape(tf.transpose(tf.stack([col1, col2, col3, col4, col5])),[5,1])
        print(features.shape)
        return features, col6  
    

if __name__ == '__main__':

    W = tf.Variable(tf.zeros([1,5]))
    b = tf.Variable(-0.9)


    def inference(x):
        y = tf.nn.softmax(tf.matmul(W,x) + b)
        return y

    def loss(y, y_):
        return -tf.reduce_sum(y_*tf.log(y))
    
    def train(total_loss):
        return tf.train.GradientDescentOptimizer(100).minimize(total_loss)
    
    
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        a = disktrain("/ssd/disk_5_csv")
        features, label = a.csv_2_tensor()
        print("########################")
        print(features, W)        
        print("########################")
        total_loss = loss(inference(features), label)
        train_op = train(total_loss)
        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord = coordinator)
        i = 1
        while i:
            try:
                sess.run([train_op])
                if not (i%1000):
                    print('file num',i)
                    print('input', features.eval())
                    print('loss:', sess.run(total_loss))
                    print('wb:', sess.run([W, b]))
                i = i + 1
            except:
                print("Something wrong")
                i = 0
        coordinator.request_stop()
        coordinator.join(threads)
