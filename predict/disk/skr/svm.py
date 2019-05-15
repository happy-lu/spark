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
        record_defaults = [[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]]
        col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13 = tf.decode_csv(value, record_defaults=record_defaults)
        features = tf.reshape(tf.transpose(tf.stack([col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11])),[11,1])
        label = tf.reshape(tf.transpose(tf.stack([col12/10])),[1,1])
        return features, label
    

if __name__ == '__main__':

    W = tf.Variable(tf.random_normal(shape=[1, 11]))
    b = tf.Variable(tf.random_normal(shape=[1, 1]))

    def inference(x):
        y = tf.subtract(tf.matmul(W,x),b)
        return y

    def loss(y, y_):
        l2_norm = tf.reduce_sum(tf.square(W))
        alpha = tf.constant([0.01])
        classification_term = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(y, y_))))
        loss = tf.add(classification_term, tf.multiply(alpha, l2_norm))
        return loss

    def train(loss):
        my_opt = tf.train.GradientDescentOptimizer(0.0001)
        train_step = my_opt.minimize(loss)   
        return train_step
    
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        a = disktrain("/ssd/disk_11_csv")
        features, label = a.csv_2_tensor()
        total_loss = loss(inference(features), label)
        train_op = train(total_loss)
        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord = coordinator)
        i = 1
        best_loss = 3
        best_wb = None
        circle_i = -1
        while i:
            try:
                sess.run([train_op])
                if best_loss > sess.run(total_loss):
                   best_loss = sess.run(total_loss)
                   best_wb = sess.run([W, b])
                   circle_i = i
                if label.eval()[0] > 0:
                    print('file num',i)
                    print('label', float(label.eval()[0]))
                    print('loss:', sess.run(total_loss))
                    print('wb:', sess.run([W, b]))
                i = i + 1
                if not(i%10000): 
                    print('file num',i)
                    print('label', label.eval()[0])
                    print('loss:', sess.run(total_loss))
                    print('wb:', sess.run([W, b]))
            except:
                i = 0
        print('############################')
        print('loss:', best_loss)
        print('wb:', best_wb)
        print('circle_i', circle_i)
        print('############################')
        coordinator.request_stop()
        coordinator.join(threads)
