import tf.mnist.input_data as input_data
import tensorflow as tf


def init_model():
    sess = tf.InteractiveSession()
    x = tf.placeholder("float", [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    return sess


if __name__ == '__main__':

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # create model
    x = tf.placeholder("float", [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    # train
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder("float", [None, 10])
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    init = tf.initialize_all_variables()

    # sess = tf.Session()
    # sess.run(init)
    sess = init_model()
    summary_writer = tf.summary.FileWriter(r'E:\tf_tmp\mnist_logs_simple', sess.graph)

    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # valid
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # print(sess.run(accuracy, feed_dict={x: mnist.validation.images, y_: mnist.validation.labels}))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
