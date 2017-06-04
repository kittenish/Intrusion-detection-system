import tensorflow as tf
import numpy as np
from tf_utils import weight_variable, bias_variable, dense_to_one_hot, weight_variable_cnn
import time

train = np.load('./feature/train.npz')
test = np.load('./feature/test.npz')

X_train = train['feature'][:,0:9]
label_2 = train['label_2']
label_5 = train['label_5']
label_23 = train['label_23']
y_train = label_2.ravel()
X_test = test['feature'][:,0:9]
test_l = test['label_2']
y_test = test_l.ravel()

Y_train = dense_to_one_hot(y_train, n_classes=2)
Y_test = dense_to_one_hot(y_test, n_classes=2)

x = tf.placeholder(tf.float32, [None, 9])
y = tf.placeholder(tf.float32,[None, 2])
keep_prob = tf.placeholder(tf.float32)

W_fc1 = weight_variable([9, 5])
b_fc1 = bias_variable([5])
h_fc1 = tf.nn.tanh(tf.matmul(x, W_fc1) + b_fc1)
W_fc2 = weight_variable([5, 2])
b_fc2 = bias_variable([2])
h_fc2 = tf.nn.tanh(tf.matmul(h_fc1, W_fc2) + b_fc2)

y_logits = h_fc2

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(y_logits, y))
opt = tf.train.AdamOptimizer()
optimizer = opt.minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
grads = opt.compute_gradients(cross_entropy, [W_fc1])

sess = tf.Session()
sess.run(tf.initialize_all_variables())

iter_per_epoch = 60
n_epochs = 1000
train_size = 125973

indices = np.linspace(0, train_size - 1, iter_per_epoch)
indices = indices.astype('int')

for epoch_i in range(n_epochs):
    for iter_i in range(iter_per_epoch - 1):
        batch_xs = X_train[indices[iter_i]:indices[iter_i+1]]
        batch_ys = Y_train[indices[iter_i]:indices[iter_i+1]]

        if iter_i % 10 == 0:
            loss = sess.run(cross_entropy,
                            feed_dict={
                                x: batch_xs,
                                y: batch_ys,
                                keep_prob: 1.0
                            })
            print('Iteration: ' + str(iter_i) + ' Loss: ' + str(loss))

        sess.run(optimizer, feed_dict={
            x: batch_xs, y: batch_ys, keep_prob: 0.75})
    time1 = time.time()
    print('test (%d): ' % epoch_i + str(sess.run(accuracy,feed_dict={x: X_test,y: Y_test,keep_prob: 1.0})))
    time2 = time.time()
    print float(time2 - time1) / float(len(y_test))
    # grad_vals = sess.run([g for (g,v) in grads], feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
    # print 'grad_vals: ', grad_vals
