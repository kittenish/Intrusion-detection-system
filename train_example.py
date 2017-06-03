import tensorflow as tf
import numpy as np
from tf_utils import weight_variable, bias_variable, dense_to_one_hot, weight_variable_cnn
from process_data import znormalization
from shuffle_data import shuffle

data = np.load('./feature/udp_private.npz')

x_train = data['x_train'][:,0:7]
y_train = data['y_train'].ravel()
x_test = data['x_test'][:,0:7]
y_test = data['y_test'].ravel()

X_train = znormalization(x_train)
X_test = znormalization(x_test)
X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)

[X_train, y_train] = shuffle(X_train, y_train, x_train.shape[0])
Y_train = dense_to_one_hot(y_train, n_classes=2)
Y_test = dense_to_one_hot(y_test, n_classes=2)

x = tf.placeholder(tf.float32, [None, 7])
y = tf.placeholder(tf.float32,[None, 2])
keep_prob = tf.placeholder(tf.float32)

W_fc1 = weight_variable_cnn([7, 5])
b_fc1 = bias_variable([5])
h_fc1 = tf.nn.tanh(tf.matmul(x, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = weight_variable_cnn([5, 2])
b_fc2 = bias_variable([2])
h_fc2 = tf.nn.tanh(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

y_logits = h_fc2

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(y_logits, y))
opt = tf.train.AdamOptimizer(1e-3)
optimizer = opt.minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
grads = opt.compute_gradients(cross_entropy, [W_fc1])

sess = tf.Session()
sess.run(tf.initialize_all_variables())

iter_per_epoch = 10
n_epochs = 10000
train_size = x_train.shape[0]

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
    print('test (%d): ' % epoch_i + str(sess.run(accuracy,feed_dict={x: X_test,y: Y_test,keep_prob: 1.0})))
    # grad_vals = sess.run([g for (g,v) in grads], feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
    # print 'grad_vals: ', grad_vals
    theta = sess.run(h_fc1, feed_dict={x: batch_xs, keep_prob: 1.0})
    print theta
    
