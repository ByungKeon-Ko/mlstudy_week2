# SGD result : [31.97, 0.65, 1.11]
# Matrix Differentiate result : [31.98, 0.65, 1.11 ]

import tensorflow as tf
sess = tf.InteractiveSession()

train_data = tf.constant( \
   [ [40.,  6.,  4.], \
     [44., 10.,  4.], \
     [46., 12.,  5.], \
     [48., 14.,  7.], \
     [52., 16.,  9.], \
     [58., 18., 12.], \
     [60., 22., 14.], \
     [68., 24., 20.], \
     [74., 26., 21.], \
     [80., 32., 24.] ] )

# Creating X array
# train_data.initializer.run()

temp_mat = tf.slice(train_data, [0,1], [10,2])
x_data = tf.concat(1, [tf.ones([10,1]), temp_mat])
y_data = tf.slice(train_data, [0,0], [10,1])


# Formula : theta = (Xt * X)-1 * Xt * y

xtx = tf.matmul(x_data, x_data, transpose_a=True, transpose_b=False)
xtx_1 = tf.matrix_inverse(xtx)

temp_a = tf.matmul(xtx_1, x_data, transpose_a=False, transpose_b=True)
theta = tf.matmul(temp_a, y_data)

print theta.eval()

sess.close()

