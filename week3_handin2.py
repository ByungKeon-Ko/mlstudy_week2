# Homework for week3
# SGD and basic linear regression
# Purpose of program : estimate the amount of corns depends on uses of fertilizer and insecticide

import tensorflow as tf

# -- Creating Traning Data ------------------------------------ #

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

# state = tf.Variable(0, name="counter")
# 
# one = tf.constant(1)
# new_value = tf.add(state,one)
# update = tf.assign(state, new_value)

# train_data_shuffle = tf.random_shuffle(train_data)
train_data_shuffle = train_data

# Creating X array
temp_mat = tf.slice(train_data_shuffle, [0,1], [10,2])
x_data = tf.concat(1, [tf.ones([10,1]), temp_mat])
y_data = tf.slice(train_data_shuffle, [0,0], [10,1])
# temp_mat = tf.slice(train_data_shuffle, [0,1], [1,2])
# x_data = tf.concat(1, [tf.ones([1,1]), temp_mat])
# y_data = tf.slice(train_data_shuffle, [0,0], [1,1])

xs = tf.placeholder(tf.float32, [None, 3], name="xs")
ys = tf.placeholder(tf.float32, [None, 1], name="ys")

# -- Defining the model ( predictor ) ------------------------ #

theta = tf.Variable([[0.,1.,1.]])
y = tf.matmul(xs, tf.reshape(theta, [3,1]))

# Loss calculation
loss = tf.reduce_mean(tf.square(ys-y))

train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

init_op = tf.initialize_all_variables()

# -- Session ----------------------------------------------------
with tf.Session() as sess:
	sess.run(init_op)

	for i in range(100000):
		batch_xs = x_data.eval()
		batch_ys = y_data.eval()
		sess.run(train_step, feed_dict={xs:batch_xs, ys:batch_ys})

		if(i%1000 == 0):
			print "i=", i
			print "theta : ", sess.run(theta)
			print "loss : ",  sess.run(loss, feed_dict={xs:batch_xs, ys:batch_ys})
