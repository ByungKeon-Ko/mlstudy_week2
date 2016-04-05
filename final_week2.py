# can support 2D- matrix of any shapes

import tensorflow as tf

# -------- input data ---------------------------------- #
input_mat = [ [1,2,3], [4,5,6], [7,8,9] ]
# input_mat = [ [1,2,3], [4,5,6], [7,8,9], [10, 11, 12]  ]
# input_mat = [ [1,2,3,4,5], [6,7,8,9,10], [11,12,13,14,15] ]

# -------- OPs ----------------------------------------- #
tmp_mat = tf.Variable(input_mat, name="tmp_mat")

# finding centor vector
measure_size = tf.shape(tmp_mat)
ts_zero = tf.constant([0])
ts_one = tf.constant([1])
center_col = tf.concat(0, [ts_zero,  tf.slice(measure_size, [1], [1]) / 2] )
shape_col = tf.concat(0, [tf.slice(measure_size, [0], [1]), ts_one ] )

# extracting the vector
vector_col = tf.slice(tmp_mat, center_col, shape_col)

# Reshaping vector : col vector -> row vector
vector_row = tf.reshape(vector_col, tf.slice(measure_size, [0], [1]) )
# vector_row = tf.reshape(vector_col, [1,3])

init_op = tf.initialize_all_variables()

# -------- Session  ------------------------------------- #
with tf.Session() as sess:
	sess.run(init_op)
	print sess.run(vector_row)
	# print sess.run(measure_size)
	# print sess.run(center_col)
	# print sess.run(shape_col)
	# sess.run(vector_col)


