import tensorflow as tf

tf.compat.v1.disable_eager_execution()

# create a placeholder for a vector of 3 elements, type tf.float32
a = tf.compat.v1.placeholder(tf.float32, shape=[3])
b = tf.constant([5, 5, 5], tf.float32)
c = tf.add(a, b)

with tf.compat.v1.Session() as sess:
    print(sess.run(c, feed_dict={a: [1, 2, 3]}))

# Mở cho lớp xem program12
