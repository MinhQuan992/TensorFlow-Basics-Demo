import tensorflow as tf

with tf.Graph().as_default():
    c1 = tf.constant(4, dtype=tf.float64, name='c')
    c2 = tf.constant(4, dtype=tf.int32, name='c')

print(c1.name)
print(c2.name)

with tf.Graph().as_default():
    c = tf.constant(2, name='c')

print(c.name)
