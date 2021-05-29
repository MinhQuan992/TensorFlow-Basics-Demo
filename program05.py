import tensorflow as tf

a = tf.constant(4.0)
print(a)
print(a.dtype)

b = tf.constant(4.0, dtype=tf.float64)
print(b)
print(b.dtype)

x = tf.constant([1, 2, 3], name='x', dtype=tf.float32)
print(x.dtype)
x = tf.cast(x, tf.int64)
print(x.dtype)
