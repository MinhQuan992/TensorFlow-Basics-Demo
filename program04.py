import tensorflow as tf

tf.compat.v1.disable_eager_execution()

a = tf.constant(5)
b = tf.constant(2)
c = tf.constant(3)
d = tf.multiply(a, b)
e = tf.add(c, b)
f = tf.subtract(d, e)

with tf.compat.v1.Session() as sess:
    fetches = [a, b, c, d, e, f]
    outs = sess.run(fetches)

print("outs = {}".format(outs))
print(type(outs[0]))
