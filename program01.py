import tensorflow as tf

tf.compat.v1.disable_eager_execution()

a = tf.constant(5)
b = tf.constant(2)
c = tf.constant(3)
d = tf.multiply(a, b)
e = tf.add(c, b)
f = tf.subtract(d, e)

sess = tf.compat.v1.Session()
outs = sess.run(f)
sess.close()

print("outs = {}".format(outs))
