import tensorflow as tf

tf.compat.v1.disable_eager_execution()

print(tf.compat.v1.get_default_graph())
g = tf.Graph()
print(g)

a = tf.constant(5)
print(a.graph is g)
print(a.graph is tf.compat.v1.get_default_graph())
