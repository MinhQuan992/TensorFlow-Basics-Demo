import tensorflow as tf

g1 = tf.compat.v1.get_default_graph()
g2 = tf.Graph()

print(g1 is tf.compat.v1.get_default_graph())

with g2.as_default():
    print(g1 is tf.compat.v1.get_default_graph())

print(g1 is tf.compat.v1.get_default_graph())
