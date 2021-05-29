import tensorflow as tf

tf.compat.v1.disable_eager_execution()

# Cách 1
a = tf.Variable(2, name="scalar")
b = tf.Variable([[0, 1], [2, 3]], name="matrix")

# Cách 2
c = tf.compat.v1.get_variable("scalar", initializer=tf.constant(2))
d = tf.compat.v1.get_variable("matrix", initializer=tf.constant([[0, 1], [2, 3]]))

print("a pre run: \n{}".format(a))
print("b pre run: \n{}".format(b))
print("c pre run: \n{}".format(c))
print("d pre run: \n{}".format(d))

# Khởi tạo tất cả các biến
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    print("a post run: \n{}".format(sess.run(a)))
    print("b post run: \n{}".format(sess.run(b)))
    print("c post run: \n{}".format(sess.run(c)))
    print("d post run: \n{}".format(sess.run(d)))

# # Khởi tạo một số biến nhất định
# with tf.compat.v1.Session() as sess:
#     sess.run(tf.compat.v1.variables_initializer([a, b]))
#     print("a post run: \n{}".format(sess.run(a)))
#     print("b post run: \n{}".format(sess.run(b)))
#     print("c post run: \n{}".format(sess.run(c)))
#     print("d post run: \n{}".format(sess.run(d)))

# # Khởi tạo một biến
# with tf.compat.v1.Session() as sess:
#     sess.run(a.initializer)
#     print("a post run: \n{}".format(sess.run(a)))
#     print("b post run: \n{}".format(sess.run(b)))
#     print("c post run: \n{}".format(sess.run(c)))
#     print("d post run: \n{}".format(sess.run(d)))
