import tensorflow as tf

tf.compat.v1.disable_eager_execution()


def user_input():
    print("Vector A:")
    a1 = int(input("a1 = "))
    a2 = int(input("a2 = "))
    a3 = int(input("a3 = "))

    print("Vector B:")
    b1 = int(input("b1 = "))
    b2 = int(input("b2 = "))
    b3 = int(input("b3 = "))

    return [a1, a2, a3], [b1, b2, b3]


def demo_placeholder(A, B):
    a = tf.compat.v1.placeholder(tf.int32, shape=3)
    b = tf.compat.v1.placeholder(tf.int32, shape=3)
    c = 2 * a + b
    with tf.compat.v1.Session() as sess:
        print(sess.run(c, feed_dict={a: A, b: B}))


if __name__ == "__main__":
    A, B = user_input()
    print(A, B)
    demo_placeholder(A, B)
