import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import imageio


def plot_for_offset(x, y, x_axis, y_axis, w, b, i, loss_value):
    # Data for plotting
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(x, y)
    ax.plot(x_axis, y_axis, color='red')
    ax.grid()
    ax.set(xlabel='X', ylabel='Y',
           title="Iter: {}, w: {}, b: {}, loss value: {}".format(i, w, b, loss_value))
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(1, 5)

    # IMPORTANT ANIMATION CODE HERE
    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return image


tf.compat.v1.disable_eager_execution()

x = np.arange(0, 1, 0.01)
y = np.arange(2, 4, 0.02) + np.random.randn(x.shape[0], ) * 0.2

x_train = tf.constant(x, dtype=np.float32, shape=[x.shape[0], 1])
y_train = tf.constant(y, dtype=np.float32, shape=[x.shape[0], 1])
with tf.compat.v1.variable_scope("foo"):
    weights = tf.compat.v1.get_variable("weights", [1, 1], initializer=tf.random_normal_initializer())
    biases = tf.compat.v1.get_variable("biases", [1, 1], initializer=tf.constant_initializer(0.0))

y_hat = tf.add(tf.matmul(x_train, weights), biases)

loss = tf.reduce_mean(tf.square(y_hat - y_train))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)

init_op = tf.compat.v1.global_variables_initializer()
image_list = []

with tf.compat.v1.Session() as sess:
    sess.run(init_op)
    x_axis = np.arange(0, 1, 0.1)
    for i in range(200):
        _, loss_value = sess.run((train, loss))
        w = sess.run(weights)
        b = sess.run(biases)
        if i % 5 == 0:
            print(loss_value)
            image_list.append(plot_for_offset(x, y, x_axis, w[0][0] * x_axis + b[0][0], w, b, i, loss_value))

imageio.mimsave('./linear_regression_fitting.gif', image_list, fps=3)
