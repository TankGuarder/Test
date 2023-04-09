import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Generate the training data
x_train = np.random.rand(1000, 2)
y_train = np.sin(2 * np.pi * x_train[:,0] + 2 * np.pi * x_train[:,1]).reshape((-1, 1))

# Generate the test data
x_test = np.linspace(0, 1, 100)
y_test = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x_test, y_test)
x_test_2d = np.column_stack((X.flatten(), Y.flatten()))

# Hyperparameters
batch_size = 1024
starter_learning_rate = 0.1
train = True

# Define the input placeholders
x = tf.placeholder(tf.float32, shape=[None, 2])
y_true = tf.placeholder(tf.float32, shape=[None, 1])

# Define the network architecture
with tf.variable_scope("model"):
    # Hidden layer with 10 units and ReLU activation
    h1 = tf.layers.dense(x, units=20, activation=tf.nn.tanh, name="hidden1")
    # Batch normalization
    h1_norm = tf.layers.batch_normalization(h1, training=True, name="batch_norm1")
    # Hidden layer with 10 units and ReLU activation
    h2 = tf.layers.dense(h1_norm, units=10, activation=tf.nn.tanh, name="hidden2")
    # Batch normalization
    h2_norm = tf.layers.batch_normalization(h2, training=True, name="batch_norm2")
    # Hidden layer with 10 units and ReLU activation
    h3 = tf.layers.dense(h2_norm, units=5, activation=tf.nn.tanh, name="hidden3")
    # Batch normalization
    h3_norm = tf.layers.batch_normalization(h3, training=True, name="batch_norm3")
    # Output layer with linear activation
    y_pred = tf.layers.dense(h3_norm, units=1, activation=None, name="output")

# Define the loss function
loss = tf.reduce_mean(tf.square(y_true - y_pred))

# Define the optimizer with learning rate decay
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.compat.v1.train.exponential_decay(starter_learning_rate, global_step,
                                           decay_steps=100, decay_rate=0.9, staircase=True)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(loss, global_step=global_step)

# Initialize the variables
init_op = tf.global_variables_initializer()

saver = tf.train.Saver()
# Train the model
with tf.Session() as sess:
    # Initialize the variables
    sess.run(init_op)

    if train == True:
        for epoch in range(2000):
            shuffled_indices = np.random.permutation(len(y_train))
            x_train_shuffle = x_train[shuffled_indices]
            y_train_shuffle = y_train[shuffled_indices]

            for i in range(0, len(y_train), batch_size):
                x_train_batch = x_train_shuffle[i:i+batch_size, :]
                y_train_batch = y_train_shuffle[i:i+batch_size, :]
                _, loss_val, lr_val = sess.run([train_op, loss, learning_rate], feed_dict={x: x_train_batch, y_true: y_train_batch})

            loss_val, lr_val = sess.run([loss, learning_rate], feed_dict={x: x_train, y_true: y_train})
            if epoch % 100 == 0:
                print("Epoch {}/{}: loss = {:.4f}, lr = {:.4f}".format(epoch+1, 2000, loss_val, lr_val))
        saver.save(sess, 'my_model.ckpt')

    else:
        saver.restore(sess, 'my_model.ckpt')


    # Predict the y values for the test data
    y_pred_test = sess.run(y_pred, feed_dict={x: x_test_2d})
    y_pred_test = y_pred_test.reshape((100, 100))

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))

    def update(frame):
        # Compute the time step for the animation
        time_step = frame / 100.0

        # Update the plot with the model predictions
        ax.clear()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Wave Propagation')
        ax.plot(x_test, np.sin(2 * np.pi * x_test + 2 * np.pi * time_step), color='blue')
        ax.plot(x_test, y_pred_test[:, frame], color='red')

    # Create the animation
    anim = FuncAnimation(fig, update, frames=np.arange(0, 100), interval=100, repeat=False)

    # Save the animation as a GIF file
    anim.save('wave_propagation.gif', writer='imagemagick')
