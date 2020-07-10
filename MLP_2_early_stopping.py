import tensorflow as tf
import numpy as np
from IPython.display import Image

Image(url= "https://raw.githubusercontent.com/minsuk-heo/deeplearning/master/img/dropout.png", width=500, height=250)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print(x_train.shape)
print(x_test.shape)

x_val  = x_train[50000:60000]
x_train = x_train[0:50000]
y_val  = y_train[50000:60000]
y_train = y_train[0:50000]

print("train data has " + str(x_train.shape[0]) + " samples")
print("every train data is " + str(x_train.shape[1])
      + " * " + str(x_train.shape[2]) + " image")

print("validation data has " + str(x_val.shape[0]) + " samples")
print("every train data is " + str(x_val.shape[1])
      + " * " + str(x_train.shape[2]) + " image")

print(x_train[0][8])

print(y_train[0:9])

print("test data has " + str(x_test.shape[0]) + " samples")
print("every test data is " + str(x_test.shape[1])
      + " * " + str(x_test.shape[2]) + " image")

Image(url= "https://raw.githubusercontent.com/minsuk-heo/deeplearning/master/img/reshape_mnist.png", width=500, height=250)

x_train = x_train.reshape(50000, 784)
x_val = x_val.reshape(10000, 784)
x_test = x_test.reshape(10000, 784)

print(x_train.shape)
print(x_test.shape)

x_train[0]

x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')

gray_scale = 255
x_train /= gray_scale
x_val /= gray_scale
x_test /= gray_scale

num_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_val = tf.keras.utils.to_categorical(y_val, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

y_train

Image(url= "https://raw.githubusercontent.com/minsuk-heo/deeplearning/master/img/simple_mlp_mnist.png", width=500, height=250)

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)


def mlp(x):
    # hidden layer1
    w1 = tf.Variable(tf.random_uniform([784, 256]))
    b1 = tf.Variable(tf.zeros([256]))
    h1 = tf.nn.relu(tf.matmul(x, w1) + b1)
    # hidden layer2
    w2 = tf.Variable(tf.random_uniform([256, 128]))
    b2 = tf.Variable(tf.zeros([128]))
    h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
    h2_drop = tf.nn.dropout(h2, keep_prob)
    # output layer
    w3 = tf.Variable(tf.random_uniform([128, 10]))
    b3 = tf.Variable(tf.zeros([10]))
    logits = tf.matmul(h2_drop, w3) + b3

    return logits

logits = mlp(x)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=logits, labels=y))

train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss_op)

Image(url= "https://raw.githubusercontent.com/minsuk-heo/deeplearning/master/img/early_stop.png", width=500, height=250)

# initialize
init = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# train hyperparameters
epoch_cnt = 300
batch_size = 1000
iteration = len(x_train) // batch_size

earlystop_threshold = 10
earlystop_cnt = 0

# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    prev_train_acc = 0.0
    max_val_acc = 0.0

    for epoch in range(epoch_cnt):
        avg_loss = 0.
        start = 0;
        end = batch_size

        for i in range(iteration):
            _, loss = sess.run([train_op, loss_op],
                               feed_dict={x: x_train[start: end], y: y_train[start: end],
                                          keep_prob: 0.9})
            start += batch_size;
            end += batch_size
            # Compute train average loss
            avg_loss += loss / iteration

        # Validate model
        preds = tf.nn.softmax(logits)  # Apply softmax to logits
        correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        # train accuracy
        cur_train_acc = accuracy.eval({x: x_train, y: y_train, keep_prob: 1.0})
        # validation accuarcy
        cur_val_acc = accuracy.eval({x: x_val, y: y_val, keep_prob: 1.0})
        # validation loss
        cur_val_loss = loss_op.eval({x: x_val, y: y_val, keep_prob: 1.0})

        print("epoch: " + str(epoch) +
              ", train acc: " + str(cur_train_acc) +
              ", val acc: " + str(cur_val_acc))
        # ', train loss: '+str(avg_loss)+
        # ', val loss: '+str(cur_val_loss))

        if cur_val_acc < max_val_acc:
            if cur_train_acc > prev_train_acc or cur_train_acc > 0.99:
                if earlystop_cnt == earlystop_threshold:
                    print("early stopped on " + str(epoch))
                    break
                else:
                    print("overfitting warning: " + str(earlystop_cnt))
                    earlystop_cnt += 1
            else:
                earlystop_cnt = 0
        else:
            earlystop_cnt = 0
            max_val_acc = cur_val_acc
            # Save the variables to file.
            save_path = saver.save(sess, "model/model.ckpt")
        prev_train_acc = cur_train_acc

# Start testing
with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, "model/model.ckpt")
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("[Test Accuracy] :", accuracy.eval({x: x_test, y: y_test, keep_prob: 1.0}))