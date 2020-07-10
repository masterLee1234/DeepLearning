import tensorflow as tf

X=tf.compat.v1.placeholder(tf.float32, shape=[4,2])
Y=tf.compat.v1.placeholder(tf.float32, shape=[4,1])

W1 = tf.Variable(tf.random.uniform([2,2]))
B1=tf.Variable(tf.zeros([2]))

Z=tf.sigmoid(tf.matmul(X, W1)+B1)

W2=tf.Variable(tf.random.uniform([2, 1]))
B2=tf.Variable(tf.zeros([1]))

Y_hat = tf.sigmoid(tf.matmul(Z, W2)+ B2)

loss = tf.reduce_mean(-1*((Y*tf.math.log(Y_hat))+((1-Y)*tf.math.log(1.0-Y_hat))))

train_step = tf.compat.v1.train.GradientDescentOptimizer(0.05).minimize(loss)

train_X=[[0,0], [0,1], [1,0], [1,1]]
train_Y=[[0],[1],[1],[0]]

init = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    sess.run(init)
    print("train data: "+str(train_X))
    for i in range(20000):
        sess.run(train_step, feed_dict={X: train_X, Y: train_Y})
        if i % 5000 ==0:
            print("Epoch : ", i)
            print("Output : ", sess.run(Y_hat, feed_dict={X: train_X, Y: train_Y}))

    print("Final Output : ", sess.run(Y_hat, feed_dict={X: train_X, Y: train_Y}))