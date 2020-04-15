import tensorflow as tf

x_data = [[0,0],[0,1],[1,0],[1,1]]
y_data = [[0],[1],[1],[0]]

X = tf.placeholder(tf.float32,[None,2])
Y = tf.placeholder(tf.float32,[None,1])

W1 = tf.Variable(tf.random_uniform([2,4], -1.0,1.0))
b1 = tf.Variable(tf.random_uniform([4], -1.0,1.0))
logits1 = tf.add(tf.matmul(X,W1),b1)
layer1 = tf.nn.sigmoid(logits1)

W2 = tf.Variable(tf.random_uniform([4,1],-1.0,1.0))
b2 = tf.Variable(tf.random_uniform([1],-1.0,1.0))
logits2 = tf.add(tf.matmul(layer1,W2),b2)
layer2 = tf.nn.sigmoid(logits2)

cost = -tf.reduce_mean(Y * tf.log(layer2) + (1 - Y) * tf.log(1 - layer2))

opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = opt.minimize(cost)

predicted = tf.cast(layer2 > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y),dtype = tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(10000):
        sess.run(train,feed_dict={X:x_data,Y:y_data})
        if step % 1000 == 0:
            print(step, sess.run(cost,feed_dict={X:x_data,Y:y_data}),sess.run([W1,W2]))
    
    h, c, a = sess.run([layer2, predicted, accuracy],feed_dict = {X:x_data, Y:y_data})
    print(h)
