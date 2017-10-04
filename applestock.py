import tensorflow as tf
import numpy as np
import pandas as pd

sess = tf.Session()

file = np.array(pd.read_csv("apg.csv"))

x_train =  np.array([x[1] for x in file])
y_train =  np.array([x[0] for x in file])
o1 = 0.000000001
o2 = 0.000000000001
o3 = 0.00000000000001
o4 = 0.000000000000001
o5 = 0.000000000000001
y_test = np.array([150.550003])
x_test = np.array([170925])

y_accuracy = ((150.550003))
x_accuracy = ((170925))

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

W = tf.Variable(0.0)
w = tf.Variable(0.0)
b = tf.Variable(0.0)
B = tf.Variable(0.0)
w_= tf.Variable(0.0)
b_= tf.Variable(0.0)
c = tf.Variable(0.0)
v = tf.Variable(0.0)
l = tf.Variable(0.0)
p = tf.Variable(0.0)
layer1 = w * x + B
layer2 = w_ * layer1 + b_
layer3 = c * layer2 + v
layer4 = l * layer3 + p
model = W * layer4 + b

loss = tf.reduce_sum(tf.square(model - y))
loss1 = tf.reduce_sum(tf.square(layer1- y))
loss2 = tf.reduce_sum(tf.square(layer2- y))
loss3 = tf.reduce_sum(tf.square(layer3- y))
loss4 = tf.reduce_sum(tf.square(layer4- y))


train0 = tf.train.GradientDescentOptimizer(o1).minimize(loss)
train1 = tf.train.GradientDescentOptimizer(o2).minimize(loss1)
train2 = tf.train.GradientDescentOptimizer(o3).minimize(loss2)
train3 = tf.train.GradientDescentOptimizer(o4).minimize(loss3)
train4 = tf.train.GradientDescentOptimizer(o5).minimize(loss4)
sess.run(tf.global_variables_initializer())

for i in range(5000):
	sess.run(train4, {x: x_train, y: y_train})
	sess.run(train3, {x: x_train, y: y_train})


	sess.run(train2, {x: x_train, y: y_train})


	sess.run(train1, {x: x_train, y: y_train})

	sess.run(train0, {x: x_train, y: y_train})





print(sess.run(model, {x: x_test}))
acc = sess.run(model, {x: x_accuracy})
accuracy =acc - y_accuracy
accuracyf = abs(accuracy)
print(accuracyf)
