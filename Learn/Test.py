import matplotlib.pyplot as plt
import math
import numpy as np
import tensorflow as tf
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data = np.empty([2000, 2],dtype = np.float32)
answer = np.empty([2000, 1],dtype = np.float32)
for i in range(1000):
	answer[i] = np.array([-1], dtype = np.float32)
	theta = 2 * np.pi * np.random.random()
	data[i] = np.array([0.25 * math.cos(theta), 0.25 * math.sin(theta)],dtype = np.float32)
for i in range(1000,2000):
	answer[i] = np.array([1], dtype = np.float32)
	theta = 2 * np.pi * np.random.random()
	data[i] = np.array([0.5 * math.cos(theta), 0.5 * math.sin(theta)],dtype = np.float32)

sess = tf.Session()
batchSize = 128
inputBatch = tf.placeholder(tf.float32, [2 ,None])
answerBatch = tf.placeholder(tf.float32, [1, None])
w0 = tf.Variable(tf.random_normal([5, 2]))
b0 = tf.Variable(tf.random_normal([5, 1]))
w1 = tf.Variable(tf.random_normal([3, 5]))
b1 = tf.Variable(tf.random_normal([3, 1]))
w2 = tf.Variable(tf.random_normal([1, 3]))
b2 = tf.Variable(tf.random_normal([1, 1]))
x1 = tf.nn.tanh(tf.matmul(w0, inputBatch) + b0)
x2 = tf.nn.tanh(tf.matmul(w1, x1) + b1)
x3 = tf.matmul(w2, x2) + b2
delta = tf.reduce_mean(tf.square(x3 - answerBatch))
accuracy = tf.reduce_mean(tf.cast(tf.logical_xor(tf.less(x3 ,0), tf.equal(answerBatch ,1)), tf.float32))
init = tf.global_variables_initializer()
sess.run(init)
option = tf.train.GradientDescentOptimizer(0.05)
train = option.minimize(delta)
begin = time.time()
acc = []
for i in range(1000):
	idx = np.random.choice(2000, batchSize)
	inputSet = np.transpose(data[idx])
	answerSet = np.transpose(answer[idx])
	sess.run(train, {inputBatch: inputSet, answerBatch: answerSet})
	if i % 20 == 0:
		acc.append(sess.run(accuracy, {inputBatch: inputSet, answerBatch: answerSet}))
end = time.time()
print("Time used: ", end - begin, "s")
inputSet = np.transpose(data)
answerSet = np.transpose(answer)
print("Final accuracy:", sess.run(accuracy, {inputBatch: inputSet, answerBatch: answerSet}))
sess.close()
plt.plot(range(0, 1000, 20), acc, 'b-', label='Accuracy')
plt.legend(loc='upper right', prop={'size': 11})
plt.show()
