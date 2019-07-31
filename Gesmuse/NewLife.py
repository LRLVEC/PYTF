import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
import os
import time
import RealTime

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

testSize = 32


print("Start to process training data:")
start = time.time()
RealTime.init()
if not os.path.exists("G:/DataSet/TestingSet.npy"):
	testingSet = np.empty([testSize, 256, 256], dtype = np.float32)
	testingSetAnswer1 = np.empty([testSize, 5], dtype = np.bool)
	file = open("G:/DataSet/test.txt","r")
	for i in range(testSize):
		testingSet[i] = cv2.imread("G:/DataSet/test/" + str(i) + ".jpg",cv2.IMREAD_COLOR)[:,:,0] / 256.0
		testingSetAnswer1[i] = [int(x) for x in file.readline().split()]
	file.close()
	testingSet = testingSet.reshape((testSize, 256, 256, 1))
	testingSetAnswer0 = np.array(testingSetAnswer1, dtype = np.float32)
	testingSetAnswer0 = testingSetAnswer0 * 2.0 - 1.0
	np.save("G:/DataSet/TestingSet.npy", testingSet)
	np.save("G:/DataSet/TestingAnswer0.npy", testingSetAnswer0)
	np.save("G:/DataSet/TestingAnswer1.npy", testingSetAnswer1)
else:
	testingSet = np.load("G:/DataSet/TestingSet.npy")
	testingSetAnswer0 = np.load("G:/DataSet/TestingAnswer0.npy")
	testingSetAnswer1 = np.load("G:/DataSet/TestingAnswer1.npy")
stop = time.time()
print("Process time:")
print(stop - start, " s")

if not os.path.exists("./NewLife"):
	w0 = tf.Variable(tf.random.truncated_normal([5, 5, 1, 32], stddev=0.1))
	b0 = tf.Variable(tf.constant(0.1,shape=[32]))
	w1 = tf.Variable(tf.random.truncated_normal([3, 3, 32, 32], stddev=0.1))
	b1 = tf.Variable(tf.constant(0.1,shape=[32]))
	w2 = tf.Variable(tf.random.truncated_normal([3, 3, 32, 32], stddev=0.1))
	b2 = tf.Variable(tf.constant(0.1,shape=[32]))
	w3 = tf.Variable(tf.random.truncated_normal([8 * 8 * 32, 512],stddev=0.1))
	b3 = tf.Variable(tf.constant(0.1,shape=[512]))
	w4 = tf.Variable(tf.random.truncated_normal([512, 256], stddev=0.1))
	b4 = tf.Variable(tf.constant(0.1,shape=[256]))
	w5 = tf.Variable(tf.random.truncated_normal([256, 5], stddev=0.1))
	b5 = tf.Variable(tf.constant(0.1,shape=[5]))
else:
	w0 = tf.Variable(tf.convert_to_tensor(np.load("NewLife/w0.npy")))
	w1 = tf.Variable(tf.convert_to_tensor(np.load("NewLife/w1.npy")))
	w2 = tf.Variable(tf.convert_to_tensor(np.load("NewLife/w2.npy")))
	w3 = tf.Variable(tf.convert_to_tensor(np.load("NewLife/w3.npy")))
	w4 = tf.Variable(tf.convert_to_tensor(np.load("NewLife/w4.npy")))
	w5 = tf.Variable(tf.convert_to_tensor(np.load("NewLife/w5.npy")))
	b0 = tf.Variable(tf.convert_to_tensor(np.load("NewLife/b0.npy")))
	b1 = tf.Variable(tf.convert_to_tensor(np.load("NewLife/b1.npy")))
	b2 = tf.Variable(tf.convert_to_tensor(np.load("NewLife/b2.npy")))
	b3 = tf.Variable(tf.convert_to_tensor(np.load("NewLife/b3.npy")))
	b4 = tf.Variable(tf.convert_to_tensor(np.load("NewLife/b4.npy")))
	b5 = tf.Variable(tf.convert_to_tensor(np.load("NewLife/b5.npy")))
x0 = tf.compat.v1.placeholder(tf.float32, shape = [None, 256, 256, 1])
x1 = tf.nn.relu(tf.nn.conv2d(x0, w0, strides = [1, 2, 2, 1], padding='SAME') + b0)
x2 = tf.nn.max_pool2d(x1, ksize = [1, 4, 4, 1], strides = [1, 4, 4, 1], padding = 'SAME')
x3 = tf.nn.relu(tf.nn.conv2d(x2, w1, strides = [1, 2, 2, 1], padding='SAME') + b1)
x4 = tf.nn.max_pool2d(x3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
x5 = tf.nn.relu(tf.nn.conv2d(x4, w2, strides = [1, 1, 1, 1], padding='SAME') + b2)
x6 = tf.nn.max_pool2d(x5, ksize = [1, 2, 2, 1], strides = [1, 1, 1, 1], padding = 'SAME')
x7 = tf.nn.relu(tf.matmul(tf.reshape(x6, [-1, 8 * 8 * 32]), w3) + b3)
x8 = tf.nn.relu(tf.matmul(x7, w4) + b4)
x9 = tf.matmul(x8, w5) + b5
answer0 = tf.compat.v1.placeholder(tf.float32, shape = [None, 5])
answer1 = tf.compat.v1.placeholder(tf.bool, shape = [None, 5])
delta = tf.reduce_mean(tf.square(x9 - answer0))

trainStep = tf.compat.v1.train.AdagradOptimizer(2e-3).minimize(delta)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.reduce_sum(tf.cast(tf.equal(answer1, tf.greater(x9, 0)), dtype = tf.int32), axis = -1), 5),dtype = tf.float32))
batchSize = 32

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)
sess.run(tf.global_variables_initializer())

trainTimes = 20000
testStride = 250
if os.path.exists("NewLife/TestAccuracy.npy"):
	acc_test_pre = np.load("NewLife/TestAccuracy.npy")
	acc_train_pre = np.load("NewLife/TrainAccuracy.npy")
	trainBegining = len(acc_test_pre) * testStride
	acc_test = np.concatenate((acc_test_pre, np.empty([int(trainTimes / testStride)],dtype = np.float32)), axis=0)
	acc_train = np.concatenate((acc_train_pre, np.empty([int(trainTimes / testStride)],dtype = np.float32)), axis=0)
else:
	trainBegining = 0
	acc_test = np.empty([int(trainTimes / testStride)],dtype = np.float32)
	acc_train = np.empty([int(trainTimes / testStride)],dtype = np.float32)
trainTimesAll = trainBegining + trainTimes

start = time.time()
for i in range(trainBegining, trainTimesAll):
	if i % 1000 == 0:
		trainingBatch, answerBatch0, answerBatch1 = RealTime.run(1000)
	if i % testStride == 0:
		trainingBatch_, answerBatch0_, answerBatch1_ = RealTime.run(32)
		test_accuracy = accuracy.eval(session = sess, feed_dict = {x0:trainingBatch_, answer0: answerBatch0_, answer1: answerBatch1_})
		acc_test[int(i / testStride)] = test_accuracy
		train_accuracy = accuracy.eval(session = sess, feed_dict = {x0:trainingBatch, answer0: answerBatch0, answer1: answerBatch1})
		acc_train[int(i / testStride)] = train_accuracy
		print("step %d: %.3f    %.3f" % (i, test_accuracy, train_accuracy))
	id = np.random.choice(1000, batchSize)
	trainStep.run(session = sess, feed_dict = {x0:trainingBatch[id], answer0:answerBatch0[id], answer1: answerBatch1[id]})
stop = time.time()
if not os.path.exists("./NewLife"):
	os.makedirs("./NewLife")
np.save("NewLife/w0.npy", w0.eval(session = sess))
np.save("NewLife/w1.npy", w1.eval(session = sess))
np.save("NewLife/w2.npy", w2.eval(session = sess))
np.save("NewLife/w3.npy", w3.eval(session = sess))
np.save("NewLife/w4.npy", w4.eval(session = sess))
np.save("NewLife/w5.npy", w5.eval(session = sess))
np.save("NewLife/b0.npy", b0.eval(session = sess))
np.save("NewLife/b1.npy", b1.eval(session = sess))
np.save("NewLife/b2.npy", b2.eval(session = sess))
np.save("NewLife/b3.npy", b3.eval(session = sess))
np.save("NewLife/b4.npy", b4.eval(session = sess))
np.save("NewLife/b5.npy", b5.eval(session = sess))

sess.close()
np.save("NewLife/TestAccuracy.npy",acc_test)
np.save("NewLife/TrainAccuracy.npy",acc_train)
plt.plot(range(0, trainTimesAll, testStride), acc_test, 'r-', label='Test Accuracy')
plt.plot(range(0, trainTimesAll, testStride), acc_train, 'b-', label='Train Accuracy')
plt.legend(loc='lower right', prop={'size': 11})
plt.show()
