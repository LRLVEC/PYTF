import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


trainSize = 32000
testSize = 32


print("Start to process training data:")
start = time.time()
if not os.path.exists("G:/DataSet/TrainingSet.npy"):
	trainingSet = np.empty([trainSize, 256, 256], dtype = np.float32)
	testingSet = np.empty([testSize, 256, 256], dtype = np.float32)
	trainingSetAnswer1 = np.empty([trainSize, 5], dtype = np.bool)
	testingSetAnswer1 = np.empty([testSize, 5], dtype = np.bool)
	file = open("G:/DataSet/train.txt","r")
	for i in range(trainSize):
		trainingSet[i] = cv2.imread("G:/DataSet/train/" + str(i) + ".png",cv2.IMREAD_COLOR)[:,:,0] / 256.0
		trainingSetAnswer1[i] = [int(x) for x in file.readline().split()]
	file.close()
	trainingSet = trainingSet.reshape((trainSize, 256, 256, 1))
	trainingSetAnswer0 = np.array(trainingSetAnswer1, dtype = np.float32)
	trainingSetAnswer0 = trainingSetAnswer0 * 2.0 - 1.0
	file = open("G:/DataSet/test.txt","r")
	for i in range(testSize):
		testingSet[i] = cv2.imread("G:/DataSet/test/" + str(i) + ".jpg",cv2.IMREAD_COLOR)[:,:,0] / 256.0
		testingSetAnswer1[i] = [int(x) for x in file.readline().split()]
	file.close()
	testingSet = testingSet.reshape((testSize, 256, 256, 1))
	testingSetAnswer0 = np.array(testingSetAnswer1, dtype = np.float32)
	testingSetAnswer0 = testingSetAnswer0 * 2.0 - 1.0
	np.save("G:/DataSet/TrainingSet.npy", trainingSet)
	np.save("G:/DataSet/TrainingAnswer0.npy", trainingSetAnswer0)
	np.save("G:/DataSet/TrainingAnswer1.npy", trainingSetAnswer1)
	np.save("G:/DataSet/TestingSet.npy", testingSet)
	np.save("G:/DataSet/TestingAnswer0.npy", testingSetAnswer0)
	np.save("G:/DataSet/TestingAnswer1.npy", testingSetAnswer1)
else:
	trainingSet = np.load("G:/DataSet/TrainingSet.npy")
	testingSet = np.load("G:/DataSet/TestingSet.npy")
	trainingSetAnswer0 = np.load("G:/DataSet/TrainingAnswer0.npy")
	testingSetAnswer0 = np.load("G:/DataSet/TestingAnswer0.npy")
	trainingSetAnswer1 = np.load("G:/DataSet/TrainingAnswer1.npy")
	testingSetAnswer1 = np.load("G:/DataSet/TestingAnswer1.npy")
stop = time.time()
#for i in trainingSet[0]:
#	for j in i:
#		if j > 0.7:print(j)
print("Process time:")
print(stop - start, " s")

if not os.path.exists("./weight"):
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
	w0 = tf.Variable(tf.convert_to_tensor(np.load("weight/w0.npy")))
	w1 = tf.Variable(tf.convert_to_tensor(np.load("weight/w1.npy")))
	w2 = tf.Variable(tf.convert_to_tensor(np.load("weight/w2.npy")))
	w3 = tf.Variable(tf.convert_to_tensor(np.load("weight/w3.npy")))
	w4 = tf.Variable(tf.convert_to_tensor(np.load("weight/w4.npy")))
	w5 = tf.Variable(tf.convert_to_tensor(np.load("weight/w5.npy")))
	b0 = tf.Variable(tf.convert_to_tensor(np.load("weight/b0.npy")))
	b1 = tf.Variable(tf.convert_to_tensor(np.load("weight/b1.npy")))
	b2 = tf.Variable(tf.convert_to_tensor(np.load("weight/b2.npy")))
	b3 = tf.Variable(tf.convert_to_tensor(np.load("weight/b3.npy")))
	b4 = tf.Variable(tf.convert_to_tensor(np.load("weight/b4.npy")))
	b5 = tf.Variable(tf.convert_to_tensor(np.load("weight/b5.npy")))
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
batchSize = 16

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)
sess.run(tf.global_variables_initializer())

trainTimes = 20000
testStride = 250
if os.path.exists("weight/TestAccuracy.npy"):
	acc_test_pre = np.load("weight/TestAccuracy.npy")
	acc_train_pre = np.load("weight/TrainAccuracy.npy")
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
	if i % testStride == 0:
		test_accuracy = accuracy.eval(session = sess, feed_dict = {x0:testingSet, answer0: testingSetAnswer0, answer1: testingSetAnswer1})
		acc_test[int(i / testStride)] = test_accuracy
		idx = np.random.choice(trainSize, 1000)
		trainingBatch = trainingSet[idx]
		answerBatch0 = trainingSetAnswer0[idx]
		answerBatch1 = trainingSetAnswer1[idx]
		train_accuracy = accuracy.eval(session = sess, feed_dict = {x0:trainingBatch, answer0: answerBatch0, answer1: answerBatch1})
		acc_train[int(i / testStride)] = train_accuracy
		print("step %d: %.3f    %.3f" % (i, test_accuracy, train_accuracy))
	idx = np.random.choice(trainSize, batchSize)
	trainingBatch = trainingSet[idx]
	answerBatch0 = trainingSetAnswer0[idx]
	answerBatch1 = trainingSetAnswer1[idx]
	trainStep.run(session = sess, feed_dict = {x0:trainingBatch, answer0: answerBatch0, answer1: answerBatch1})
stop = time.time()
print(stop - start, " s")
print("test accuracy %g" % accuracy.eval(session = sess, feed_dict = {x0:testingSet, answer0: testingSetAnswer0, answer1: testingSetAnswer1}))
if not os.path.exists("./weight"):
	os.makedirs("./weight")
np.save("weight/w0.npy", w0.eval(session = sess))
np.save("weight/w1.npy", w1.eval(session = sess))
np.save("weight/w2.npy", w2.eval(session = sess))
np.save("weight/w3.npy", w3.eval(session = sess))
np.save("weight/w4.npy", w4.eval(session = sess))
np.save("weight/w5.npy", w5.eval(session = sess))
np.save("weight/b0.npy", b0.eval(session = sess))
np.save("weight/b1.npy", b1.eval(session = sess))
np.save("weight/b2.npy", b2.eval(session = sess))
np.save("weight/b3.npy", b3.eval(session = sess))
np.save("weight/b4.npy", b4.eval(session = sess))
np.save("weight/b5.npy", b5.eval(session = sess))


sess.close()
np.save("weight/TestAccuracy.npy",acc_test)
np.save("weight/TrainAccuracy.npy",acc_train)
plt.plot(range(0, trainTimesAll, testStride), acc_test, 'r-', label='Test Accuracy')
plt.plot(range(0, trainTimesAll, testStride), acc_train, 'b-', label='Train Accuracy')
plt.legend(loc='lower right', prop={'size': 11})
plt.show()
