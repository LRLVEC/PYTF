import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

inputBatch = tf.placeholder(tf.float32, shape=[None, 65536])
answer = tf.placeholder(tf.float32,shape=[None,5])


