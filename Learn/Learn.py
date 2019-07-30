import matplotlib.pyplot as plt
import numpy as np

acc_test = np.load("TestAccuracy.npy")
acc_train = np.load("TrainAccuracy.npy")
plt.figure(figsize=(9, 5))
plt.plot(range(0, 100000, 500), acc_test, 'r-', label='Test Accuracy')
plt.plot(range(0, 100000, 500), acc_train, 'b-', label='Train Accuracy')
plt.xlabel('Number of trainings')
plt.ylabel('Accuracy')
plt.legend(loc='lower right', prop={'size': 11})
plt.title('Deep learning gesture recognition accuracy relationship with number of trainings',fontsize=12,color='black')
#plt.show()
plt.savefig('Figure.png', dpi=300)