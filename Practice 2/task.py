import random
import numpy as np
import time
from glob import glob

m = 10000
n = 500

# return x_train, y_train, x_test, y_test
# get those of set, use np.array to vectorize
def make_train_test_set():
    global m,n,k,alpha
    #Make Training Set
    x_train = []
    x1_train = []
    x2_train = []
    y_train = []

    for i in range(m):
        x1_train.append(random.uniform(-10,10))
        x2_train.append(random.uniform(-10,10))
        x_train.append(np.array([x1_train[-1],x2_train[-1]]))
        if x1_train[-1]+x2_train[-1]>0:
            y_train.append(1)
        else: y_train.append(0)

    #Make Test Set
    x_test = []
    x1_test = []
    x2_test = []
    y_test = []

    for i in range(n):
        x1_test.append(random.uniform(-10,10))
        x2_test.append(random.uniform(-10,10))
        x_test.append(np.array([x1_test[-1],x2_test[-1]]))
        if x1_test[-1]+x2_test[-1]>0:
            y_test.append(1)
        else: y_test.append(0)

    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

x_train, y_train, x_test, y_test = make_train_test_set()

f = open("train.txt", 'w')

for i in range(m):
	line = "%f\t%f\t%f\n" %(x_train[i][0], x_train[i][1], y_train[i])
	f.write(line)

f.close()

f = open("test.txt", 'w')

for i in range(n):
	line = "%f\t%f\t%f\n" %(x_test[i][0], x_test[i][1], y_test[i])
	f.write(line)

f.close()

f = open("result.txt", 'w')

line = "\tTrain_Time\tTest_Time\tTrain_Accuracy\tTest_Accuracy\n"
f.write(line)

f.close()

file_list = glob("*.py")

for file in file_list:
    if file == "test.py" or file == "task.py": continue
    exec(open(file).read())