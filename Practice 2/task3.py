import random
import numpy as np
import time

# generating training data and test data
m = 10000
n = 500
k = 5000
alpha = 0.5

def sigmoid(x):
	return 1/(1+np.exp(-x))

def sigmoid_diff(x):
    return sigmoid(x) * (1 - sigmoid(x))

# return x_train, y_train, x_test, y_test
# get those of set, use np.array to vectorize
def make_train_test_set():
    
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    f = open("train.txt", 'r')

    while True:
        line = f.readline()
        if not line: break
        array = line.strip().split("\t")
        x_train.append([float(array[0]), float(array[1])])
        y_train.append(float(array[2]))

    f.close()

    f = open("test.txt", 'r')

    while True:
        line = f.readline()
        if not line: break
        array = line.strip().split("\t")
        x_test.append([float(array[0]), float(array[1])])
        y_test.append(float(array[2]))

    f.close()

    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

x_train, y_train, x_test, y_test = make_train_test_set()

start = time.time()

W1 = np.zeros((3,2)) + random.uniform(0,1)
B1 = np.zeros((3,1)) + random.uniform(0,1)
W2 = np.zeros((1,3)) + random.uniform(0,1)
B2 = random.uniform(0,1)

for i in range(k):
    Z1 = np.dot(W1, x_train.T) + B1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + B2
    A2 = sigmoid(Z2)

    Y_train = y_train.reshape(10000,1).T
    
    dZ2 = A2 - Y_train
    dW2 = np.dot(dZ2, A1.T) / m
    dB2 = np.sum(dZ2, axis = 1, keepdims = True) / m
    dZ1 = np.dot(W2.T, dZ2) * sigmoid_diff( Z1 )
    dW1 = np.dot(dZ1, x_train) / m
    dB1 = np.sum(dZ1, axis = 1, keepdims = True) / m

    W1 -= dW1 * alpha
    W2 -= dW2 * alpha
    B1 -= dB1 * alpha
    B2 -= dB2 * alpha

    if i % 50 == 0:
        print(W1, B1, W2, B2)

train_time = time.time()-start
print("Train Time: %.3f"%(train_time))
start = time.time()

train_correct = 0.
Z1 = np.dot(W1, x_train.T) + B1
A1 = sigmoid(Z1)
Z2 = np.dot(W2, A1) + B2
A2 = sigmoid(Z2)

for i in range(m):
	if round(A2[0][i]) == y_train[i]: train_correct += 1
print("Accuracy with m train set: %.3f" %(train_correct * 100 / m))

test_correct = 0.
Z1 = np.dot(W1, x_test.T) + B1
A1 = sigmoid(Z1)
Z2 = np.dot(W2, A1) + B2
A2 = sigmoid(Z2)

for i in range(n):
	if round(A2[0][i]) == y_test[i]: test_correct += 1
print("Accuracy with n test set: %.3f" %(test_correct * 100 / n))

test_time = time.time()-start
print("Test Time: %.3f"%(test_time))
print("Task3 Done\n")

f = open("result.txt", 'a')
line = "task3\t%.3f\t%.3f\t%.3f\t%.3f\n"%(train_time, test_time, (train_correct * 100 / m), (test_correct * 100 / n))
f.write(line)
f.close()