import random
import numpy as np
import time

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

x = np.array(x_train)
y = np.array(y_train)

weight = np.zeros((1,2))
bias = 0

start = time.time()
for iteration in range(k):
	j = 0
	dweight = np.zeros((1,2))
	dbias = 0

	z = np.dot(x,weight.T)+bias
	a = sigmoid(z)

	y = np.reshape(y,(m,1))
	dz = a-y
	dweight = np.zeros((2,1))
	x0 = np.reshape(x[:,0],(m,1))
	x1 = np.reshape(x[:,1],(m,1))
	dweight[0] += np.sum(x0*dz)
	dweight[1] += np.sum(x1*dz)
		
	dbias = np.sum(dz)/m

	j = np.sum(-(y*np.log(a+(1e-9))+(1-y)*np.log(1-a+(1e-9))))/m
	dweight /= m
	weight -= dweight.T*alpha
	bias -= alpha*dbias

	if iteration%50 == 0:
		print(weight, bias)

train_time = time.time()-start
print("Train Time: %.3f"%(train_time))
start = time.time()

#Calculate Accuracy of Training Set
train_correct = 0.
z = np.dot(x,weight.T)+bias
a = sigmoid(z)

for i in range(m):
	if round(a[i][0]) == y[i]: train_correct += 1
print("Accuracy with m train set: %.3f" %(train_correct * 100 / m))

#Calculate Accuracy of Test Set
x = np.array(x_test)
y = np.array(y_test)
test_correct = 0.
z = np.dot(x,weight.T)+bias
a = sigmoid(z)

for i in range(n):
	if round(a[i][0])==y[i]: test_correct += 1
print("Accuracy with n test set: %.3f"%(test_correct*100/n))

test_time = time.time()-start
print("Test Time: %.3f"%(test_time))
print("Task1 Done\n")

f = open("result.txt", 'a')
line = "task1\t%.3f\t%.3f\t%.3f\t%.3f\n"%(train_time, test_time, (train_correct * 100 / m), (test_correct * 100 / n))
f.write(line)
f.close()