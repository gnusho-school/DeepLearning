import random
import numpy as np
import time

m=1000
n=100
k=2000
alpha=0.01

alpha_list=[0.0001,0.001,0.01,0.1,1,10]

def sigmoid(x):

	return 1/(1+np.exp(-x))

#Make Training Set
x_train=[]
x1_train=[]
x2_train=[]
y_train=[]

for i in range(m):
	x1_train.append(random.uniform(-10,10))
	x2_train.append(random.uniform(-10,10))
	x_train.append(np.array([x1_train[-1],x2_train[-1]]))
	if x1_train[-1]+x2_train[-1]>0:
		y_train.append(1)
	else: y_train.append(0)

#Make Test Set
x_test=[]
x1_test=[]
x2_test=[]
y_test=[]

for i in range(n):
	x1_test.append(random.uniform(-10,10))
	x2_test.append(random.uniform(-10,10))
	x_test.append(np.array([x1_test[-1],x2_test[-1]]))
	if x1_test[-1]+x2_test[-1]>0:
		y_test.append(1)
	else: y_test.append(0)

def vectorized(alpha):
#Let's Learning
	#alpha=alpha
	print(alpha)
	x=np.array(x_train)
	y=np.array(y_train)

	weight=np.zeros((1,2))
	bias=0
	minimum_cost=1
	start=time.time()
	for iteration in range(k):
		j=0
		dweight=np.zeros((1,2))
		dbias=0

		z=np.dot(x,weight.T)+bias

		#for i in range(m):
		#	if z[i]>700:z[i]=500

		a=sigmoid(z)

		#j=np.zeros()
		y=np.reshape(y,(m,1))
		dz=a-y
		#print(x.shape,dz.shape)
		#print((x[:,0]*dz).shape,x[:,0].shape,dz.shape)
		dweight=np.zeros((2,1))
		x0=np.reshape(x[:,0],(m,1))
		x1=np.reshape(x[:,1],(m,1))
		dweight[0]+=np.sum(x0*dz)
		dweight[1]+=np.sum(x1*dz)
		
		dbias=np.sum(dz)/m

		j=np.sum(-(y*np.log(a+1e-9)+(1-y)*np.log(1-a+1e-9)))/m
		dweight/=m
		#dbias/=m
		#print(dweight.shape,weight.shape)
		weight-=dweight.T*alpha
		bias-=alpha*dbias

		if iteration%10==0:
		#	print(weight,bias)
			if minimum_cost>j:minimum_cost=j	

	print("minimum_cost:%.5f"%minimum_cost)
	print("weight=[%.5f, %.5f]" %((weight.T)[0],(weight.T)[1]))
	print("bias=%.5f"%bias)
	#Calculate Accuracy of Training Set
	train_correct=0.
	z=np.dot(x,weight.T)+bias
	a=sigmoid(z)
	for i in range(m):
		if round(a[i])==y[i]: train_correct+=1
	#print(train_correct)
	print("Accuracy with m train set: %.1f"%(train_correct*100/m))

	#Calculate Accuracy of Test Set
	x=np.array(x_test)
	y=np.array(y_test)
	test_correct=0.
	z=np.dot(x,weight.T)+bias
	a=sigmoid(z)
	for i in range(n):
		if round(a[i])==y[i]: test_correct+=1
	#print(test_correct)
	print("Accuracy with n test set: %.1f"%(test_correct*100/n))
	print("VectorizedTime: %.3f"%(time.time()-start))

def main():
	for i in alpha_list: 
		vectorized(i)

if __name__ == '__main__':
	main()













