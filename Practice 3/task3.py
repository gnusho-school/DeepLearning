import tensorflow as tf
import random
import numpy as np
import time

m = 10000
n = 500
k = 1000

x_train = np.array([[random.uniform(-10,10),random.uniform(-10,10)] for i in range(m)])
y_train = np.array([1 if x_train[i][0] + x_train[i][1] > 0 else 0 for i in range(m)])
x_test = np.array([[random.uniform(-10,10), random.uniform(-10,10)] for i in range(n)])
y_test = np.array([1 if x_test[i][0] + x_test[i][1] > 0 else 0 for i in range(n)])

#print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
#shape: (1000,2) (1000,) (500,2) (500,)

start = time.time()

model = tf.keras.models.Sequential([
  #tf.keras.layers.Flatten(input_shape=(2,)),
  tf.keras.layers.Dense(3, activation='sigmoid'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='SGD',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=k, batch_size = 32)
model.evaluate(x_train,y_train)
model.evaluate(x_test,y_test)
print("Total time: %f"%(time.time() - start))