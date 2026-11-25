import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

data = np.sin(np.arange(100))

X = np.array([data[i:i+5] for i in range(95)])
y = data[5:]

X = X.reshape(95,5,1)

model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(20,input_shape=(5,1)),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam',loss='mse')
model.fit(X,y,epochs=10,verbose=0)

pred = model.predict(X)

plt.plot(y,label='Actual')
plt.plot(pred,label='Predicted')
plt.legend()
plt.show()