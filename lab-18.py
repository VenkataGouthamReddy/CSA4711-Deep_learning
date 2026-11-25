import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

data = np.sin(np.arange(100))
X = np.array([data[i:i+5] for i in range(95)])
y = data[5:]
X = X.reshape(95,5,1)

for name in ['RNN','LSTM','GRU']:
    model = tf.keras.Sequential()

    if name == 'RNN':
        model.add(tf.keras.layers.SimpleRNN(20,input_shape=(5,1)))
    elif name == 'LSTM':
        model.add(tf.keras.layers.LSTM(20,input_shape=(5,1)))
    else:
        model.add(tf.keras.layers.GRU(20,input_shape=(5,1)))

    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer='adam',loss='mse')
    model.fit(X,y,epochs=5,verbose=0)

    print(name,"Loss:",model.evaluate(X,y,verbose=0))