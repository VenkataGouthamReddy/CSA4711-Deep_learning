import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

data = np.sin(np.arange(100))

X = np.array([data[i:i+5] for i in range(95)])
y = data[5:]
X = X.reshape(95,5,1)

for layer in ['RNN','LSTM']:
    model = tf.keras.Sequential()

    if layer == 'RNN':
        model.add(tf.keras.layers.SimpleRNN(20,input_shape=(5,1)))
    else:
        model.add(tf.keras.layers.LSTM(20,input_shape=(5,1)))

    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer='adam',loss='mse')
    model.fit(X,y,epochs=5,verbose=0)

    pred = model.predict(X,verbose=0)
    print(layer,"Error:",np.mean((y-pred.flatten())**2))

    plt.plot(y,label='Actual')
    plt.plot(pred,label=layer)
    plt.legend()
    plt.show()