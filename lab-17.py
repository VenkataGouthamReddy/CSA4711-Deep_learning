import tensorflow as tf
import numpy as np

X = np.random.rand(100,5,1)
y = np.random.rand(100)

model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(20,input_shape=(5,1)),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(clipnorm=1.0),
    loss='mse'
)

model.fit(X,y,epochs=5)
print("Gradient clipping completed")