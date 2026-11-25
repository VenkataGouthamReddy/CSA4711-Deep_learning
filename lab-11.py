import tensorflow as tf
import matplotlib.pyplot as plt

(x,y),(xt,yt) = tf.keras.datasets.mnist.load_data()

x = x.reshape(-1,28,28,1)/255.0
xt = xt.reshape(-1,28,28,1)/255.0

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,3,activation='relu',
                          input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x,y,epochs=3)
print("Accuracy:",model.evaluate(xt,yt)[1])

plt.imshow(xt[0].reshape(28,28),cmap='gray')
plt.show()