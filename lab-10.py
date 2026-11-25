import tensorflow as tf

(x,y),(xt,yt) = tf.keras.datasets.mnist.load_data()

x = x/255.0
xt = xt/255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

stop = tf.keras.callbacks.EarlyStopping(patience=2)

model.fit(x,y,epochs=10,validation_split=0.2,
          callbacks=[stop])

print("Accuracy:", model.evaluate(xt,yt)[1])