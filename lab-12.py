import tensorflow as tf

(x,y),(xt,yt) = tf.keras.datasets.mnist.load_data()

x = x.reshape(-1,28,28,1)/255.0
xt = xt.reshape(-1,28,28,1)/255.0

for f in [16,32,64]:
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(f,3,activation='relu',
                               input_shape=(28,28,1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10,activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x,y,epochs=2,verbose=0)
    print("Filters:",f)
    print("Accuracy:",model.evaluate(xt,yt,verbose=0)[1])