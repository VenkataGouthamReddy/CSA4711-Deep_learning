import tensorflow as tf
import matplotlib.pyplot as plt

# Load MNIST dataset
(x,y),_ = tf.keras.datasets.mnist.load_data()
x = x[:5000].astype('float32')/255.0
x = x.reshape(-1,784)

# Simple VAE
encoder = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(784,)),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(16)
])

decoder = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(16,)),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(784,activation='sigmoid')
])

vae = tf.keras.Sequential([encoder,decoder])
vae.compile(optimizer='adam',loss='binary_crossentropy')
vae.fit(x,x,epochs=3,verbose=0)

# Generate VAE output
z = tf.random.normal((1,16))
vae_image = decoder(z).numpy().reshape(28,28)

# Simple GAN
generator = tf.keras.Sequential([
    tf.keras.layers.Dense(128,activation='relu',
                          input_shape=(100,)),
    tf.keras.layers.Dense(784,activation='sigmoid')
])

discriminator = tf.keras.Sequential([
    tf.keras.layers.Dense(128,activation='relu',
                          input_shape=(784,)),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

discriminator.compile(optimizer='adam',
                      loss='binary_crossentropy')

discriminator.trainable = False

gan = tf.keras.Sequential([generator,discriminator])
gan.compile(optimizer='adam',loss='binary_crossentropy')

# Train GAN
for i in range(100):
    noise = tf.random.normal((32,100))
    fake = generator.predict(noise,verbose=0)

    real = x[:32]

    discriminator.trainable = True
    discriminator.train_on_batch(
        tf.concat([real,fake],axis=0),
        tf.concat([tf.ones((32,1)),tf.zeros((32,1))],axis=0)
    )

    discriminator.trainable = False
    gan.train_on_batch(noise,tf.ones((32,1)))

# Generate GAN image
noise = tf.random.normal((1,100))
gan_image = generator.predict(noise,verbose=0).reshape(28,28)

# Display images
plt.figure(figsize=(6,3))

plt.subplot(1,2,1)
plt.imshow(vae_image,cmap='gray')
plt.title("VAE")

plt.subplot(1,2,2)
plt.imshow(gan_image,cmap='gray')
plt.title("GAN")

plt.show()