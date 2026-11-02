import numpy as np

# Grayscale image matrix
image = np.array([
    [10, 20],
    [30, 40]
])

# Step 1: Flatten the image into a vector
x = image.flatten()

# Step 2: Weight matrix and bias vector
W = np.array([
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9],
    [1.0, 1.1, 1.2]
])

b = np.array([1, 2, 3])

# Step 3: Fully connected layer
output = np.dot(x, W) + b

print("Flattened Image:", x)
print("Output:", output)