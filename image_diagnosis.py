import numpy as np

# Extracted features from 3 medical images
X = np.array([
    [0.2, 0.5, 0.8, 0.3],
    [0.9, 0.4, 0.2, 0.7],
    [0.3, 0.8, 0.6, 0.5]
])

# Hidden layer weights and bias
W1 = np.random.rand(4, 3)
b1 = np.array([0.1, 0.2, 0.3])

# Output layer weights and bias
W2 = np.random.rand(3, 1)
b2 = np.array([0.1])

# Step 1: Matrix multiplication and bias addition
Z1 = np.dot(X, W1) + b1

# Step 2: Element-wise multiplication for ReLU
relu_mask = (Z1 > 0).astype(float)
A1 = Z1 * relu_mask

# Step 3: Output layer matrix multiplication
Z2 = np.dot(A1, W2) + b2

# Step 4: Sigmoid activation for classification
output = 1 / (1 + np.exp(-Z2))

print("Predicted Probabilities:\n", output)

# Classify: 1 = Abnormal, 0 = Normal
prediction = (output >= 0.5).astype(int)

print("Predicted Classes:\n", prediction)