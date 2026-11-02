import numpy as np

# 100 students and 3 features
# Columns: attendance, assignment score, internal marks
X = np.random.randint(40, 101, size=(100, 3))

# Weight matrix and bias
W = np.array([
    [0.4],
    [0.3],
    [0.3]
])

b = np.array([5])

# Calculate predicted performance
predictions = np.dot(X, W) + b

print("Feature Matrix Shape:", X.shape)
print("Weight Matrix Shape:", W.shape)
print("Predicted Outputs Shape:", predictions.shape)
print("First 5 Predictions:\n", predictions[:5])