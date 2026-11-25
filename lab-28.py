import numpy as np
import matplotlib.pyplot as plt

X = np.random.randint(0,2,(10,6))

W1 = np.random.rand(6,4)
W2 = np.random.rand(4,2)

h1 = 1/(1+np.exp(-X @ W1))
h2 = 1/(1+np.exp(-h1 @ W2))

reconstructed = 1/(1+np.exp(-h2 @ W2.T @ W1.T))

print("Original:\n",X)
print("Hidden Layer 1:\n",h1)
print("Hidden Layer 2:\n",h2)

plt.imshow(reconstructed,cmap='gray')
plt.title("Reconstructed Data")
plt.show()