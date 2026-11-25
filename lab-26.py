import numpy as np

np.random.seed(1)

W = np.random.rand(4,2)
v = np.array([[1,0,1,0]])

h_prob = 1/(1+np.exp(-v @ W))
h = (h_prob > 0.5).astype(int)

v_prob = 1/(1+np.exp(-h @ W.T))
v_new = (v_prob > 0.5).astype(int)

print("Original:",v)
print("Hidden:",h)
print("Reconstructed:",v_new)