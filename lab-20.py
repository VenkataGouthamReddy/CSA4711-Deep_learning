import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

X = np.random.rand(100,10)

pca = PCA(n_components=2)
reduced = pca.fit_transform(X)

reconstructed = pca.inverse_transform(reduced)

error = np.mean((X-reconstructed)**2)

print("Reduced Data:\n",reduced)
print("Reconstruction Error:",error)

plt.plot(X[0],label="Original")
plt.plot(reconstructed[0],label="Reconstructed")
plt.legend()
plt.show()