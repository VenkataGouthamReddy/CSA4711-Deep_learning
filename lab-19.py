import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X = np.random.rand(100,4)

X = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
Y = pca.fit_transform(X)

print("Reduced Data:\n",Y)
print("Eigenvalues:",pca.explained_variance_)
print("Explained Variance:",pca.explained_variance_ratio_)

plt.scatter(Y[:,0],Y[:,1])
plt.show()