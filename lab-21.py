import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FactorAnalysis

X = np.random.rand(100,5)

model = FactorAnalysis(n_components=2)
result = model.fit_transform(X)

print("Factors:\n",result)
print("Loadings:\n",model.components_)

plt.scatter(result[:,0],result[:,1])
plt.xlabel("Factor 1")
plt.ylabel("Factor 2")
plt.show()