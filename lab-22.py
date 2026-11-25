import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

np.random.seed(1)

t = np.linspace(0,8,500)

s1 = np.sin(2*t)
s2 = np.sign(np.sin(3*t))

S = np.c_[s1,s2]
X = S @ np.random.rand(2,2)

ica = FastICA(n_components=2)
result = ica.fit_transform(X)

plt.plot(result)
plt.title("Separated Signals")
plt.show()

print("Separated Signals:\n",result[:5])