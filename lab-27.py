import numpy as np
from sklearn.neural_network import BernoulliRBM

X = np.random.randint(0,2,(100,10))

rbm1 = BernoulliRBM(n_components=8,n_iter=10)
rbm2 = BernoulliRBM(n_components=4,n_iter=10)

h1 = rbm1.fit_transform(X)
h2 = rbm2.fit_transform(h1)

print("Original Data:\n",X[:5])
print("First Layer:\n",h1[:5])
print("Second Layer:\n",h2[:5])