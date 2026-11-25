import numpy as np
from sklearn.neural_network import BernoulliRBM

X = np.array([
    [1,0,1,0],
    [0,1,0,1],
    [1,1,0,0],
    [0,0,1,1]
])

model = BernoulliRBM(n_components=2,
                     learning_rate=0.1,
                     n_iter=100,
                     random_state=1)

model.fit(X)

result = model.transform(X)

print("Original Data:\n",X)
print("Hidden Features:\n",result)