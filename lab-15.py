import numpy as np
import matplotlib.pyplot as plt

x = [1,2,3,4,5]
w = 0.5
h = 0
outputs = []

for i in x:
    h = np.tanh(w*i+h)
    outputs.append(h)

print("Outputs:",outputs)

gradients = []
for i in range(1,len(outputs)):
    gradients.append(1-outputs[i]**2)

print("Gradients:",gradients)

plt.plot(gradients,marker='o')
plt.xlabel("Time Steps")
plt.ylabel("Gradient")
plt.show()