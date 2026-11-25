import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()

G.add_edges_from([
    ("Disease","Fever"),
    ("Disease","Cough"),
    ("Disease","Fatigue")
])

samples = np.random.randint(0,2,100)

print("Disease Samples:",samples)
print("Disease Probability:",np.mean(samples))

nx.draw(G,with_labels=True,node_color='lightgreen',
        node_size=2000)
plt.show()