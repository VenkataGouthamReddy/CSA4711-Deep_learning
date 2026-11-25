import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()

for i in range(3):
    G.add_edge("Input"+str(i),"Hidden"+str(i))
    G.add_edge("Hidden"+str(i),"Output"+str(i))

    if i > 0:
        G.add_edge("Hidden"+str(i-1),"Hidden"+str(i))

nx.draw(G,with_labels=True,node_color='lightblue',
        node_size=2000)
plt.show()