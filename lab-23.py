import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()

G.add_edges_from([
    ("Cloudy","Rain"),
    ("Humidity","Rain"),
    ("Rain","Umbrella")
])

nx.draw(G,with_labels=True,node_color='lightblue',
        node_size=2000,arrows=True)
plt.show()

p_rain = 0.7
p_no_rain = 1-p_rain

print("Probability of Rain:",p_rain)
print("Probability of No Rain:",p_no_rain)