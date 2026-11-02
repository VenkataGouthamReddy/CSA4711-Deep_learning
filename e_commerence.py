import numpy as np

# 500 products, each with 20 features
products = np.random.rand(500, 20)

# Customer preference vector
customer = np.random.rand(20)

# Calculate similarity scores
scores = np.dot(products, customer)

# Select the top 5 products
top_products = np.argsort(scores)[-5:][::-1]

print("Similarity Scores Shape:", scores.shape)
print("Recommended Product IDs:", top_products + 1)
print("Top 5 Similarity Scores:", scores[top_products])