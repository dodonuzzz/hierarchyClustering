from sklearn.cluster import AgglomerativeClustering
import numpy as np

x = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
clustering = AgglomerativeClustering(n_clusters=2).fit(x)

print(clustering.labels_)
