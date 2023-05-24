import p2dtw
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

distance_matrix,ids = p2dtw.main()
# make the distance matrix symmetric and non-negative and condense it
distance_matrix = np.maximum(distance_matrix, distance_matrix.T)
distance_matrix = distance_matrix / distance_matrix.max()
distance_matrix = 1 - distance_matrix
distance_matrix = np.triu(distance_matrix)
distance_matrix = distance_matrix[~np.all(distance_matrix == 0, axis=1)]
# print(distance_matrix)

Z = linkage(distance_matrix, method='ward')  # 'complete' linkage method
# print(Z)





dendrogram(Z,labels=ids)
plt.show()

# k = 2
# cutree = dendrogram(Z, truncate_mode='lastp', p=k, show_leaf_counts=True)

# # Retrieve the cluster assignments
# cluster_assignments = cutree['leaves']

# # Print the cluster assignments
# print("Cluster assignments:", cluster_assignments)