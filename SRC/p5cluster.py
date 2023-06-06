
import p5dtw_path
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# distance_matrix1,ids1 = p5dtw.main()
distance_matrix,ids,interaction_lists = p5dtw_path.main()


# distance_matrix = np.maximum(distance_matrix, distance_matrix.T)
# distance_matrix = distance_matrix / distance_matrix.max()
# distance_matrix = 1 - distance_matrix
# distance_matrix = np.triu(distance_matrix)
# distance_matrix = distance_matrix[~np.all(distance_matrix == 0, axis=1)]

Z = linkage(distance_matrix, method='median') 
print(Z)


plt.figure(figsize=(25, 10))
dendrogram(Z,labels=ids)
plt.title('Hierarchical Clustering Dendrogram based on path')
plt.savefig('/home/erfan/Documents/Puzzle/puzzlesdata/Plots_Text/Hierarchical Clustering Dendrogram based on path puzzle 5.png', dpi=300)
plt.show()

#print the index of the clusters   









