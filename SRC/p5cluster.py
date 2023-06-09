import p5dtw_path
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.patches as mpatches

def stacked_barplot_interaction(interaction_lists, ids, cluster_id):
    unique_labels = ['free', 'obj1', 'obj2','obj3', 'obj4','box1']
    color_map = {label: i for i, label in enumerate(unique_labels)}
    legend_patches = [mpatches.Patch(color=plt.cm.tab10(color_map[label]), label=label) for label in unique_labels]

    plt.figure(figsize=(25, 10))
    for j in range(len(interaction_lists)):
        sequence = interaction_lists[j]
        # Separate the string elements and float elements into separate lists
        labels = [item[0] for item in sequence]
        lengths = [item[1] for item in sequence]

        # Get unique string elements and assign colors based on the unique labels
        colors = [color_map[label] for label in labels]
        #map colors to rgb values
        colors = [plt.cm.tab10(color) for color in colors]

        plt.bar(j, lengths[0], color=colors[0],edgecolor='white')
        for i in range(1, len(lengths)):
            plt.bar(j, lengths[i], bottom=sum(lengths[:i]), color=colors[i],edgecolor='white' )
  
        plt.xticks(range(len(ids)), ids, rotation=90)
        plt.ylabel('Time (s)')
        plt.xlabel('Participant ID')
        plt.title(f'Interaction sequence for cluster {cluster_id}')


    plt.legend(handles=legend_patches)

# distance_matrix1,ids1 = p5dtw.main()
distance_matrix,ids,interaction_lists = p5dtw_path.main()
# # print(interaction_lists)



# distance_matrix = np.maximum(distance_matrix, distance_matrix.T)
# distance_matrix = distance_matrix / distance_matrix.max()
# distance_matrix = 1 - distance_matrix
# distance_matrix = np.triu(distance_matrix)
# distance_matrix = distance_matrix[~np.all(distance_matrix == 0, axis=1)]

Z = linkage(distance_matrix, method='ward', metric='euclidean')
# # print(Z)

num_clusters = 4
clusters = fcluster(Z, num_clusters, criterion='maxclust')
# # print(clusters)

cluster_ids = {}

# Iterate over the data points and assign them to their respective clusters
for i, cluster_id in enumerate(clusters):
    if cluster_id not in cluster_ids:
        cluster_ids[cluster_id] = []
    cluster_ids[cluster_id].append(ids[i])

# # Print the IDs in each cluster
# for cluster_id, data_ids in cluster_ids.items():
#     print(f"Cluster {cluster_id}: {data_ids}")



plt.figure(figsize=(25, 10))
dendrogram(Z,labels=ids)
plt.title('Hierarchical Clustering Dendrogram based on path')
plt.savefig('/home/erfan/Documents/Puzzle/puzzlesdata/Plots_Text/clustering/puzzle 5/Hierarchical Clustering Dendrogram based on path puzzle 5.png', dpi=300)
# plt.show()

#for each cluster plot the stacked bar plot
for cluster_id, data_ids in cluster_ids.items():
    # print(f"Cluster {cluster_id}: {data_ids}")
    #get the interaction lists for each cluster
    interaction_lists_cluster = []
    for data_id in data_ids:
        for i in range(len(ids)):
            if ids[i] == data_id:
                interaction_lists_cluster.append(interaction_lists[i])
                break
    #plot the stacked bar plot for each cluster
    stacked_barplot_interaction(interaction_lists_cluster, data_ids,cluster_id)
    plt.savefig(f'/home/erfan/Documents/Puzzle/puzzlesdata/Plots_Text/clustering/puzzle 5/Interaction sequence for cluster {cluster_id} puzzle 5.png', dpi=300)
    # plt.show()

        




