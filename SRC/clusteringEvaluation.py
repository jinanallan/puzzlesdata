import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.spatial.distance import  squareform
from scipy.cluster.hierarchy import fcluster

def clusteringEvaluation(Z,distanceMat, puzzleNumber):
    """
    This function calculates the evaluation metrics for the clustering results
    :param Z: linkage matrix
    :param distanceMat: distance matrix
    :param puzzleNumber: the number of the puzzle
    :return: a figure with the evaluation metrics
    """

    distanceMatSQ = squareform(distanceMat)
    silhouette_avg = []
    dbi = []
    chs = []

    for n_clusters in range(2, 11):
        # hierarchical clustering
        # cut the dendrogram at n_clusters
        cluster_labels = fcluster(Z, n_clusters, criterion='maxclust')
        # calculate evaluation metrics
        silhouette_avg.append(silhouette_score(distanceMatSQ, cluster_labels, metric='precomputed'))
        dbi.append(davies_bouldin_score(distanceMatSQ, cluster_labels))
        chs.append(calinski_harabasz_score(distanceMatSQ, cluster_labels))

    fig = plt.figure(figsize=(15, 5))
    plt.rcParams.update({'font.size': 14})
    plt.suptitle('Evaluation metrics for puzzle ' + str(puzzleNumber))
    plt.subplot(1, 3, 1)
    plt.plot(range(2, 11), silhouette_avg)
    # highlight the maximum silhouette score
    ymax = max(silhouette_avg)
    xpos = silhouette_avg.index(ymax) + 2
    plt.plot(xpos, ymax, 'o', color='red')
    plt.xlabel('Number of clusters')
    plt.xticks(range(2, 11))
    plt.ylabel('Silhouette score')
    plt.subplot(1, 3, 2)
    plt.plot(range(2, 11), dbi)
    # highlight the minimum DBI
    ymin = min(dbi)
    xpos = dbi.index(ymin) + 2
    plt.plot(xpos, ymin, 'o', color='red')
    plt.xlabel('Number of clusters')
    plt.xticks(range(2, 11))
    plt.ylabel('Davies-Bouldin index')
    plt.subplot(1, 3, 3)
    plt.plot(range(2, 11), chs)
    # highlight the maximum Calinski-Harabasz score
    ymax = max(chs)
    xpos = chs.index(ymax) + 2
    plt.plot(xpos, ymax, 'o', color='red')
    plt.xlabel('Number of clusters')
    plt.xticks(range(2, 11))
    plt.ylabel('Calinski-Harabasz score')
    plt.tight_layout()
    return fig





