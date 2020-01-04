"""
This file contains the different method necessary to perform the K-Means algorithm:
    - Computing the K-Means algortihm
    - Gently Visualizing Clusters
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy.spatial.distance as distance


def k_means_iteration(data, clusters, k=4):
    """
    Performing 1 iteration of the K-means algorithm
    """

    # computing the new means
    means = np.empty((0,2))
    for i in range(k):
        idx = np.where(clusters==i)[0]
        mean = np.mean(data[idx,:], axis=0)
        if(np.isnan(mean[0])):
            continue
        means = np.vstack((means, mean))

    # reassigning points to the clusters
    clusters = np.array([],dtype=np.int)
    for point in data:
        distances = np.linalg.norm(means-point, ord=2, axis=1)
        cluster = np.argmin(distances)
        clusters = np.append(clusters, cluster)

    return clusters, means


def k_means_clustering(data, k=4, iters=4, display=False):
    """
    Performing N iterations of the K-means clustering algorithm
    """

    if(display):
        fig, ax = plt.subplots(2,2)
        fig.set_size_inches(12, 10)

    # drawing random initial centers
    mean = np.mean(data)
    std = np.std(data)/2
    means = std*np.random.randn(k,2)+mean

    # reassigning points to the clusters
    previous_clusters = np.empty(0)
    clusters = np.array([],dtype=np.int)
    for point in data:
        distances = np.linalg.norm(means-point, ord=2, axis=1)
        cluster = np.argmin(distances)
        clusters = np.append(clusters, cluster)


    # computing iterations
    for i in range(iters):
        clusters, means = k_means_iteration(data=data, clusters=clusters, k=k)

        if(display):
            row = i//2
            col = i%2
            display_clusters(data=data, clusters=clusters, means=means, ax=ax[row,col])

        # checking for convergence
        if( len(previous_clusters) != 0 ):
            if( np.array_equal(previous_clusters, clusters) ):
                break

        previous_clusters = clusters

    return clusters, means



def display_clusters(data, clusters, means=[], ax=None, title="", xlabel="", ylabel="", legend=True):
    """
    Dispalying clusters with differnet colors for visualization purposes
    """

    k = np.max(clusters)+1

    colors = {
        "0":"bo",
        "1":"go",
        "2":"co",
        "3":"mo",
        "4":"yo",
        "5":"ko"
    }

    if(ax==None):
        fig,ax = plt.figure()

    # displaying each cluster with a different color
    for i in range(k):
        idx = np.where(clusters==i)[0]
        color = colors[str(i)]
        ax.plot(data[idx,0],data[idx,1], color, label=f"Cluster {i+1}")

    # displaying cluster centers if given
    if( len(means)>0 ):
        ax.plot(means[:,0],means[:,1], "ro", label="Cluster Centers" )

    if(legend):
        ax.legend(loc="best")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return



def within_cluster_distance(data, labels):
    """
    Computing the within clusted distance of all pairs of samples in all clusters
    """
    W = 0
    N = int(np.max(labels)+1)
    for i in range(N):
        clusterData = data[np.where(labels==i)]
        W += np.sum(
            np.square(distance.cdist(clusterData, clusterData, 'euclidean'))
            )

    return W



def createUniformSamples( numSamples, mean=0, variance=10 ):
    """
    Creating N uniformly distributed samples with certain mean and variance
    """

    offset = variance/2
    uniform_samples = variance*np.random.rand(numSamples,2)-offset+mean
    uniform_labels = np.ones( numSamples, dtype=np.int )

    return uniform_samples, uniform_labels



def compute_gap_statistics( data, initial_k=1, end_k=6, num_synthetic_distributions=20, display=False ):
    """
    Computing the optimal K for the k-means algorithm using the gap statistics, and given
    a range of possible Ks and the data
    """

    numSamples = data.shape[0]

    synthetic_distributions = []
    synthetic_labels = []

    # computing B synthetic distributions
    for i in range(num_synthetic_distributions):
        samples, labels = createUniformSamples( numSamples, mean=0, variance=10 )
        synthetic_distributions.append(samples)
        synthetic_labels.append(labels)

    real_data_distance = []
    mean_uniform_data_distance = []
    std_uniform_data_distance = []

    # computing the K-means clustering for all the given Ks
    for i in range(initial_k, end_k+1):

        # computing the distance for the real distribution
        clusters, means = k_means_clustering( data, k=i, iters=20)
        cluster_distance = within_cluster_distance(data, clusters)
        real_data_distance.append(cluster_distance)

        # computing the distance for the synthetic distributions
        uniform_data_distance = []
        for distribution in synthetic_distributions:
            clusters, means = k_means_clustering( distribution, k=i, iters=20)
            cluster_distance = within_cluster_distance(distribution, clusters)
            uniform_data_distance.append(cluster_distance)

        mean_distance = np.mean(np.log10(uniform_data_distance))
        std_distance = np.std(np.log10(uniform_data_distance))
        mean_uniform_data_distance.append(mean_distance)
        std_uniform_data_distance.append(std_distance)


    # coputing the gap statistics
    real_data_distance = np.log10(real_data_distance/real_data_distance[0])
    mean_uniform_data_distance = mean_uniform_data_distance - mean_uniform_data_distance[0]
    gap = mean_uniform_data_distance - real_data_distance

    ks = np.arange(initial_k,end_k+1)
    optimum_k = 0
    for i in range(0,len(gap)-1):
        if( gap[i] >= gap[i+1] - std_uniform_data_distance[i+1] ):
            optimum_k = i + initial_k
            break

    if(display):
        plt.figure(figsize=(6,6))
        plt.errorbar(ks, gap, std_uniform_data_distance)
        plt.xlabel("# clusters")
        plt.ylabel("Gap statistics")
        plt.title(f"Optimum K = {optimum_k}")
        plt.tight_layout()
        plt.show()

    return optimum_k

#
