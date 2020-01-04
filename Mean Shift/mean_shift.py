"""
This file contains the different method necessary to perform the Mean-shift algorithm:
    - Computing distance
    - Finding the neighboring points
    - Evaluating different kernels
    - Computing the Mean-Shift algortihm
"""

import numpy as np
from matplotlib import pyplot as plt
import numba
import cv2
import sys


def get_distance(x1, x2 ):
    """
    Computing the euclidean distance between two feature vectors
    """

    distance = np.linalg.norm(x1-x2, ord=2)
    return distance


def is_point_in_neighborhood(center, y, thr=1):
    """
    checking if two points are neighbors, that means that the distance is smaller than a threshold
    """

    distance = get_distance(center, y)
    if( distance < thr ):
        return True
    else:
        return False



def epanechnikov_kernel(x):
    """
    evaluating the epanechnikov kernel on a feature vector
    """

    value = (1-np.dot(x,x))
    return value



def gaussian_kernel(points, center, sigma=1, thr=1):
    """
    evaluating  a gaussian kernel on a feature vector
    """

    distance = get_matrix_distance(points, center)
    value = (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((distance / sigma))**2)

    return value



def get_matrix_distance(points, center):
    """
    Method that obtains the efficient matrix distance
    """

    n_points = len(points)

    matrix_point = np.tile( center, (n_points,1) )
    distance = np.linalg.norm( matrix_point-points, ord=2, axis=1 )

    return distance



def get_neighbors( center, points, thr=3 ):
    """
    algorithm that obtains the nieghbors in a vectorized and efficeint manner
    """

    distance = get_matrix_distance(points, center)
    idx = np.where( distance<=thr )[0]
    neighbors = points[idx,:]

    return neighbors



def iterate_data( data, kernel="epanechnikov", thr=3, sigma=1 ):
    """
    computing one iteration of the mean-shift algorithm using the specified constants and kernel
    """
    new_data = np.copy(data)

    for i,data_point in enumerate(data):

        # obtaining neighbors
        neighbors = get_neighbors( data_point, data, thr=thr )
        n_neighbors = len(neighbors)

        # computing the mean of the points
        # case for the epanechnikov (most efficient)
        if( kernel=="epanechnikov" ):
            updated_point = np.sum(neighbors,axis=0)/n_neighbors
        # case for other kernels
        else:
            value = gaussian_kernel(neighbors, data_point, sigma=sigma, thr=thr)

            #numerator = np.sum( np.multiply(value[:,np.newaxis],neighbors) , axis=0)
            numerator = np.dot( value[np.newaxis,:] , neighbors)
            denominator = np.sum(value, axis=0)
            # updating the position
            updated_point = numerator/denominator


        new_data[i] = updated_point

    return new_data



def mean_shift(data, num_iterations=3, kernel="epanechnikov", sigma=1, thr=3):
    """
    computing the mean-shift algorithm for a given number of iterations
    """

    centers = np.copy(data)

    # iterating over the data the given number of times
    for i in range(num_iterations):
        centers = iterate_data( centers, kernel=kernel, thr=thr, sigma=sigma )

    return centers, data



def create_feature_vectors( img, color_space="RGB" ):
    """
    Creating feature vectors (X,Y,R,G,B) or (X,Y,L,U,V)
    """

    xaxis = np.arange( img.shape[1] )
    yaxis = np.arange( img.shape[0] )
    x,y = np.meshgrid(xaxis,yaxis)
    x = x.flatten()
    y = y.flatten()

    if( color_space=="LUV" ):
        img = rgb_to_luv(img)

    r_vector = img[:,:,0].flatten()
    g_vector = img[:,:,1].flatten()
    b_vector = img[:,:,2].flatten()

    n_vectors = len(x)
    feature_vectors = np.zeros((n_vectors,5))

    feature_vectors[:,0] = x
    feature_vectors[:,1] = y
    feature_vectors[:,2] = r_vector
    feature_vectors[:,3] = g_vector
    feature_vectors[:,4] = b_vector

    return feature_vectors


def normalize_feature_vectors( feature_vectors, weights=[] ):
    """
    Normalizing the feature vectors to have unit zero mean and unit variance to be able
    to be compared in the same scale
    """
    if( len(weights)==0 ):
        weights = np.ones(feature_vectors[0].shape)

    means = []
    variances = []
    normalized_feature_vectors = np.copy(feature_vectors)

    for i in range( feature_vectors.shape[1] ):

        mean = np.mean( feature_vectors[:,i] )
        variance = np.var( feature_vectors[:,i] )

        eps = 1e-8
        normalized_feature_vectors[:,i] -= mean
        normalized_feature_vectors[:,i] /= (variance+eps)
        normalized_feature_vectors[:,i] *= weights[i]

        means.append(mean)
        variances.append(variance)

    return normalized_feature_vectors, means, variances



def mean_shift_denoising( img, weights=[], num_iterations=5, thr=0.05, kernel="gaussian", sigma=1 ):
    """
    method that performs denoising of an image using the mean shift algorithm
    """

    # obtaining the feature vectors and normalizing the dimensions
    print("Obtaining feature vectors")
    feature_vectors = create_feature_vectors( img, color_space="LUV" )

    # this weights force the clustering of nearby points
    print("Normalizing feature vectors")
    normalized_feature_vectors, _, _ = normalize_feature_vectors( feature_vectors, weights=weights )

    print("Computing mean shift algorithm")
    points,_ = mean_shift(normalized_feature_vectors, num_iterations=num_iterations, thr=thr,
                          kernel=kernel, sigma=sigma)

    # assigning poitns to corresponding cluster
    clusters = assign_points_to_clusters(points, feature_vectors)

    # avergaing the color of points that belong to the same cluster
    denoised_image = average_colors(img, clusters, color_space="LUV")

    return denoised_image


def rgb_to_luv( rgb ):
    """
    Converting from RGB to LUV
    """
    luv = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2Luv)
    return luv


def luv_to_rgb( luv ):
    """
    Converting from LUV to RGB
    """
    rgb = cv2.cvtColor(luv, cv2.COLOR_Luv2RGB)
    return rgb



def average_colors(img, clusters, color_space="RGB"):
    """
    method that averages the color of the points that belong to the same cluster
    """

    keys = list(clusters.keys())
    denoised_image = np.zeros(img.shape, dtype=np.uint8)

    # averaging colors for every cluster and assigning
    for i,key in enumerate(keys):

        points_in_current_cluster = clusters[key]

        y = np.array(points_in_current_cluster[:,0], dtype=np.int)
        x = np.array(points_in_current_cluster[:,1], dtype=np.int)
        r_mean = int(np.round(np.mean(points_in_current_cluster[:,2])))
        g_mean = int(np.round(np.mean(points_in_current_cluster[:,3])))
        b_mean = int(np.round(np.mean(points_in_current_cluster[:,4])))

        denoised_image[x,y,0] = r_mean
        denoised_image[x,y,1] = g_mean
        denoised_image[x,y,2] = b_mean

    # converting back to the RGB color space
    if( color_space=="LUV" ):
        denoised_image = luv_to_rgb(denoised_image)

    return denoised_image



def assign_points_to_clusters(points, feature_vectors):
    """
    method that assigns poitns to a clusters
    clusters = {
        center_1_l2: [p1,p2,...,pN],
        center_2_l2: [p1,p2,...,pN],
        ...
        center_M_l2: [p1,p2,...,pN]
    }
    """

    thr = 1e-10
    clusters = {}

    # grouping all points that belong to the same cluster
    # we use a dictionary as data structure
    for i, point in enumerate(points):

        # obtaining the already found centers
        keys = list(clusters.keys())
        centers = np.array(keys,dtype=np.float)

        l2_norm = np.linalg.norm(point, ord=2)
        current_feature_vector = feature_vectors[i,:]

        #initializing the dictionary in the first case
        if(len(keys)==0):
            clusters[l2_norm] = np.array([feature_vectors[i,:]])
            continue

        # checking to which cluster the current point belongs
        substraction = (l2_norm-centers)
        distances = np.linalg.norm( substraction[:,np.newaxis], ord=2, axis=1)
        idx = np.where(distances<=thr)[0]

        # in case of a new cluster
        if(len(idx)==0):
            clusters[l2_norm] = np.array([feature_vectors[i,:]])
        # appending the point to the corresponding
        else:
            idx = idx[0]
            clusters[keys[idx]] = np.concatenate((clusters[keys[idx]], current_feature_vector[np.newaxis,:] ), axis=0)


    return clusters



def mean_shift_segmentation( img, weights=[], num_iterations=5, thr=0.2, kernel="gaussian", sigma=1 ):
    """
    method that performs color segmentation of an image using the mean shift algorithm
    """

    # obtaining the feature vectors and normalizing the dimensions
    print("Obtaining feature vectors")
    feature_vectors = create_feature_vectors( img, color_space="LUV" )
    color_feature_vectors = feature_vectors[:,2:]

    # this weights force the clustering of nearby points
    print("Normalizing feature vectors")
    normalized_feature_vectors, _, _ = normalize_feature_vectors( color_feature_vectors, weights=weights )

    print("Computing mean shift algorithm")
    points,_ = mean_shift(normalized_feature_vectors, num_iterations=num_iterations, thr=thr,
                          kernel=kernel, sigma=sigma)

    # assigning poitns to corresponding cluster
    clusters = assign_points_to_clusters(points, feature_vectors)

    # avergaing the color of points that belong to the same cluster
    segmented_image = average_colors(img, clusters, color_space="LUV")

    return segmented_image


#
