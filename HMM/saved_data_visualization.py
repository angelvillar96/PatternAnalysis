import numpy as np
import os.path
import csv
import random
from hmmlearn import hmm
import matplotlib.pyplot as plt
import math
import itertools
from sklearn.cluster import KMeans
import sklearn
from sklearn.mixture import GaussianMixture
import traceback
from scipy import interpolate
from sklearn.decomposition import PCA
import pickle


feature_dict = {'x': 0, 'y': 1, 'timestamp': 2,
                'pressure': 3, 'fingerarea': 4,
                'velocityx': 5, 'velocityy': 6,
                'accelx': 7, 'accely': 8,
                'accelz': 9, 'gyrox': 10,
                'gyroy': 11, 'gyroz': 12,}



if __name__=="__main__":

    pickle_gauss = open("results_Gaussian.pickle","rb")
    results_Gaussian = pickle.load(pickle_gauss)
    pickle_gauss.close()
    pickle_gmm = open("results_GMM.pickle","rb")
    results_GMM = pickle.load(pickle_gmm)
    pickle_gmm.close()

    fig1 = plt.figure(1)
    plt.subplot_tool(targetfig=fig1)
    for i, result in enumerate(results_GMM):
        plt.subplot(4, 3, i+1)
        plt.imshow(result['img'])
        plt.title(result['title'])
        plt.subplot(4, 3, i+4)
        colors = ["blue"]*25 + ["green"]*20 + ["red"]*20
        x = np.arange(len(colors))
        y = result['scores'][9] #log prob of user 9
        plt.scatter(x, y, c=colors)
        plt.title(result['title'])
    for i, result in enumerate(results_Gaussian):
        plt.subplot(4, 3, i+7)
        plt.imshow(result['img'])
        plt.title(result['title'])
        plt.subplot(4, 3, i+10)
        colors = ["blue"]*25 + ["green"]*20 + ["red"]*20
        x = np.arange(len(colors))
        y = result['scores'][9] #log prob of user 9
        plt.scatter(x, y, c=colors)
        plt.title(result['title'])
    plt.show()

    fig2 = plt.figure(3)
    plt.subplot_tool(targetfig=fig2)
    result = results_GMM[-1]
    for user in range(1, 13):
        plt.title(result['title'])
        plt.subplot(4, 3, user)
        colors = ["blue"]*25 + ["green"]*20 + ["red"]*20
        x = np.arange(len(colors))
        y = result['scores'][user-1] #log prob of user 9
        plt.scatter(x, y, c=colors)
        plt.title(result['title'])
    plt.show()


    fig3 = plt.figure(5)
    result = results_Gaussian[-1]
    for user in range(1, 13):
        plt.title(result['title'])
        plt.subplot(4, 3, user)
        colors = ["blue"]*25 + ["green"]*20 + ["red"]*20
        x = np.arange(len(colors))
        y = result['scores'][user-1] #log prob of user 9
        plt.scatter(x, y, c=colors)
        plt.title(result['title'])
    plt.subplot_tool(targetfig=fig3)
    plt.show()


    print("save results (pickle)")
    import pickle

    print("end")
