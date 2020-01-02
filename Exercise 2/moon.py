import numpy as np
import scipy.misc as misc
import scipy.ndimage as image
from matplotlib import pyplot as plt
import os, sys, datetime, json
import sklearn.ensemble as ensemble
import sklearn.datasets as datasets


def visualize_labeled_moons():
    X, y = datasets.make_moons(10000)
    colordict = {0: 'red',
                1: 'blue',
                }
    plt.figure()
    plt.scatter(X[:,0], X[:,1], c=[colordict[i] for i in y])
    plt.show()


def make_background( n_samples  ):
    x1 = np.random.random_sample(n_samples )*3 - 1
    x2 = np.random.random_sample(n_samples )*1.5 - 0.5
    samples = np.zeros( (n_samples , 2) )
    samples[:,0] = x1
    samples[:,1] = x2
    return samples


def make_testpoints( n_samples ):
    x1test = np.linspace(-1, 2, n_samples*2)
    x2test = np.linspace(-0.5, 1, n_samples)
    x1test, x2test = np.meshgrid(x1test, x2test)
    samples = np.zeros((2*n_samples**2, 2))
    samples[:,0] = x1test.flat
    samples[:,1] = x2test.flat
    return samples



if __name__ == "__main__":


    n_samples  = [100, 1000, 10000]
    depths = [10, 15, 20]
    n_trees = [10, 20, 50, 100, 200]
    params = {
        'n_estimators' : 100,
        'max_depth' : 20,
        }
    for n in n_samples :
        for d in depths:
            for n_tr in n_trees:
                print("{} samples, {} depth, {} trees".format(n, d, n_tr))
                params['n_estimators']
                params['max_depth'] = d

                # setup training data
                pdfsamples, _ = datasets.make_moons( n)
                bgsamples = make_background(n)
                X = np.concatenate((pdfsamples, bgsamples))
                y = np.array([100]*pdfsamples.shape[0] + [0]*bgsamples.shape[0])

                # initialize and train our Random Forests
                RandomForest = ensemble.RandomForestRegressor(**params)
                ExtraTree = ensemble.ExtraTreesRegressor(**params)
                RandomForest.fit(X, y)
                ExtraTree.fit(X, y)

                # generate testpoints and predict "density values"
                testpoints = make_testpoints(100)
                yExtraTree = ExtraTree.predict(testpoints)
                yRandomForest = RandomForest.predict(testpoints)

                fig = plt.figure()
                plt.subplot(1,2,1)
                plt.title('RandomForest, #samp: {} d:{}, , #trees: {}'.format(n, d, n_tr), fontsize=6)
                plt.scatter(testpoints[:,0], testpoints[:,1], c=yRandomForest)
                plt.subplot(1,2,2)
                plt.title('ExtraTrees,  #samp: {}, d:{}, #trees: {}'.format(n, d, n_tr), fontsize=6)
                plt.scatter(testpoints[:,0], testpoints[:,1], c=yExtraTree)
                fig.savefig("S{}D{}T{}.png".format(n, d, n_tr))
                plt.close(fig)
