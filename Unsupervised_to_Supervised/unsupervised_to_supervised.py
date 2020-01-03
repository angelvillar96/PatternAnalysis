import numpy as np
import scipy.misc as misc
import scipy.ndimage as image
from matplotlib import pyplot as plt
import os, sys, datetime, json
import sklearn.ensemble as ensemble
import time


def measure(func):
    """
    Method to measure the duration of the program
    """
    def wrapper(*args, **kwargs):
        print("call: ", func.__name__)
        t = time.time()
        r =func(*args, **kwargs)
        print("took {} s".format(time.time()-t))
        return r
    return wrapper


class Unsupervised_to_Supervised():


    def __init__(self, sigma=50):
        """
        Initializer of the object
        """

        self.sigma = sigma
        self.original_image = misc.face(gray=True)
        self.blurred_image = image.gaussian_filter( self.original_image, sigma )

        self.racoon = None
        self.background = None
        self.model = None


    def getSamples(self, numberOfSamples=1e5):
        """
        Method that draws a given number of samples
        """

        #drawing n random numbers between 0 and 1
        randomVector = np.random.random(int(numberOfSamples))

        #creating the CDF function by linearizing and normalizing the image
        CDF = np.cumsum(self.blurred_image)/np.sum(self.blurred_image)

        #sample the CDF function
        Indexes1D = np.searchsorted(CDF, randomVector)
        Indexes2D = np.unravel_index(Indexes1D, self.blurred_image.shape)

        sampledPDF = np.zeros(self.blurred_image.shape)

        Indexes2D = list(zip(Indexes2D[0], Indexes2D[1]))
        for (Idx, Idy) in Indexes2D:
            sampledPDF[Idx, Idy] = 100

        self.racoon = sampledPDF

        return sampledPDF, Indexes2D


    def getBackgroundSamples(self, numberOfSamples=1e5):
        """
        Drawing samples for the background
        """
        randomRow = np.random.randint( low=0, high=self.blurred_image.shape[0], size=numberOfSamples)
        randomCol = np.random.randint( low=0, high=self.blurred_image.shape[1], size=numberOfSamples)

        sampledDensity = np.zeros( self.blurred_image.shape )

        for i in range(len(randomRow)):
            sampledDensity[randomRow[i],randomCol[i]] = 1

        self.background = sampledDensity
        Indexes2D = np.array((randomRow, randomCol)).T
        return sampledDensity, Indexes2D


    def displayDensities(self):
        """
        Displaying original, auxiliary and overlapping densities
        """

        plt.figure(figsize=(12,6))
        plt.subplot(1,3,1)
        plt.imshow(self.racoon, cmap="gray")
        plt.title("Original Density")
        plt.subplot(1,3,2)
        plt.imshow(self.background, cmap="gray")
        plt.title("Auxiliary Density")
        plt.subplot(1,3,3)
        plt.imshow(100*self.background+self.racoon, cmap="gray")
        plt.title("Combined Densities")


    def RandomForestRegressorTrain( self, n_trees=8, max_depth=None, mode="RandomForest" ):
        """
        Training the Random forest Regressor on the labelled background vs racoon
        """

        #merging background with racoon
        labels = self.racoon + self.background

        #obtaining only the samples
        idx = np.where( labels>0 )
        labels = labels[idx]
        features = np.array(idx).T

        #training the model
        if( mode=="ExtraTrees" ):
            self.model = ensemble.ExtraTreesRegressor( n_estimators=n_trees, max_depth=max_depth, random_state=42 )
        else:
            self.model = ensemble.RandomForestRegressor( n_estimators=n_trees, max_depth=max_depth, random_state=42 )
        self.model.fit(features, labels);


    def RandomForestRegressorTest( self ):
        """
        Using the random forest regressor to infere predictions
        """

        #reshaping the image
        size = self.blurred_image.shape[0]*self.blurred_image.shape[1]
        x1 = np.arange(self.blurred_image.shape[0])
        x2 = np.arange(self.blurred_image.shape[1])
        X1, X2 = np.meshgrid(x1, x2)
        data_to_predict = np.zeros((size, 2))
        data_to_predict[:,0] = X1.flat
        data_to_predict[:,1] = X2.flat

        #predicting the data
        predictions = self.model.predict( data_to_predict )

        #making a matrix to display
        reconstruction = np.zeros( self.blurred_image.shape )
        for i, idx in enumerate(data_to_predict):
            reconstruction[int(idx[0]),int(idx[1])] = predictions[i]
        #predictions = np.reshape( predictions, self.blurred_image.shape )

        return reconstruction


    def compareMethods(self):
        """
        Comparing the performance of scikit ExtraTrees with the regular RandomForests
        """

        n_trees = [8, 20, 50]
        depth_trees = [8, 12, 20]

        #evaluating method for different number of trees
        fig = plt.figure()
        fig.suptitle("Comparing for different number of trees")
        for i,n in enumerate(n_trees):

            self.RandomForestRegressorTrain( n_trees=n, max_depth=12 )
            predictions = self.RandomForestRegressorTest()
            plt.subplot(2,3,i+1)
            plt.imshow(predictions, cmap="gray")
            plt.title(f"Random Forest n={n}")

            self.RandomForestRegressorTrain( n_trees=n, mode="ExtraTrees", max_depth=12 )
            predictions = self.RandomForestRegressorTest()
            plt.subplot(2,3,i+4)
            plt.imshow(predictions, cmap="gray")
            plt.title(f"ExtraTrees n={n}")


        #evaluating methods for different depth with 8 trees
        fig = plt.figure()
        fig.suptitle("Comparing for different tree depths")
        for i,n in enumerate(depth_trees):

            self.RandomForestRegressorTrain( max_depth=n )
            predictions = self.RandomForestRegressorTest()
            plt.subplot(2,3,i+1)
            plt.imshow(predictions, cmap="gray")
            plt.title(f"Random Forest depth={n}")

            self.RandomForestRegressorTrain( max_depth=n, mode="ExtraTrees" )
            predictions = self.RandomForestRegressorTest()
            plt.subplot(2,3,i+4)
            plt.imshow(predictions, cmap="gray")
            plt.title(f"ExtraTrees depth={n}")




def convertFeatures2D(racoonSamples, BackgroundSamples):

    xrac = np.where(racoonSamples>0.1)
    yrac = np.array(racoonSamples[xrac])
    xrac = np.array(xrac).T
    xbg = np.where(BackgroundSamples>0.1)
    ybg = np.array(BackgroundSamples[xbg])
    xbg = np.array(xbg).T

    X = np.concatenate((xrac, xbg))
    y = np.concatenate((yrac, ybg))
    return X, y



def predict2D( forest, dimx, dimy ):
    x1test = np.arange(dimx)
    x2test = np.arange(dimy)
    X1test, X2test = np.meshgrid(x1test, x2test)
    xtest = np.zeros((dimx*dimy, 2))
    xtest[:,0] = X1test.flat
    xtest[:,1] = X2test.flat

    ytest = forest.predict(xtest)
    reconstructed = np.zeros((dimx, dimy))
    for i, id_ in enumerate(xtest):
        reconstructed[int(id_[0]),int(id_[1])] = ytest[i]
    return reconstructed




def test1():

    #clearing the console
    os.system("clear")
    sigma = 3
    numberOfSamples = int(1e5)
    n_trees = 50
    max_depth = 10

    converter = Unsupervised_to_Supervised( sigma=sigma )

    #obtaining and displaying samples from the racoon
    racoonSamples,_ = converter.getSamples( numberOfSamples )
    plt.figure()
    plt.imshow( racoonSamples, cmap="gray" )
    plt.title("Sample from the racoon")


    #obtaining and displaying samples from the background
    BackgroundSamples, _ = converter.getBackgroundSamples( numberOfSamples )
    plt.figure()
    plt.imshow( BackgroundSamples, cmap="gray" )
    plt.title("Sample from the background")


    # applying the random forest to compuite the regression problem
    converter.RandomForestRegressorTrain( n_trees=n_trees, max_depth=max_depth, mode="ExtraTrees" )
    predictions = converter.RandomForestRegressorTest()
    plt.figure()
    plt.imshow( predictions, cmap="gray" )
    plt.title("Predictions")

    # setup variables for 2D Regression Forest
    X, y = convertFeatures2D(racoonSamples, BackgroundSamples)

    # applying the random forest for 2D
    forest = ensemble.ExtraTreesRegressor(n_estimators=n_trees, max_depth=max_depth)
    forest.fit = measure(forest.fit)
    forest.predict = measure(forest.predict)

    forest.fit(X, y)
    reconstructed = predict2D( forest, *racoonSamples.shape )


    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow( reconstructed, cmap="gray" )
    plt.title("Reconstructed")
    plt.subplot(1,3,2)
    plt.imshow( racoonSamples, cmap="gray" )
    plt.title("racoonSamples")
    plt.subplot(1,3,3)
    plt.imshow( predictions, cmap="gray" )
    plt.title("predictions")


    #difference between random forest and extra trees
    # - RandomForest take random features and then calculates the best split among this random
    # subset of the features
    # - ExtraTress also takes random features, but instead of searching the optimum threshold,
    # it gets the best from a randomly selected subset of points for the features (as we have seen in the lecture)
    converter.compareMethods()


    plt.tight_layout()
    plt.show()




def test2():
    #clearing the console
    os.system("clear")
    sigma = 3

    numberofSamples = [1e4, 10e4, 50e4, 100e4, 150e4, 200e4]

    for n_samples in numberofSamples:
        n_samples = int(n_samples)


        n_trees = [15, 30, 60, 120]

        for n_tr in n_trees:

            depths = [10, 15, 20, 25, 30]
            for d in depths:

                print("#samples: {}, #trees: {}, max_depth: {}".format(n_samples, n_tr, d))

                converter = Unsupervised_to_Supervised( sigma=sigma )

                #obtaining and displaying samples from the racoon
                racoonSamples,_ = converter.getSamples( n_samples )

                #obtaining and displaying samples from the background
                BackgroundSamples, _ = converter.getBackgroundSamples( n_samples )

                # applying the random forest to compuite the regression problem
                converter.RandomForestRegressorTrain( n_trees=n_tr, max_depth=d, mode="ExtraTrees" )
                prediction1D = converter.RandomForestRegressorTest()

                # setup variables for 2D Regression Forest
                X, y = convertFeatures2D(racoonSamples, BackgroundSamples)

                # applying the random forest for 2D
                #forest = ensemble.RandomForestRegressor(n_estimators=40, max_depth=20)
                forest = ensemble.ExtraTreesRegressor(n_estimators=n_tr, max_depth= d)
                forest.fit = measure(forest.fit)
                forest.predict = measure(forest.predict)

                forest.fit(X, y)
                prediction2D = predict2D( forest, *racoonSamples.shape )


                fig1 = plt.figure()
                plt.imshow(prediction1D, cmap="gray")
                plt.title("prediction1D #trees:{}, #depth:{}, #samples:{}".format(n_tr, d, n_samples))
                plt.tight_layout()
                fig1.savefig("prediction1DTr{}D{}S{}.png".format(n_tr, d, n_samples))
                plt.close(fig1)

                fig2 = plt.figure()
                plt.imshow(prediction2D, cmap="gray")
                plt.title("prediction2D #trees:{}, #depth:{}, #samples:{}".format(n_tr, d, n_samples))
                plt.tight_layout()
                fig2.savefig("prediction2DTr{}D{}S{}.png".format(n_tr, d, n_samples))
                plt.close(fig2)

                fig3 = plt.figure()
                plt.subplot(1,3,1)
                plt.imshow( prediction2D, cmap="gray" )
                plt.title("Prediction2D #trees{}".format(n_tr), fontsize=5)
                plt.subplot(1,3,2)
                plt.imshow( racoonSamples, cmap="gray" )
                plt.title("racoonSamples #samples:{}".format(n_samples), fontsize=5)
                plt.subplot(1,3,3)
                plt.imshow( prediction1D, cmap="gray" )
                plt.title("Prediction1D, #depth:{}".format(d), fontsize=5)
                #plt.tight_layout()
                fig3.savefig("summaryTr{}D{}S{}.png".format(n_tr, d, n_samples))
                plt.close(fig3)



##########################################
# Main code
##########################################
if __name__=="__main__":

    # this trains random forest with few parameters and compares with extra trees
    test1()

    # this method generates figures using different parameters
    # test2()
