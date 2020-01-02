import numpy as np
import scipy.misc as misc
import scipy.ndimage as image
from matplotlib import pyplot as plt
from numba import jit
import os, sys, datetime, json

#################################################################
# Class that contains the methods to carry out density estimation
#################################################################
class densityEstimation(object):


    def __init__(self, sigma=80, kernel_sizes=[1, 5, 10, 20, 30, 50], trainSize=80):
        """
        Initializer of the density estimator object

        Args:
        -----
        sigma: Integer
            variance of the Gaussiam Smoothing Filter
        kernel_sizes: list of floats

        trainSize: float [0:100]
            size of the total pixels used for training
        """

        self.sigma = sigma
        self.original_image = misc.face(gray=True)
        self.blurred_image = image.gaussian_filter( self.original_image, sigma )
        self.kernel_sizes = kernel_sizes
        self.trainSize = trainSize
        self.trainingSet = []
        self.testSet = []


    def displayImages(self):
        """
        Method that displays the original and blurred images
        """

        plt.figure(figsize=(8,6))
        plt.subplot(1,2,1)
        plt.imshow( self.original_image, cmap="gray")
        plt.title("Original Image")
        plt.subplot(1,2,2)
        plt.imshow( self.blurred_image, cmap="gray")
        plt.title("Blurred Image")


    def drawSamples(self, numberOfSamples, from_where="image"):
        """
        Method that draws a given number of samples

        Args:
        -----
        numberOfSamples: Integer
            number of pixels to sample from the Cumulative
            Desity Function of the image
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
            sampledPDF[Idx, Idy] += 1

        return sampledPDF, Indexes2D


    def estimateDensity(self, sampledPDF, h=1, kernel='box'):
        """
        Estimating the PDF via the Parzen window estimator

        Args:
        -----
        sampledPDF:
            samples taken
        h: Integer
            size of the box kernel
        kernel: string (only box kernel was implemented)
            Type of Parzen kernel used for density estimation
        """

        if kernel=='box':
            kernel = np.ones((h,h))
        else:
            kernel = np.ones((h,h))
        PDF = image.correlate(input=sampledPDF, weights=kernel, mode='mirror')
        PDF = PDF/np.sum(PDF) #normalize pdf

        return PDF


    def evaluateSet( self, numberOfSamples=10000, kernels_to_test=[], threshold=10e-12 ):
        """
        Applying maximum likelihood to compute the optimal kernel size

        Args:
        -----
        numberOfSamples: Integer
            Number of samples to be drawn from the original image
        kernels_to_test: list of Integers
            Sizes of the kernels to test and compare
        threshold: float
            small number to avoid logarithms of 0
        """

        if( len(kernels_to_test)==0 ):
            kernels_to_test = self.kernel_sizes

        trainSet_likelihoods = []
        testSet_likelihoods = []

        _,indexes = self.drawSamples(numberOfSamples)
        likelihoods = {}

        for j in range(0, int(100/(100-self.trainSize)) ):

            print("Process Fold #" + str(j))
            stepsize = int(np.ceil(len(indexes)*(100-self.trainSize)/100))

            self.testSet = indexes[j*stepsize:(j+1)*stepsize]
            self.trainingSet = indexes[:(j)*stepsize] + indexes[(j+1)*stepsize:]

            emptyMatrix = np.zeros( self.blurred_image.shape )
            for k in self.trainingSet:
                emptyMatrix[ k ] += 1

            for i,kernelSize in enumerate(kernels_to_test):
                estimated_pdf = self.estimateDensity( emptyMatrix, kernelSize )
                estimated_pdf = estimated_pdf + np.ones( estimated_pdf.shape )*threshold
                likelihoods.setdefault(kernelSize, []).append( np.sum( np.log( estimated_pdf.T[ self.testSet ] ) ) )

        print(likelihoods)
        averageLikelihoods = []

        for kernelSize in kernels_to_test:
            averageLikelihoods.append( np.sum(likelihoods[kernelSize])/len(likelihoods[kernelSize]) )

        print(kernels_to_test)
        print(averageLikelihoods)

        maxIndex = np.argmax( averageLikelihoods )
        maxLikelihood = np.max(averageLikelihoods)
        print( "maxLikelihood: " + str(maxLikelihood) )
        print( "optimum Kernel: " + str(kernels_to_test[maxIndex]))


    def displayReconstruction(self, samples, numberOfSamples):
        """
        Displaying the reconstructed density

        Args:
        -----
        samples: numpy array
            samples taken from the original image and used to estimate the pdf
        numberOfSamples: integer
            number of samples taken from the original pdf
        """

        plt.figure(figsize=(12,8))
        plt.suptitle( "Reconstruction using " + str(int(numberOfSamples)) + " samples")

        pdfs = []

        for i, h in enumerate( self.kernel_sizes ):
            pdf = self.estimateDensity( samples, h)
            pdfs.append(pdf)
            plt.subplot(2, 3, i+1)
            plt.imshow(pdfs[i], cmap="gray")
            plt.title("Kernel Size=" + str(h))
        plt.tight_layout()


##########################################
# Main code
##########################################
if __name__=="__main__":

    #clearing the console
    os.system("cls")
    sigma = 3
    trainSize = 80
    kernel_sizes = [1, 5, 10, 25, 50]
    sampleSizes = [1e4, 5e4, 1e5, 2e5, 4e5]

    densityEstimator = densityEstimation(sigma, kernel_sizes, trainSize)
    densityEstimator.displayImages()

    threshold = 10e-8

    for numberOfSamples in sampleSizes:
        print("\n\n")
        print("Processing the iteration with " + str(numberOfSamples) + " samples" )
        print(datetime.datetime.now())
        samples, _ = densityEstimator.drawSamples(numberOfSamples=numberOfSamples, from_where="training")
        densityEstimator.displayReconstruction( samples, numberOfSamples )
        densityEstimator.evaluateSet( numberOfSamples=numberOfSamples, threshold=threshold )

    plt.show()
