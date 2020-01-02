import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import scipy.cluster.vq as vq
import scipy.spatial.distance as distance

###############################################
# helper functions
###############################################
'''
if arg is scalar: convert to vector of shape (1, dim)
'''
def scalar_to_vector(arg, dim):
    if ( not (isinstance(arg, np.ndarray)) or arg.shape[0] != dim ):
        vector = np.array([arg]*dim)
    else:
        vector = arg
    return vector


'''###################################################
calculate the within cluster distance W
###################################################'''
def within_cluster_distance(samples, labels):
    W = 0
    for i in range(int(np.min(labels)), int(np.max(labels)+1)):
        clustersamples = samples[np.where(labels==i)]
        W += np.sum(
            np.square(distance.cdist(clustersamples, clustersamples, 'euclidean'))
            )

    return W



'''###################################################
class for data generation/sampling
###################################################'''
class DataCreation():

    def __init__(self, numCenters, sigma):
        self.numCenters = numCenters
        # if sigma is given as a scalar, convert it to a vector
        self.sigma = scalar_to_vector(sigma, numCenters)
        self.centers = self.createCenters(numCenters)

        self.numSamples = None
        self.samples = None
        self.labels = None

        self.num_repetitions = 1
        self.uniform_samples = []
        self.uniform_labels = []


    def createCenters(self, numCenters):
        shape = (numCenters, 2)
        #return np.random.random_sample(shape)
        return 5*np.random.random_sample(shape) - 2.5


    def createSamples(self, numSamples):

        # if numSamples is given as a scalar, convert it to a vector:
        self.numSamples = scalar_to_vector(numSamples, self.numCenters)

        # init empty samples array
        shape = (np.sum(self.numSamples), 2)
        self.samples = np.zeros(shape)
        self.labels = np.zeros(shape[0])

        # create random gaussian samples
        idx = 0
        for i, center in enumerate(self.centers):

            #create samples for center i
            samples = (np.random.randn(self.numSamples[i], 2) )*self.sigma[i] + center

            #add class labels
            labels = i*np.ones( self.numSamples[i], dtype=np.int )

            #collect samples in self.samples
            self.samples[idx:idx+self.numSamples[i], :] = samples
            self.labels[idx:idx+self.numSamples[i]] = labels
            idx += self.numSamples[i]


        permutation = np.random.permutation(shape[0])
        self.samples = self.samples[permutation, :]
        self.labels = self.labels[permutation]

        return ([self.samples, self.labels])


    def createUniformSamples( self, numSamples, num_repetitions=10 ):

        self.num_repetitions = num_repetitions
        self.uniform_samples = []
        self.uniform_labels = []

        for i in range(0, num_repetitions):

            uniform_samples = 5*np.random.rand(numSamples,2)-2.5
            uniform_labels = np.ones( numSamples, dtype=np.int )

            #collect samples in self.samples
            self.uniform_samples.append(uniform_samples)
            self.uniform_labels.append(uniform_labels)

        return [self.uniform_samples, self.uniform_labels]





'''###############################################
class for data visalization/plotting
###############################################'''
class DataVisualization():

    #class variable
    figcounter = 0

    def __init__(self):
        pass


    def createColorDict(self, numClassLabels):
        self.colors = cm.get_cmap('viridis', numClassLabels).colors
        self.colorDict = {}
        for i in range(numClassLabels):
            self.colorDict[i] = self.colors[i]


    def plotLabels(self, samples, labels, title=""):
        DataVisualization.figcounter += 1
        self.fig1 = plt.figure(DataVisualization.figcounter)
        plt.scatter( samples[:,0], samples[:,1],
                     c=[self.colorDict[label] for label in labels] )
        plt.xlabel("x1")
        plt.ylabel("x2")
        if(title!=""):
            plt.title(title)





####################################################
# MAIN
####################################################
if __name__ == "__main__":

    numCenters = 4
    sigma = 0.2

    DataCreation = DataCreation(numCenters, sigma)
    labeledSamples = DataCreation.createSamples(500)
    samples = labeledSamples[0]
    labels = labeledSamples[1]

    labeledUniformSamples = DataCreation.createUniformSamples(500, num_repetitions=20)
    uniform_samples = labeledUniformSamples[0]
    uniform_labels = labeledUniformSamples[1]

    visualizer = DataVisualization()
    visualizer.createColorDict(numCenters)
    visualizer.plotLabels(samples, labels, title=f"Samples using {numCenters} centers")

    visualizer = DataVisualization()
    visualizer.createColorDict(numCenters)
    visualizer.plotLabels(uniform_samples[0], uniform_labels[0], title=f"Uniform samples")

    # predict the results using a different number of clusters 2-6
    ks = range(2,7)
    real_data_distance = []
    mean_uniform_data_distance = []
    std_uniform_data_distance = []

    #iterating vector of possible clusters
    for k in ks:

        print(f"Computing for k={k}")

        #predicting the real data
        real_prediction = vq.kmeans2(samples, k, iter=1000, minit="random" )
        real_predicted_labels = real_prediction[1]

        #displaying the real data predicitions
        visualizer = DataVisualization()
        visualizer.createColorDict(k)
        visualizer.plotLabels(samples, real_predicted_labels, title=f"Predicted labels using {k} clusters")

        #computing the distance for real distribution
        real_data_distance.append( within_cluster_distance(samples, real_predicted_labels) )

        auxiliar_distances = []

        #calculating distance for all synthetic distribution
        for i in range( 0, DataCreation.num_repetitions ):
            uniform_prediction = vq.kmeans2(uniform_samples[i], k, iter=1000, minit="random")
            uniform_predicted_labels = uniform_prediction[1]

            #displaying the simulated data predicitions for one of the distributions
            if( i==0 ):
                visualizer = DataVisualization()
                visualizer.createColorDict(k)
                visualizer.plotLabels(uniform_samples[i], uniform_predicted_labels, title=f"Uniform Predicted labels using {k} clusters")

            auxiliar_distances.append( within_cluster_distance(uniform_samples[i], uniform_predicted_labels) )

        #computing mean and standard deviation for the vector of distances
        mean = np.mean(np.log10(auxiliar_distances))
        unbiased_factor = np.sqrt(1+1/DataCreation.num_repetitions)
        std = np.std(np.log10(auxiliar_distances))*unbiased_factor
        mean_uniform_data_distance.append(mean)
        std_uniform_data_distance.append(std)


    #printing and displaying the within-cluster-distance
    plt.figure()
    real_curve = np.log10(real_data_distance/real_data_distance[0])
    simulated_curve = mean_uniform_data_distance - mean_uniform_data_distance[0]
    plt.plot(ks, real_curve, label="Real Data")
    plt.plot(ks, simulated_curve, 'r', label="Uniform Data")
    plt.legend(loc="best")
    plt.xlabel("# clusters")
    plt.ylabel("Within-Cluster-Distance")
    plt.title(f"Real Data vs Synthetic Data")


    #calculatin and displaying the gap statistics

    gap_curve = simulated_curve - real_curve
    optimum_k = 0
    for i in range(0,len(gap_curve)-1):
        if( gap_curve[i] >= gap_curve[i+1] - std_uniform_data_distance[i+1] ):
            optimum_k = i + ks[0]
            break

    plt.figure()
    plt.errorbar(ks, gap_curve, std_uniform_data_distance)
    plt.xlabel("# clusters")
    plt.ylabel("Gap statistics")
    plt.title(f"Optimum K = {optimum_k}")


    plt.tight_layout()
    plt.show()
