import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import signal
from hmmlearn import hmm
import os, datetime, csv, json
import sklearn.mixture


###############################################
############# DATASET CLASS ###################
###############################################
class Dataset():
    """
    Class that handles the data loading from csv files into canva.
    It also creates the training and test sets.
    """

    ###############################################
    # init method from the dataset class
    def __init__(self):

        self.current_path = os.getcwd()
        self.data_path = self.current_path + "/data"

        self.original_files = {}
        self.imitation_files = {}
        self.original_test_files = {}
        self.imitation_test_files = {}

        self.training_set = None
        self.original_test_set = None
        self.imitation_test_set = None

        self.accuracy = 0.
        self.threshold = 0.

        self.get_files()


    ###############################################
    # Method that obtains and clasifies original and imitation csv files from the file directory
    def get_files(self):

        for path, dirs, files in os.walk(self.data_path):
            for dir in dirs:
                self.original_files[dir] = []
                self.imitation_files[dir] = []
                for file in os.listdir(path + "/" + dir):
                    if( "original" in file ):
                        self.original_files[dir].append(path + "/" + dir + "/" + file)
                    else:
                        self.imitation_files[dir].append(path + "/" + dir + "/" + file)

        return


    ###############################################
    # Method that creates equilibrated train and test sets
    def obtain_sets(self, user="user_0", size_training=30):

        original_files_permutated = np.random.permutation(self.original_files[user])
        self.training_files = original_files_permutated[:size_training]
        self.imitation_test_files = self.imitation_files[user]
        self.original_test_files = original_files_permutated[-len(self.imitation_test_files):]

        return


    #############################################
    # Method that creates the feature vectors
    def create_feature_vector(self, files=[], name=""):

        if( len(files)==0 ):
            return

        epsilon = 1e-8
        set = []

        #iterating all files obtaining the significant data to compute the feature vectors
        for file in files:

            #reading the csv files and keeping the first 3 columns (x,y,time)
            file_data = pd.read_csv(file)
            file_data = file_data.to_numpy()
            data = np.zeros((file_data.shape[0],7))
            data[:,0:3] = file_data[:,0:3]

            #computing the other interesting features
            angle = np.arctan(data[:,1]/(data[:,0]+epsilon))
            velocity = np.sqrt( np.square(data[:,1]) + np.square(data[:,0]) )
            log_curvature = np.log10( velocity/(angle+epsilon) )
            acceleration = np.sqrt( np.square(velocity) + np.square(velocity*angle) )

            #assigning the new computed features
            data[:,3] = angle
            data[:,4] = velocity
            data[:,5] = log_curvature
            data[:,6] = acceleration

            #normalizing the data
            data = self.normalization(data)
            set.append(data)

        return set


    #############################################
    # Method that normalizes the feature vectors
    def normalization(self, set):

        #epsilon = np.random.randn(np.array(set).shape[0],np.array(set).shape[1])
        #variance = np.var(set+epsilon, axis=0)
        mean = np.mean(set, axis=0)
        variance = np.var(set, axis=0)
        set = (set - mean)/variance

        return set



###############################################
############## MODEL CLASS ####################
###############################################
class Model():

    accuracies = []

    ###############################################
    # Method that initialized the HMM
    def __init__(self, n_components=5, user="user_0", model="gaussian"):

        self.user = user
        self.n_components = n_components

        if(model=="gaussian"):
            self.model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag", \
                                         init_params="cm", params="cmt")
        elif(model=="GMMHMM"):
            self.model = hmm.GMMHMM(n_components=n_components, n_mix=3, covariance_type="diag", \
                                         init_params="cm", params="cmt")
            self.model.gmms_ = [sklearn.mixture.GaussianMixture()]*3

        self.model.startprob_ = np.concatenate(([1],np.zeros(n_components-1)))
        self.model.transmat_ = self.compute_trans_matrix( n_components )

        self.overall_accuracy = 0


    ##############################################
    # Method that computes the initial transition probabilities
    def compute_trans_matrix( self, n_components ):

        matrix = np.zeros((n_components,n_components))
        matrix[-1,-1] = 1.

        for i in range(0,matrix.shape[0]-1):
            matrix[i,i:i+2] = [0.5,0.5]

        return matrix


    ###############################################
    # Method that trains the HMM on a given set
    def train(self, training_set):

        lengths =  [ seq.shape[0] for seq in training_set ]

        #preprocessing the training set to obtain the desired shape
        concatenated_set = self.preprocessing( training_set )

        #fitting the model
        self.model.fit(concatenated_set, lengths)

        return


    ###############################################
    # Method that test the HMM on all signatures
    def test(self, training_set, original_test_set, imitation_test_set ):

        plt.figure()

        training_axis = np.arange(len(training_set))
        original_test_axis = np.arange(len(original_test_set))+len(training_axis)
        imitation_test_axis = np.arange(len(imitation_test_set))+len(training_axis)+len(original_test_set)

        training_scores = []
        original_test_scores = []
        imitation_test_scores = []

        for signature in training_set:
            vectorized_signature = signature.tolist()
            score = -1*self.model.score(vectorized_signature)
            training_scores.append(score)

        for signature in original_test_set:
            vectorized_signature = signature.tolist()
            score = -1*self.model.score(vectorized_signature)
            original_test_scores.append(score)

        for signature in imitation_test_set:
            vectorized_signature = signature.tolist()
            score = -1*self.model.score(vectorized_signature)
            imitation_test_scores.append(score)

        accuracy, threshold = self.evaluate(training_scores, original_test_scores, imitation_test_scores)

        xaxis = np.arange(len(imitation_test_set)+len(training_axis)+len(original_test_set))
        plt.plot( xaxis, threshold*np.ones(len(xaxis)), "--", label="Threshold" )
        plt.scatter(training_axis,training_scores, label="Training data")
        plt.scatter(original_test_axis, original_test_scores, c="g", label="Original Test data")
        plt.scatter(imitation_test_axis, imitation_test_scores, c="r", label="Imitated Test data")
        plt.legend(loc="best")
        plt.title(f"{self.user} data. Accuracy={accuracy} ")
        plt.ylabel("Score")
        plt.xlabel("File")
        plt.savefig(f"{self.n_components}_{self.user}.png")

        Model.accuracies.append(accuracy)


    ###############################################
    # Method that preprocesses a set to obtain the desired shape
    def preprocessing(self, set):

        concatenated_set = []
        for signature in set:
            vectorized_signature = signature.tolist()
            concatenated_set.extend(vectorized_signature)
        concatenated_set = np.array(concatenated_set)

        return concatenated_set


    #############################################
    # Method that evaluates the performance of the model
    def evaluate(self, training_scores, original_test_scores, imitation_test_scores):

        #finding a threshold: third to smallest training score
        sorted_scores =  np.sort(training_scores)
        threshold = sorted_scores[2]

        #computing the number of errors
        errors = len(np.where(original_test_scores < threshold)[0])
        errors += len(np.where(imitation_test_scores > threshold)[0])

        #computing the local accuracy
        accuracy = 1 - errors/(len(original_test_scores)+len(imitation_test_scores))
        return accuracy, threshold


    #############################################
    # Method that obtains the overall_accuracy
    def get_overall_accuracy(self):

        self.overall_accuracy = np.sum(Model.accuracies)/len(Model.accuracies)

        #saving the accuracies
        if( not os.path.isfile("logs.txt") ):
            with open("logs.txt", 'a') as f:
                f.write("States,  U_0,  U_1,  U_2,  U_3,  U_4,  U_5,  U_6,  U_7," + \
                "  U_8,  U_9,  accuracy\n")

        with open("logs.txt", 'a+') as f:
            f.write(f"  {self.n_components}     " )
            for i in Model.accuracies:
                f.write(f"{i}  " )
            f.write(f"{self.overall_accuracy}\n")

        return self.overall_accuracy



###############################################
############### MAIN CODE #####################
###############################################
if __name__=="__main__":

    users = ["user_0","user_1","user_2","user_3","user_4",\
             "user_5","user_6","user_7","user_8","user_9"]
    size_training = 30
    n_components = 5

    model_type="GMMHMM"
    model_type = "gaussian"

    os.system("clear")
    print(f"Code started: {datetime.datetime.now()}")

    #intializing dataset and model
    dataset = Dataset()

    # carrying out the algorithm for all users
    for user in users:

        #obraining the sets correspondign to the current user
        dataset.obtain_sets(user=user, size_training=size_training)

        #computing a training set
        training_set = dataset.create_feature_vector( files=dataset.training_files, name="Training" )
        dataset.training_set = training_set

        #computing a test set
        original_test_set = dataset.create_feature_vector( files=dataset.original_test_files, name="Test" )
        dataset.original_test_set = original_test_set

        #computing a test set
        imitation_test_set = dataset.create_feature_vector( files=dataset.imitation_test_files, name="Test" )
        dataset.imitation_test_set = imitation_test_set

        #intializing the HMM model
        model = Model( n_components=n_components, user=user, model=model_type )

        model.train( training_set )
        model.test( training_set, original_test_set, imitation_test_set )

    overall_accuracy = model.get_overall_accuracy()
    print(f"Overall accuracy: {overall_accuracy*100}%")

    print(f"\nCode finished: {datetime.datetime.now()}\n")

    plt.tight_layout()
    plt.show()



#
