from __future__ import division  # floating point division
import math
import numpy as np
import random

####### Main load functions
def load_skin(trainsize=500, testsize=1000,numruns = 1):
    """ A physics classification dataset """
    filename = 'datasets/result_1.txt'
    dataset = np.loadtxt(filename)
    trainset, testset = splitdataset(numruns,dataset,trainsize, testsize)    
    return trainset,testset
 
####### Helper functions
def splitdataset(numruns,dataset, trainsize, testsize, testdataset=None, featureoffset=None, outputfirst=None):
    """
    Splits the dataset into a train and test split
    If there is a separate testfile, it can be specified in testfile
    If a subset of features is desired, this can be specifed with featureinds; defaults to all
    Assumes output variable is the last variable
    """
    # Generate random indices without replacement, to make train and test sets disjoint
    #np.random.seed(123)
    #randindices = np.random.choice(dataset.shape[0],trainsize+testsize, replace=False)
    randindices = []
    trainsizeRange = int(trainsize/2)
    testsizeRange = int(testsize/2)
    numruns += 1
    print numruns
    print (numruns-1)*5000,numruns*5000
    print (numruns+9)*5000,(numruns+10)*5000
    # train data
    for i in range(((numruns-1)*trainsizeRange),(numruns*trainsizeRange)):
        #index = random.randint(((numruns-1)*10000),(numruns*10000))
        randindices.append(i)
    for i in range(((numruns+9)*trainsizeRange),((numruns+10)*trainsizeRange)):
       # index = random.randint(((numruns+4)*10000),((numruns+5)*10000))
        randindices.append(i)
   # print randindices

    # test data
    for i in range(testsizeRange):
        index = random.randint(40000,50000)
        randindices.append(index)
    for i in range(testsizeRange):
        index = random.randint(90000,99999)
        randindices.append(index)

    featureend = dataset.shape[1]-1
    outputlocation = featureend    
    if featureoffset is None:
        featureoffset = 0
    if outputfirst is not None:
        featureoffset = featureoffset + 1
        featureend = featureend + 1
        outputlocation = 0
    
    Xtrain = dataset[randindices[0:trainsize],featureoffset:featureend]
    ytrain = dataset[randindices[0:trainsize],outputlocation]
    Xtest = dataset[randindices[trainsize:trainsize+testsize],featureoffset:featureend]
    ytest = dataset[randindices[trainsize:trainsize+testsize],outputlocation]


    if testdataset is not None:
        Xtest = dataset[:,featureoffset:featureend]
        ytest = dataset[:,outputlocation]        

    # Normalize features, with maximum value in training set
    # as realistically, this would be the only possibility    
    for ii in range(Xtrain.shape[1]):
        maxval = np.max(np.abs(Xtrain[:,ii]))
        if maxval > 0:
            Xtrain[:,ii] = np.divide(Xtrain[:,ii], maxval)
            Xtest[:,ii] = np.divide(Xtest[:,ii], maxval)
                        
    # Add a column of ones; done after to avoid modifying entire dataset
    Xtrain = np.hstack((Xtrain, np.ones((Xtrain.shape[0],1))))
    Xtest = np.hstack((Xtest, np.ones((Xtest.shape[0],1))))
                              
    return ((Xtrain,ytrain), (Xtest,ytest))

