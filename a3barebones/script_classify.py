from __future__ import division  # floating point division
import csv
import random
import math
import numpy as np

import dataloader as dtl
import classalgorithms as algs

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def getaccuracy(ytest, predictions):
    correct = 0
    for i in range(len(ytest)):
        if ytest[i] == predictions[i]:
            correct += 1
    return (correct/float(len(ytest))) * 100.0

def geterror(ytest, predictions):
    return (100.0-getaccuracy(ytest, predictions))


def fun(x1,x2,w1,w2,w3,w0):
    '''
    return x3 accroding weights and x1 x2
    '''
    return (-x1*w1-x2*w2-w0)/w3


if __name__ == '__main__':
    trainsize = 9000
    testsize =  1000
    numruns = 1

    classalgs = {#'Random': algs.Classifier(),
                 'Linear Regression': algs.LinearRegressionClass(),
                 #'Logistic Regression': algs.LogitReg(),
                 #'Neural Network': algs.NeuralNet({'epochs': 100}),

                }
    numalgs = len(classalgs)

    parameters = (
       # {'regwgt': 0.0, 'nh': 4},
        {'regwgt': 0.01, 'nh': 8},
        #{'regwgt': 0.05, 'nh': 16},
        #{'regwgt': 0.1, 'nh': 32},
                      )
    numparams = len(parameters)

    errors = {}
    for learnername in classalgs:
        errors[learnername] = np.zeros((numparams,numruns))

    for r in range(numruns):
        trainset, testset = dtl.load_skin(trainsize,testsize,r)
        print (trainset)

        print(('Running on train={0} and test={1} samples for run {2}').format(trainset[0].shape[0], testset[0].shape[0],r))

        for p in range(numparams):
            params = parameters[p]
            for learnername, learner in classalgs.items():
                # Reset learner for new parameters
                learner.reset(params)
                print ('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
                # Train model
                learner.learn(trainset[0], trainset[1])
                # Test model
                predictions = learner.predict(testset[0])
                error = geterror(testset[1], predictions)
                print ('Error for ' + learnername + ': ' + str(error))
                errors[learnername][p,r] = error

    w=[]
    for learnername, learner in classalgs.items():
        besterror = np.mean(errors[learnername][0,:])
        bestparams = 0
        for p in range(numparams):
            aveerror = np.mean(errors[learnername][p,:])
            if aveerror < besterror:
                besterror = aveerror
                bestparams = p
        w=learner.weights
        # Extract best parameters
        learner.reset(parameters[bestparams])
        print("")
        print ('Best parameters for ' + learnername + ': ' + str(learner.getparams()))
        #print ('Average error for ' + learnername + ': ' + str(besterror) + ' +- ' + str(np.std(errors[learnername][bestparams,:])/math.sqrt(numruns)))

    for i,j in classalgs.items():
        #print i + ":"
        errorList = errors[i][0]
        print (errorList)
        #sample_std = np.std(errors[learnername],ddof=1) / np.sqrt(numparams)
        #print("Standard error for " +learnername + ": " + str(sample_std))

    #draw points in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs=[]
    ys=[]
    zs=[]
    for item in range(len(testset)-1):
        for i in testset[item]:
            xs.append(i[0])
            ys.append(i[1])
            zs.append(i[2])

    ax.scatter(xs, ys, zs)
    #draw surface
    x = y = np.arange(0, 1.0, 0.005)
    X, Y = np.meshgrid(x, y)
    zs = np.array([fun(x,y,w[0],w[1],w[2],w[3]) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    ax.plot_surface(X, Y, Z,edgecolors='r')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
