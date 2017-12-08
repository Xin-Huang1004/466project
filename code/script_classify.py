from __future__ import division  # floating point division
import csv
import random
import math
import numpy as np
import utilities as utils

import dataloader as dtl
import classalgorithms as algs

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from scipy import stats

import matplotlib as mpl


def getaccuracy(ytest, predictions):
    correct = 0
    for i in range(len(ytest)):
        if ytest[i] == predictions[i]:
            correct += 1
    return (correct/float(len(ytest))) * 100.0

def fun(x1,x2,w1,w2,w3,w0):
    '''
    return x3 accroding weights and x1 x2
    '''
    return ((-x1*w1-x2*w2-w0)/w3)

def plotgbr(temp,w,is_NN,learnername):
    if(is_NN == 1):
        temp = utils.sigmoid(np.dot(testset[0], w1.T))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs0=[]
    ys0=[]
    zs0=[]
    xs1=[]
    ys1=[]
    zs1=[]
    for i in range(len(temp)):
        if(testset[1][i] == 0):
            xs0.append(temp[i][0])
            ys0.append(temp[i][1])
            zs0.append(temp[i][2])
        else:
            xs1.append(temp[i][0])
            ys1.append(temp[i][1])
            zs1.append(temp[i][2])
    ax.scatter(xs0, ys0, zs0,c = 'r',marker = "^")
    ax.scatter(xs1, ys1, zs1,c = 'b',marker = "o")

    #draw surface
    x = y = np.arange(0, 1.0, 0.005)
    X, Y = np.meshgrid(x, y)
    if(is_NN == 1):
        zs = np.array([fun(x,y,w[0][0],w[0][1],w[0][2],w[0][3]) for x,y in zip(np.ravel(X), np.ravel(Y))])
    else:
        zs = np.array([fun(x,y,w[0],w[1],w[2],w[3]) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    ax.plot_surface(X, Y, Z,edgecolors='black')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title(learnername)
    plt.show()

def t_test(errorslist):
    NaiveBayes = np.array(errorslist[0])
    Neural_Network = np.array(errorslist[1])
    Logistic = np.array(errorslist[2])
    # test NN is better than linear?
    alpha = 0.01
    t,p = stats.ttest_ind(Neural_Network,NaiveBayes,equal_var=False)
    print('t-value is: %f,  p-vaule is: %f'%(t,p))
    if p < alpha:
        print("Reject h0, Neural Network and NaiveBayes are different!")
    else:
        print("Fail to reject h0")
    # test NN is better than logsitic?
    t,p = stats.ttest_ind(Neural_Network,Logistic,equal_var=False)
    print('t-value is: %f,  p-vaule is: %f'%(t,p))
    if p < alpha:
        print("Reject h0,Neural_Network and Logistic are different!")
    else:
        print("Fail to reject h0")
    # test liner is better than logistic?
    t,p = stats.ttest_ind(NaiveBayes,Logistic,equal_var=False)
    print('t-value is: %f,  p-vaule is: %f'%(t,p))
    if p < alpha:
        print("Reject h0,NaiveBayes and Logistic are different!")
    else:
        print("Fail to reject h0")


if __name__ == '__main__':
    trainsize = 90000
    testsize =  10000
    numruns = 10

    classalgs = {
                 #'Linear Regression': algs.LinearRegressionClass(),
                 'Naive Bayes': algs.NaiveBayes({'usecolumnones': False}),
                 'Logistic Regression': algs.LogitReg(),
                 'Neural Network': algs.NeuralNet({'epochs': 100}),
                 #'KernelLogitReg': algs.KernelLogitReg({'kernel': 'linear'})
                 
                }

    numalgs = len(classalgs)
    parameters = (
        # best parameters for now
        {'regwgt': 0.1, 'nh': 12,'eta':0.0001},
       # {'regwgt': 0.1, 'nh': 2,'eta':0.01},
       # {'regwgt': 0.01, 'nh': 4,'eta':0.001},
       # {'regwgt': 0.02, 'nh': 6,'eta':0.0001},
       # {'regwgt': 0.03, 'nh': 8,'eta':0.0005},
       # {'regwgt': 0.04, 'nh': 10,'eta':0.00001},
       # {'regwgt': 0.05, 'nh': 12,'eta':0.00005},
       # {'regwgt': 0.06, 'nh': 14,'eta':0.000001},
       # {'regwgt': 0.1,  'nh': 16,'eta':0.000005},
       #{'regwgt': 0.0,  'nh': 32,'eta':0.0000001},
                )
    numparams = len(parameters)

    errors = {}
    for learnername in classalgs:
        errors[learnername] = np.zeros((numparams,numruns))

    for r in range(numruns):
        trainset, testset = dtl.load_skin(trainsize,testsize,r)


        print(('Running on train={0} and test={1} samples for run {2}').format(trainset[0].shape[0], testset[0].shape[0],r))

        for p in range(numparams):
            #trainset, testset = dtl.load_skin(trainsize,testsize,p)
            #print(('Running on train={0} and test={1} samples for run {2}').format(trainset[0].shape[0], testset[0].shape[0],r))

            params = parameters[p]
            for learnername, learner in classalgs.items():
                # Reset learner for new parameters
                learner.reset(params)
                print ('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
                # Train model
                learner.learn(trainset[0], trainset[1])
                # Test model
                predictions = learner.predict(testset[0])
                error = getaccuracy(testset[1], predictions)
                print ('Accuracy for ' + learnername + ': ' + str(error))
                errors[learnername][p,r] = error


    #w=[]
    #w1 = []
    #is_NN = 0
    for learnername, learner in classalgs.items():
        besterror = np.mean(errors[learnername][0,:])
        bestparams = 0
        for p in range(numparams):
            aveerror = np.mean(errors[learnername][p,:])
            if aveerror < besterror:
                besterror = aveerror
                bestparams = p

        # if(learnername == "Neural Network"):
        #     is_NN = 1
        #     w = learner.w_output
        # else:
        #     w = learner.weights
        # # Extract best parameters
        # learner.reset(parameters[bestparams])
        #
        # temp = testset[0]
        # plotgbr(temp,w,is_NN,learnername)

        print("")
        print ('Best parameters for ' + learnername + ': ' + str(learner.getparams()))

        print ('Average error for ' + learnername + ': ' + str(besterror) + ' +- ' + str(np.std(errors[learnername][bestparams,:])/math.sqrt(numruns)))
        sample_std = np.std(errors[learnername],ddof=1) / np.sqrt(numparams)
        print("Standard error for " +learnername + ": " + str(sample_std))

    errorsList = []
    for i,j in classalgs.items():
        errorList = []
        #print i + ":"
        for error in range(numparams):
            for item in errors[i][error]:
                errorList.append(item)
        errorsList.append(errorList)
    print (errors)
    print (errorsList)
    t_test(errorsList)
