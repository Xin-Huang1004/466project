from __future__ import division  # floating point division
import numpy as np
import utilities as utils
import math
import random

class Classifier:
    """
    Generic classifier interface; returns random classification
    Assumes y in {0,1}, rather than {-1, 1}
    """

    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}

    def reset(self, parameters):
        """ Reset learner """
        self.resetparams(parameters)

    def resetparams(self, parameters):
        """ Can pass parameters to reset with new parameters """
        try:
            utils.update_dictionary_items(self.params,parameters)
        except AttributeError:
            # Variable self.params does not exist, so not updated
            # Create an empty set of params for future reference
            self.params = {}

    def getparams(self):
        return self.params

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """

    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])
        ytest = utils.threshold_probs(probs)
        return ytest

class LinearRegressionClass(Classifier):
    """
    Linear Regression with ridge regularization
    Simply solves (X.T X/t + lambda eye)^{-1} X.T y/t
    """
    def __init__( self, parameters={} ):
        self.params = {'regwgt': 0.01}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Ensure ytrain is {-1,1}
        yt = np.copy(ytrain)
        yt[yt == 0] = -1

        # Dividing by numsamples before adding ridge regularization
        # for additional stability; this also makes the
        # regularization parameter not dependent on numsamples
        # if want regularization disappear with more samples, must pass
        # such a regularization parameter lambda/t
        numsamples = Xtrain.shape[0]
        self.weights = np.dot(np.dot(np.linalg.pinv(np.add(np.dot(Xtrain.T,Xtrain)/numsamples,self.params['regwgt']*np.identity(Xtrain.shape[1]))), Xtrain.T),yt)/numsamples

    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        ytest[ytest > 0] = 1
        ytest[ytest < 0] = 0
        return ytest


class LogitReg(Classifier):

    def __init__(self, parameters={}):
        # Default: no regularization
        self.params = {'regwgt': 0.0, 'regularizer': 'None'}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None
        if self.params['regularizer'] is 'l1':
            self.regularizer = (utils.l1, utils.dl1)
        elif self.params['regularizer'] is 'l2':
            self.regularizer = (utils.l2, utils.dl2)
        else:
            self.regularizer = (lambda w: 0, lambda w: np.zeros(w.shape,))


    def logit_cost(self, theta, X, y):
        """
        Compute cost for logistic regression using theta as the parameters.
        """

        cost = 0.0

        ### YOUR CODE HERE

        Xw = np.dot(X,theta)
        sigXw = utils.sigmoid(Xw)
        #print sigXw
        cost_temp = np.dot(y, np.log(sigXw)) + (np.dot((1-y), np.log((1-sigXw))))
        cost = -(np.sum(cost_temp) / len(X))

        #cost = self.Err_(np.dot(X,theta),y)


        ### END YOUR CODE

        return cost


    def logit_cost_grad(self, theta, X, y):
        """
        Compute gradients of the cost with respect to theta.
        """

        grad = np.zeros(len(theta))

        ### YOUR CODE HERE

        sig = utils.sigmoid(np.dot(X,theta))
        grad = np.dot(X.T, np.subtract(sig,y))

        ### END YOUR CODE
        return grad


    def Err_(self,prediction,ytest):
        return utils.l2(np.subtract(utils.sigmoid(prediction),ytest))

    def learn(self, Xtrain, ytrain):
        """
        Learn the weights using the training data
        """

        self.weights = np.zeros(Xtrain.shape[1],)

        ### YOUR CODE HERE
        gradient = self.weights
        err = float("inf")
        tolerance = 0.00001
        while (abs(self.logit_cost(gradient, Xtrain, ytrain) - err) > tolerance):
            #update err, g
            err = self.logit_cost(gradient, Xtrain, ytrain)
            # update eta by line search
            eta = 0.00001
            # update seights
            gradient = np.subtract(gradient,np.dot(eta,self.logit_cost_grad(gradient,Xtrain,ytrain)))


        self.weights = gradient

        ### END YOUR CODE



    def predict(self, Xtest):
        """
        Use the parameters computed in self.learn to give predictions on new
        observations.
        """
        ytest = np.zeros(Xtest.shape[0], dtype=int)

        ### YOUR CODE HERE

        ytest = utils.sigmoid(np.dot(Xtest, self.weights))

        for i in range(len(ytest)):

            if ytest[i] >= 0.5:
                ytest[i] = 1
            if ytest[i] < 0.5:
                ytest[i] = 0

        ### END YOUR CODE

        assert len(ytest) == Xtest.shape[0]
        return ytest

class NeuralNet(Classifier):
    """ Implement a neural network with a single hidden layer. Cross entropy is
    used as the cost function.

    Parameters:
    nh -- number of hidden units
    transfer -- transfer function, in this case, sigmoid
    stepsize -- stepsize for gradient descent
    epochs -- learning epochs

    Note:
    1) feedforword will be useful! Make sure it can run properly.
    2) Implement the back-propagation algorithm with one layer in ``backprop`` without
    any other technique or trick or regularization. However, you can implement
    whatever you want outside ``backprob``.
    3) Set the best params you find as the default params. The performance with
    the default params will affect the points you get.
    """
    def __init__(self, parameters={}):
        self.params = {'nh': 8,
                    'transfer': 'sigmoid',
                    'stepsize': 0.01,
                    'epochs': 100}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        if self.params['transfer'] is 'sigmoid':
            self.transfer = utils.sigmoid
            self.dtransfer = utils.dsigmoid
        else:
            # For now, only allowing sigmoid transfer
            raise Exception('NeuralNet -> can only handle sigmoid transfer, must set option transfer to string sigmoid')
        self.w_input = None
        self.w_output = None

    def feedforward(self, inputs):
        """
        Returns the output of the current neural network for the given input
        """
        # hidden activations
        a_hidden = self.transfer(np.dot(self.w_input, inputs))

        # output activations
        a_output = self.transfer(np.dot(self.w_output, a_hidden))

        return (a_hidden, a_output)

    def backprop(self, x, y):
        """
        Return a tuple ``(nabla_input, nabla_output)`` representing the gradients
        for the cost function with respect to self.w_input and self.w_output.
        """

        ### YOUR CODE HERE

        # get hidden and y hat
        a_h, y_Hat = self.feedforward(x)

        # get delta1
        delta1 = (y_Hat - y)
        # get delta2
        delta2 =(np.dot(self.w_output.T, delta1) * (a_h * (1 - a_h)))
        # reshape delta2 and x
        delta2 = np.reshape(delta2, (self.nh,self.no))
        x = np.reshape(x, (self.no, self.ni))

        # get nabla input and output
        nabla_output = (delta1 * a_h)
        nabla_output = np.reshape(nabla_output, (self.no, self.nh))
        nabla_input = np.dot(delta2, x)

        ### END YOUR CODE

        assert nabla_input.shape == self.w_input.shape
        assert nabla_output.shape == self.w_output.shape
        return (nabla_input, nabla_output)

    # TODO: implement learn and predict functions
    def learn(self, Xtrain, ytrain):
        #print Xtrain
        #print ytrain
        zerocount = 0
        onecount = 0
        for i in ytrain:
            if(i == 1):
                zerocount+=1
            else:
                onecount+=1
        #print zerocount, onecount
        # set value
        self.stepsize = self.params['stepsize']
        self.epochs = self.params['epochs']
        # ni is number of input units
        # nh is number of hidden units
        # no is number of output units
        self.ni = Xtrain.shape[1]
        self.nh = self.params['nh']
        self.no = 1

        # w_input and w_output are two random matrix(value from -1 to 1)
        self.w_input = - 1 + 2 * np.random.random((self.nh,self.ni))
        self.w_output = - 1 + 2 * np.random.random((self.no,self.nh))

        # get shuffle list
        index_list = []
        for i in range(len(Xtrain)):
            index_list.append(i)
        np.random.shuffle(index_list)
        shuffle_x = [];
        shuffle_y = [];

        for epochs in range(self.epochs):
            # shuffle Xtrian and ytrian
            for item in range(len(Xtrain)):
                shuffle_x.append(Xtrain[index_list[item]])
                shuffle_y.append(ytrain[index_list[item]])

            for i in range(Xtrain.shape[0]):
                #update w_input and w_output
                n_input,n_output = self.backprop(shuffle_x[i], shuffle_y[i])
                self.w_input -= self.stepsize * n_input
                self.w_output -= self.stepsize * n_output

    def predict(self, Xtest):
        hidden = utils.sigmoid(np.dot(Xtest, self.w_input.T))
        ytest = utils.sigmoid(np.dot(hidden, self.w_output.T))

        for i in range(len(ytest)):
            #print Xtest[i],ytest[i]
            if ytest[i] <= 0.5:
                ytest[i] = 0
            else:
                ytest[i] = 1
        return ytest
